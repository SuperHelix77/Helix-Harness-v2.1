# SPDX-License-Identifier: AGPL-3.0-only
"""Pressure-aware automatic context selection for local inference.

The governor is deliberately a policy layer rather than another loader. Existing
model policies propose a context; this module may lower that *automatic* proposal
when the current machine cannot defend it. A positive user-selected context is
never rewritten by callers.

Signals are best-effort. macOS unified memory does not have one trustworthy
"free RAM" number, so decisions combine host availability, swap/compression,
Metal allocation when MLX is already imported, and bounded empirical history.
The selected band is sticky for the process and upgrades require several clean
observations, preventing 24K -> 16K -> 24K oscillation near a pressure boundary.
"""

from __future__ import annotations

import json
import math
import platform
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from utils.paths import account_path, ensure_dir

ContextBand = Literal["safe", "balanced", "max"]

_LOCK = threading.RLock()
_SESSION: dict[str, "SessionState"] = {}
_ACTIVE_PROFILES: dict[str, "ContextDecision"] = {}
_MAX_HISTORY = 64
_HISTORY_VERSION = 1
_MIN_CONTEXT = 4096
_CONTEXT_GRANULARITY = 1024
_QUALIFIED_CONTEXT_RATIO = 0.80


def backend_kind_before_model_config(
    *,
    model_identifier: str,
    gguf_variant: str | None,
    host_serves_mlx: bool,
) -> Literal["llama.cpp", "mlx", "transformers"]:
    """Classify the serving backend using only pre-ModelConfig load evidence.

    Automatic context governance intentionally runs before model metadata resolution
    so its selected window participates in the already-loaded dedupe.  This helper
    makes that ordering explicit and prevents an accidental read of the not-yet-bound
    ``config`` local from disabling the governor on every fresh load.
    """

    if gguf_variant or str(model_identifier or "").lower().endswith(".gguf"):
        return "llama.cpp"
    return "mlx" if host_serves_mlx else "transformers"


@dataclass(frozen=True)
class PressureSample:
    total_bytes: int | None = None
    available_bytes: int | None = None
    swap_used_bytes: int | None = None
    swap_in_bytes: int | None = None
    swap_out_bytes: int | None = None
    compressed_bytes: int | None = None
    metal_active_bytes: int | None = None
    metal_cache_bytes: int | None = None
    metal_peak_bytes: int | None = None
    captured_at_ms: int = 0

    @property
    def available_ratio(self) -> float | None:
        if not self.total_bytes or self.total_bytes <= 0 or self.available_bytes is None:
            return None
        return max(0.0, min(1.0, self.available_bytes / self.total_bytes))

    @property
    def swap_ratio(self) -> float | None:
        if not self.total_bytes or self.total_bytes <= 0 or self.swap_used_bytes is None:
            return None
        return max(0.0, self.swap_used_bytes / self.total_bytes)

    @property
    def compression_ratio(self) -> float | None:
        if not self.total_bytes or self.total_bytes <= 0 or self.compressed_bytes is None:
            return None
        return max(0.0, self.compressed_bytes / self.total_bytes)


@dataclass(frozen=True)
class ContextDecision:
    context_length: int
    band: ContextBand
    pressure_level: Literal["normal", "elevated", "critical"]
    reason: str
    profile_key: str
    proposed_context: int
    empirical_cap: int | None = None


@dataclass
class SessionState:
    band: ContextBand
    safe_streak: int = 0
    elevated_streak: int = 0
    last_sample: PressureSample | None = None


def _history_path() -> Path:
    path = account_path("runtime/context-governor-v1.json")
    ensure_dir(path.parent)
    return path


def _bounded_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return max(0, parsed)


def _vm_stat_sample() -> tuple[int | None, int | None, int | None]:
    """Return compressed bytes, swapins bytes, swapouts bytes on Darwin."""
    if sys.platform != "darwin":
        return None, None, None
    try:
        result = subprocess.run(
            ["/usr/bin/vm_stat"],
            capture_output=True,
            text=True,
            check=False,
            timeout=0.25,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None, None
    if result.returncode != 0:
        return None, None, None
    page_match = re.search(r"page size of\s+(\d+) bytes", result.stdout)
    page_size = int(page_match.group(1)) if page_match else 4096
    values: dict[str, int] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        match = re.search(r"(\d+)", raw.replace(".", ""))
        if match:
            values[name.strip().casefold()] = int(match.group(1)) * page_size
    compressed = values.get("pages occupied by compressor")
    swap_in = values.get("swapins")
    swap_out = values.get("swapouts")
    return compressed, swap_in, swap_out


def _mlx_memory_sample() -> tuple[int | None, int | None, int | None]:
    """Read Metal counters only when MLX is already resident; never import it for telemetry."""
    mx = sys.modules.get("mlx.core")
    if mx is None:
        return None, None, None

    def call(name: str) -> int | None:
        fn = getattr(mx, name, None)
        if not callable(fn):
            metal = getattr(mx, "metal", None)
            fn = getattr(metal, name, None) if metal is not None else None
        if not callable(fn):
            return None
        try:
            return _bounded_int(fn())
        except Exception:
            return None

    return call("get_active_memory"), call("get_cache_memory"), call("get_peak_memory")


def sample_runtime_pressure() -> PressureSample:
    total = available = swap_used = swap_in = swap_out = None
    try:
        import psutil

        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        total = _bounded_int(getattr(virtual, "total", None))
        available = _bounded_int(getattr(virtual, "available", None))
        swap_used = _bounded_int(getattr(swap, "used", None))
        swap_in = _bounded_int(getattr(swap, "sin", None))
        swap_out = _bounded_int(getattr(swap, "sout", None))
    except Exception:
        pass
    compressed, vm_swap_in, vm_swap_out = _vm_stat_sample()
    if vm_swap_in is not None:
        swap_in = vm_swap_in
    if vm_swap_out is not None:
        swap_out = vm_swap_out
    metal_active, metal_cache, metal_peak = _mlx_memory_sample()
    return PressureSample(
        total_bytes=total,
        available_bytes=available,
        swap_used_bytes=swap_used,
        swap_in_bytes=swap_in,
        swap_out_bytes=swap_out,
        compressed_bytes=compressed,
        metal_active_bytes=metal_active,
        metal_cache_bytes=metal_cache,
        metal_peak_bytes=metal_peak,
        captured_at_ms=int(time.time() * 1000),
    )


def classify_pressure(sample: PressureSample, previous: PressureSample | None = None) -> Literal[
    "normal", "elevated", "critical"
]:
    """Conservative multi-signal pressure classification.

    Availability alone can only elevate; critical requires either a very small
    reserve plus another pressure signal, or unmistakable swap/compression growth.
    """
    available = sample.available_ratio
    swap = sample.swap_ratio
    compression = sample.compression_ratio
    metal_active = (
        sample.metal_active_bytes / sample.total_bytes
        if sample.total_bytes and sample.total_bytes > 0 and sample.metal_active_bytes is not None
        else None
    )
    metal_peak = (
        sample.metal_peak_bytes / sample.total_bytes
        if sample.total_bytes and sample.total_bytes > 0 and sample.metal_peak_bytes is not None
        else None
    )
    swap_growth = 0
    if previous is not None and sample.swap_out_bytes is not None and previous.swap_out_bytes is not None:
        swap_growth = max(0, sample.swap_out_bytes - previous.swap_out_bytes)
    compressed_growth = 0
    if (
        previous is not None
        and sample.compressed_bytes is not None
        and previous.compressed_bytes is not None
    ):
        compressed_growth = max(0, sample.compressed_bytes - previous.compressed_bytes)

    secondary = bool(
        (swap is not None and swap >= 0.08)
        or (compression is not None and compression >= 0.18)
        or (metal_active is not None and metal_active >= 0.78)
        or (metal_peak is not None and metal_peak >= 0.88)
        or swap_growth >= 256 * 1024**2
        or compressed_growth >= 512 * 1024**2
    )
    if (available is not None and available <= 0.08 and secondary) or (
        swap_growth >= 1024 * 1024**2
    ):
        return "critical"
    if (
        (available is not None and available <= 0.18)
        or (swap is not None and swap >= 0.05)
        or (compression is not None and compression >= 0.12)
        or (metal_active is not None and metal_active >= 0.72)
        or (metal_peak is not None and metal_peak >= 0.82)
        or swap_growth > 0
        or compressed_growth >= 128 * 1024**2
    ):
        return "elevated"
    return "normal"


def _round_context(value: int) -> int:
    return max(_MIN_CONTEXT, (max(_MIN_CONTEXT, int(value)) // _CONTEXT_GRANULARITY) * _CONTEXT_GRANULARITY)


def _profile_contexts(proposed: int, empirical_cap: int | None) -> dict[ContextBand, int]:
    ceiling = min(proposed, empirical_cap) if empirical_cap else proposed
    ceiling = _round_context(ceiling)
    return {
        "safe": min(ceiling, _round_context(max(_MIN_CONTEXT, ceiling * 0.5))),
        "balanced": min(ceiling, _round_context(max(_MIN_CONTEXT, ceiling * 0.75))),
        "max": ceiling,
    }


def exact_profile_key(
    *,
    model_id: str,
    backend: str,
    kv: str | int | None,
    speculative: str | None,
    total_bytes: int | None,
    runtime_version: str = "",
) -> str:
    total_gib = round((total_bytes or 0) / 1024**3)
    machine = f"{platform.system().lower()}-{platform.machine().lower()}-{total_gib}g"
    return "|".join(
        (
            machine,
            str(model_id).strip().casefold(),
            str(backend).strip().casefold(),
            str(kv or "auto").strip().casefold(),
            str(speculative or "auto").strip().casefold(),
            str(runtime_version or "unknown").strip().casefold(),
        )
    )


def seeded_empirical_cap(model_id: str, sample: PressureSample) -> int | None:
    """Return a qualification-backed cap for an exact machine/model class.

    Helix's M3 Max 36-GB qualification established Qwen3.8-27B at 32K with
    q4/4-bit KV as the practical maximum: the 26.2K occupancy checkpoint left
    only about eleven percent free memory during prefill. This seed is deliberately
    narrow; every other configuration must learn its own cap from observations.
    """
    if sys.platform != "darwin" or not sample.total_bytes:
        return None
    total_gib = sample.total_bytes / 1024**3
    compact = re.sub(r"[^a-z0-9]+", "", str(model_id or "").casefold())
    if not (34.0 <= total_gib <= 38.5):
        return None
    if "qwen38" not in compact or "27b" not in compact:
        return None
    return 32_768


def activate_profile(decision: ContextDecision, *model_ids: str) -> None:
    """Associate the loaded runtime with the decision that created it."""
    with _LOCK:
        for model_id in model_ids:
            key = str(model_id or "").strip().casefold()
            if key:
                _ACTIVE_PROFILES[key] = decision


def active_profile(model_id: str) -> ContextDecision | None:
    with _LOCK:
        return _ACTIVE_PROFILES.get(str(model_id or "").strip().casefold())


def _load_history() -> dict[str, Any]:
    try:
        data = json.loads(_history_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, UnicodeError):
        return {"version": _HISTORY_VERSION, "profiles": {}}
    if not isinstance(data, dict) or not isinstance(data.get("profiles"), dict):
        return {"version": _HISTORY_VERSION, "profiles": {}}
    return data


def _save_history(data: dict[str, Any]) -> None:
    path = _history_path()
    profiles = data.get("profiles") if isinstance(data.get("profiles"), dict) else {}
    ordered = sorted(
        profiles.items(),
        key=lambda item: int((item[1] if isinstance(item[1], dict) else {}).get("updated_at_ms") or 0),
        reverse=True,
    )[:_MAX_HISTORY]
    payload = {"version": _HISTORY_VERSION, "profiles": dict(ordered)}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def empirical_cap(profile_key: str) -> int | None:
    with _LOCK:
        entry = _load_history().get("profiles", {}).get(profile_key)
    if not isinstance(entry, dict):
        return None
    return _bounded_int(entry.get("validated_max_context")) or None


def empirical_recommended_band(profile_key: str) -> ContextBand | None:
    with _LOCK:
        entry = _load_history().get("profiles", {}).get(profile_key)
    if not isinstance(entry, dict):
        return None
    band = str(entry.get("recommended_band") or "")
    return band if band in {"safe", "balanced", "max"} else None


def _more_conservative(left: ContextBand, right: ContextBand) -> ContextBand:
    rank = {"safe": 0, "balanced": 1, "max": 2}
    return left if rank[left] <= rank[right] else right


def choose_automatic_context(
    *,
    model_id: str,
    proposed_context: int,
    backend: str,
    kv: str | int | None = None,
    speculative: str | None = None,
    runtime_version: str = "",
    sample: PressureSample | None = None,
    previous_sample: PressureSample | None = None,
    known_empirical_cap: int | None = None,
) -> ContextDecision:
    """Choose a sticky context band for one automatic load proposal."""
    proposed = _round_context(proposed_context)
    current = sample or sample_runtime_pressure()
    key = exact_profile_key(
        model_id=model_id,
        backend=backend,
        kv=kv,
        speculative=speculative,
        total_bytes=current.total_bytes,
        runtime_version=runtime_version,
    )
    history_cap = empirical_cap(key)
    history_band = empirical_recommended_band(key)
    cap_candidates = [value for value in (known_empirical_cap, history_cap) if value and value > 0]
    cap = min(cap_candidates) if cap_candidates else None
    contexts = _profile_contexts(proposed, cap)

    with _LOCK:
        state = _SESSION.get(key)
        effective_previous = previous_sample or (state.last_sample if state is not None else None)
        pressure = classify_pressure(current, effective_previous)
        if state is None:
            band: ContextBand = "max"
            if pressure == "critical":
                band = "safe"
            elif pressure == "elevated":
                band = "balanced"
            if history_band is not None:
                band = _more_conservative(band, history_band)
            state = SessionState(band=band, last_sample=current)
            _SESSION[key] = state
        elif pressure == "critical":
            state.band = "safe"
            state.safe_streak = 0
            state.elevated_streak = 0
        elif pressure == "elevated":
            state.safe_streak = 0
            state.elevated_streak += 1
            if state.elevated_streak >= 2 and state.band == "max":
                state.band = "balanced"
            elif state.elevated_streak >= 2 and state.band == "balanced":
                state.band = "safe"
        else:
            state.elevated_streak = 0
            state.safe_streak += 1
            # Upgrades deliberately require multiple clean observations. A single
            # post-unload sample must not immediately undo a pressure downgrade.
            if state.safe_streak >= 3:
                if state.band == "safe":
                    state.band = "balanced"
                    state.safe_streak = 0
                elif state.band == "balanced":
                    state.band = "max"
                    state.safe_streak = 0
        state.last_sample = current
        band = state.band

    context = contexts[band]
    reason = f"pressure={pressure}; band={band}"
    if cap:
        reason += f"; empirical_cap={cap}"
    return ContextDecision(
        context_length=context,
        band=band,
        pressure_level=pressure,
        reason=reason,
        profile_key=key,
        proposed_context=proposed,
        empirical_cap=cap,
    )


def record_observation(
    *,
    profile_key: str,
    context_length: int,
    occupancy_tokens: int | None = None,
    sample: PressureSample | None = None,
    outcome: Literal["ok", "near_oom", "oom"] = "ok",
    qualified_boundary: bool = False,
) -> None:
    """Persist bounded empirical evidence for a concrete runtime configuration."""
    current = sample or sample_runtime_pressure()
    with _LOCK:
        data = _load_history()
        profiles = data.setdefault("profiles", {})
        entry = profiles.get(profile_key)
        if not isinstance(entry, dict):
            entry = {}
        validated = _bounded_int(entry.get("validated_max_context")) or 0
        near_oom = _bounded_int(entry.get("near_oom_count")) or 0
        oom = _bounded_int(entry.get("oom_count")) or 0
        clean_streak = _bounded_int(entry.get("clean_streak")) or 0
        recommended = str(entry.get("recommended_band") or "max")
        if recommended not in {"safe", "balanced", "max"}:
            recommended = "max"
        occupancy = _bounded_int(occupancy_tokens)
        qualified = bool(
            qualified_boundary
            or (
                occupancy is not None
                and int(context_length) > 0
                and occupancy / int(context_length) >= _QUALIFIED_CONTEXT_RATIO
            )
        )
        if outcome == "ok":
            # Merely loading a large window and using a small fraction of it is
            # not evidence that the host can safely serve the whole window.
            # Promote the empirical cap only from a qualified boundary/stress
            # observation. Pressure failures may still downgrade it below.
            if qualified:
                validated = max(validated, int(context_length))
            clean_streak += 1
            # Persisted hysteresis mirrors the session policy. Three clean exact
            # boundary observations permit one step back toward max; a single
            # post-pressure turn can never erase a learned downgrade.
            if clean_streak >= 3:
                if recommended == "safe":
                    recommended = "balanced"
                    clean_streak = 0
                elif recommended == "balanced":
                    recommended = "max"
                    clean_streak = 0
        elif outcome == "near_oom":
            near_oom += 1
            clean_streak = 0
            recommended = "balanced" if recommended == "max" else "safe"
            # A pressure boundary is a ceiling, not proof that a shorter context
            # failed. Keep the last validated context but never learn this one up.
        else:
            oom += 1
            clean_streak = 0
            recommended = "safe"
            if validated >= int(context_length):
                validated = max(_MIN_CONTEXT, int(context_length) - _CONTEXT_GRANULARITY)
        entry.update(
            {
                "validated_max_context": validated or None,
                "near_oom_count": near_oom,
                "oom_count": oom,
                "clean_streak": clean_streak,
                "recommended_band": recommended,
                "last_context": int(context_length),
                "last_occupancy": occupancy,
                "last_qualified_boundary": qualified,
                "last_pressure": classify_pressure(current),
                "last_sample": asdict(current),
                "updated_at_ms": int(time.time() * 1000),
            }
        )
        profiles[profile_key] = entry
        _save_history(data)


def record_active_observation(
    *,
    model_id: str,
    context_length: int,
    occupancy_tokens: int | None = None,
    sample: PressureSample | None = None,
    outcome: Literal["ok", "near_oom", "oom"] | None = None,
    qualified_boundary: bool = False,
) -> bool:
    """Record an observation against the active load decision, when one exists."""
    decision = active_profile(model_id)
    if decision is None:
        return False
    current = sample or sample_runtime_pressure()
    with _LOCK:
        state = _SESSION.get(decision.profile_key)
        previous = state.last_sample if state is not None else None
    resolved_outcome = outcome
    if resolved_outcome is None:
        resolved_outcome = (
            "near_oom" if classify_pressure(current, previous) != "normal" else "ok"
        )
    record_observation(
        profile_key=decision.profile_key,
        context_length=context_length,
        occupancy_tokens=occupancy_tokens,
        sample=current,
        outcome=resolved_outcome,
        qualified_boundary=qualified_boundary,
    )
    with _LOCK:
        state = _SESSION.get(decision.profile_key)
        if state is not None:
            if resolved_outcome == "oom":
                state.band = "safe"
                state.safe_streak = 0
                state.elevated_streak = 0
            elif resolved_outcome == "near_oom":
                state.safe_streak = 0
                state.elevated_streak += 1
                if state.band == "max":
                    state.band = "balanced"
                elif state.band == "balanced" and state.elevated_streak >= 2:
                    state.band = "safe"
            else:
                state.elevated_streak = 0
                state.safe_streak += 1
                if state.safe_streak >= 3:
                    if state.band == "safe":
                        state.band = "balanced"
                        state.safe_streak = 0
                    elif state.band == "balanced":
                        state.band = "max"
                        state.safe_streak = 0
            state.last_sample = current
    return True


def reset_session_state_for_tests() -> None:
    with _LOCK:
        _SESSION.clear()
        _ACTIVE_PROFILES.clear()
