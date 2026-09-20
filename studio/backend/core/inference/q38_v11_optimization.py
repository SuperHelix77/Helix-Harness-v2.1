"""Safe V1.1 inference policy for Unsloth's stock llama.cpp runtime.

Q38 V1.1 adds exact-prefix state management around a pinned llama.cpp runtime. Unsloth already
has the compatible prompt/slot persistence primitives, so this port keeps the production
llama-server path (including SSE, tools, vision, and fallback recovery) and applies the measured
policy that matters most on Apple unified memory: spend the KV budget on one long-lived slot by
default. Explicit user settings always win.
"""

from __future__ import annotations

import re
from typing import Optional


Q38_V11_PROFILE = "q38-v1.1"
Q38_V11_PREFERRED_CONTEXT = 65_536
Q38_V11_PREFERRED_CACHE_TYPE_KV = "q4_0"
Q38_V11_PREFERRED_N_BATCH = 1_024
Q38_V11_PREFERRED_N_UBATCH = 256
# Measured Unsloth v1.1 path on M3 Max is speculation-off (~13 tok/s Q4).
# DFlash/MTP remain an explicit user opt-in; Auto must not rewrite to dflash.
Q38_V11_PREFERRED_SPECULATIVE_TYPE = None
Q38_V11_UPSTREAM_LLAMA_COMMIT = "4df29be4f4c3673f428170fda944a5b19f743bb8"


def q38_v11_is_qwen38(model_identifier: Optional[str]) -> bool:
    """Return whether an identifier names the Qwen3.8 family.

    The profile is deliberately scoped to this family. Applying its 64K/q4 KV policy to
    every GGUF would make unrelated models inherit a memory trade-off they were not tested
    with, while this is the model family the V1.1 measurements cover.
    """

    compact = re.sub(r"[^a-z0-9]+", "", str(model_identifier or "").lower())
    return "qwen38" in compact and "gguf" in compact


def q38_v11_default_load_updates(
    model_identifier: Optional[str],
    *,
    platform_name: str,
    max_seq_length: int,
    cache_type_kv: Optional[str],
    speculative_type: Optional[str],
    n_batch: Optional[int],
    n_ubatch: Optional[int],
) -> dict[str, object]:
    """Return safe V1.1 defaults for an unpinned Mac Qwen3.8 load.

    Positive/user-selected values remain authoritative. ``0``/``None``/``auto`` are the
    existing Unsloth sentinels for automatic sizing, so they receive the tested V1.1
    profile rather than silently falling back to the old short-context defaults.
    """

    if platform_name != "darwin" or not q38_v11_is_qwen38(model_identifier):
        return {}
    updates: dict[str, object] = {}
    if max_seq_length <= 0:
        updates["max_seq_length"] = Q38_V11_PREFERRED_CONTEXT
    if cache_type_kv is None:
        updates["cache_type_kv"] = Q38_V11_PREFERRED_CACHE_TYPE_KV
    # Leave speculative_type untouched. The ledger's measured 13 tok/s path is
    # ordinary Q4 decode; forcing DFlash on Auto was a Helix drift.
    _ = speculative_type
    if n_batch is None:
        updates["n_batch"] = Q38_V11_PREFERRED_N_BATCH
    if n_ubatch is None:
        updates["n_ubatch"] = Q38_V11_PREFERRED_N_UBATCH
    return updates


def q38_v11_default_extra_args(
    model_identifier: Optional[str],
    extra_args: Optional[list[str]],
    *,
    platform_name: str,
) -> Optional[list[str]]:
    """Add unified-KV to the scoped default while honoring an explicit opposite flag."""

    if platform_name != "darwin" or not q38_v11_is_qwen38(model_identifier):
        return extra_args
    args = list(extra_args or [])
    flags = {str(arg).split("=", 1)[0] for arg in args}
    if not ({"--kv-unified", "-kvu", "--no-kv-unified", "-no-kvu"} & flags):
        args.append("--kv-unified")
    return args


def q38_v11_default_parallel_slots(
    platform_name: str,
    requested_slots: Optional[int],
    configured_slots: int,
) -> int:
    """Choose a safe default slot count for a GGUF load.

    A caller-provided slot count is an explicit concurrency choice and is never changed. On
    Darwin, one slot leaves the context fitter enough KV capacity to reach the V1.1 64K target;
    other platforms retain their existing configured default.
    """

    if requested_slots is not None:
        return max(1, int(requested_slots))
    if platform_name == "darwin":
        return 1
    return max(1, int(configured_slots))


def q38_v11_context_target(native_context: Optional[int]) -> Optional[int]:
    """Return the preferred context target without exceeding a model's native window."""

    if native_context is None or native_context <= 0:
        return Q38_V11_PREFERRED_CONTEXT
    return min(Q38_V11_PREFERRED_CONTEXT, int(native_context))
