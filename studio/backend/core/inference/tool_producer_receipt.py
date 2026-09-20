# SPDX-License-Identifier: AGPL-3.0-only
"""Private, backend-authored evidence for local process tool results.

The classes in this module are deliberately not part of the tool/model schema.
Only the backend can attach a :class:`ProducerReceipt` to a string result, so
text printed by a command cannot impersonate receipt metadata.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator


PRODUCER_RECEIPT_VERSION = 1
OBSERVATION_SEED_MIN_BYTES = 10 * 1024
OBSERVATION_SEED_MAX_BYTES = 8 * 1024 * 1024
RESULT_BUDGET_PRICING_MODES = frozenset(
    {"unpriced", "measured_model", "conservative_estimate"}
)


def _utf8_facts(text: str) -> tuple[str, int]:
    encoded = text.encode("utf-8", "surrogatepass")
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


def _directory_identity(path: str | None) -> dict[str, int] | None:
    """Return a directory identity observed at this instant, when available."""

    if not path:
        return None
    try:
        stat = os.stat(path, follow_symlinks=True)
    except (OSError, TypeError, ValueError):
        return None
    if not os.path.isdir(path):
        return None
    return {"device": int(stat.st_dev), "inode": int(stat.st_ino)}


@dataclass(frozen=True)
class LaunchCwdObservation:
    """Best-effort launch-CWD facts sampled before the producer calls Popen.

    This does not claim that the child necessarily entered the same inode: a
    path can be replaced between this observation and the kernel resolving the
    child's cwd.  The phase is explicit so consumers cannot over-read it.
    """

    resolved_path_observed_before_spawn: str | None
    identity_observed_before_spawn: dict[str, int] | None


def observe_launch_cwd(path: str | None) -> LaunchCwdObservation:
    return LaunchCwdObservation(
        resolved_path_observed_before_spawn=(os.path.realpath(path) if path else None),
        identity_observed_before_spawn=_directory_identity(path),
    )


@dataclass(frozen=True)
class ProcessOutputCapture:
    """Bounded capture plus facts accumulated while the whole pipe was drained.

    Iteration preserves the historical ``output, timed_out = drain(...)`` API.
    Consumers that need evidence use the named fields instead of parsing a cap
    notice from the returned text.
    """

    output: str
    timed_out: bool
    cancelled: bool
    complete: bool
    output_digest: str
    output_byte_length: int
    captured_byte_length: int
    discarded_byte_length: int
    capture_budget_chars: int
    read_error: str | None = None

    def __iter__(self) -> Iterator[Any]:
        yield self.output
        yield self.timed_out


@dataclass(frozen=True)
class ResultBudgetExposure:
    """Private pricing facts latched at the ordinary producer fit.

    ``active_result_budget_tokens`` is the caller-provided remaining room for
    this result, not the historical character ceiling.  The served context
    and pricing branch travel with it because a later observation publisher
    must not reinterpret an integer using a different request or tokenizer
    assumption.  This object is intentionally never serialized.
    """

    active_result_budget_tokens: int | None
    served_context_tokens: int | None
    pricing_mode: str

    def is_valid(self) -> bool:
        if self.pricing_mode not in RESULT_BUDGET_PRICING_MODES:
            return False
        if self.active_result_budget_tokens is not None and (
            not isinstance(self.active_result_budget_tokens, int)
            or isinstance(self.active_result_budget_tokens, bool)
            or self.active_result_budget_tokens < 0
        ):
            return False
        if self.served_context_tokens is not None and (
            not isinstance(self.served_context_tokens, int)
            or isinstance(self.served_context_tokens, bool)
            or self.served_context_tokens <= 0
        ):
            return False
        if self.active_result_budget_tokens is None:
            return self.pricing_mode == "unpriced"
        if self.pricing_mode == "measured_model" and self.served_context_tokens is None:
            return False
        return self.pricing_mode != "unpriced"


@dataclass(frozen=True)
class ObservationSeed:
    """Private canonical output candidate for a future ObservationPack.

    A seed is intentionally carried only in-process.  It is not part of the
    producer receipt, durable event, model-facing result, or any other
    serialized contract.  The eventual ObservationPack builder must verify
    every field again against the authoritative execution receipt before it can
    publish evidence.
    """

    canonical_text: str
    canonical_text_utf8_surrogatepass_sha256: str
    canonical_text_utf8_surrogatepass_byte_length: int
    result_budget_exposure: ResultBudgetExposure = field(
        default_factory=lambda: ResultBudgetExposure(
            active_result_budget_tokens=None,
            served_context_tokens=None,
            pricing_mode="unpriced",
        )
    )

    def is_valid(self) -> bool:
        digest, length = _utf8_facts(self.canonical_text)
        return (
            self.canonical_text_utf8_surrogatepass_sha256 == digest
            and self.canonical_text_utf8_surrogatepass_byte_length == length
            and OBSERVATION_SEED_MIN_BYTES <= length <= OBSERVATION_SEED_MAX_BYTES
            and isinstance(self.result_budget_exposure, ResultBudgetExposure)
            and self.result_budget_exposure.is_valid()
        )


def observation_seed_for_text(
    text: str | None,
    *,
    result_budget_exposure: ResultBudgetExposure | None = None,
) -> ObservationSeed | None:
    """Build a bounded private seed from canonical post-defuse text."""

    if not isinstance(text, str):
        return None
    digest, length = _utf8_facts(text)
    if not OBSERVATION_SEED_MIN_BYTES <= length <= OBSERVATION_SEED_MAX_BYTES:
        return None
    exposure = result_budget_exposure or ResultBudgetExposure(
        active_result_budget_tokens=None,
        served_context_tokens=None,
        pricing_mode="unpriced",
    )
    if not isinstance(exposure, ResultBudgetExposure) or not exposure.is_valid():
        return None
    return ObservationSeed(
        canonical_text=text,
        canonical_text_utf8_surrogatepass_sha256=digest,
        canonical_text_utf8_surrogatepass_byte_length=length,
        result_budget_exposure=exposure,
    )


@dataclass(frozen=True)
class ProducerReceipt:
    schema_version: str
    producer_version: int
    tool: str
    resolved_launch_cwd_observed_before_spawn: str | None
    cwd_identity_observed_before_spawn: dict[str, int] | None
    confinement: str
    process_outcome_kind: str
    return_code: int | None
    return_code_available: bool
    timed_out: bool
    cancelled: bool
    spawn_error: bool
    decoded_output_utf8_surrogatepass_sha256: str
    decoded_output_utf8_surrogatepass_byte_length: int
    capture_complete: bool
    captured_byte_length: int
    discarded_byte_length: int
    capture_budget_chars: int
    read_error: str | None
    fallback_result_utf8_surrogatepass_sha256: str
    fallback_result_utf8_surrogatepass_byte_length: int
    fallback_budget_chars: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProducedToolResult(str):
    """String-compatible result carrying unforgeable in-process receipt data."""

    __slots__ = ("_producer_receipt", "_observation_seed")

    def __new__(
        cls,
        value: str,
        receipt: ProducerReceipt,
        observation_seed: ObservationSeed | None = None,
    ):
        instance = super().__new__(cls, value)
        instance._producer_receipt = receipt
        instance._observation_seed = observation_seed
        return instance

    @property
    def producer_receipt(self) -> ProducerReceipt:
        return self._producer_receipt

    @property
    def observation_seed(self) -> ObservationSeed | None:
        return self._observation_seed


def produced_result(
    value: str,
    *,
    tool: str,
    workdir: str | None,
    cwd_observation: LaunchCwdObservation | None = None,
    confinement: str,
    outcome_kind: str,
    return_code: int | None,
    timed_out: bool = False,
    cancelled: bool = False,
    spawn_error: bool = False,
    capture: ProcessOutputCapture | None = None,
    fallback_budget_chars: int | None = None,
    observation_seed: ObservationSeed | None = None,
) -> ProducedToolResult:
    """Attach one typed receipt without changing any external string behavior."""

    # Keep the attachment policy at the private transport boundary as well as
    # at each producer call site.  A future caller cannot accidentally attach a
    # seed to a bypass, failed, incomplete, or otherwise ineligible result by
    # merely passing an ObservationSeed through this helper.
    if not (
        isinstance(observation_seed, ObservationSeed)
        and observation_seed.is_valid()
        and str(tool) in {"python", "terminal"}
        and str(confinement) != "bypass"
        and str(outcome_kind) == "exited"
        and return_code == 0
        and not timed_out
        and not cancelled
        and not spawn_error
        and capture is not None
        and capture.complete
        and capture.read_error is None
    ):
        observation_seed = None

    fallback_digest, fallback_length = _utf8_facts(str(value))
    if capture is None:
        output_digest, output_length = _utf8_facts("")
        captured_length = 0
        discarded_length = 0
        capture_budget = 0
        capture_complete = outcome_kind not in {"spawn_error", "read_error"}
        read_error = None
    else:
        output_digest = capture.output_digest
        output_length = capture.output_byte_length
        captured_length = capture.captured_byte_length
        discarded_length = capture.discarded_byte_length
        capture_budget = capture.capture_budget_chars
        capture_complete = capture.complete
        read_error = capture.read_error
    observed_cwd = cwd_observation or observe_launch_cwd(workdir)
    receipt = ProducerReceipt(
        schema_version="helix.tool-producer-receipt.v1",
        producer_version=PRODUCER_RECEIPT_VERSION,
        tool=str(tool or "unknown"),
        resolved_launch_cwd_observed_before_spawn=(
            observed_cwd.resolved_path_observed_before_spawn
        ),
        cwd_identity_observed_before_spawn=(
            observed_cwd.identity_observed_before_spawn
        ),
        confinement=confinement,
        process_outcome_kind=outcome_kind,
        return_code=return_code,
        return_code_available=return_code is not None,
        timed_out=bool(timed_out),
        cancelled=bool(cancelled),
        spawn_error=bool(spawn_error),
        decoded_output_utf8_surrogatepass_sha256=output_digest,
        decoded_output_utf8_surrogatepass_byte_length=output_length,
        capture_complete=bool(capture_complete),
        captured_byte_length=captured_length,
        discarded_byte_length=discarded_length,
        capture_budget_chars=capture_budget,
        read_error=read_error,
        fallback_result_utf8_surrogatepass_sha256=fallback_digest,
        fallback_result_utf8_surrogatepass_byte_length=fallback_length,
        fallback_budget_chars=fallback_budget_chars,
    )
    return ProducedToolResult(str(value), receipt, observation_seed)


def receipt_dict(value: Any) -> dict[str, Any] | None:
    receipt = getattr(value, "producer_receipt", None)
    if isinstance(receipt, ProducerReceipt):
        return receipt.to_dict()
    return None
