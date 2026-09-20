# SPDX-License-Identifier: AGPL-3.0-only
"""Helix Harness 5x DFlash/GDN path vs frozen v1.1 speculation-off baseline."""

from core.inference.helix_speed_policy import (
    helix_5x_load_updates,
    helix_gdn_host_path,
    helix_may_claim_5x,
    helix_speculation_fail_open,
    helix_speed_ratio,
    write_speed_json,
)
from core.inference.q38_v11_optimization import (
    Q38_V11_PREFERRED_CONTEXT,
    Q38_V11_PREFERRED_N_BATCH,
    Q38_V11_PREFERRED_N_UBATCH,
)


def test_helix_5x_auto_selects_dflash_on_darwin_qwen38_and_keeps_v11_knobs():
    updates = helix_5x_load_updates(
        "unsloth/Qwen3.8-27B-GGUF",
        platform_name="darwin",
        max_seq_length=0,
        cache_type_kv=None,
        speculative_type="auto",
        n_batch=None,
        n_ubatch=None,
    )
    assert updates["max_seq_length"] == Q38_V11_PREFERRED_CONTEXT
    assert updates["cache_type_kv"] == "q4_0"
    assert updates["n_batch"] == Q38_V11_PREFERRED_N_BATCH
    assert updates["n_ubatch"] == Q38_V11_PREFERRED_N_UBATCH
    assert updates["speculative_type"] == "dflash"
    assert helix_gdn_host_path("unsloth/Qwen3.8-27B-GGUF", "darwin") is True
    assert helix_gdn_host_path("unsloth/Qwen3.8-27B-GGUF", "linux") is False


def test_helix_5x_explicit_off_is_authoritative():
    updates = helix_5x_load_updates(
        "unsloth/Qwen3.8-27B-GGUF",
        platform_name="darwin",
        max_seq_length=65536,
        cache_type_kv="q4_0",
        speculative_type="off",
        n_batch=1024,
        n_ubatch=256,
    )
    assert updates.get("speculative_type") != "dflash"


def test_helix_speculation_fail_open_when_sidecar_missing_or_zero_drafts():
    assert helix_speculation_fail_open("dflash", "dflash", accepted_drafts=12, sidecar_ok=False) == "off"
    assert helix_speculation_fail_open("dflash", "off", accepted_drafts=0, sidecar_ok=True) == "off"
    assert helix_speculation_fail_open("dflash", "dflash", accepted_drafts=0, sidecar_ok=True) == "off"
    assert helix_speculation_fail_open("dflash", "dflash", accepted_drafts=40, sidecar_ok=True) == "dflash"


def test_helix_may_not_claim_5x_without_accepted_drafts():
    assert helix_speed_ratio(13.0, 65.0) == 5.0
    assert helix_may_claim_5x(5.0, accepted_drafts=0) is False
    assert helix_may_claim_5x(5.0, accepted_drafts=12) is True
    assert helix_may_claim_5x(4.9, accepted_drafts=100) is False


def test_write_speed_json_records_honest_failure(tmp_path):
    path = tmp_path / "speed.json"
    payload = write_speed_json(
        path,
        baseline_tok_s=None,
        candidate_tok_s=None,
        accepted_drafts=0,
        error="model not loaded",
    )
    assert payload["ratio"] is None
    assert payload["claim_5x"] is False
    assert payload["error"] == "model not loaded"
    assert path.is_file()
