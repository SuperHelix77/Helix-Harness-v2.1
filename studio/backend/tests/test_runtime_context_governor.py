# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from core.inference import runtime_context_governor as governor

GIB = 1024**3


def _sample(*, available_gib: float, swap_gib: float = 0, compressed_gib: float = 0):
    return governor.PressureSample(
        total_bytes=36 * GIB,
        available_bytes=int(available_gib * GIB),
        swap_used_bytes=int(swap_gib * GIB),
        compressed_bytes=int(compressed_gib * GIB),
        swap_out_bytes=0,
        captured_at_ms=1,
    )


def test_pressure_does_not_call_low_free_memory_critical_by_itself():
    assert governor.classify_pressure(_sample(available_gib=5.0)) == "elevated"
    assert governor.classify_pressure(_sample(available_gib=2.0)) == "elevated"
    assert (
        governor.classify_pressure(_sample(available_gib=2.0, swap_gib=4.0))
        == "critical"
    )


def test_pre_config_backend_classification_needs_no_model_config():
    assert (
        governor.backend_kind_before_model_config(
            model_identifier="org/model.gguf",
            gguf_variant=None,
            host_serves_mlx=True,
        )
        == "llama.cpp"
    )
    assert (
        governor.backend_kind_before_model_config(
            model_identifier="org/model",
            gguf_variant="Q4_K_M",
            host_serves_mlx=True,
        )
        == "llama.cpp"
    )
    assert (
        governor.backend_kind_before_model_config(
            model_identifier="mlx-community/Qwen3.8-27B-4bit",
            gguf_variant=None,
            host_serves_mlx=True,
        )
        == "mlx"
    )
    assert (
        governor.backend_kind_before_model_config(
            model_identifier="org/safetensors-model",
            gguf_variant=None,
            host_serves_mlx=False,
        )
        == "transformers"
    )


def test_swap_and_compression_growth_raise_pressure():
    before = governor.PressureSample(
        total_bytes=36 * GIB,
        available_bytes=12 * GIB,
        swap_out_bytes=0,
        compressed_bytes=2 * GIB,
    )
    elevated = governor.PressureSample(
        total_bytes=36 * GIB,
        available_bytes=12 * GIB,
        swap_out_bytes=300 * 1024**2,
        compressed_bytes=2 * GIB + 256 * 1024**2,
    )
    critical = governor.PressureSample(
        total_bytes=36 * GIB,
        available_bytes=12 * GIB,
        swap_out_bytes=2 * GIB,
        compressed_bytes=2 * GIB,
    )
    assert governor.classify_pressure(elevated, before) == "elevated"
    assert governor.classify_pressure(critical, before) == "critical"


def test_qualified_27b_36gb_profile_seeds_32k_only(monkeypatch):
    monkeypatch.setattr(governor.sys, "platform", "darwin")
    sample = _sample(available_gib=20)
    assert (
        governor.seeded_empirical_cap("mlx-community/Qwen3.8-27B-4bit", sample)
        == 32_768
    )
    assert governor.seeded_empirical_cap("mlx-community/Qwen3.8-9B-4bit", sample) is None
    too_large = governor.PressureSample(total_bytes=64 * GIB, available_bytes=50 * GIB)
    assert (
        governor.seeded_empirical_cap("mlx-community/Qwen3.8-27B-4bit", too_large)
        is None
    )


def test_context_bands_are_sticky_and_upgrade_only_after_clean_streak(monkeypatch, tmp_path):
    monkeypatch.setattr(governor, "_history_path", lambda: tmp_path / "history.json")
    governor.reset_session_state_for_tests()
    model = "mlx-community/Qwen3.8-27B-4bit"

    first = governor.choose_automatic_context(
        model_id=model,
        proposed_context=65_536,
        backend="mlx",
        kv=4,
        sample=_sample(available_gib=20),
        known_empirical_cap=32_768,
    )
    assert (first.band, first.context_length) == ("max", 32_768)

    pressured = governor.choose_automatic_context(
        model_id=model,
        proposed_context=65_536,
        backend="mlx",
        kv=4,
        sample=_sample(available_gib=2, swap_gib=4),
        known_empirical_cap=32_768,
    )
    assert (pressured.band, pressured.context_length) == ("safe", 16_384)

    # One or two clean samples cannot immediately bounce a pressure downgrade.
    for _ in range(2):
        still_safe = governor.choose_automatic_context(
            model_id=model,
            proposed_context=65_536,
            backend="mlx",
            kv=4,
            sample=_sample(available_gib=20),
            known_empirical_cap=32_768,
        )
        assert (still_safe.band, still_safe.context_length) == ("safe", 16_384)
    balanced = governor.choose_automatic_context(
        model_id=model,
        proposed_context=65_536,
        backend="mlx",
        kv=4,
        sample=_sample(available_gib=20),
        known_empirical_cap=32_768,
    )
    assert (balanced.band, balanced.context_length) == ("balanced", 24_576)


def test_empirical_history_caps_future_auto_choice(monkeypatch, tmp_path):
    path = tmp_path / "history.json"
    monkeypatch.setattr(governor, "_history_path", lambda: path)
    governor.reset_session_state_for_tests()
    sample = _sample(available_gib=20)
    key = governor.exact_profile_key(
        model_id="org/model",
        backend="llama.cpp",
        kv="q4_0",
        speculative="off",
        total_bytes=sample.total_bytes,
    )
    governor.record_observation(
        profile_key=key,
        context_length=24_576,
        occupancy_tokens=20_000,
        sample=sample,
        outcome="ok",
    )
    decision = governor.choose_automatic_context(
        model_id="org/model",
        proposed_context=65_536,
        backend="llama.cpp",
        kv="q4_0",
        speculative="off",
        sample=sample,
    )
    assert decision.context_length == 24_576
    assert decision.empirical_cap == 24_576


def test_active_profile_records_exact_adaptive_boundary(monkeypatch, tmp_path):
    path = tmp_path / "history.json"
    monkeypatch.setattr(governor, "_history_path", lambda: path)
    governor.reset_session_state_for_tests()
    sample = _sample(available_gib=20)
    decision = governor.choose_automatic_context(
        model_id="org/model",
        proposed_context=32_768,
        backend="mlx",
        kv=4,
        sample=sample,
    )
    governor.activate_profile(decision, "org/model", "alias/model")
    assert governor.record_active_observation(
        model_id="alias/model",
        context_length=32_768,
        occupancy_tokens=26_215,
        sample=sample,
        outcome="ok",
    ) is True
    data = governor._load_history()
    entry = data["profiles"][decision.profile_key]
    assert entry["validated_max_context"] == 32_768
    assert entry["last_occupancy"] == 26_215
    assert entry["last_qualified_boundary"] is True


def test_short_clean_turn_cannot_validate_an_unused_large_window(monkeypatch, tmp_path):
    path = tmp_path / "history.json"
    monkeypatch.setattr(governor, "_history_path", lambda: path)
    governor.reset_session_state_for_tests()
    sample = _sample(available_gib=20)
    key = governor.exact_profile_key(
        model_id="org/model",
        backend="mlx",
        kv=4,
        speculative="auto",
        total_bytes=sample.total_bytes,
    )
    governor.record_observation(
        profile_key=key,
        context_length=65_536,
        occupancy_tokens=4_096,
        sample=sample,
        outcome="ok",
    )
    entry = governor._load_history()["profiles"][key]
    assert entry["validated_max_context"] is None
    assert entry["last_qualified_boundary"] is False


def test_session_remembers_previous_sample_so_swap_growth_is_not_inert(monkeypatch, tmp_path):
    monkeypatch.setattr(governor, "_history_path", lambda: tmp_path / "history.json")
    governor.reset_session_state_for_tests()
    first = governor.PressureSample(
        total_bytes=36 * GIB,
        available_bytes=20 * GIB,
        swap_used_bytes=0,
        swap_out_bytes=0,
        compressed_bytes=0,
    )
    initial = governor.choose_automatic_context(
        model_id="org/model",
        proposed_context=32_768,
        backend="mlx",
        kv=4,
        sample=first,
    )
    assert initial.band == "max"

    # Same amount of available RAM, but the machine swapped 2 GiB since the
    # previous exact-profile observation. Production callers do not have to pass
    # previous_sample manually for this signal to work.
    later = governor.PressureSample(
        total_bytes=36 * GIB,
        available_bytes=20 * GIB,
        swap_used_bytes=0,
        swap_out_bytes=2 * GIB,
        compressed_bytes=0,
    )
    pressured = governor.choose_automatic_context(
        model_id="org/model",
        proposed_context=32_768,
        backend="mlx",
        kv=4,
        sample=later,
    )
    assert pressured.pressure_level == "critical"
    assert (pressured.band, pressured.context_length) == ("safe", 16_384)


def test_metal_working_set_participates_in_pressure_classification():
    sample = governor.PressureSample(
        total_bytes=36 * GIB,
        available_bytes=20 * GIB,
        swap_used_bytes=0,
        compressed_bytes=0,
        metal_active_bytes=27 * GIB,
        metal_peak_bytes=27 * GIB,
    )
    assert governor.classify_pressure(sample) == "elevated"


def test_near_oom_band_survives_process_session_reset(monkeypatch, tmp_path):
    path = tmp_path / "history.json"
    monkeypatch.setattr(governor, "_history_path", lambda: path)
    governor.reset_session_state_for_tests()
    sample = _sample(available_gib=20)
    key = governor.exact_profile_key(
        model_id="org/model",
        backend="mlx",
        kv=4,
        speculative="auto",
        total_bytes=sample.total_bytes,
    )
    governor.record_observation(
        profile_key=key,
        context_length=32_768,
        occupancy_tokens=26_215,
        sample=sample,
        outcome="near_oom",
    )
    assert governor._load_history()["profiles"][key]["recommended_band"] == "balanced"

    # Simulate a fresh process/session: the local empirical machine profile, not
    # a module-global streak, must keep the prior pressure lesson.
    governor.reset_session_state_for_tests()
    decision = governor.choose_automatic_context(
        model_id="org/model",
        proposed_context=32_768,
        backend="mlx",
        kv=4,
        sample=sample,
    )
    assert (decision.band, decision.context_length) == ("balanced", 24_576)
