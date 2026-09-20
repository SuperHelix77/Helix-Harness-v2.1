from core.inference.q38_v11_optimization import (
    Q38_V11_PREFERRED_CONTEXT,
    Q38_V11_PREFERRED_N_BATCH,
    Q38_V11_PREFERRED_N_UBATCH,
    q38_v11_default_extra_args,
    q38_v11_default_load_updates,
    q38_v11_is_qwen38,
    q38_v11_context_target,
    q38_v11_default_parallel_slots,
)


def test_darwin_defaults_to_one_slot_for_long_context():
    assert q38_v11_default_parallel_slots("darwin", None, 4) == 1


def test_explicit_slots_are_authoritative():
    assert q38_v11_default_parallel_slots("darwin", 4, 1) == 4
    assert q38_v11_default_parallel_slots("linux", 2, 4) == 2


def test_other_platforms_keep_configured_default():
    assert q38_v11_default_parallel_slots("linux", None, 4) == 4


def test_context_target_never_exceeds_native_window():
    assert q38_v11_context_target(None) == Q38_V11_PREFERRED_CONTEXT
    assert q38_v11_context_target(32_768) == 32_768
    assert q38_v11_context_target(262_144) == Q38_V11_PREFERRED_CONTEXT


def test_qwen38_profile_is_scoped_to_darwin_and_auto_settings():
    assert q38_v11_is_qwen38("unsloth/Qwen3.8-27B-GGUF")
    assert not q38_v11_is_qwen38("Qwen3.6-35B")
    updates = q38_v11_default_load_updates(
        "unsloth/Qwen3.8-27B-GGUF",
        platform_name = "darwin",
        max_seq_length = 0,
        cache_type_kv = None,
        speculative_type = "auto",
        n_batch = None,
        n_ubatch = None,
    )
    assert updates == {
        "max_seq_length": Q38_V11_PREFERRED_CONTEXT,
        "cache_type_kv": "q4_0",
        "n_batch": Q38_V11_PREFERRED_N_BATCH,
        "n_ubatch": Q38_V11_PREFERRED_N_UBATCH,
    }
    assert "speculative_type" not in updates


def test_qwen38_profile_does_not_force_dflash_or_mtp():
    """V1.1 measured path is speculation-off. Auto/None must not rewrite to dflash."""
    for speculative in (None, "auto", "default"):
        updates = q38_v11_default_load_updates(
            "unsloth/Qwen3.8-27B-GGUF",
            platform_name = "darwin",
            max_seq_length = 65_536,
            cache_type_kv = "q4_0",
            speculative_type = speculative,
            n_batch = 1024,
            n_ubatch = 256,
        )
        assert updates == {}


def test_qwen38_profile_preserves_explicit_settings():
    assert q38_v11_default_load_updates(
        "Qwen3.8-27B-GGUF",
        platform_name = "darwin",
        max_seq_length = 32_768,
        cache_type_kv = "f16",
        speculative_type = "mtp",
        n_batch = 512,
        n_ubatch = 128,
    ) == {}
    assert q38_v11_default_load_updates(
        "Qwen3.8-27B-GGUF",
        platform_name = "linux",
        max_seq_length = 0,
        cache_type_kv = None,
        speculative_type = None,
        n_batch = None,
        n_ubatch = None,
    ) == {}


def test_qwen38_profile_adds_unified_kv_without_overriding_explicit_flag():
    assert q38_v11_default_extra_args(
        "Qwen3.8-27B-GGUF", [], platform_name = "darwin"
    ) == ["--kv-unified"]
    assert q38_v11_default_extra_args(
        "Qwen3.8-27B-GGUF", ["--no-kv-unified"], platform_name = "darwin"
    ) == ["--no-kv-unified"]
