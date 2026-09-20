from core.inference.mlx_inference import (
    MLX_QWYTHOS_PREFERRED_CONTEXT,
    MLXInferenceBackend,
    mlx_effective_context_length,
    mlx_is_qwythos_model,
)


def test_qwythos_is_detected_and_gets_the_durable_64k_context():
    model = "empero-ai/Qwythos-9B-v2"
    assert mlx_is_qwythos_model(model)
    assert mlx_effective_context_length(model, 0) == MLX_QWYTHOS_PREFERRED_CONTEXT
    assert mlx_effective_context_length(model, 8192) == 8192


def test_qwythos_uses_the_published_qwen35_dflash_pair_but_never_fakes_dflare():
    model = "empero-ai/Qwythos-9B-v2"
    assert (
        MLXInferenceBackend._default_mlx_draft_model(model, "dflash")
        == "z-lab/Qwen3.5-9B-DFlash"
    )
    assert MLXInferenceBackend._default_mlx_draft_model(model, "dflare") is None
