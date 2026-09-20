# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Small Apple-Silicon correctness tests for the native DFlash/DFlare loop."""

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import RotatingKVCache, make_prompt_cache  # noqa: E402
from mlx_lm.models.qwen3 import Model, ModelArgs  # noqa: E402

from core.inference.mlx_speculative import (  # noqa: E402
    DFlashConfig,
    DFlashDraftModel,
    DFlareDraftModel,
    MLX_QUANTIZED_VERIFY_MAX_BLOCK_SIZE,
    resolve_speculative_block_size,
    stream_generate,
)


class _TinyTokenizer:
    eos_token_id = 127
    bos_token_id = None
    chat_template = None
    clean_up_tokenization_spaces = False

    def get_vocab(self):
        return {}

    def encode(self, _text, add_special_tokens=True):
        return [1]

    def decode(self, ids):
        return "".join(chr(65 + (int(token) % 26)) for token in ids)


def _target():
    args = ModelArgs(
        "qwen3", 64, 2, 128, 4, 1e-6, 128, 4, 256, 10000.0, 16, False, None
    )
    return Model(args)


def _draft(draft_class, target):
    config = DFlashConfig(
        64,
        2,
        4,
        4,
        16,
        128,
        128,
        1e-6,
        10000.0,
        256,
        4,
        (0, 1),
        2,
        mask_token_id=0,
        layer_types=("full_attention", "full_attention"),
    )
    draft = draft_class(config)
    draft.bind(target)
    return draft


def _greedy(target, prompt, count, max_kv_size):
    cache = make_prompt_cache(target, max_kv_size=max_kv_size)
    inputs = mx.array([prompt])
    result = []
    for _ in range(count):
        logits = target(inputs, cache)
        mx.eval(logits)
        token = int(mx.argmax(logits[:, -1, :], axis=-1)[0].item())
        result.append(token)
        inputs = mx.array([[token]])
    return result


@pytest.mark.parametrize("configured, expected", [(16, 5), (5, 5), (3, 3), (None, 5)])
def test_quantized_mlx_verify_width_uses_the_official_safe_cap(configured, expected):
    assert MLX_QUANTIZED_VERIFY_MAX_BLOCK_SIZE == 5
    assert resolve_speculative_block_size(configured, 16, quantized=True) == expected
    assert resolve_speculative_block_size(configured, 16, quantized=False) == (
        16 if configured is None else configured
    )


def test_non_positive_speculative_width_is_rejected():
    with pytest.raises(ValueError, match="block_size"):
        resolve_speculative_block_size(0, 16, quantized=True)


@pytest.mark.parametrize("draft_class", [DFlashDraftModel, DFlareDraftModel])
def test_native_speculative_decode_matches_greedy_and_honors_kv_window(draft_class):
    mx.random.seed(17)
    baseline_target = _target()
    mx.eval(baseline_target.parameters())

    mx.random.seed(99)
    speculative_target = _target()
    speculative_target.update(baseline_target.parameters())
    mx.eval(speculative_target.parameters())

    draft = _draft(draft_class, speculative_target)
    mx.eval(draft.parameters())
    max_kv_size = 16
    assert all(
        isinstance(cache, RotatingKVCache)
        for cache in draft.make_cache(max_kv_size=max_kv_size)
    )

    prompt = [3, 7, 11, 19, 23]
    expected = _greedy(baseline_target, prompt, 12, max_kv_size)
    actual = []
    for response in stream_generate(
        speculative_target,
        draft,
        _TinyTokenizer(),
        mx.array(prompt),
        block_size=4,
        max_tokens=12,
        temperature=0.0,
        max_kv_size=max_kv_size,
    ):
        actual.extend(response.tokens)

    assert actual == expected
