# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import threading
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator


_TRAINING_LIFECYCLE_LOCK = threading.RLock()


@contextmanager
def training_lifecycle_guard() -> Iterator[None]:
    with _TRAINING_LIFECYCLE_LOCK:
        yield


@asynccontextmanager
async def training_inference_admission_guard() -> AsyncIterator[None]:
    """Serialize an autonomous training claim with new local inference admission.

    The chat keep-warm middleware publishes a request as pending before it enters its
    lifecycle gate, then marks it inflight while holding that gate. Holding the same gate
    across a final idle recheck and the training backend's start claim closes the gap where
    a new GGUF/MLX/safetensors request could otherwise start after an idle poll but before
    training unloads resident models and publishes ``is_training_active()``.

    Import lazily so the training backend itself does not acquire inference modules during
    ordinary lifecycle operations (stop/reset/worker finalization).
    """
    from core.inference.llama_keepwarm import inference_lifecycle_gate

    async with inference_lifecycle_gate():
        yield
