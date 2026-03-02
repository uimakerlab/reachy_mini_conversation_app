"""Shared helpers for parakeet_mlx and parakeet_mlx_streaming providers."""

from __future__ import annotations
import time
import wave
import logging
import tempfile
from typing import Any
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


def load_parakeet_model(model_name: str, precision: str) -> Any:
    """Load a Parakeet-MLX model with the given precision."""
    import mlx.core as mx
    from parakeet_mlx import from_pretrained

    dtype_map = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}
    dtype = dtype_map.get(precision)
    if dtype is None:
        logger.warning(f"Unknown precision '{precision}', using fp32")
        dtype = mx.float32

    return from_pretrained(model_name, dtype=dtype)


def warmup_parakeet_model(model: Any) -> None:
    """Run a dummy transcription to pre-compile MLX kernels."""
    logger.info("Warming up Parakeet model (pre-compiling MLX kernels)...")
    t0 = time.perf_counter()

    try:
        sample_rate = 16000
        silence = np.zeros(int(sample_rate * 1.0), dtype=np.int16)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(silence.tobytes())

        _ = model.transcribe(tmp.name)
        Path(tmp.name).unlink(missing_ok=True)

        logger.info(f"Parakeet warmup complete in {(time.perf_counter() - t0) * 1000:.0f}ms")
    except Exception as e:
        logger.warning(f"Parakeet warmup failed (non-critical): {e}")
