"""Parakeet-MLX progressive ASR provider with sentence-aware sliding window.

Uses mlx_audio's decode_chunk() for periodic re-transcription of a growing audio buffer.
Completed sentences become "fixed" and immutable; only the active tail is re-transcribed.

Adapted from Andi Marafioti's work on https://github.com/huggingface/speech-to-speech#
"""

from __future__ import annotations
import time
import logging
from typing import Any, Optional

import numpy as np
import mlx.core as mx
import numpy.typing as npt

from .audio_utils import wav_to_float32
from .base_streaming import StreamingASRProvider


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class ParakeetMLXProgressiveASR(StreamingASRProvider):
    """Progressive ASR using periodic re-transcription with sentence-aware sliding window.

    Strategy:
    1. Accumulate audio in a growing buffer
    2. Every ~250ms of new audio, re-transcribe from the last fixed sentence boundary
    3. When the window exceeds max_window_size, fix completed sentences and slide forward
    4. Fixed sentences never change; only the active tail is re-transcribed
    """

    def __init__(
        self,
        model: str = "mlx-community/parakeet-tdt-0.6b-v3",
        precision: str = "float16",
        max_window_size: float = 15.0,
        sentence_buffer: float = 2.0,
    ):
        """Initialize progressive ASR with mlx_audio model."""
        self.model_name = model
        self.precision = precision
        self.max_window_size = max_window_size
        self.sentence_buffer = sentence_buffer
        self.target_sample_rate = SAMPLE_RATE

        # Load model
        logger.info(f"Loading Parakeet model via mlx_audio: {model}")
        from mlx_audio.stt.generate import load_model

        self._model: Any = load_model(model)
        logger.info("Parakeet model loaded")

        # Warmup with silence (no temp file needed)
        logger.info("Warming up model...")
        t0 = time.perf_counter()
        silence = mx.zeros(SAMPLE_RATE, dtype=mx.float32)  # 1s
        self._model.decode_chunk(silence, verbose=False)
        logger.info(f"Warmup done in {(time.perf_counter() - t0) * 1000:.0f}ms")

        # Streaming state (initialized in start_stream)
        self._audio_buffer: list[npt.NDArray[np.float32]] = []
        self._total_samples = 0
        self._fixed_sentences: list[str] = []
        self._fixed_end_time = 0.0
        self._last_transcribed_length = 0
        self._last_partial: str | None = None
        self._min_new_samples = int(SAMPLE_RATE * 0.25)  # 250ms between transcriptions

    # -- StreamingASRProvider interface --

    async def start_stream(self) -> None:
        """Reset state for new streaming session."""
        self._audio_buffer = []
        self._total_samples = 0
        self._fixed_sentences = []
        self._fixed_end_time = 0.0
        self._last_transcribed_length = 0
        self._last_partial = None
        logger.info("Progressive ASR stream started")

    async def send_audio_chunk(self, audio_chunk: bytes) -> None:
        """Append audio and run incremental transcription if enough new audio."""
        audio_np = wav_to_float32(audio_chunk, self.target_sample_rate)
        if len(audio_np) == 0:
            return

        self._audio_buffer.append(audio_np)
        self._total_samples += len(audio_np)

        # Check if enough new audio since last transcription
        new_samples = self._total_samples - self._last_transcribed_length
        if new_samples < self._min_new_samples:
            return

        # Concatenate full buffer and transcribe
        full_audio = np.concatenate(self._audio_buffer)
        fixed_text, active_text = self._transcribe_incremental(full_audio)

        # Build partial
        parts = [p for p in (fixed_text, active_text) if p]
        self._last_partial = " ".join(parts) or None

    async def get_partial_transcript(self) -> Optional[str]:
        """Return latest partial transcript."""
        return self._last_partial

    async def end_stream(self) -> str:
        """Finalize stream: re-transcribe full audio for maximum accuracy."""
        if self._total_samples == 0:
            return ""

        last_partial = self._last_partial or ""

        # Always transcribe the full audio for the final result (only runs once).
        # Progressive partials use the sliding window, but the final LLM input
        # benefits from full-context transcription.
        full_audio = np.concatenate(self._audio_buffer)
        audio_mx = mx.array(full_audio, dtype=mx.float32)
        result = self._model.decode_chunk(audio_mx, verbose=False)
        transcript = result.text.strip()

        # Fall back to last partial if re-transcription returns empty
        if not transcript and last_partial:
            logger.warning(
                f"Full re-transcription returned empty ({self._total_samples} samples), "
                f"falling back to last partial: '{last_partial}'"
            )
            transcript = last_partial

        # Reset
        self._audio_buffer = []
        self._total_samples = 0
        self._fixed_sentences = []
        self._fixed_end_time = 0.0
        self._last_transcribed_length = 0
        self._last_partial = None

        logger.info(f"Progressive ASR final: '{transcript}'")
        return transcript

    async def transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        """Batch fallback: decode_chunk on entire audio (no temp files)."""
        audio_np = wav_to_float32(audio_bytes, self.target_sample_rate)
        if len(audio_np) == 0:
            return ""
        audio_mx = mx.array(audio_np, dtype=mx.float32)
        result = self._model.decode_chunk(audio_mx, verbose=False)
        return result.text.strip()

    # -- Core sliding window logic (ported from s2s SmartProgressiveStreamingHandler) --

    def _transcribe_incremental(self, audio: npt.NDArray[np.float32]) -> tuple[str, str]:
        """Transcribe incrementally with sentence-aware sliding window.

        Returns (fixed_text, active_text).
        """
        current_length = len(audio)

        # Need at least 500ms of audio
        if current_length < SAMPLE_RATE // 2:
            return " ".join(self._fixed_sentences), ""

        self._last_transcribed_length = current_length

        # Extract window from last fixed sentence boundary to end
        window_start = int(self._fixed_end_time * SAMPLE_RATE)
        audio_window = audio[window_start:]

        # Transcribe current window
        audio_mx = mx.array(audio_window, dtype=mx.float32)
        result = self._model.decode_chunk(audio_mx, verbose=False)

        # Check if window exceeds max size → fix some sentences
        window_duration = len(audio_window) / SAMPLE_RATE
        if window_duration >= self.max_window_size and len(result.sentences) > 1:
            cutoff_time = window_duration - self.sentence_buffer
            new_fixed = []
            new_fixed_end = self._fixed_end_time

            for sentence in result.sentences:
                if sentence.end < cutoff_time:
                    new_fixed.append(sentence.text.strip())
                    new_fixed_end = self._fixed_end_time + sentence.end
                else:
                    break

            if new_fixed:
                self._fixed_sentences.extend(new_fixed)
                self._fixed_end_time = new_fixed_end
                logger.info(f"Fixed {len(new_fixed)} sentences, window slides to {self._fixed_end_time:.1f}s")

                # Re-transcribe from new fixed point
                window_start = int(self._fixed_end_time * SAMPLE_RATE)
                audio_window = audio[window_start:]
                audio_mx = mx.array(audio_window, dtype=mx.float32)
                result = self._model.decode_chunk(audio_mx, verbose=False)

        return " ".join(self._fixed_sentences), result.text.strip()
