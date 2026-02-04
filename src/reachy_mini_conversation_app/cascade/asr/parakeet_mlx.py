"""Parakeet-MLX ASR provider (Apple Silicon optimized)."""

from __future__ import annotations
import wave
import asyncio
import logging
from typing import Any, Optional

import numpy as np
import mlx.core as mx

from .base import ASRProvider


logger = logging.getLogger(__name__)


class ParakeetMLXASR(ASRProvider):
    """Parakeet-MLX ASR (local, Apple Silicon batch)."""

    def __init__(
        self,
        model: str = "mlx-community/parakeet-tdt-0.6b-v3",
        precision: str = "fp32",
    ):
        """Initialize Parakeet-MLX ASR.

        Args:
            model: Model ID from HuggingFace (parakeet-tdt-0.6b-v3 recommended)
            precision: Inference precision (fp32 for quality, bf16 for speed)

        """
        self.model_name = model
        self.precision = precision
        self.target_sample_rate = 16000  # Parakeet requires 16kHz
        self.model: Any = None  # Will be set by _ensure_model

        # Preload model immediately to avoid first-call delay
        logger.info(f"Loading Parakeet model: {model} (precision: {precision})...")
        self._ensure_model()
        logger.info("Parakeet model loaded successfully")

        # Warmup: Run a dummy inference to pre-compile kernels
        self._warmup_model()

    def _ensure_model(self) -> None:
        """Load the Parakeet model."""
        if self.model is None:
            from parakeet_mlx import from_pretrained

            # Convert string precision to MLX dtype
            if self.precision == "fp32":
                dtype = mx.float32
            elif self.precision == "bf16":
                dtype = mx.bfloat16
            elif self.precision == "fp16":
                dtype = mx.float16
            else:
                logger.warning(f"Unknown precision '{self.precision}', using fp32")
                dtype = mx.float32

            self.model = from_pretrained(
                self.model_name,
                dtype=dtype,
            )

    def _warmup_model(self) -> None:
        """Warmup model with dummy inference to pre-compile MLX kernels."""
        import time
        import tempfile

        logger.info("Warming up Parakeet model (pre-compiling MLX kernels)...")
        warmup_start = time.perf_counter()

        try:
            # Create a short dummy audio file (1 second of silence)
            sample_rate = 16000
            duration = 1.0  # 1 second
            num_samples = int(sample_rate * duration)
            silence = np.zeros(num_samples, dtype=np.int16)

            # Write to temp WAV file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            with wave.open(temp_file.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(silence.tobytes())

            # Run dummy inference (this compiles MLX kernels)
            _ = self.model.transcribe(temp_file.name)

            # Cleanup
            from pathlib import Path

            Path(temp_file.name).unlink(missing_ok=True)

            warmup_duration = (time.perf_counter() - warmup_start) * 1000
            logger.info(
                f"Parakeet warmup complete! First inference took {warmup_duration:.0f}ms (subsequent calls will be ~4-5x faster)"
            )

        except Exception as e:
            logger.warning(f"Parakeet warmup failed (non-critical): {e}")

    async def transcribe(self, audio_bytes: bytes, language: Optional[str] = None) -> str:
        """Transcribe audio using Parakeet-MLX (batch mode).

        Args:
            audio_bytes: Audio data in WAV format
            language: Language code (currently unused - Parakeet auto-detects)

        Returns:
            Transcribed text

        """
        import tempfile
        from pathlib import Path

        from reachy_mini_conversation_app.cascade.timing import tracker

        logger.debug(f"Transcribing {len(audio_bytes)} bytes of audio with Parakeet-MLX")

        tracker.mark("asr_start", {"bytes": len(audio_bytes)})

        # Create temporary WAV file for Parakeet
        # Parakeet expects a file path, not raw audio data
        temp_file = None
        try:
            # Save audio bytes to temporary file
            tracker.mark("asr_temp_file_write_start")
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.write(audio_bytes)
            temp_file.close()
            tracker.mark("asr_temp_file_write_end")

            temp_path = temp_file.name
            logger.debug(f"Saved audio to temporary file: {temp_path}")

            # Run transcription in thread pool (Parakeet is synchronous)
            def _transcribe_sync() -> str:
                from reachy_mini_conversation_app.cascade.timing import tracker

                tracker.mark("asr_model_inference_start")
                # Transcribe using Parakeet (pass file path)
                result = self.model.transcribe(temp_path)
                tracker.mark("asr_model_inference_end")

                # Extract text from result
                if hasattr(result, "text"):
                    return str(result.text)
                elif isinstance(result, str):
                    return result
                elif isinstance(result, dict) and "text" in result:
                    return str(result["text"])
                else:
                    # Fallback: try to convert to string
                    logger.warning(f"Unexpected result type from Parakeet: {type(result)}")
                    return str(result)

            transcript = await asyncio.to_thread(_transcribe_sync)

            # Clean up transcript
            transcript = transcript.strip()

            logger.info(f"Parakeet transcription successful: '{transcript[:100]}...'")

            tracker.mark("asr_complete", {"transcript_len": len(transcript)})
            return transcript

        except Exception as e:
            logger.error(f"Parakeet transcription failed: {e}")
            raise

        finally:
            # Clean up temporary file
            if temp_file is not None:
                try:
                    Path(temp_file.name).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file.name}: {e}")
