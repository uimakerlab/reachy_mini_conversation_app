"""Parakeet-MLX ASR provider (Apple Silicon optimized)."""

from __future__ import annotations
import asyncio
import logging
from typing import Any, Optional

from .base import ASRProvider
from ._parakeet_helpers import load_parakeet_model, warmup_parakeet_model


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
        self.model = load_parakeet_model(self.model_name, self.precision)
        logger.info("Parakeet model loaded successfully")

        # Warmup: Run a dummy inference to pre-compile kernels
        warmup_parakeet_model(self.model)

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
