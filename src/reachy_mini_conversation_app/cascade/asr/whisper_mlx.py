"""Whisper MLX ASR provider (local, Apple Silicon).

Uses mlx-whisper for efficient inference on M1/M2/M3 chips.
Implements pseudo-streaming via incremental batch transcription.
"""

from __future__ import annotations
import io
import wave
import logging
from typing import Optional

import numpy as np

from .base_streaming import StreamingASRProvider


logger = logging.getLogger(__name__)


class WhisperMLXASR(StreamingASRProvider):
    """Whisper ASR via MLX on Apple Silicon (local, pseudo-streaming).

    This provider accumulates audio and transcribes incrementally.
    True streaming is simulated by processing accumulated audio chunks.
    """

    def __init__(
            self,
            model: str = "mlx-community/whisper-large-v3-turbo",
            language: str = "en",
            sample_rate: int = 16000,
            chunk_interval_ms: int = 500,
    ) -> None:
        """Initialize Whisper MLX ASR.

        Args:
            model: HuggingFace model name (mlx-community models recommended)
            language: Language code for transcription
            sample_rate: Audio sample rate (must be 16000 for Whisper)
            chunk_interval_ms: How often to run transcription on accumulated audio

        """
        self.model_name = model
        self.language = language
        self.sample_rate = sample_rate
        self.chunk_interval_ms = chunk_interval_ms

        # Calculate samples between transcription runs
        self.samples_per_interval = int(sample_rate * chunk_interval_ms / 1000)

        # Model (lazy loaded)
        self._model_loaded = False

        # Streaming state
        self._audio_buffer: list[np.ndarray] = []
        self._samples_since_last_transcribe = 0
        self._partial_transcript = ""
        self._final_transcript = ""

        logger.info(
            f"WhisperMLXASR configured (model={model}, "
            f"language={language}, interval={chunk_interval_ms}ms)"
        )

    def _ensure_model_loaded(self) -> None:
        """Lazy load mlx-whisper on first use."""
        if self._model_loaded:
            return

        logger.info(f"Loading MLX Whisper model: {self.model_name}")

        try:
            import mlx_whisper

            # Just verify mlx_whisper is available
            # The actual model loading happens during transcription
            self._mlx_whisper = mlx_whisper
            self._model_loaded = True
            logger.info("MLX Whisper ready")

        except ImportError as e:
            raise ImportError(
                f"mlx-whisper not installed. Install with: pip install mlx-whisper\nError: {e}"
            )

    async def start_stream(self) -> None:
        """Initialize streaming session."""
        # Ensure model is loaded
        self._ensure_model_loaded()

        # Reset state
        self._audio_buffer = []
        self._samples_since_last_transcribe = 0
        self._partial_transcript = ""
        self._final_transcript = ""

        logger.info("MLX streaming session started")

    async def send_audio_chunk(self, audio_chunk: bytes) -> None:
        """Send audio chunk for processing.

        Args:
            audio_chunk: Audio data (WAV format or raw PCM int16)

        """
        # Convert to numpy array
        audio_array = self._wav_to_array(audio_chunk)
        if audio_array is None or len(audio_array) == 0:
            return

        # Add to buffer
        self._audio_buffer.append(audio_array)
        self._samples_since_last_transcribe += len(audio_array)

        # Run incremental transcription at intervals
        if self._samples_since_last_transcribe >= self.samples_per_interval:
            await self._transcribe_accumulated()
            self._samples_since_last_transcribe = 0

    async def get_partial_transcript(self) -> Optional[str]:
        """Get current partial transcript.

        Returns:
            Partial transcript or None

        """
        return self._partial_transcript if self._partial_transcript else None

    async def end_stream(self) -> str:
        """Finalize stream and get final transcript.

        Returns:
            Final complete transcript

        """
        # Final transcription of all accumulated audio
        if self._audio_buffer:
            await self._transcribe_accumulated(is_final=True)

        # Get final result
        transcript = self._final_transcript.strip()
        if not transcript and self._partial_transcript:
            transcript = self._partial_transcript.strip()

        logger.info(f"MLX final transcript: '{transcript}'")

        # Reset state
        self._audio_buffer = []
        self._partial_transcript = ""
        self._final_transcript = ""

        return transcript

    async def _transcribe_accumulated(self, is_final: bool = False) -> None:
        """Transcribe accumulated audio buffer.

        Args:
            is_final: If True, this is the final transcription

        """
        if not self._audio_buffer:
            return

        try:
            # Concatenate all audio
            audio = np.concatenate(self._audio_buffer)

            # Convert to float32 normalized to [-1, 1]
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0

            # Transcribe using mlx-whisper
            result = self._mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model_name,
                language=self.language,
                fp16=True,  # Use FP16 for faster inference
            )

            # Extract text
            if result and "text" in result:
                text = result["text"].strip()

                if is_final:
                    self._final_transcript = text
                    logger.debug(f"MLX final: '{text}'")
                else:
                    self._partial_transcript = text
                    logger.debug(f"MLX partial: '{text}'")

        except Exception as e:
            logger.warning(f"Error in MLX transcription: {e}")

    def _wav_to_array(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        """Convert WAV bytes to numpy array.

        Args:
            audio_bytes: WAV or raw PCM bytes

        Returns:
            Numpy array of audio samples, or None if conversion fails

        """
        try:
            # Try to parse as WAV
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()

                # Read frames
                frames = wav_file.readframes(n_frames)

                # Convert to numpy
                if sample_width == 2:  # 16-bit
                    audio = np.frombuffer(frames, dtype=np.int16)
                elif sample_width == 4:  # 32-bit
                    audio = np.frombuffer(frames, dtype=np.int32)
                else:
                    audio = np.frombuffer(frames, dtype=np.int16)

                # Resample if needed
                if sample_rate != self.sample_rate:
                    import librosa

                    audio = audio.astype(np.float32) / 32768.0
                    audio = librosa.resample(
                        audio,
                        orig_sr=sample_rate,
                        target_sr=self.sample_rate,
                    )
                    audio = (audio * 32768).astype(np.int16)

                return audio

        except Exception:
            # Assume raw PCM int16
            try:
                return np.frombuffer(audio_bytes, dtype=np.int16)
            except Exception as e:
                logger.warning(f"Failed to convert audio: {e}")
                return None
