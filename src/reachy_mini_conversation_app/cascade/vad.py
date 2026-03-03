"""Voice Activity Detection using Silero VAD.

Silero VAD is a pre-trained ML model for detecting speech in audio.
It works with 16kHz audio and returns speech probability for each chunk.
"""

import logging

import numpy as np
import torch


logger = logging.getLogger(__name__)

# Silero VAD requires specific sample rates and chunk sizes
SILERO_SAMPLE_RATE = 16000


class SileroVAD:
    """Voice Activity Detection using Silero VAD model.

    The model processes audio chunks and returns speech probability.
    It maintains internal state for streaming audio processing.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
    ) -> None:
        """Initialize Silero VAD.

        Args:
            threshold: Speech probability threshold (0.0-1.0). Higher = stricter.
            min_speech_duration_ms: Minimum speech duration to trigger detection.
            min_silence_duration_ms: Minimum silence duration to end speech segment.

        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms

        # Load Silero VAD model
        logger.info("Loading Silero VAD model...")
        self.model, _utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
        )
        self.model.eval()

        # State tracking for streaming
        self._speech_frames = 0
        self._silence_frames = 0
        self._is_speaking = False

        logger.info(f"Silero VAD initialized (threshold={threshold})")

    def get_speech_prob(self, audio: np.ndarray, sample_rate: int = SILERO_SAMPLE_RATE) -> float:
        """Get speech probability for an audio chunk.

        Args:
            audio: Audio samples as int16 or float32 numpy array.
            sample_rate: Sample rate of the audio (must be 16000).

        Returns:
            Speech probability between 0.0 and 1.0.

        """
        if sample_rate != SILERO_SAMPLE_RATE:
            raise ValueError(f"Silero VAD requires {SILERO_SAMPLE_RATE}Hz audio, got {sample_rate}Hz")

        # Convert to float32 tensor normalized to [-1, 1]
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.float32:
            audio_float = audio
        else:
            audio_float = audio.astype(np.float32)

        tensor = torch.from_numpy(audio_float)

        # Run VAD model
        with torch.no_grad():
            speech_prob = self.model(tensor, sample_rate).item()

        return speech_prob

    def is_speech(self, audio: np.ndarray, sample_rate: int = SILERO_SAMPLE_RATE) -> bool:
        """Check if audio chunk contains speech above threshold.

        Args:
            audio: Audio samples as numpy array.
            sample_rate: Sample rate of the audio.

        Returns:
            True if speech probability exceeds threshold.

        """
        prob = self.get_speech_prob(audio, sample_rate)
        return prob >= self.threshold

    def process_chunk(self, audio: np.ndarray, sample_rate: int = SILERO_SAMPLE_RATE) -> tuple[bool, bool]:
        """Process audio chunk with hysteresis for robust speech detection.

        Implements state machine with minimum duration requirements:
        - Speech must persist for min_speech_duration_ms to trigger start
        - Silence must persist for min_silence_duration_ms to trigger end

        Args:
            audio: Audio samples as numpy array.
            sample_rate: Sample rate of the audio.

        Returns:
            Tuple of (speech_started, speech_ended) events.

        """
        chunk_duration_ms = len(audio) / sample_rate * 1000
        is_speech = self.is_speech(audio, sample_rate)

        speech_started = False
        speech_ended = False

        if is_speech:
            self._silence_frames = 0
            self._speech_frames += chunk_duration_ms

            # Trigger speech start after minimum duration
            if not self._is_speaking and self._speech_frames >= self.min_speech_duration_ms:
                self._is_speaking = True
                speech_started = True
                logger.debug(f"Speech started (accumulated {self._speech_frames:.0f}ms)")
        else:
            self._speech_frames = 0
            if self._is_speaking:
                self._silence_frames += chunk_duration_ms

                # Trigger speech end after minimum silence duration
                if self._silence_frames >= self.min_silence_duration_ms:
                    self._is_speaking = False
                    speech_ended = True
                    logger.debug(f"Speech ended (silence {self._silence_frames:.0f}ms)")

        return speech_started, speech_ended

    @property
    def is_speaking(self) -> bool:
        """Returns True if currently in a speech segment."""
        return self._is_speaking

    def reset(self) -> None:
        """Reset VAD state between sessions."""
        self._speech_frames = 0
        self._silence_frames = 0
        self._is_speaking = False
        self.model.reset_states()
        logger.debug("VAD state reset")

