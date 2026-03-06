"""Audio recording for VAD-based continuous recording."""

from __future__ import annotations
import logging
import threading
from enum import Enum
from typing import TYPE_CHECKING, Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from reachy_mini_conversation_app.cascade.vad import VADEvent, VADStateMachine
from reachy_mini_conversation_app.cascade.asr.audio_utils import pcm_to_wav


if TYPE_CHECKING:
    from reachy_mini_conversation_app.cascade.vad import SileroVAD

logger = logging.getLogger(__name__)


class ContinuousState(Enum):
    """State machine for continuous VAD-based recording."""

    IDLE = "idle"
    LISTENING = "listening"
    RECORDING = "recording"
    PROCESSING = "processing"


@dataclass
class StreamingASRCallbacks:
    """Callbacks for streaming ASR integration.

    Allows decoupling recording from ASR provider by injecting callbacks.
    """

    on_start: Callable[[], None]
    """Called when recording starts to initialize streaming session."""

    on_chunk: Callable[[bytes], None]
    """Called for each audio chunk (as WAV bytes) during recording."""



class ContinuousVADRecorder:
    """VAD-based continuous recording mode.

    Automatically detects speech start/end using Silero VAD.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        streaming_asr_callbacks: StreamingASRCallbacks | None = None,
        on_speech_captured: Callable[[bytes], None] | None = None,
        vad_threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 700,
    ) -> None:
        """Initialize VAD recorder.

        Args:
            sample_rate: Recording sample rate (default 16kHz)
            streaming_asr_callbacks: Optional callbacks for streaming ASR
            on_speech_captured: Callback when complete utterance is captured (receives WAV bytes)
            vad_threshold: VAD detection threshold (0-1)
            min_speech_duration_ms: Minimum speech duration to trigger detection
            min_silence_duration_ms: Silence duration to end speech segment

        """
        self.sample_rate = sample_rate
        self.streaming_callbacks = streaming_asr_callbacks
        self.on_speech_captured = on_speech_captured
        self.vad_threshold = vad_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms

        self._active = False
        self._vad_sm: VADStateMachine | None = None
        self._continuous_thread: threading.Thread | None = None
        self._vad: SileroVAD | None = None

    @property
    def state(self) -> ContinuousState:
        """Current VAD state."""
        if not self._active:
            return ContinuousState.IDLE
        if self._vad_sm is None:
            return ContinuousState.IDLE
        return ContinuousState(self._vad_sm.state.value)

    @property
    def is_active(self) -> bool:
        """Whether continuous mode is active."""
        return self._active

    def start(self) -> str:
        """Start continuous VAD-based recording mode.

        Returns:
            Status message

        """
        if self._active:
            return "Already in continuous mode"

        # Initialize VAD lazily (avoids ~1-2s model load at startup)
        if self._vad is None:
            logger.info("Initializing Silero VAD...")
            from reachy_mini_conversation_app.cascade.vad import SileroVAD

            self._vad = SileroVAD(
                threshold=self.vad_threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
            )

        self._vad_sm = VADStateMachine(self._vad)
        self._active = True

        # Start continuous recording thread
        self._continuous_thread = threading.Thread(target=self._continuous_record_loop, daemon=True)
        self._continuous_thread.start()

        logger.info("Continuous mode started")
        return "Listening... (VAD active)"

    def stop(self) -> str:
        """Stop continuous VAD-based recording mode.

        Returns:
            Status message

        """
        if not self._active:
            return "Not in continuous mode"

        self._active = False

        # Wait for thread to finish
        if self._continuous_thread:
            self._continuous_thread.join(timeout=2.0)
            self._continuous_thread = None

        # Reset VAD state
        if self._vad:
            self._vad.reset()
        self._vad_sm = None

        logger.info("Continuous mode stopped")
        return "Continuous mode stopped"

    def _continuous_record_loop(self) -> None:
        """Continuous recording loop with VAD-based speech detection."""
        from reachy_mini_conversation_app.cascade.vad import VAD_CHUNK_SIZE, SILERO_SAMPLE_RATE
        from reachy_mini_conversation_app.cascade.timing import tracker

        assert self._vad_sm is not None

        # Log which mic we'll use (system default)
        default_dev = sd.query_devices(kind="input")
        logger.info(
            f"Mic input: '{default_dev['name']}' (system default, "
            f"{default_dev['default_samplerate']:.0f} Hz)"
        )

        logger.info("Continuous mode started - listening for speech...")

        # VAD processes 16kHz audio, but we record at self.sample_rate
        record_chunk_samples = int(VAD_CHUNK_SIZE * self.sample_rate / SILERO_SAMPLE_RATE)

        try:
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                dtype=np.int16,
                blocksize=record_chunk_samples,
            ) as stream:
                while self._active:
                    data, overflowed = stream.read(record_chunk_samples)
                    if overflowed:
                        logger.warning("Audio buffer overflowed in continuous mode")

                    # Resample to 16kHz for VAD if needed
                    if self.sample_rate != SILERO_SAMPLE_RATE:
                        import librosa

                        audio_float = data.flatten().astype(np.float32) / 32768.0
                        audio_resampled = librosa.resample(
                            audio_float,
                            orig_sr=self.sample_rate,
                            target_sr=SILERO_SAMPLE_RATE,
                        )
                        vad_audio = (audio_resampled * 32768).astype(np.int16)
                    else:
                        vad_audio = data.flatten()

                    event = self._vad_sm.process_chunk(vad_audio)

                    if event == VADEvent.SPEECH_STARTED:
                        tracker.reset("user_conversation_turn")
                        tracker.mark("vad_speech_start")

                        if self.streaming_callbacks:
                            self.streaming_callbacks.on_start()
                            # Send pre-roll + current chunk so ASR sees speech onset
                            for frame in self._vad_sm.speech_chunks:
                                self.streaming_callbacks.on_chunk(
                                    pcm_to_wav(frame.tobytes(), self.sample_rate)
                                )

                    elif event == VADEvent.SPEECH_ENDED:
                        tracker.mark("vad_speech_end")

                        audio_data = np.concatenate(self._vad_sm.speech_chunks)
                        duration = len(audio_data) / self.sample_rate
                        logger.info(f"VAD: Captured {duration:.2f}s of speech")
                        tracker.mark("recording_captured", {"duration_s": round(duration, 2)})

                        wav_bytes = pcm_to_wav(audio_data.tobytes(), self.sample_rate)
                        if self.on_speech_captured:
                            self.on_speech_captured(wav_bytes)

                        self._vad_sm.finish_processing()
                        logger.info("VAD: Ready for next utterance")

                    elif self._vad_sm.state.value == ContinuousState.RECORDING.value:
                        # Mid-recording: stream current chunk to ASR
                        if self.streaming_callbacks:
                            self.streaming_callbacks.on_chunk(
                                pcm_to_wav(data.tobytes(), self.sample_rate)
                            )

        except Exception as e:
            logger.exception(f"Error in continuous recording loop: {e}")
        finally:
            logger.info("Continuous mode stopped")
