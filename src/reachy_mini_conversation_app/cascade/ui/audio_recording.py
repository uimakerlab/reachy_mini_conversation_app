"""Audio recording classes for push-to-talk and VAD-based continuous recording."""

from __future__ import annotations
import io
import wave
import asyncio
import logging
import threading
from enum import Enum
from typing import TYPE_CHECKING, List, Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import numpy.typing as npt


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


class PushToTalkRecorder:
    """Manual push-to-talk recording mode.

    Records audio while active, supports optional streaming ASR callbacks.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        streaming_asr_callbacks: StreamingASRCallbacks | None = None,
        event_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Initialize recorder.

        Args:
            sample_rate: Recording sample rate (default 16kHz)
            streaming_asr_callbacks: Optional callbacks for streaming ASR
            event_loop: Event loop for async callback execution

        """
        self.sample_rate = sample_rate
        self.streaming_callbacks = streaming_asr_callbacks
        self.event_loop = event_loop

        self._recording = False
        self._audio_frames: List[npt.NDArray[np.int16]] = []
        self._audio_data: npt.NDArray[np.int16] | None = None
        self._record_thread: threading.Thread | None = None

    @property
    def is_recording(self) -> bool:
        """Whether currently recording."""
        return self._recording

    @property
    def audio_data(self) -> npt.NDArray[np.int16] | None:
        """Captured audio data after recording stops."""
        return self._audio_data

    def start(self) -> str:
        """Start recording audio.

        Returns:
            Status message

        """
        if self._recording:
            return "Already recording"

        self._recording = True
        self._audio_frames = []
        self._audio_data = None

        # Start recording in background thread
        self._record_thread = threading.Thread(target=self._record_audio, daemon=True)
        self._record_thread.start()

        logger.info("Recording started...")
        return "Recording..."

    def stop(self) -> tuple[str, str]:
        """Stop recording and finalize audio.

        Returns:
            Tuple of (button_label, status_message)

        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        # Reset tracker at the start of the full pipeline
        tracker.reset("user_conversation_turn")
        tracker.mark("user_stop_click")

        if not self._recording:
            return "START Recording", "Not recording. Click START first."

        self._recording = False
        tracker.mark("recording_stop_signal_sent")

        # OPTIMIZATION: Don't wait for thread to finish!
        # We already have all the audio frames in self._audio_frames
        # Concatenate immediately and start ASR without delay
        if self._audio_frames and len(self._audio_frames) > 0:
            audio_data = np.concatenate(self._audio_frames)
            self._audio_data = audio_data

            logger.info(
                f"Recording captured immediately: {len(audio_data)} frames ({len(audio_data) / self.sample_rate:.2f}s)"
            )
            tracker.mark(
                "recording_captured",
                {"frames": len(audio_data), "duration_s": round(len(audio_data) / self.sample_rate, 2)},
            )
            tracker.mark("recording_ready")

            return "START Recording", "Processing your message..."
        else:
            # No frames captured - wait for thread to see if it has data
            if self._record_thread:
                self._record_thread.join(timeout=0.5)
            tracker.mark("recording_thread_joined")

            if self._audio_data is not None and len(self._audio_data) > 0:
                tracker.mark("recording_ready")
                return "START Recording", "Processing your message..."
            else:
                return "START Recording", "Failed to capture recording."

    def get_wav_bytes(self) -> bytes | None:
        """Get recorded audio as WAV bytes.

        Returns:
            WAV bytes or None if no recording

        """
        if self._audio_data is None or len(self._audio_data) == 0:
            return None

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(self._audio_data.tobytes())

        return wav_buffer.getvalue()

    def _record_audio(self) -> None:
        """Record audio from microphone in background thread."""
        self._audio_frames = []

        # Initialize streaming ASR session if callbacks provided
        if self.streaming_callbacks:
            logger.info("Initializing streaming ASR session...")
            self.streaming_callbacks.on_start()

        try:
            # Create audio stream (use small blocksize for lower latency)
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                dtype=np.int16,
                blocksize=1024,  # 1024 for lower latency (64ms)
            ) as stream:
                logger.info("Recording started...")
                while self._recording:
                    data, overflowed = stream.read(1024)
                    if overflowed:
                        logger.warning("Audio buffer overflowed")
                    self._audio_frames.append(data.copy())

                    # Send chunk to streaming ASR if callbacks provided
                    if self.streaming_callbacks:
                        # Convert chunk to WAV bytes
                        wav_buffer = io.BytesIO()
                        with wave.open(wav_buffer, "wb") as wav_file:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)  # 16-bit
                            wav_file.setframerate(self.sample_rate)
                            wav_file.writeframes(data.tobytes())

                        chunk_wav = wav_buffer.getvalue()
                        self.streaming_callbacks.on_chunk(chunk_wav)

            # Thread cleanup - only set audio_data if main thread hasn't already done it
            # (main thread may have concatenated early for low latency)
            if self._audio_data is None and self._audio_frames:
                audio_data = np.concatenate(self._audio_frames)
                self._audio_data = audio_data
                logger.debug(f"Recording thread finished: {len(audio_data)} frames")

        except Exception as e:
            logger.exception(f"Error recording audio: {e}")
            self._recording = False


class ContinuousVADRecorder:
    """VAD-based continuous recording mode.

    Automatically detects speech start/end using Silero VAD.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        streaming_asr_callbacks: StreamingASRCallbacks | None = None,
        on_speech_captured: Callable[[bytes], None] | None = None,
        event_loop: asyncio.AbstractEventLoop | None = None,
        vad_threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 700,
    ) -> None:
        """Initialize VAD recorder.

        Args:
            sample_rate: Recording sample rate (default 16kHz)
            streaming_asr_callbacks: Optional callbacks for streaming ASR
            on_speech_captured: Callback when complete utterance is captured (receives WAV bytes)
            event_loop: Event loop for async callback execution
            vad_threshold: VAD detection threshold (0-1)
            min_speech_duration_ms: Minimum speech duration to trigger detection
            min_silence_duration_ms: Silence duration to end speech segment

        """
        self.sample_rate = sample_rate
        self.streaming_callbacks = streaming_asr_callbacks
        self.on_speech_captured = on_speech_captured
        self.event_loop = event_loop
        self.vad_threshold = vad_threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms

        self._active = False
        self._state = ContinuousState.IDLE
        self._audio_frames: List[npt.NDArray[np.int16]] = []
        self._continuous_thread: threading.Thread | None = None
        self._vad: SileroVAD | None = None

    @property
    def state(self) -> ContinuousState:
        """Current VAD state."""
        return self._state

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

        self._active = True
        self._audio_frames = []

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

        self._state = ContinuousState.IDLE
        logger.info("Continuous mode stopped")
        return "Continuous mode stopped"

    def _continuous_record_loop(self) -> None:
        """Continuous recording loop with VAD-based speech detection.

        State machine: IDLE -> LISTENING -> RECORDING -> PROCESSING -> LISTENING
        """
        import time

        from reachy_mini_conversation_app.cascade.vad import SILERO_SAMPLE_RATE
        from reachy_mini_conversation_app.cascade.timing import tracker

        logger.info("Continuous mode started - listening for speech...")
        self._state = ContinuousState.LISTENING

        # VAD processes 16kHz audio, but we record at self.sample_rate
        # Silero VAD works best with chunks of 512-1536 samples at 16kHz
        vad_chunk_samples = 512  # 32ms at 16kHz
        record_chunk_samples = int(vad_chunk_samples * self.sample_rate / SILERO_SAMPLE_RATE)

        try:
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                dtype=np.int16,
                blocksize=record_chunk_samples,
            ) as stream:
                while self._active:
                    # Read audio chunk
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

                    # Process through VAD
                    if self._vad is None:
                        continue

                    speech_started, speech_ended = self._vad.process_chunk(vad_audio, SILERO_SAMPLE_RATE)

                    if self._state == ContinuousState.LISTENING:
                        # Waiting for speech to start
                        if speech_started:
                            logger.info("VAD: Speech started - recording")
                            self._state = ContinuousState.RECORDING
                            self._audio_frames = []
                            tracker.reset("user_conversation_turn")
                            tracker.mark("vad_speech_start")

                            # Initialize streaming ASR if callbacks provided
                            if self.streaming_callbacks:
                                self.streaming_callbacks.on_start()

                        # Always collect audio (we might need pre-roll)
                        self._audio_frames.append(data.copy())
                        # Keep only last ~500ms of audio as pre-roll buffer
                        max_preroll_chunks = int(0.5 * self.sample_rate / record_chunk_samples)
                        if len(self._audio_frames) > max_preroll_chunks:
                            self._audio_frames = self._audio_frames[-max_preroll_chunks:]

                    elif self._state == ContinuousState.RECORDING:
                        # Recording speech
                        self._audio_frames.append(data.copy())

                        # Send chunk to streaming ASR if callbacks provided
                        if self.streaming_callbacks:
                            wav_buffer = io.BytesIO()
                            with wave.open(wav_buffer, "wb") as wav_file:
                                wav_file.setnchannels(1)
                                wav_file.setsampwidth(2)
                                wav_file.setframerate(self.sample_rate)
                                wav_file.writeframes(data.tobytes())
                            chunk_wav = wav_buffer.getvalue()
                            self.streaming_callbacks.on_chunk(chunk_wav)

                        if speech_ended:
                            logger.info("VAD: Speech ended - processing")
                            tracker.mark("vad_speech_end")
                            self._state = ContinuousState.PROCESSING

                            # Process the recorded audio
                            if self._audio_frames:
                                audio_data = np.concatenate(self._audio_frames)
                                duration = len(audio_data) / self.sample_rate
                                logger.info(f"VAD: Captured {duration:.2f}s of speech")
                                tracker.mark("recording_captured", {"duration_s": round(duration, 2)})

                                # Convert to WAV and call callback
                                wav_buffer = io.BytesIO()
                                with wave.open(wav_buffer, "wb") as wav_file:
                                    wav_file.setnchannels(1)
                                    wav_file.setsampwidth(2)
                                    wav_file.setframerate(self.sample_rate)
                                    wav_file.writeframes(audio_data.tobytes())

                                wav_bytes = wav_buffer.getvalue()

                                # Call speech captured callback
                                if self.on_speech_captured:
                                    self.on_speech_captured(wav_bytes)

                            # Reset for next utterance
                            self._audio_frames = []
                            self._vad.reset()
                            self._state = ContinuousState.LISTENING
                            logger.info("VAD: Ready for next utterance")

                    elif self._state == ContinuousState.PROCESSING:
                        # Wait for processing to complete (shouldn't stay here long)
                        time.sleep(0.01)

        except Exception as e:
            logger.exception(f"Error in continuous recording loop: {e}")
        finally:
            self._state = ContinuousState.IDLE
            logger.info("Continuous mode stopped")
