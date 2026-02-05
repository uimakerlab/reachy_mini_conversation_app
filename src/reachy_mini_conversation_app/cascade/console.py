"""Console mode for cascade pipeline using VAD-based speech detection.

This module provides a console interface for the cascade pipeline (ASR→LLM→TTS)
without requiring Gradio UI. It uses Silero VAD for automatic speech detection.
"""

from __future__ import annotations
import io
import wave
import asyncio
import logging
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import resample

from reachy_mini_conversation_app.cascade.vad import SILERO_SAMPLE_RATE, SileroVAD


# Minimum chunk size for Silero VAD (512 samples = 32ms at 16kHz)
VAD_MIN_CHUNK_SIZE = 512

if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from reachy_mini_conversation_app.cascade.handler import CascadeHandler


logger = logging.getLogger(__name__)

# TTS output sample rate (OpenAI TTS outputs PCM at 24kHz)
TTS_SAMPLE_RATE = 24000


class VADState(Enum):
    """VAD state machine states."""

    LISTENING = auto()  # Waiting for speech to start
    RECORDING = auto()  # Recording speech in progress
    PROCESSING = auto()  # Processing recorded audio through pipeline


class CascadeLocalStream:
    """Console stream for cascade pipeline using VAD-based speech detection."""

    def __init__(self, handler: CascadeHandler, robot: ReachyMini) -> None:
        """Initialize the console stream.

        Args:
            handler: CascadeHandler instance for processing audio
            robot: ReachyMini instance for audio I/O

        """
        self.handler = handler
        self._robot = robot

        # VAD for speech detection
        self._vad = SileroVAD(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=700,
        )

        # State
        self._state = VADState.LISTENING
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

        # Audio buffers
        self._vad_input_buffer: list[np.ndarray] = []  # Buffer for accumulating VAD input
        self._audio_buffer: list[np.ndarray] = []  # Buffer for recording speech
        self._playback_queue: asyncio.Queue[bytes] = asyncio.Queue()

        # Set playback callback on handler
        self.handler.set_playback_callback(self._queue_audio_for_playback)

        logger.info("CascadeLocalStream initialized")

    async def _queue_audio_for_playback(self, audio_bytes: bytes) -> None:
        """Queue audio bytes for playback on the robot speaker."""
        await self._playback_queue.put(audio_bytes)

    def launch(self) -> None:
        """Start the console stream and run the async processing loops."""
        self._stop_event.clear()

        # Start media
        logger.info("Starting media recording and playback...")
        self._robot.media.start_recording()
        self._robot.media.start_playing()

        # Give pipelines time to start
        import time

        time.sleep(1)

        print("\n[CASCADE] Console mode ready. Speak to interact with the robot.")
        print("[CASCADE] Press Ctrl+C to stop.\n")

        asyncio.run(self._main_loop())

    async def _main_loop(self) -> None:
        """Run record and play loops concurrently."""
        self._tasks = [
            asyncio.create_task(self._record_loop(), name="cascade-record-loop"),
            asyncio.create_task(self._play_loop(), name="cascade-play-loop"),
        ]
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Tasks cancelled during shutdown")

    async def _record_loop(self) -> None:
        """Read mic frames and process through VAD state machine."""
        input_sample_rate = self._robot.media.get_input_audio_samplerate()
        logger.info(f"Audio recording at {input_sample_rate} Hz")

        print("[CASCADE] Listening... (speak to begin)")

        while not self._stop_event.is_set():
            audio_frame = self._robot.media.get_audio_sample()

            if audio_frame is None:
                await asyncio.sleep(0.01)
                continue

            # Resample to 16kHz if needed (VAD requires 16kHz)
            if input_sample_rate != SILERO_SAMPLE_RATE:
                num_samples = int(len(audio_frame) * SILERO_SAMPLE_RATE / input_sample_rate)
                audio_frame = resample(audio_frame, num_samples).astype(np.float32)

            # Ensure audio is 1D (mono) - take first channel if stereo
            if audio_frame.ndim > 1:
                audio_frame = audio_frame[:, 0] if audio_frame.shape[1] <= audio_frame.shape[0] else audio_frame[0, :]

            # Convert float32 to int16 for VAD
            audio_int16 = (audio_frame * 32767).astype(np.int16)

            # Accumulate in VAD input buffer
            self._vad_input_buffer.append(audio_int16)

            # Check if we have enough samples for VAD processing
            total_samples = sum(chunk.size for chunk in self._vad_input_buffer)
            while total_samples >= VAD_MIN_CHUNK_SIZE:
                # Concatenate buffered audio
                all_audio = np.concatenate(self._vad_input_buffer).flatten()

                # Take exactly VAD_MIN_CHUNK_SIZE samples (Silero requires exact size)
                vad_chunk = all_audio[:VAD_MIN_CHUNK_SIZE]

                # Keep remainder in buffer
                remainder = all_audio[VAD_MIN_CHUNK_SIZE:]
                self._vad_input_buffer = [remainder] if remainder.size > 0 else []
                total_samples = remainder.size

                # Process through VAD state machine
                await self._process_vad(vad_chunk)

            await asyncio.sleep(0)  # Yield to event loop

    async def _process_vad(self, audio_chunk: np.ndarray) -> None:
        """Process audio chunk through VAD state machine.

        Args:
            audio_chunk: Audio samples (int16, 16kHz, at least VAD_MIN_CHUNK_SIZE samples)

        """
        if self._state == VADState.LISTENING:
            speech_started, _ = self._vad.process_chunk(audio_chunk, SILERO_SAMPLE_RATE)
            if speech_started:
                self._state = VADState.RECORDING
                self._audio_buffer = [audio_chunk]
                print("[VAD] Speech detected - recording...")
                logger.info("VAD: Speech started, recording...")

        elif self._state == VADState.RECORDING:
            self._audio_buffer.append(audio_chunk)
            _, speech_ended = self._vad.process_chunk(audio_chunk, SILERO_SAMPLE_RATE)
            if speech_ended:
                self._state = VADState.PROCESSING
                print("[VAD] Speech ended - processing...")
                logger.info(f"VAD: Speech ended, buffer has {len(self._audio_buffer)} chunks")

                # Process the recorded audio
                await self._process_recorded_audio()

                # Reset for next utterance
                self._audio_buffer = []
                self._vad.reset()
                self._state = VADState.LISTENING
                print("[CASCADE] Listening... (speak to begin)")

        # PROCESSING state is handled within _process_recorded_audio

    async def _process_recorded_audio(self) -> None:
        """Process recorded audio through the cascade pipeline."""
        if not self._audio_buffer:
            logger.warning("Empty audio buffer, skipping processing")
            return

        # Concatenate all audio chunks
        audio_data = np.concatenate(self._audio_buffer)
        logger.info(f"Processing {len(audio_data)} samples ({len(audio_data) / SILERO_SAMPLE_RATE:.2f}s)")

        # Convert to WAV bytes for the handler
        wav_bytes = self._audio_to_wav(audio_data, SILERO_SAMPLE_RATE)

        print("[ASR] Transcribing...")

        # Process through cascade pipeline
        transcript = await self.handler.process_audio_manual(wav_bytes)

        if transcript:
            print(f"[USER] {transcript}")
        else:
            print("[ASR] (no speech detected)")

    def _audio_to_wav(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert audio numpy array to WAV bytes.

        Args:
            audio: Audio samples (int16)
            sample_rate: Sample rate

        Returns:
            WAV file bytes

        """
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return buffer.getvalue()

    async def _play_loop(self) -> None:
        """Pull audio from playback queue and push to robot speaker."""
        output_sample_rate = self._robot.media.get_output_audio_samplerate()
        logger.info(f"Audio playback at {output_sample_rate} Hz")

        while not self._stop_event.is_set():
            try:
                # Non-blocking get with timeout
                audio_bytes = await asyncio.wait_for(
                    self._playback_queue.get(),
                    timeout=0.1,
                )
            except asyncio.TimeoutError:
                continue

            # Convert PCM int16 bytes to numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

            # Convert to float32 [-1, 1]
            audio_float = audio_data.astype(np.float32) / 32768.0

            # Resample from TTS rate (24kHz) to output rate if needed
            if TTS_SAMPLE_RATE != output_sample_rate:
                num_samples = int(len(audio_float) * output_sample_rate / TTS_SAMPLE_RATE)
                audio_float = resample(audio_float, num_samples).astype(np.float32)

            # Push to robot speaker
            self._robot.media.push_audio_sample(audio_float)

    def close(self) -> None:
        """Stop the stream and cleanup."""
        logger.info("Stopping CascadeLocalStream...")

        # Stop media first
        try:
            self._robot.media.stop_recording()
        except Exception as e:
            logger.debug(f"Error stopping recording: {e}")

        try:
            self._robot.media.stop_playing()
        except Exception as e:
            logger.debug(f"Error stopping playback: {e}")

        # Signal async loops to stop
        self._stop_event.set()

        # Cancel tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        logger.info("CascadeLocalStream stopped")
