"""Console mode for cascade pipeline using VAD-based speech detection.

This module provides a console interface for the cascade pipeline (ASR→LLM→TTS)
without requiring Gradio UI. It uses Silero VAD for automatic speech detection.
"""

from __future__ import annotations
import io
import time
import wave
import asyncio
import logging
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import resample

from reachy_mini_conversation_app.cascade.vad import SILERO_SAMPLE_RATE, SileroVAD
from reachy_mini_conversation_app.cascade.timing import tracker


if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from reachy_mini_conversation_app.cascade.handler import CascadeHandler

logger = logging.getLogger(__name__)

# Minimum chunk size for Silero VAD (512 samples = 32ms at 16kHz)
VAD_MIN_CHUNK_SIZE = 512


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo audio to mono by taking first channel."""
    if audio.ndim == 1:
        return audio
    if audio.shape[0] > audio.shape[1]:
        return audio[:, 0]  # (samples, channels)
    return audio[0, :]  # (channels, samples)


class AudioChunkBuffer:
    """Accumulates audio samples and yields fixed-size chunks for VAD."""

    def __init__(self, chunk_size: int, max_samples: int = 16000) -> None:
        """Initialize buffer with given chunk size."""
        self._buffer = np.zeros(max_samples, dtype=np.int16)
        self._pos = 0
        self._chunk_size = chunk_size

    def add(self, samples: np.ndarray) -> None:
        """Add samples to buffer, growing if needed."""
        n = samples.size
        if self._pos + n > self._buffer.size:
            new_size = max(self._buffer.size * 2, self._pos + n)
            new_buffer = np.zeros(new_size, dtype=np.int16)
            new_buffer[: self._pos] = self._buffer[: self._pos]
            self._buffer = new_buffer
        self._buffer[self._pos : self._pos + n] = samples.flatten()
        self._pos += n

    def get_chunks(self) -> list[np.ndarray]:
        """Return all complete chunks, keep remainder."""
        chunks = []
        while self._pos >= self._chunk_size:
            chunks.append(self._buffer[: self._chunk_size].copy())
            self._buffer[: self._pos - self._chunk_size] = self._buffer[self._chunk_size : self._pos]
            self._pos -= self._chunk_size
        return chunks

    def clear(self) -> None:
        """Reset buffer."""
        self._pos = 0


class VADState(Enum):
    """VAD state machine states."""

    LISTENING = auto()
    RECORDING = auto()
    PROCESSING = auto()


class CascadeLocalStream:
    """Console stream for cascade pipeline using VAD-based speech detection."""

    def __init__(self, handler: CascadeHandler, robot: ReachyMini) -> None:
        """Initialize the console stream."""
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
        self._vad_chunk_buffer = AudioChunkBuffer(VAD_MIN_CHUNK_SIZE)
        self._speech_chunks: list[np.ndarray] = []
        self._playback_queue: asyncio.Queue[bytes] = asyncio.Queue()

        # Wire speech output so handler plays audio through robot speaker
        from reachy_mini_conversation_app.cascade.speech_output import ConsoleSpeechOutput

        self.handler.speech_output = ConsoleSpeechOutput(
            tts=self.handler.tts,
            head_wobbler=self.handler.deps.head_wobbler,
            playback_callback=self._queue_audio_for_playback,
        )
        logger.info("CascadeLocalStream initialized")

    async def _queue_audio_for_playback(self, audio_bytes: bytes) -> None:
        """Queue audio bytes for playback on the robot speaker."""
        await self._playback_queue.put(audio_bytes)

    def launch(self) -> None:
        """Start the console stream and run the async processing loops."""
        self._stop_event.clear()

        logger.info("Starting media recording and playback...")
        self._robot.media.start_recording()
        self._robot.media.start_playing()
        time.sleep(1)  # Give pipelines time to start

        logger.info("Console mode ready. Speak to interact with the robot. Press Ctrl+C to stop.")
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
        logger.info(f"Audio recording at {input_sample_rate} Hz, listening...")

        while not self._stop_event.is_set():
            audio_frame = self._robot.media.get_audio_sample()
            if audio_frame is None:
                await asyncio.sleep(0.01)
                continue

            # Resample to 16kHz if needed (VAD requires 16kHz)
            if input_sample_rate != SILERO_SAMPLE_RATE:
                num_samples = int(len(audio_frame) * SILERO_SAMPLE_RATE / input_sample_rate)
                audio_frame = resample(audio_frame, num_samples).astype(np.float32)

            audio_frame = to_mono(audio_frame)
            audio_int16 = (audio_frame * 32767).astype(np.int16)

            self._vad_chunk_buffer.add(audio_int16)
            for vad_chunk in self._vad_chunk_buffer.get_chunks():
                await self._process_vad(vad_chunk)

            await asyncio.sleep(0)

    async def _process_vad(self, audio_chunk: np.ndarray) -> None:
        """Process audio chunk through VAD state machine."""
        streaming = self.handler.is_streaming_asr

        if self._state == VADState.LISTENING:
            speech_started, _ = self._vad.process_chunk(audio_chunk, SILERO_SAMPLE_RATE)
            if speech_started:
                self._state = VADState.RECORDING
                self._speech_chunks = [audio_chunk]
                logger.info("Speech detected, recording...")
                if streaming:
                    await self.handler.process_audio_streaming_start()
                    wav_bytes = self._audio_to_wav(audio_chunk, SILERO_SAMPLE_RATE)
                    await self.handler.process_audio_streaming_chunk(wav_bytes)

        elif self._state == VADState.RECORDING:
            self._speech_chunks.append(audio_chunk)
            if streaming:
                wav_bytes = self._audio_to_wav(audio_chunk, SILERO_SAMPLE_RATE)
                await self.handler.process_audio_streaming_chunk(wav_bytes)
            _, speech_ended = self._vad.process_chunk(audio_chunk, SILERO_SAMPLE_RATE)
            if speech_ended:
                self._state = VADState.PROCESSING
                logger.info(f"Speech ended, {len(self._speech_chunks)} chunks")
                # Start latency tracking from speech end
                tracker.reset("vad_speech_end")
                tracker.mark("vad_speech_end")

                if streaming:
                    # Mark recording_captured before ASR finalization
                    # (audio was already streamed, so capture is instant)
                    audio_data = np.concatenate(self._speech_chunks)
                    duration = len(audio_data) / SILERO_SAMPLE_RATE
                    tracker.mark("recording_captured", {"duration_s": round(duration, 2)})
                    turn = await self.handler.process_audio_streaming_end()
                    transcript = turn.transcript
                    if transcript:
                        logger.info(f"User: {transcript}")
                    else:
                        logger.info("No speech detected")
                    tracker.print_summary()
                else:
                    await self._process_recorded_audio()

                # Reset for next utterance
                self._speech_chunks = []
                self._vad.reset()
                self._state = VADState.LISTENING
                logger.info("Listening...")

    async def _process_recorded_audio(self) -> None:
        """Process recorded audio through the cascade pipeline."""
        if not self._speech_chunks:
            logger.warning("Empty audio buffer, skipping")
            return

        audio_data = np.concatenate(self._speech_chunks)
        duration = len(audio_data) / SILERO_SAMPLE_RATE
        logger.info(f"Processing {len(audio_data)} samples ({duration:.2f}s)")
        tracker.mark("recording_captured", {"duration_s": round(duration, 2)})

        wav_bytes = self._audio_to_wav(audio_data, SILERO_SAMPLE_RATE)
        logger.info("Transcribing...")

        turn = await self.handler.process_audio_manual(wav_bytes)
        transcript = turn.transcript
        if transcript:
            logger.info(f"User: {transcript}")
        else:
            logger.info("No speech detected")

        # Print latency summary for this turn
        tracker.print_summary()

    def _audio_to_wav(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert int16 audio array to WAV bytes."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
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

            # Resample from TTS rate to output rate if needed
            tts_rate = self.handler.tts.sample_rate
            if tts_rate != output_sample_rate:
                num_samples = int(len(audio_float) * output_sample_rate / tts_rate)
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
