"""Console mode for cascade pipeline using VAD-based speech detection.

This module provides a console interface for the cascade pipeline (ASR→LLM→TTS)
without requiring Gradio UI. It uses Silero VAD for automatic speech detection.
"""

from __future__ import annotations
import time
import asyncio
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import sounddevice as sd
import numpy.typing as npt
from scipy.signal import resample

from reachy_mini_conversation_app.cascade.vad import (
    VAD_CHUNK_SIZE,
    SILERO_SAMPLE_RATE,
    VADEvent,
    VADState,
    SileroVAD,
    VADStateMachine,
)
from reachy_mini_conversation_app.cascade.timing import tracker
from reachy_mini_conversation_app.cascade.asr.audio_utils import pcm_to_wav


if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from reachy_mini_conversation_app.cascade.handler import CascadeHandler

logger = logging.getLogger(__name__)


class CascadeLocalStream:
    """Console stream for cascade pipeline using VAD-based speech detection."""

    def __init__(self, handler: CascadeHandler, robot: ReachyMini) -> None:
        """Initialize the console stream."""
        self.handler = handler
        self._robot = robot

        # VAD state machine
        vad = SileroVAD(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=700,
        )
        self._vad_sm = VADStateMachine(vad)

        # State
        self._stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []
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

        # start_recording() also starts the camera pipeline
        logger.info("Starting media recording and playback...")
        self._robot.media.start_recording()
        self._robot.media.start_playing()
        time.sleep(1)  # Give pipelines time to start

        # Log which mic we'll use (system default, not robot USB device)
        default_dev = sd.query_devices(kind="input")
        logger.info(
            f"Mic input: '{default_dev['name']}' (system default, "
            f"{default_dev['default_samplerate']:.0f} Hz)"
        )
        output_sr = self._robot.media.get_output_audio_samplerate()
        logger.info(f"Speaker output: {output_sr} Hz (robot)")

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
        """Read mic audio from system default device and process through VAD."""
        logger.info(f"Recording from system default mic at {SILERO_SAMPLE_RATE} Hz, listening...")

        stream = sd.InputStream(
            samplerate=SILERO_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=VAD_CHUNK_SIZE,
        )
        stream.start()

        try:
            while not self._stop_event.is_set():
                # Read exactly one VAD chunk (512 samples = 32ms at 16kHz)
                audio_frame, overflowed = stream.read(VAD_CHUNK_SIZE)
                if overflowed:
                    logger.debug("Audio input overflowed")

                audio_int16 = audio_frame[:, 0].astype(np.int16)  # (samples, 1) → (samples,)
                await self._process_vad(audio_int16)
                await asyncio.sleep(0)
        finally:
            stream.stop()
            stream.close()

    async def _process_vad(self, audio_chunk: npt.NDArray[np.int16]) -> None:
        """Process audio chunk through VAD state machine."""
        streaming = self.handler.is_streaming_asr
        event = self._vad_sm.process_chunk(audio_chunk)

        if event == VADEvent.SPEECH_STARTED:
            if streaming:
                await self.handler.process_audio_streaming_start()
                for chunk in self._vad_sm.speech_chunks:
                    wav_bytes = pcm_to_wav(chunk.tobytes(), SILERO_SAMPLE_RATE)
                    await self.handler.process_audio_streaming_chunk(wav_bytes)

        elif event == VADEvent.SPEECH_ENDED:
            # Start latency tracking from speech end
            tracker.reset("vad_speech_end")
            tracker.mark("vad_speech_end")

            if streaming:
                audio_data = np.concatenate(self._vad_sm.speech_chunks)
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

            self._vad_sm.finish_processing()
            logger.info("Listening...")

        elif self._vad_sm.state == VADState.RECORDING and streaming:
            # Mid-recording: stream current chunk to ASR
            wav_bytes = pcm_to_wav(audio_chunk.tobytes(), SILERO_SAMPLE_RATE)
            await self.handler.process_audio_streaming_chunk(wav_bytes)

    async def _process_recorded_audio(self) -> None:
        """Process recorded audio through the cascade pipeline."""
        if not self._vad_sm.speech_chunks:
            logger.warning("Empty audio buffer, skipping")
            return

        audio_data = np.concatenate(self._vad_sm.speech_chunks)
        duration = len(audio_data) / SILERO_SAMPLE_RATE
        logger.info(f"Processing {len(audio_data)} samples ({duration:.2f}s)")
        tracker.mark("recording_captured", {"duration_s": round(duration, 2)})

        wav_bytes = pcm_to_wav(audio_data.tobytes(), SILERO_SAMPLE_RATE)
        logger.info("Transcribing...")

        turn = await self.handler.process_audio_manual(wav_bytes)
        transcript = turn.transcript
        if transcript:
            logger.info(f"User: {transcript}")
        else:
            logger.info("No speech detected")

        # Print latency summary for this turn
        tracker.print_summary()

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
