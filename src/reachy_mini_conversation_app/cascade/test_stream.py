"""Test file mode for cascade pipeline — feeds text utterances through TTS→ASR→LLM→TTS.

Reads utterances from a text file, synthesizes them to audio via TTS, and feeds
the audio through the full cascade pipeline without human interaction.
"""

from __future__ import annotations
import io
import wave
import asyncio
import logging
import threading
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import sounddevice as sd

from reachy_mini_conversation_app.cascade.timing import tracker
from reachy_mini_conversation_app.cascade.speech_output import ConsoleSpeechOutput
from reachy_mini_conversation_app.cascade.provider_factory import init_tts_provider


if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from reachy_mini_conversation_app.cascade.handler import CascadeHandler

logger = logging.getLogger(__name__)

DEFAULT_DELAY_BETWEEN_TURNS = 2.0


class CascadeTestStream:
    """Feeds text utterances from a file through the full cascade pipeline."""

    def __init__(self, handler: CascadeHandler, robot: ReachyMini, test_file: str) -> None:
        """Initialize test stream with a separate TTS for input audio generation."""
        self.handler = handler
        self._robot = robot
        self._test_file = Path(test_file)
        self._delay = DEFAULT_DELAY_BETWEEN_TURNS

        # Separate TTS instance for generating "user voice" audio
        self._input_tts = init_tts_provider()
        sample_rate = self._input_tts.sample_rate

        # Audio buffer for callback-based sounddevice playback.
        # The callback pulls from this buffer at a steady rate, avoiding
        # gaps that occur when pushing chunks with blocking write().
        self._audio_buffer = bytearray()
        self._buffer_lock = threading.Lock()

        # Sounddevice output stream in callback mode — pulls audio smoothly
        self._sd_stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            callback=self._audio_callback,
        )

        # Wire speech output so handler plays audio through computer speakers
        self.handler.speech_output = ConsoleSpeechOutput(
            tts=self.handler.tts,
            head_wobbler=self.handler.deps.head_wobbler,
            playback_callback=self._queue_audio_chunk,
        )

        logger.info(f"CascadeTestStream initialized (file={self._test_file}, rate={sample_rate}Hz)")

    def _audio_callback(
        self, outdata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags
    ) -> None:
        """Sounddevice callback — fills output buffer from our audio buffer."""
        bytes_needed = frames * 2  # int16 = 2 bytes per frame
        with self._buffer_lock:
            available = len(self._audio_buffer)
            if available >= bytes_needed:
                data = bytes(self._audio_buffer[:bytes_needed])
                del self._audio_buffer[:bytes_needed]
                outdata[:] = np.frombuffer(data, dtype=np.int16).reshape(-1, 1)
            elif available > 0:
                data = bytes(self._audio_buffer)
                self._audio_buffer.clear()
                frames_available = available // 2
                outdata[:frames_available] = np.frombuffer(data, dtype=np.int16).reshape(-1, 1)
                outdata[frames_available:] = 0
            else:
                outdata[:] = 0

    async def _queue_audio_chunk(self, audio_bytes: bytes) -> None:
        """Append audio chunk to the playback buffer (non-blocking)."""
        with self._buffer_lock:
            self._audio_buffer.extend(audio_bytes)

    def launch(self) -> None:
        """Start the test stream (blocking)."""
        self._sd_stream.start()
        logger.info("Test file mode ready. Running utterances...")
        asyncio.run(self._main_loop())

    async def _main_loop(self) -> None:
        """Process each utterance through the full pipeline."""
        utterances = self._load_utterances()
        logger.info(f"Loaded {len(utterances)} utterances from {self._test_file}")

        for i, text in enumerate(utterances):
            logger.info(f"\n{'='*60}")
            logger.info(f"Utterance {i+1}/{len(utterances)}: \"{text}\"")
            logger.info(f"{'='*60}")

            tracker.reset("test_utterance_start")
            tracker.mark("test_utterance_start")

            # Generate WAV audio from text
            wav_bytes = await self._text_to_wav(text)
            tracker.mark("input_tts_complete")
            logger.info(f"Generated input audio ({len(wav_bytes)} bytes)")

            # Feed through the cascade pipeline (ASR → LLM → TTS)
            tracker.mark("recording_captured")
            turn = await self.handler.process_audio_manual(wav_bytes)

            # Log results
            if turn.transcript:
                logger.info(f"ASR heard: \"{turn.transcript}\"")
            for item in turn.items:
                logger.info(f"  {item}")
            if turn.cost > 0:
                logger.info(f"Turn cost: ${turn.cost:.4f}")

            tracker.print_summary()

            # Delay between turns to let movements complete
            logger.info(f"Waiting {self._delay}s before next utterance...")
            await asyncio.sleep(self._delay)

        logger.info(f"\nAll {len(utterances)} utterances processed.")

    async def _text_to_wav(self, text: str) -> bytes:
        """Synthesize text to WAV bytes and play on speakers as "user voice"."""
        pcm_chunks: list[bytes] = []
        async for chunk in self._input_tts.synthesize(text):
            pcm_chunks.append(chunk)

        pcm_data = b"".join(pcm_chunks)

        # Play the user utterance on speakers so it sounds like a real conversation
        with self._buffer_lock:
            self._audio_buffer.extend(pcm_data)
        duration = len(pcm_data) / (2 * self._input_tts.sample_rate)
        await asyncio.sleep(duration)

        audio = np.frombuffer(pcm_data, dtype=np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._input_tts.sample_rate)
            wf.writeframes(audio.tobytes())
        return buffer.getvalue()

    def _load_utterances(self) -> list[str]:
        """Load utterances from text file, stripping comments and blank lines."""
        lines = self._test_file.read_text().splitlines()
        utterances = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                utterances.append(stripped)
        return utterances

    def close(self) -> None:
        """Stop sounddevice stream."""
        logger.info("Stopping CascadeTestStream...")
        self._sd_stream.stop()
        self._sd_stream.close()
        logger.info("CascadeTestStream stopped")
