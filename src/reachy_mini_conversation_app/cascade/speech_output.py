"""SpeechOutput protocol and implementations for cascade TTS playback."""

from __future__ import annotations
import re
import time
import base64
import asyncio
import logging
from typing import TYPE_CHECKING, Callable, Protocol, Awaitable

import numpy as np


if TYPE_CHECKING:
    import numpy.typing as npt

    from reachy_mini_conversation_app.cascade.tts import TTSProvider
    from reachy_mini_conversation_app.audio.head_wobbler import HeadWobbler
    from reachy_mini_conversation_app.cascade.ui.audio_playback import AudioPlaybackSystem


logger = logging.getLogger(__name__)


class SpeechOutput(Protocol):
    """Protocol for TTS playback backends."""

    async def speak(self, text: str) -> None:
        """Synthesize and play speech."""
        ...


# ---------------------------------------------------------------------------
# Console mode
# ---------------------------------------------------------------------------


class ConsoleSpeechOutput:
    """TTS playback for console mode — streams chunks to a playback callback with rate limiting."""

    def __init__(
        self,
        tts: TTSProvider,
        head_wobbler: HeadWobbler | None,
        playback_callback: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """Initialize with TTS provider, optional head wobbler, and playback callback."""
        self.tts = tts
        self.head_wobbler = head_wobbler
        self.playback_callback = playback_callback

    async def speak(self, text: str) -> None:
        """Stream TTS chunks to playback callback with rate limiting and head wobble."""
        from reachy_mini_conversation_app.cascade.timing import tracker

        if self.head_wobbler:
            self.head_wobbler.reset()

        print(f"[TTS] Speaking: {text}")

        audio_chunks: list[bytes] = []
        first_chunk_sent = False

        async for chunk in self.tts.synthesize(text):
            audio_chunks.append(chunk)

            if self.head_wobbler:
                self.head_wobbler.feed(base64.b64encode(chunk).decode("utf-8"))

            await self.playback_callback(chunk)
            if not first_chunk_sent:
                tracker.mark("audio_playback_started")
                first_chunk_sent = True

            # Rate limiting: match audio generation speed
            chunk_duration = len(chunk) / (2 * self.tts.sample_rate)
            await asyncio.sleep(chunk_duration * 0.95)

        logger.info(f"Generated {len(audio_chunks)} audio chunks for head animation")

        # Small buffer to let remaining audio drain
        await asyncio.sleep(0.5)

        if self.head_wobbler:
            self.head_wobbler.reset()


# ---------------------------------------------------------------------------
# Gradio mode
# ---------------------------------------------------------------------------


class GradioSpeechOutput:
    """TTS playback for Gradio mode — parallel sentence generation with ordered queuing."""

    def __init__(self, tts: TTSProvider, playback: AudioPlaybackSystem) -> None:
        """Initialize with TTS provider and Gradio audio playback system."""
        self.tts = tts
        self.playback = playback

    async def speak(self, text: str) -> None:
        """Split text into sentences, generate TTS in parallel, and queue for ordered playback."""
        from reachy_mini_conversation_app.cascade.timing import tracker

        logger.info(f"Synthesizing speech: '{text[:50]}...'")

        audio_chunks: list[npt.NDArray[np.int16]] = []
        first_chunk_queued = False

        sentences = split_into_sentences(text)
        logger.debug(f"Split text into {len(sentences)} sentence chunks for streaming TTS")
        for i, s in enumerate(sentences):
            logger.debug(f"  Sentence {i + 1}: '{s}'")

        logger.debug("Using pre-warmed audio playback system")

        total_chunks = 0

        # Gate events ensure chunks queue in sentence order even if TTS responses arrive out of order
        queue_events = [asyncio.Event() for _ in sentences]
        queue_events[0].set()  # Sentence 0 can queue immediately

        async def generate_and_queue_sentence(idx: int, sentence: str) -> list[npt.NDArray[np.int16]]:
            """Generate TTS for one sentence and queue chunks in order."""
            nonlocal total_chunks, first_chunk_queued

            logger.debug(f"TTS sentence {idx + 1}/{len(sentences)}: '{sentence}' (PARALLEL)")
            sentence_chunks: list[npt.NDArray[np.int16]] = []
            sentence_start = time.time()

            gate_is_open = queue_events[idx].is_set()

            if gate_is_open:
                async for chunk in self.tts.synthesize(sentence):
                    total_chunks += 1
                    audio_array = np.frombuffer(chunk, dtype=np.int16)
                    sentence_chunks.append(audio_array)
                    audio_chunks.append(audio_array)
                    self.playback.put_audio(audio_array)
                    self.playback.put_wobbler(chunk)
                    if not first_chunk_queued:
                        first_chunk_queued = True
                        tracker.mark("audio_playback_started")
                        logger.info("First audio chunk playing - playback started while TTS continues in background")
            else:
                raw_chunks: list[bytes] = []
                async for chunk in self.tts.synthesize(sentence):
                    total_chunks += 1
                    audio_array = np.frombuffer(chunk, dtype=np.int16)
                    sentence_chunks.append(audio_array)
                    raw_chunks.append(chunk)

                await queue_events[idx].wait()

                for audio_array, raw_chunk in zip(sentence_chunks, raw_chunks):
                    audio_chunks.append(audio_array)
                    self.playback.put_audio(audio_array)
                    self.playback.put_wobbler(raw_chunk)
                    if not first_chunk_queued:
                        first_chunk_queued = True
                        tracker.mark("audio_playback_started")
                        logger.info("First audio chunk playing - playback started while TTS continues in background")

            gen_duration = time.time() - sentence_start
            if sentence_chunks:
                total_samples = sum(len(c) for c in sentence_chunks)
                logger.debug(
                    f"Sentence {idx + 1} generated: {len(sentence_chunks)} chunks "
                    f"({total_samples} samples, {total_samples / self.tts.sample_rate:.2f}s) "
                    f"in {gen_duration:.2f}s"
                )

            if idx + 1 < len(queue_events):
                queue_events[idx + 1].set()

            logger.debug(f"Sentence {idx + 1} queued for playback")
            return sentence_chunks

        # Generate sentences with intelligent overlap
        tasks: list[asyncio.Task] = []
        for idx, sentence in enumerate(sentences):
            if idx == 0:
                task = asyncio.create_task(generate_and_queue_sentence(idx, sentence))
                tasks.append(task)
            elif idx == 1:
                await asyncio.sleep(0.3)
                task = asyncio.create_task(generate_and_queue_sentence(idx, sentence))
                tasks.append(task)
            else:
                if idx >= 2 and tasks:
                    await tasks[idx - 1]
                task = asyncio.create_task(generate_and_queue_sentence(idx, sentence))
                tasks.append(task)

        await asyncio.gather(*tasks)
        logger.info(f"Parallel TTS complete: All {len(sentences)} sentences generated")
        logger.info(f"Generated {total_chunks} total audio chunks from {len(sentences)} sentences")

        self.playback.signal_end_of_turn()

        # Wait for audio to finish playing
        if audio_chunks:
            total_samples = sum(len(chunk) for chunk in audio_chunks)
            duration_seconds = total_samples / self.tts.sample_rate
            logger.info(f"Waiting {duration_seconds:.1f}s for playback to complete...")
            await asyncio.sleep(duration_seconds + 0.5)

        logger.info("Playback complete (using pre-warmed system)")

        tracker.print_summary()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def split_into_sentences(text: str, min_length: int = 8) -> list[str]:
    """Split text into sentence-like chunks for streaming TTS.

    Splits on: . ! ? , ; — (but keeps punctuation with the sentence)

    Args:
        text: Text to split
        min_length: Minimum characters per segment (default 8)

    Returns:
        List of text segments, each at least min_length characters (except possibly the last)

    """
    pattern = r"([.!?,;—]\s+)"
    parts = re.split(pattern, text)

    raw_sentences: list[str] = []
    current = ""
    for part in parts:
        current += part
        if re.match(pattern, part):
            if current.strip():
                raw_sentences.append(current.strip())
            current = ""

    if current.strip():
        raw_sentences.append(current.strip())

    if not raw_sentences:
        return [text]

    merged_sentences: list[str] = []
    accumulator = ""

    for sentence in raw_sentences:
        if accumulator:
            accumulator += " " + sentence
        else:
            accumulator = sentence

        if len(accumulator) >= min_length:
            merged_sentences.append(accumulator)
            accumulator = ""

    if accumulator:
        if merged_sentences and len(merged_sentences[-1]) < min_length * 2:
            merged_sentences[-1] += " " + accumulator
        else:
            merged_sentences.append(accumulator)

    return merged_sentences if merged_sentences else [text]
