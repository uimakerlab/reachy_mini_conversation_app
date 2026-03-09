"""Main cascade handler orchestrating ASR → LLM → TTS pipeline."""

from __future__ import annotations
import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Union

from reachy_mini_conversation_app.cascade import pipeline
from reachy_mini_conversation_app.cascade.asr import ASRProvider, StreamingASRProvider
from reachy_mini_conversation_app.cascade.llm import LLMProvider
from reachy_mini_conversation_app.cascade.tts import TTSProvider
from reachy_mini_conversation_app.cascade.config import get_config
from reachy_mini_conversation_app.cascade.pipeline import PROMPT_LOG, PipelineContext
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
)
from reachy_mini_conversation_app.cascade.turn_result import TurnItem, TurnResult, PipelineResult
from reachy_mini_conversation_app.cascade.provider_factory import (
    init_asr_provider,
    init_llm_provider,
    init_tts_provider,
    init_transcript_analysis,
)
from reachy_mini_conversation_app.cascade.transcript_analysis import (
    NoOpTranscriptManager,
    TranscriptAnalysisManager,
)


if TYPE_CHECKING:
    from reachy_mini_conversation_app.cascade.speech_output import SpeechOutput


logger = logging.getLogger(__name__)


def convert_tool_specs_to_chat_format(realtime_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert tool specs from Realtime API format to Chat Completions API format."""
    chat_specs = []
    for spec in realtime_specs:
        if spec["type"] == "function":
            chat_spec = {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            }
            chat_specs.append(chat_spec)
    return chat_specs


class CascadeHandler:
    """Main handler for cascade pipeline mode."""

    def __init__(self, deps: ToolDependencies):
        """Initialize cascade handler."""
        self.deps = deps

        # Speech output backend (set by console or Gradio frontend)
        self.speech_output: SpeechOutput | None = None

        # Initialize providers based on config
        self.asr = init_asr_provider()
        self.llm = init_llm_provider()
        self.tts = init_tts_provider()

        # Conversation state
        self.conversation_history: List[Dict[str, Any]] = []
        self.processing_lock = asyncio.Lock()
        self.running = False

        # Event loop for async operations
        self.loop: asyncio.AbstractEventLoop | None = None
        self.loop_thread: threading.Thread | None = None

        # Track last partial transcript to avoid log spam
        self._last_partial_transcript = ""

        # Store streaming status based on config
        self.is_streaming_asr = get_config().is_asr_streaming()

        # Dynamic tool gating based on available capabilities
        exclusion_list: list[str] = []
        if deps.vision_manager is None:
            exclusion_list.append("describe_camera_image")

        # Get tool specs and convert to Chat Completions format
        # Note : get_tool_specs() returns Realtime API format, so we need Chat Completions format
        self.tool_specs = convert_tool_specs_to_chat_format(get_tool_specs(exclusion_list=exclusion_list))

        # Side-channel storage for see_image_through_camera frames (JPEG bytes, indexed)
        self._captured_frames: list[bytes] = []

        # Transcript analysis (NoOp if no reactions configured)
        self.transcript_manager: TranscriptAnalysisManager | NoOpTranscriptManager = (
            init_transcript_analysis(deps)
        )

        # Cost tracking
        self.cumulative_cost: float = 0.0
        self._turn_cost: float = 0.0

        # Turn result tracking
        self._current_turn_items: list[TurnItem] = []
        self._turn_results: list[TurnResult] = []

        logger.info(f"Cascade handler initialized (streaming_asr={self.is_streaming_asr})")

    # ─────────────────────────────────────────────────────────────────────────────
    # Transcript Analysis Helpers (fire-and-forget, never block pipeline)
    # ─────────────────────────────────────────────────────────────────────────────

    def _get_stable_text(self, partial: str) -> str:
        """Get stable text for analysis (if ASR supports it)."""
        if hasattr(self.asr, "get_stable_text"):
            stable = self.asr.get_stable_text()
            if stable and stable != partial:
                logger.debug(f"📌 Using stable text for analysis: '{stable[:60]}...'")
                return stable  # type: ignore[no-any-return]
        return partial

    async def _on_transcript_partial(self, text: str) -> None:
        """Notify partial transcript for real-time reactions (streaming only)."""
        await self.transcript_manager.analyze_partial(text)

    def _on_transcript_final(self, text: str) -> None:
        """Notify final transcript (fire-and-forget, parallel with LLM)."""
        task = asyncio.create_task(self.transcript_manager.analyze_final(text))
        if hasattr(self.transcript_manager, '_pending_tasks'):
            self.transcript_manager._pending_tasks.append(task)

    def _on_turn_complete(self) -> None:
        """Reset transcript analysis between conversation turns."""
        self.transcript_manager.reset()

    @property
    def turn_results(self) -> list[TurnResult]:
        """Completed conversation turns (read by UI poller)."""
        return self._turn_results

    def _aggregate_cost(self, provider: Union[ASRProvider, LLMProvider, TTSProvider], provider_name: str) -> None:
        """Aggregate cost from a provider if it tracks costs."""
        if hasattr(provider, "last_cost") and provider.last_cost > 0:
            cost = provider.last_cost
            self.cumulative_cost += cost
            self._turn_cost += cost
            logger.info(f"Cost ({provider_name}): ${cost:.4f} | Cumulative: ${self.cumulative_cost:.4f}")
            provider.last_cost = 0.0  # Reset for next call

    async def _run_pipeline_after_transcription(self, transcript: str) -> TurnResult:
        """Run the shared post-ASR pipeline: validate → history → LLM → TTS → result.

        Called by both manual and streaming paths after transcription is obtained.
        Caller must hold self.processing_lock.
        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        if not transcript.strip():
            logger.warning("Empty transcript, ignoring")
            if self.deps.movement_manager:
                self.deps.movement_manager.set_listening(False)
            return TurnResult()

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": transcript})

        # Update robot state - done listening
        if self.deps.movement_manager:
            self.deps.movement_manager.set_listening(False)

        # Analyze final transcript (parallel with LLM, fire-and-forget)
        self._on_transcript_final(transcript)

        # LLM: Text → Response + Tool Calls
        logger.info("Generating LLM response...")
        tracker.mark("llm_start")
        ctx = PipelineContext(
            llm=self.llm, tts=self.tts, speech_output=self.speech_output,
            conversation_history=self.conversation_history,
            tool_specs=self.tool_specs,
            deps=self.deps,
            result=PipelineResult(),
        )
        result = await pipeline.process_llm_response(ctx)
        tracker.mark("llm_complete")

        # Apply pipeline outputs to handler state
        self._current_turn_items.extend(result.turn_items)
        self._captured_frames.extend(result.captured_frames)
        self._turn_cost += result.cost
        self.cumulative_cost += result.cost

        # Reset transcript analysis for next turn
        self._on_turn_complete()

        # Build and store TurnResult
        turn = TurnResult(
            transcript=transcript,
            items=list(self._current_turn_items),
            cost=self._turn_cost,
        )
        self._turn_results.append(turn)
        return turn

    async def process_audio_manual(self, audio_bytes: bytes) -> TurnResult:
        """Process recorded audio through the cascade pipeline.

        Called manually from Gradio UI.

        Args:
            audio_bytes: WAV audio bytes from Gradio recording

        Returns:
            TurnResult with transcript, displayable items, and cost

        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        # Note: tracker.reset() is called in gradio_ui._stop_recording()
        # to capture user_stop_click in the same timeline

        # Reset per-turn state
        self._current_turn_items = []
        self._turn_cost = 0.0

        async with self.processing_lock:
            try:
                # Update robot state - user is speaking
                if self.deps.movement_manager:
                    self.deps.movement_manager.set_listening(True)

                # ASR: Audio → Text
                logger.info("Transcribing...")
                tracker.mark("transcribing_start")
                transcript = await self.asr.transcribe(audio_bytes, language="en")
                tracker.mark("asr_complete", {"transcript_len": len(transcript)})
                self._aggregate_cost(self.asr, "ASR")
                logger.info(f"User said: {transcript}")

                return await self._run_pipeline_after_transcription(transcript)

            except Exception as e:
                logger.exception(f"Error processing audio: {e}")
                if self.deps.movement_manager:
                    self.deps.movement_manager.set_listening(False)
                raise

    async def process_audio_streaming_start(self) -> None:
        """Initialize streaming ASR session.

        Called from Gradio UI when user starts recording with a streaming ASR provider.
        """
        if isinstance(self.asr, StreamingASRProvider):
            logger.info("Starting streaming ASR session")
            await self.asr.start_stream()

            # Update robot state - user is about to speak
            if self.deps.movement_manager:
                self.deps.movement_manager.set_listening(True)
        else:
            logger.warning("ASR provider does not support streaming")

    async def process_audio_streaming_chunk(self, chunk: bytes) -> str | None:
        """Send audio chunk to streaming ASR and get partial transcript.

        Called from Gradio UI during recording to stream audio in real-time.

        Args:
            chunk: Audio chunk bytes (WAV format)

        Returns:
            Partial transcript if available, None otherwise

        """
        if isinstance(self.asr, StreamingASRProvider):
            await self.asr.send_audio_chunk(chunk)
            partial = await self.asr.get_partial_transcript()

            # Log partial transcript (debounced to reduce spam)
            if partial and partial != self._last_partial_transcript:
                logger.info(f"🎤 Partial: {partial}")
                self._last_partial_transcript = partial

            # Analyze partial transcript (debounced, fire-and-forget)
            if partial:
                # Use stable text for entity extraction to avoid noisy draft tokens
                stable_text = self._get_stable_text(partial)
                await self._on_transcript_partial(stable_text)

            return partial
        return None

    async def process_audio_streaming_end(self) -> TurnResult:
        """Finalize streaming session, get final transcript, and run LLM pipeline.

        Called from Gradio UI when user stops recording with a streaming ASR provider.

        Returns:
            TurnResult with transcript, displayable items, and cost

        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        # Reset per-turn state
        self._current_turn_items = []
        self._turn_cost = 0.0

        async with self.processing_lock:
            try:
                # Get final transcript from streaming ASR
                if isinstance(self.asr, StreamingASRProvider):
                    logger.info("Finalizing streaming ASR session")
                    tracker.mark("transcribing_start")
                    transcript = await self.asr.end_stream()
                    tracker.mark("asr_complete", {"transcript_len": len(transcript)})
                    self._aggregate_cost(self.asr, "ASR")
                else:
                    logger.warning("ASR provider does not support streaming, this shouldn't happen")
                    return TurnResult()

                logger.info(f"User said: {transcript}")

                turn = await self._run_pipeline_after_transcription(transcript)

                # Reset partial transcript tracking (streaming-specific)
                self._last_partial_transcript = ""

                return turn

            except Exception as e:
                logger.exception(f"Error processing streaming audio: {e}")
                if self.deps.movement_manager:
                    self.deps.movement_manager.set_listening(False)
                raise

    def _run_event_loop(self, ready: threading.Event) -> None:
        """Run the asyncio event loop in a background thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.call_soon(ready.set)
        logger.debug("Event loop started in background thread")
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()

    def start(self) -> None:
        """Start the cascade handler (Gradio mode)."""
        if self.running:
            logger.warning("Cascade handler already running")
            return

        logger.info("Starting cascade handler (Gradio mode)...")
        self.running = True

        # Reset prompt log for this run
        PROMPT_LOG.write_text("", encoding="utf-8")

        # Start event loop in background thread for async operations
        loop_ready = threading.Event()
        self.loop_thread = threading.Thread(target=self._run_event_loop, args=(loop_ready,), daemon=True)
        self.loop_thread.start()
        loop_ready.wait(timeout=5)

        # Warmup LLM connection
        if self.loop:
            logger.info("Pre-warming LLM connection...")
            asyncio.run_coroutine_threadsafe(self.llm.warmup(), self.loop)

        logger.info("Cascade handler started")

    def stop(self) -> None:
        """Stop the cascade handler."""
        if not self.running:
            return

        logger.info("Stopping cascade handler...")
        self.running = False

        # Stop event loop
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.loop_thread:
            self.loop_thread.join(timeout=5)

        logger.info("Cascade handler stopped")

    def clear_state(self) -> None:
        """Reset all conversation and turn state (called from UI clear button)."""
        self.conversation_history.clear()
        self._captured_frames.clear()
        self._current_turn_items.clear()
        self._turn_results.clear()
