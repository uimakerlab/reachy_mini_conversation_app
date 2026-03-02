"""Main cascade handler orchestrating ASR → LLM → TTS pipeline."""

from __future__ import annotations
import json
import base64
import asyncio
import logging
import importlib
import threading
from typing import Any, Dict, List, Union, Callable, Awaitable

from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.cascade.asr import ASRProvider, StreamingASRProvider
from reachy_mini_conversation_app.cascade.llm import LLMProvider
from reachy_mini_conversation_app.cascade.tts import TTSProvider
from reachy_mini_conversation_app.cascade.config import config
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)
from reachy_mini_conversation_app.cascade.turn_result import TurnItem, TurnResult
from reachy_mini_conversation_app.cascade.transcript_analysis import (
    NoOpTranscriptManager,
    TranscriptAnalysisManager,
)


logger = logging.getLogger(__name__)


CASCADE_EXTRA_INSTRUCTIONS = """\n\n**IMPORTANT:**
## SPEAKING TO THE USER
- To talk to the user, you *MUST* use the 'speak' tool, there is no other way to generate speech.
- When you want to say something, always use the 'speak' tool, even for short acknowledgments like "OK" or "Sure".

## ISSUING SEVERAL TOOLS IN ONE RESPONSE
- You can always issue several tools in one response if needed.
- You can combine the 'speak' tool with other tools in the same response.
- Do not hesitate to use multiple tools if the situation requires it, especially for complex tasks.
"""


class CascadeHandler:
    """Main handler for cascade pipeline mode."""

    def __init__(self, deps: ToolDependencies, skip_audio_playback: bool = False):
        """Initialize cascade handler.

        Args:
            deps: Tool dependencies for robot control
            skip_audio_playback: If True, don't play audio in _speak() (for Gradio mode)

        """
        self.deps = deps
        self.skip_audio_playback = skip_audio_playback

        # Playback callback for console mode
        self._playback_callback: Callable[[bytes], Awaitable[None]] | None = None

        # Initialize providers based on config
        self.asr = self._init_asr_provider()
        self.llm = self._init_llm_provider()
        self.tts = self._init_tts_provider()

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
        self.is_streaming_asr = config.is_asr_streaming()

        # Dynamic tool gating based on available capabilities
        exclusion_list: list[str] = []
        if deps.vision_manager is None:
            exclusion_list.append("describe_scene")

        # Get tool specs and convert to Chat Completions format
        # Note : get_tool_specs() returns Realtime API format, so we need Chat Completions format
        self.tool_specs = self._convert_tool_specs_to_chat_format(get_tool_specs(exclusion_list=exclusion_list))

        # Side-channel storage for see_image frames (JPEG bytes, indexed)
        self._captured_frames: list[bytes] = []

        # Transcript analysis (NoOp if no reactions configured)
        self.transcript_manager: TranscriptAnalysisManager | NoOpTranscriptManager = (
            self._init_transcript_analysis()
        )

        # Cost tracking
        self.cumulative_cost: float = 0.0
        self._turn_cost: float = 0.0

        # Turn result tracking
        self._current_turn_items: list[TurnItem] = []
        self._turn_results: list[TurnResult] = []

        logger.info(
            f"Cascade handler initialized (skip_audio_playback={skip_audio_playback}, streaming_asr={self.is_streaming_asr})"
        )

    def _convert_tool_specs_to_chat_format(self, realtime_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

    def _init_provider(self, provider_type: str, extra_kwargs: Dict[str, Any] | None = None) -> Any:
        """Initialize a provider (ASR/LLM/TTS) from cascade.yaml config.

        Args:
            provider_type: One of "asr", "llm", "tts"
            extra_kwargs: Additional kwargs to pass to provider constructor

        Returns:
            Initialized provider instance

        """
        # All API keys that any provider might need
        api_key_map = {
            "OPENAI_API_KEY": config.OPENAI_API_KEY,
            "DEEPGRAM_API_KEY": config.DEEPGRAM_API_KEY,
            "GEMINI_API_KEY": config.GEMINI_API_KEY,
            "ELEVENLABS_API_KEY": config.ELEVENLABS_API_KEY,
        }

        # Get provider name, info, and settings using dynamic attribute access
        name = getattr(config, f"{provider_type}_provider")
        info = getattr(config, f"get_{provider_type}_provider_info")(name)
        kwargs = getattr(config, f"get_{provider_type}_settings")(name)

        # Validate and add API keys
        for required in info["requires"]:
            if not api_key_map.get(required):
                raise ValueError(f"{required} not set (required by {name})")
            kwargs["api_key"] = api_key_map[required]

        # Merge extra kwargs if provided
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        # Dynamic import and instantiate
        module = importlib.import_module(f"reachy_mini_conversation_app.cascade.{provider_type}.{info['module']}")
        ProviderClass = getattr(module, info["class"])

        # Log with provider-specific details
        extra_info = f", streaming={info['streaming']}" if "streaming" in info else ""
        logger.info(f"Initializing {provider_type.upper()}: {name} (location={info['location']}{extra_info})")

        return ProviderClass(**kwargs)

    def _init_asr_provider(self) -> ASRProvider:
        """Initialize ASR provider from cascade.yaml config."""
        return self._init_provider("asr")

    def _init_llm_provider(self) -> LLMProvider:
        """Initialize LLM provider from cascade.yaml config."""
        # Add cascade-specific instructions (computed at runtime, not from config)
        cascade_instructions = get_session_instructions() + CASCADE_EXTRA_INSTRUCTIONS
        return self._init_provider("llm", {"system_instructions": cascade_instructions})

    def _init_tts_provider(self) -> TTSProvider:
        """Initialize TTS provider from cascade.yaml config."""
        return self._init_provider("tts")

    def set_playback_callback(self, callback: Callable[[bytes], Awaitable[None]]) -> None:
        """Set callback for audio playback (console mode).

        Args:
            callback: Async function that receives audio bytes (PCM int16, 24kHz)

        """
        self._playback_callback = callback

    def _init_transcript_analysis(self) -> TranscriptAnalysisManager | NoOpTranscriptManager:
        """Initialize transcript analysis from profile reactions."""
        from reachy_mini_conversation_app.cascade.transcript_analysis import get_profile_reactions

        reactions = get_profile_reactions()
        if not reactions:
            logger.info("No profile reactions configured, transcript analysis disabled")
            return NoOpTranscriptManager()

        return TranscriptAnalysisManager(reactions=reactions, deps=self.deps)

    # ─────────────────────────────────────────────────────────────────────────────
    # Transcript Analysis Helpers (fire-and-forget, never block pipeline)
    # ─────────────────────────────────────────────────────────────────────────────

    def _get_stable_text(self, partial: str) -> str:
        """Get stable text for analysis (if ASR supports it)."""
        if hasattr(self.asr, "get_stable_text"):
            stable = self.asr.get_stable_text()
            if stable and stable != partial:
                logger.debug(f"📌 Using stable text for analysis: '{stable[:60]}...'")
                return stable
        return partial

    async def _on_transcript_partial(self, text: str) -> None:
        """Notify partial transcript for real-time reactions (streaming only)."""
        await self.transcript_manager.analyze_partial(text)

    def _on_transcript_final(self, text: str) -> None:
        """Notify final transcript (fire-and-forget, parallel with LLM)."""
        asyncio.create_task(self.transcript_manager.analyze_final(text))

    def _on_turn_complete(self) -> None:
        """Reset transcript analysis between conversation turns."""
        self.transcript_manager.reset()

    def _aggregate_cost(self, provider: Union[ASRProvider, LLMProvider, TTSProvider], provider_name: str) -> None:
        """Aggregate cost from a provider if it tracks costs."""
        if hasattr(provider, "last_cost") and provider.last_cost > 0:
            cost = provider.last_cost
            self.cumulative_cost += cost
            self._turn_cost += cost
            logger.info(f"Cost ({provider_name}): ${cost:.4f} | Cumulative: ${self.cumulative_cost:.4f}")
            provider.last_cost = 0.0  # Reset for next call

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

                # 1. ASR: Audio → Text
                logger.info("Transcribing...")
                tracker.mark("transcribing_start")
                transcript = await self.asr.transcribe(audio_bytes, language="en")
                tracker.mark("asr_complete", {"transcript_len": len(transcript)})
                self._aggregate_cost(self.asr, "ASR")
                logger.info(f"User said: {transcript}")

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

                # 2. LLM: Text → Response + Tool Calls
                logger.info("Generating LLM response...")
                tracker.mark("llm_start")
                await self._process_llm_response()
                tracker.mark("llm_complete")

                # Reset transcript analysis for next conversation
                self._on_turn_complete()

                # Build and store TurnResult
                turn = TurnResult(
                    transcript=transcript,
                    items=list(self._current_turn_items),
                    cost=self._turn_cost,
                )
                self._turn_results.append(turn)
                return turn

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
                # Only log if transcript changed (reduce spam)
                if partial != self._last_partial_transcript:
                    logger.debug(f"🎤 Got partial transcript: '{partial[:60]}...'")
                    self._last_partial_transcript = partial

                # Use stable text for entity extraction to avoid noisy draft tokens
                stable_text = self._get_stable_text(partial)
                await self._on_transcript_partial(stable_text)

            if partial and partial != self._last_partial_transcript:
                logger.debug(f"Partial transcript: {partial}")
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
                    # Fallback to batch (shouldn't happen if UI checks properly)
                    logger.warning("ASR provider does not support streaming, this shouldn't happen")
                    return TurnResult()

                logger.info(f"User said: {transcript}")

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

                # 2. LLM: Text → Response + Tool Calls
                logger.info("Generating LLM response...")
                tracker.mark("llm_start")
                await self._process_llm_response()
                # llm_complete is already marked inside the LLM provider

                # Reset transcript analysis for next conversation
                self._on_turn_complete()

                # Reset partial transcript tracking
                self._last_partial_transcript = ""

                # Build and store TurnResult
                turn = TurnResult(
                    transcript=transcript,
                    items=list(self._current_turn_items),
                    cost=self._turn_cost,
                )
                self._turn_results.append(turn)
                return turn

            except Exception as e:
                logger.exception(f"Error processing streaming audio: {e}")
                if self.deps.movement_manager:
                    self.deps.movement_manager.set_listening(False)
                raise

    async def _process_llm_response(self) -> None:
        """Process LLM response with streaming, tool calls, and TTS."""
        try:
            # Generate streaming response
            text_chunks: List[str] = []
            tool_calls: List[Dict[str, Any]] = []

            async for chunk in self.llm.generate(
                messages=self.conversation_history,
                tools=self.tool_specs,
                temperature=1.0,  # TODO: maybe move temperature parameter in config ?
            ):
                if chunk.type == "text_delta" and chunk.content:
                    text_chunks.append(chunk.content)
                    logger.debug(f"LLM text delta: {chunk.content}")

                elif chunk.type == "tool_call" and chunk.tool_call:
                    tool_calls.append(chunk.tool_call)
                    logger.info(f"LLM tool call: {chunk.tool_call}")

                elif chunk.type == "done":
                    logger.debug("LLM generation complete")
                    break

            # Aggregate LLM cost after generator completes
            self._aggregate_cost(self.llm, "LLM")

            # Create assistant message with text, tool calls...
            assistant_message: Dict[str, Any] = {"role": "assistant"}
            full_text = ""
            if text_chunks:
                full_text = "".join(text_chunks)
                assistant_message["content"] = full_text
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls

            logger.debug(
                f"_process_llm_response: text_chunks={len(text_chunks)}, tool_calls={len(tool_calls)}, full_text_len={len(full_text)}"
            )

            if text_chunks or tool_calls:
                self.conversation_history.append(assistant_message)
                logger.debug(f"Added assistant message to history, history_len={len(self.conversation_history)}")

            # Handle text-only responses: auto-inject speak tool call
            # This handles cases where LLM returns text without using the speak tool
            # In principle it should not happen thanks to the extra instructions.
            # If it happens, we create a synthetic tool call for speaking
            if full_text and not tool_calls:
                logger.info("❓LLM returned text without speak tool - auto-injecting speak call")

                synthetic_tool_call = {
                    "id": "auto_speak",
                    "type": "function",
                    "function": {"name": "speak", "arguments": json.dumps({"message": full_text})},
                }
                await self._execute_tool_calls([synthetic_tool_call])
            elif tool_calls and not any(tc.get("function", {}).get("name") == "speak" for tc in tool_calls):
                # Tool calls but no speak — record assistant text if present
                if full_text:
                    self._current_turn_items.append(TurnItem(kind="assistant", text=full_text))
            if tool_calls:
                # Process normal tool calls
                await self._execute_tool_calls(tool_calls)

        except Exception as e:
            logger.exception(f"Error processing LLM response: {e}")

    async def _execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """Execute tool calls and handle camera/see_image and speak tool specially."""
        camera_image_bytes: bytes | None = None

        # First pass: execute all tools and add ALL tool results to conversation
        # This must be done before adding any other messages (OpenAI requires all tool
        # responses immediately after the assistant message with tool_calls)
        for tool_call in tool_calls:
            try:
                call_id, tool_name, arguments = self.llm.parse_tool_call(tool_call)

                logger.info(f"Executing tool: {tool_name}({arguments})")

                # Execute tool
                result = await dispatch_tool_call(
                    tool_name,
                    json.dumps(arguments),
                    self.deps,
                )

                # Do not log full result if the tool returned base64 (huge)
                if tool_name in ("camera", "see_image") and "b64_im" in result:
                    logger.info("Tool result: [image in base64, not shown]")
                else:
                    logger.info(f"Tool result: {result}")

                # Add tool result to conversation
                self.conversation_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": json.dumps(result),
                    }
                )
                logger.debug(
                    f"Added tool result to history: name={tool_name}, history_len={len(self.conversation_history)}"
                )

                # Special handling for see_image tool - store frame, replace heavy b64
                if tool_name == "see_image":
                    if "b64_im" in result:
                        b64_im = result["b64_im"]
                        camera_image_bytes = base64.b64decode(b64_im)
                        frame_index = len(self._captured_frames)
                        self._captured_frames.append(camera_image_bytes)
                        # Replace the heavy b64 blob in conversation history with a lightweight marker
                        self.conversation_history[-1]["content"] = json.dumps(
                            {"status": "image_captured", "frame_index": frame_index}
                        )
                        self._current_turn_items.append(TurnItem(kind="image", image_jpeg=camera_image_bytes))
                        logger.info("see_image: stored frame %d, will add image to conversation", frame_index)
                    else:
                        logger.warning(f"see_image returned error: {result}")

                # Special handling for camera tool (backward compat) - store image for later
                elif tool_name == "camera":
                    if "b64_im" in result:
                        b64_im = result["b64_im"]
                        logger.info("Camera tool executed - will add image to conversation for LLM analysis")
                        camera_image_bytes = base64.b64decode(b64_im)
                        self._current_turn_items.append(TurnItem(kind="image", image_jpeg=camera_image_bytes))
                    else:
                        # Camera failed - error already in tool result, LLM will see it
                        logger.warning(f"Camera tool returned error: {result}")

                # Special handling for speak tool
                elif tool_name == "speak" and "message" in result:
                    message = result["message"]
                    logger.info(f"Speaking: {message}")
                    self._current_turn_items.append(TurnItem(kind="speak", text=message))

                    # Only synthesize audio if not in Gradio mode (Gradio UI handles audio playback separately)
                    if not self.skip_audio_playback:
                        await self._speak(message)
                    else:
                        logger.debug("Skipping audio playback (Gradio mode will handle it)")

                # Other tools
                elif tool_name not in ("speak", "see_image", "camera"):
                    self._current_turn_items.append(
                        TurnItem(kind="tool", tool_name=tool_name, tool_content=json.dumps(result))
                    )

            except Exception as e:
                logger.exception(f"Error executing tool {tool_name}: {e}")

                # Add error to conversation
                self.conversation_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": json.dumps({"error": str(e)}),
                    }
                )

        # After all tool results are added, add camera image as user message and call LLM
        if camera_image_bytes is not None:
            self.conversation_history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": camera_image_bytes,  # Will be converted to provider format in LLM
                        }
                    ],
                }
            )
            logger.info("Camera image added to conversation - calling LLM to analyze it")
            await self._process_llm_response()

    async def _speak(self, text: str) -> None:
        """Synthesize speech and feed to head wobbler for animation.

        Audio Playback Strategy:
        - Console mode: Audio sent via _playback_callback for robot speaker playback
        - Gradio mode: Audio is NOT played here - Gradio UI handles playback via browser

        This method:
        1. Generates TTS audio chunks
        2. Feeds chunks to head_wobbler for synchronized head animation
        3. Sends audio to playback callback (console mode only)
        4. Rate-limits to match real-time audio playback speed
        """
        from reachy_mini_conversation_app.cascade.timing import tracker

        try:
            # Start head wobbler if available
            if self.deps.head_wobbler:
                self.deps.head_wobbler.reset()

            # Log what we're saying in console mode
            if not self.skip_audio_playback:
                print(f"[TTS] Speaking: {text}")

            # Stream TTS audio for head wobbler animation
            audio_chunks = []
            first_chunk_sent = False
            async for chunk in self.tts.synthesize(text):
                audio_chunks.append(chunk)

                # Feed to head wobbler for motion
                if self.deps.head_wobbler:
                    # Convert to base64 for the wobbler's feed() method
                    self.deps.head_wobbler.feed(base64.b64encode(chunk).decode("utf-8"))

                # Send to console playback callback (console mode only)
                if not self.skip_audio_playback and self._playback_callback:
                    await self._playback_callback(chunk)
                    if not first_chunk_sent:
                        tracker.mark("audio_playback_started")
                        first_chunk_sent = True

                # Rate limiting: match audio generation speed
                # PCM int16: 2 bytes per sample
                chunk_duration = len(chunk) / (2 * self.tts.sample_rate)
                # Sleep for 95% of chunk duration to stay slightly ahead
                await asyncio.sleep(chunk_duration * 0.95)

            # Aggregate TTS cost after generator completes
            self._aggregate_cost(self.tts, "TTS")

            logger.info(f"Generated {len(audio_chunks)} audio chunks for head animation")

            # Small buffer to let remaining audio drain from queue/speaker
            # (we already waited 95% of duration during streaming)
            await asyncio.sleep(0.5)

            # Reset head wobbler
            if self.deps.head_wobbler:
                self.deps.head_wobbler.reset()

        except Exception as e:
            logger.exception(f"Error speaking: {e}")

    def _run_event_loop(self) -> None:
        """Run the asyncio event loop in a background thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
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

        # Start event loop in background thread for async operations
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()

        # Wait for event loop to start
        import time

        time.sleep(0.5)

        # Warmup LLM connection
        if hasattr(self.llm, "warmup") and self.loop:
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
