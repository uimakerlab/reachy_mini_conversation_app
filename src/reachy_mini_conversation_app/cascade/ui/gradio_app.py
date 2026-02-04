"""Gradio UI for cascade mode."""

from __future__ import annotations
import re
import json
import time
import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import gradio as gr

from .audio_playback import AudioPlaybackSystem
from .audio_recording import ContinuousState, PushToTalkRecorder, ContinuousVADRecorder, StreamingASRCallbacks


if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from reachy_mini_conversation_app.cascade.handler import CascadeHandler

from reachy_mini_conversation_app.cascade.asr import StreamingASRProvider


logger = logging.getLogger(__name__)


class CascadeGradioUI:
    """Gradio interface for cascade pipeline."""

    def __init__(self, cascade_handler: CascadeHandler, robot: ReachyMini | None = None) -> None:
        """Initialize Gradio UI.

        Args:
            cascade_handler: Cascade pipeline handler
            robot: Robot instance (if running on robot hardware, enables robot speaker output)

        """
        self.handler = cascade_handler
        self.robot = robot
        # Tell handler to skip audio playback, since it will be done in gradio UI.
        self.handler.skip_audio_playback = True

        self.shutdown_event = threading.Event()
        self._last_click_time: float = 0.0

        # Create playback system (pre-warmed threads)
        self.playback = AudioPlaybackSystem(
            robot=robot,
            head_wobbler=self.handler.deps.head_wobbler,
            shutdown_event=self.shutdown_event,
        )

        # Recorders created lazily after handler.start() provides event loop
        self._ptt_recorder: PushToTalkRecorder | None = None
        self._vad_recorder: ContinuousVADRecorder | None = None
        self.continuous_mode = False

    def _is_streaming_asr(self) -> bool:
        """Check if the ASR provider supports streaming."""
        return isinstance(self.handler.asr, StreamingASRProvider)

    def _create_streaming_callbacks(self) -> StreamingASRCallbacks | None:
        """Create streaming ASR callbacks if provider supports it."""
        if not self._is_streaming_asr():
            return None

        def on_start() -> None:
            future = asyncio.run_coroutine_threadsafe(
                self.handler.process_audio_streaming_start(), self.handler.loop
            )
            try:
                future.result(timeout=5.0)
            except Exception as e:
                logger.error(f"Failed to start streaming ASR: {e}")

        def on_chunk(chunk_wav: bytes) -> None:
            asyncio.run_coroutine_threadsafe(
                self.handler.process_audio_streaming_chunk(chunk_wav), self.handler.loop
            )

        return StreamingASRCallbacks(on_start=on_start, on_chunk=on_chunk)

    def _get_ptt_recorder(self) -> PushToTalkRecorder:
        """Get or create push-to-talk recorder (lazy initialization)."""
        if self._ptt_recorder is None:
            self._ptt_recorder = PushToTalkRecorder(
                sample_rate=16000,
                streaming_asr_callbacks=self._create_streaming_callbacks(),
                event_loop=self.handler.loop,
            )
        return self._ptt_recorder

    def _get_vad_recorder(self) -> ContinuousVADRecorder:
        """Get or create VAD recorder (lazy initialization)."""
        if self._vad_recorder is None:
            self._vad_recorder = ContinuousVADRecorder(
                sample_rate=16000,
                streaming_asr_callbacks=self._create_streaming_callbacks(),
                on_speech_captured=self._on_vad_speech_captured,
                event_loop=self.handler.loop,
            )
        return self._vad_recorder

    def _on_vad_speech_captured(self, wav_bytes: bytes) -> None:
        """Handle speech captured by VAD recorder."""
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._process_audio_async(wav_bytes), self.handler.loop
            )
            result = future.result(timeout=60)

            if result["success"]:
                logger.info(f"Continuous mode: Processed transcript: '{result.get('transcript', '')[:50]}...'")
            else:
                logger.error(f"Continuous mode processing error: {result.get('error')}")

        except Exception as e:
            logger.exception(f"Error processing continuous audio: {e}")

    def create_interface(self) -> gr.Blocks:
        """Create and return Gradio interface."""
        # Custom CSS to change button color based on recording state
        custom_css = """
        button:has(span:contains("STOP")) {
            background: linear-gradient(to bottom right, #dc2626, #991b1b) !important;
            border-color: #991b1b !important;
        }
        button:has(span:contains("START")) {
            background: linear-gradient(to bottom right, #16a34a, #15803d) !important;
            border-color: #15803d !important;
        }
        """

        with gr.Blocks(title="Reachy Mini - Cascade Mode", css=custom_css) as demo:
            gr.Markdown("# Reachy Mini Conversation (Cascade Mode)")
            gr.Markdown(
                "**Instructions:** Click the button to **START** recording, speak your message, "
                "then click again to **STOP** and process automatically!"
            )

            # Chat display
            chatbot = gr.Chatbot(
                label="Conversation",
                type="messages",
                height=400,
            )

            # Recording toggle button (single button for start/stop)
            with gr.Row():
                record_btn = gr.Button(
                    "🎤 START Recording",
                    scale=2,
                    variant="primary",
                    size="lg",
                )
                continuous_checkbox = gr.Checkbox(
                    label="Continuous Mode (VAD)",
                    value=False,
                    scale=1,
                    info="Auto-detect speech start/end",
                )

            # Status display
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                value="Ready to record...",
                placeholder="Status updates will appear here",
            )

            # Clear button
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear History", scale=1)

            # Toggle recording button logic
            def toggle_recording_wrapper(chat_history: List[Dict[str, Any]]) -> tuple[str, str, List[Dict[str, Any]]]:
                """Toggle between start and stop recording."""
                # Prevent rapid double-clicks (debounce)
                current_time = time.time()
                if current_time - self._last_click_time < 0.5:  # 500ms debounce
                    recorder = self._get_ptt_recorder()
                    if recorder.is_recording:
                        return "STOP Recording", "Recording... (speak now)", chat_history
                    else:
                        return "START Recording", "Ready to record...", chat_history
                self._last_click_time = current_time

                recorder = self._get_ptt_recorder()
                if not recorder.is_recording:
                    # Start recording
                    recorder.start()
                    return "STOP Recording", "Recording... (speak now)", chat_history
                else:
                    # Stop recording and process
                    _, save_status = recorder.stop()

                    # Auto-process if recording was successful
                    wav_bytes = recorder.get_wav_bytes()
                    if wav_bytes:
                        chat_history, process_status = self._process_audio_sync(wav_bytes, chat_history)
                        return "START Recording", process_status, chat_history
                    else:
                        return "START Recording", save_status, chat_history

            record_btn.click(
                fn=toggle_recording_wrapper,
                inputs=[chatbot],
                outputs=[record_btn, status_box, chatbot],
            )

            # Continuous mode toggle handler
            def toggle_continuous_mode(
                enabled: bool, chat_history: List[Dict[str, Any]]
            ) -> tuple[str, str, List[Dict[str, Any]], bool]:
                """Toggle continuous VAD mode on/off."""
                if enabled:
                    # Start continuous mode
                    recorder = self._get_vad_recorder()
                    status = recorder.start()
                    self.continuous_mode = True
                    return "🎤 (Continuous Active)", status, chat_history, True
                else:
                    # Stop continuous mode
                    recorder = self._get_vad_recorder()
                    status = recorder.stop()
                    self.continuous_mode = False
                    return "🎤 START Recording", status, chat_history, False

            continuous_checkbox.change(
                fn=toggle_continuous_mode,
                inputs=[continuous_checkbox, chatbot],
                outputs=[record_btn, status_box, chatbot, continuous_checkbox],
            )

            # Polling for continuous mode updates (updates chat when VAD detects speech)
            def poll_continuous_updates(
                chat_history: List[Dict[str, Any]],
            ) -> tuple[List[Dict[str, Any]], str]:
                """Poll for updates from continuous mode processing."""
                if not self.continuous_mode:
                    return chat_history, "Ready to record..."

                # Get status based on current state
                recorder = self._get_vad_recorder()
                state_messages = {
                    ContinuousState.IDLE: "Continuous mode stopped",
                    ContinuousState.LISTENING: "Listening... (speak now)",
                    ContinuousState.RECORDING: "Recording speech...",
                    ContinuousState.PROCESSING: "Processing...",
                }
                status = state_messages.get(recorder.state, "Listening...")

                # Update chat history from handler's conversation history
                # This syncs any new messages added during continuous processing
                # Use same logic as _process_audio_async: prefer speak tool over assistant content
                has_speak_tool = any(
                    msg.get("role") == "tool" and msg.get("name") == "speak"
                    for msg in self.handler.conversation_history
                )

                new_history = []
                for msg in self.handler.conversation_history:
                    role = msg.get("role")
                    content = msg.get("content", "")

                    if role == "user" and content:
                        # User messages (skip multimodal content like images)
                        if isinstance(content, str):
                            new_history.append({"role": "user", "content": content})

                    elif role == "assistant":
                        # Only show assistant content if no speak tool (to avoid duplicates)
                        if not has_speak_tool and content:
                            new_history.append({"role": "assistant", "content": content})

                    elif role == "tool" and msg.get("name") == "speak":
                        # Speak tool results contain the actual response text
                        try:
                            tool_content = json.loads(content) if isinstance(content, str) else content
                            if "message" in tool_content:
                                new_history.append({"role": "assistant", "content": tool_content["message"]})
                        except (json.JSONDecodeError, TypeError):
                            pass

                return new_history if new_history else chat_history, status

            # Set up polling timer for continuous mode (every 500ms)
            poll_timer = gr.Timer(0.5, active=True)
            poll_timer.tick(
                fn=poll_continuous_updates,
                inputs=[chatbot],
                outputs=[chatbot, status_box],
            )

            # Clear button
            clear_btn.click(
                fn=self._clear_history,
                inputs=None,
                outputs=[chatbot, status_box],
            )

        return demo  # type: ignore[no-any-return]

    def _process_audio_sync(
        self, audio_bytes: bytes, chat_history: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], str]:
        """Process audio bytes (sync wrapper for async handler).

        Args:
            audio_bytes: WAV audio bytes
            chat_history: Current chat history

        Returns:
            Tuple of (updated_chat_history, status_message)

        """
        if not audio_bytes or len(audio_bytes) == 0:
            return chat_history, "No audio data. Please record first."

        logger.debug(f"Processing {len(audio_bytes)} bytes of audio")

        try:
            # Use handler's event loop via run_coroutine_threadsafe
            # This avoids conflicts with the handler's background event loop
            if not self.handler.loop:
                return chat_history, "Error: Event loop not available"

            future = asyncio.run_coroutine_threadsafe(self._process_audio_async(audio_bytes), self.handler.loop)

            # Wait for result with timeout
            result = future.result(timeout=60)

            logger.debug(f"_process_audio_sync: received result - success={result.get('success')}, responses_count={len(result.get('responses', []))}")

            # Update chat history
            if result["success"]:
                # Add user message
                if result.get("transcript"):
                    chat_history.append({"role": "user", "content": result["transcript"]})

                # Add assistant responses
                if result.get("responses"):
                    for response in result["responses"]:
                        chat_history.append({"role": "assistant", "content": response})
                        logger.debug(f"Added assistant response to chat_history: {response[:50]}...")

                status = "Message processed successfully!"
            else:
                status = f"Error: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.exception(f"Error processing audio: {e}")
            status = f"Error processing audio: {str(e)}"

        return chat_history, status

    async def _process_audio_async(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Process audio through cascade pipeline (async).

        Args:
            audio_bytes: Audio file bytes

        Returns:
            Dictionary with processing results

        """
        result: Dict[str, Any] = {
            "success": False,
            "transcript": None,
            "responses": [],
            "error": None,
        }

        try:
            # Store the current conversation length to track new messages
            initial_history_length = len(self.handler.conversation_history)

            # Choose streaming or batch processing based on ASR provider
            if self._is_streaming_asr():
                # Streaming path: Finalize stream and get transcript
                # Note: Streaming session was already started in recorder
                # and chunks were sent during recording
                logger.info("Finalizing streaming ASR session...")
                transcript = await self.handler.process_audio_streaming_end()
            else:
                # Batch path: Use handler's process_audio_manual which handles ASR->LLM->TTS pipeline
                # Note: This will execute speak tools but won't play audio (that's done separately below)
                transcript = await self.handler.process_audio_manual(audio_bytes)

            result["transcript"] = transcript

            if not transcript.strip():
                result["error"] = "Empty transcript"
                return result

            new_messages = self.handler.conversation_history[initial_history_length:]
            logger.debug(f"initial_history_length={initial_history_length}, new_messages_count={len(new_messages)}")
            for i, msg in enumerate(new_messages):
                role = msg.get("role")
                name = msg.get("name", "")
                content_preview = str(msg.get("content", ""))[:100]
                logger.debug(f"new_msg[{i}] role={role}, name={name}, content_preview={content_preview}")

            # Collect all speech messages first (to play them all in one stream)
            speak_messages = []

            # Check if there are any speak tool calls (to avoid duplicate messages)
            has_speak_tool = any(
                msg.get("role") == "tool" and msg.get("name") == "speak"
                for msg in self.handler.conversation_history[initial_history_length:]
            )

            # Extract responses from conversation history that were added during processing
            # The handler adds: user message, assistant message(s), and tool results
            for message in self.handler.conversation_history[initial_history_length:]:
                role = message.get("role")

                if role == "assistant":
                    # Only add assistant text if there's NO speak tool (to avoid duplicates)
                    # When speak tool is present, we'll show that message instead
                    if not has_speak_tool and "content" in message and message["content"]:
                        result["responses"].append(message["content"])

                elif role == "tool":
                    tool_name = message.get("name")
                    if tool_name == "speak":
                        # Extract speak tool results for display, since this tool is used in place of a message
                        try:
                            tool_content = json.loads(message.get("content", "{}"))
                            if "message" in tool_content:
                                result["responses"].append(f"{tool_content['message']}")
                                # Collect speak messages to synthesize all at once
                                speak_messages.append(tool_content["message"])
                        except json.JSONDecodeError:
                            pass
                    elif tool_name == "camera":
                        # Display camera image in chat
                        try:
                            tool_content = json.loads(message.get("content", "{}"))
                            if "b64_im" in tool_content and self.handler.deps.camera_worker is not None:
                                # Get the latest camera frame as numpy array
                                np_img = self.handler.deps.camera_worker.get_latest_frame()

                                # Save to temporary file for Gradio Chatbot display
                                import tempfile

                                from PIL import Image

                                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir="/tmp")
                                # Convert RGB numpy array to PIL Image and save
                                pil_img = Image.fromarray(np_img)
                                pil_img.save(temp_file.name, format="JPEG", quality=85)
                                temp_file.close()

                                # Add file path as string (Gradio Chatbot accepts file paths)
                                result["responses"].append(temp_file.name)
                                logger.info(f"Added camera image to chat display: {temp_file.name}")
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse camera tool result")
                        except Exception as e:
                            logger.exception(f"Failed to add camera image to chat: {e}")
                    elif tool_name:
                        # Display other tool executions
                        result["responses"].append(f"Used tool: {tool_name}")

            # In case there are several speech messages in a single stream (it shouldn't happen too much), concatenate them.
            if speak_messages:
                combined_text = ". ".join(speak_messages)
                logger.info(f"Synthesizing {len(speak_messages)} speak message(s) as one stream")
                await self._synthesize_for_gradio(combined_text)
            else:
                # No speech - print summary here since _synthesize_for_gradio won't be called
                from reachy_mini_conversation_app.cascade.timing import tracker

                logger.info("No speech output - printing latency summary")
                tracker.print_summary()

            result["success"] = True

            logger.debug(f"Returning result - success={result['success']}, responses_count={len(result['responses'])}")

        except Exception as e:
            logger.exception(f"Error in async processing: {e}")
            result["error"] = str(e)

        return result

    async def _synthesize_for_gradio(self, text: str) -> None:
        """Synthesize speech and play through pre-warmed audio system.

        1. Generates TTS audio chunks
        2. Plays audio immediately through pre-warmed sounddevice (robot speaker)
        3. Feeds chunks to head_wobbler for animation (synchronized with playback)

        Args:
            text: Text to synthesize

        """
        import numpy.typing as npt

        from reachy_mini_conversation_app.cascade.timing import tracker

        logger.info(f"Synthesizing speech: '{text[:50]}...'")

        audio_chunks: list[npt.NDArray[np.int16]] = []
        first_chunk_queued = False

        try:
            # Split text into sentences for streaming, so TTS can send a first audio faster.
            sentences = self._split_into_sentences(text)
            logger.debug(f"Split text into {len(sentences)} sentence chunks for streaming TTS")
            for i, s in enumerate(sentences):
                logger.debug(f"  Sentence {i + 1}: '{s}'")

            # Use pre-warmed persistent queues (no thread creation overhead!)
            logger.debug("Using pre-warmed audio playback system")

            # Generate TTS with PARALLEL sentence generation and STREAMING playback
            # Key optimization: Start playing sentence 1 while generating sentence 2, etc.
            total_chunks = 0

            async def generate_and_queue_sentence(idx: int, sentence: str) -> List[npt.NDArray[np.int16]]:
                """Generate TTS for one sentence and queue chunks immediately."""
                nonlocal total_chunks, first_chunk_queued

                logger.debug(f"TTS sentence {idx + 1}/{len(sentences)}: '{sentence}' (PARALLEL)")
                sentence_chunks: list[npt.NDArray[np.int16]] = []
                sentence_start = time.time()

                async for chunk in self.handler.tts.synthesize(sentence):
                    total_chunks += 1

                    # Convert to numpy array
                    audio_array = np.frombuffer(chunk, dtype=np.int16)
                    sentence_chunks.append(audio_array)

                    # Collect for Gradio output
                    audio_chunks.append(audio_array)

                    # Queue for immediate playback (pre-warmed thread picks it up instantly!)
                    self.playback.put_audio(audio_array)

                    # Queue for wobbler (pre-warmed thread picks it up too)
                    self.playback.put_wobbler(chunk)

                    # Track first chunk queued (should be nearly instant playback)
                    if not first_chunk_queued:
                        first_chunk_queued = True
                        tracker.mark("audio_first_chunk_queued")
                        tracker.mark("audio_playback_started")  # With pre-warmed system, queueing = playing
                        tracker.mark("wobbler_first_chunk")
                        logger.info("First audio chunk playing - playback started while TTS continues in background")

                if sentence_chunks:
                    duration = time.time() - sentence_start
                    total_samples = sum(len(c) for c in sentence_chunks)
                    logger.debug(
                        f"Sentence {idx + 1} complete: {len(sentence_chunks)} chunks ({total_samples} samples, {total_samples / 24000:.2f}s) generated in {duration:.2f}s"
                    )

                return sentence_chunks

            # Strategy: Generate sentences with intelligent overlap
            # - Sentence 1: Start immediately, play as soon as first chunk arrives
            # - Sentence 2: Start while sentence 1 is still generating/playing
            # - Sentence 3+: Start with small delay to avoid overwhelming TTS API

            tasks = []
            for idx, sentence in enumerate(sentences):
                if idx == 0:
                    # Sentence 1: Start immediately
                    task = asyncio.create_task(generate_and_queue_sentence(idx, sentence))
                    tasks.append(task)
                elif idx == 1:
                    # Sentence 2: Start after a small delay (let sentence 1 begin streaming)
                    await asyncio.sleep(0.3)  # 300ms delay - allows sentence 1 to start playing
                    task = asyncio.create_task(generate_and_queue_sentence(idx, sentence))
                    tasks.append(task)
                else:
                    # Sentence 3+: Start after previous task completes (avoid API overload)
                    # But playback is already happening from sentence 1 & 2!
                    if idx >= 2 and tasks:
                        await tasks[idx - 1]  # Wait for previous sentence to finish generating
                    task = asyncio.create_task(generate_and_queue_sentence(idx, sentence))
                    tasks.append(task)

            # Wait for all sentence generations to complete
            await asyncio.gather(*tasks)
            logger.info(f"Parallel TTS complete: All {len(sentences)} sentences generated")

            logger.info(f"Generated {total_chunks} total audio chunks from {len(sentences)} sentences")

            # Signal end of this playback session to persistent threads
            self.playback.signal_end_of_turn()

            # Wait for audio to finish playing (estimate based on total duration)
            if audio_chunks:
                total_samples = sum(len(chunk) for chunk in audio_chunks)
                duration_seconds = total_samples / 24000
                logger.info(f"Waiting {duration_seconds:.1f}s for playback to complete...")
                await asyncio.sleep(duration_seconds + 0.5)  # Add 500ms buffer

            logger.info("Playback complete (using pre-warmed system)")

            # Print latency summary now that full pipeline is complete (speech path)
            from reachy_mini_conversation_app.cascade.timing import tracker

            tracker.print_summary()

        except Exception as e:
            logger.exception(f"Error synthesizing speech: {e}")

    def _split_into_sentences(self, text: str, min_length: int = 8) -> list[str]:
        """Split text into sentence-like chunks for streaming TTS.

        Splits on: . ! ? , ; — (but keeps punctuation with the sentence)
        This allows starting TTS synthesis earlier for long texts.

        Args:
            text: Text to split
            min_length: Minimum characters per segment (default 8)
                       Segments shorter than this are combined with the next segment.

        Returns:
            List of text segments, each at least min_length characters (except possibly the last)

        """
        # Split on sentence boundaries but keep the punctuation
        # Pattern: split after punctuation + optional whitespace
        pattern = r"([.!?,;—]\s+)"
        parts = re.split(pattern, text)

        # Recombine parts to keep punctuation with sentences
        raw_sentences = []
        current = ""
        for part in parts:
            current += part
            # If this part ends with punctuation + space, it's a sentence boundary
            if re.match(pattern, part):
                if current.strip():
                    raw_sentences.append(current.strip())
                current = ""

        # Add any remaining text
        if current.strip():
            raw_sentences.append(current.strip())

        # If no splits, return original text
        if not raw_sentences:
            return [text]

        # Merge short segments to meet minimum length requirement
        merged_sentences = []
        accumulator = ""

        for sentence in raw_sentences:
            # Add to accumulator
            if accumulator:
                accumulator += " " + sentence
            else:
                accumulator = sentence

            # If accumulator is long enough, add it as a segment
            if len(accumulator) >= min_length:
                merged_sentences.append(accumulator)
                accumulator = ""

        # Add any remaining text (last segment can be shorter)
        if accumulator:
            # If we have previous segments, try to append to last one if it's not too long
            # Otherwise, add as separate segment
            if merged_sentences and len(merged_sentences[-1]) < min_length * 2:
                merged_sentences[-1] += " " + accumulator
            else:
                merged_sentences.append(accumulator)

        return merged_sentences if merged_sentences else [text]

    def _clear_history(self) -> tuple[List[Dict[str, Any]], str]:
        """Clear conversation history."""
        self.handler.conversation_history = []
        return [], "History cleared"

    def launch(self, **kwargs: Any) -> None:
        """Launch Gradio interface."""
        demo = self.create_interface()
        demo.launch(**kwargs)

    def close(self) -> None:
        """Close Gradio interface and shutdown all subsystems."""
        # Stop recording if still active
        if self._ptt_recorder and self._ptt_recorder.is_recording:
            self._ptt_recorder.stop()

        # Stop continuous mode if active
        if self._vad_recorder and self._vad_recorder.is_active:
            self._vad_recorder.stop()

        # Shutdown playback system
        self.playback.close()
