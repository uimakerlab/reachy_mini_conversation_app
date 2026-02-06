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
from .audio_recording import ContinuousState, ContinuousVADRecorder, StreamingASRCallbacks


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

        # Create playback system (pre-warmed threads)
        self.playback = AudioPlaybackSystem(
            robot=robot,
            head_wobbler=self.handler.deps.head_wobbler,
            shutdown_event=self.shutdown_event,
        )

        # VAD recorder created lazily after handler.start() provides event loop
        self._vad_recorder: ContinuousVADRecorder | None = None
        self.continuous_mode = False
        self._shown_camera_indices: set[int] = set()  # Track which camera messages have been shown

    def _is_streaming_asr(self) -> bool:
        """Check if the ASR provider supports streaming."""
        return isinstance(self.handler.asr, StreamingASRProvider)

    def _create_streaming_callbacks(self) -> StreamingASRCallbacks | None:
        """Create streaming ASR callbacks if provider supports it."""
        if not self._is_streaming_asr():
            return None

        def on_start() -> None:
            future = asyncio.run_coroutine_threadsafe(self.handler.process_audio_streaming_start(), self.handler.loop)
            try:
                future.result(timeout=5.0)
            except Exception as e:
                logger.error(f"Failed to start streaming ASR: {e}")

        def on_chunk(chunk_wav: bytes) -> None:
            asyncio.run_coroutine_threadsafe(self.handler.process_audio_streaming_chunk(chunk_wav), self.handler.loop)

        return StreamingASRCallbacks(on_start=on_start, on_chunk=on_chunk)

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
            future = asyncio.run_coroutine_threadsafe(self._process_audio_async(wav_bytes), self.handler.loop)
            result = future.result(timeout=60)

            if result["success"]:
                logger.info(f"Continuous mode: Processed transcript: '{result.get('transcript', '')[:50]}...'")
            else:
                logger.error(f"Continuous mode processing error: {result.get('error')}")

        except Exception as e:
            logger.exception(f"Error processing continuous audio: {e}")

    def create_interface(self) -> gr.Blocks:
        """Create and return Gradio interface."""
        with gr.Blocks(title="Reachy Mini - Cascade Mode") as demo:
            gr.Markdown("# Reachy Mini Conversation (Cascade Mode)")

            # Chat display
            chatbot = gr.Chatbot(
                label="Conversation",
                type="messages",
                height=400,
            )

            # Status display
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                value="Ready. Toggle 'Listening' to start.",
            )

            # Controls
            with gr.Row():
                listening_checkbox = gr.Checkbox(
                    label="Listening",
                    value=False,
                    scale=1,
                    info="Toggle microphone on/off",
                )
                clear_btn = gr.Button("Clear History", scale=1)

            # Listening toggle handler
            def toggle_listening(
                enabled: bool, chat_history: List[Dict[str, Any]]
            ) -> tuple[str, List[Dict[str, Any]], bool, gr.Timer]:
                """Toggle continuous VAD listening on/off."""
                if enabled:
                    recorder = self._get_vad_recorder()
                    status = recorder.start()
                    self.continuous_mode = True
                    return status, chat_history, True, gr.Timer(active=True)
                else:
                    recorder = self._get_vad_recorder()
                    status = recorder.stop()
                    self.continuous_mode = False
                    return status, chat_history, False, gr.Timer(active=False)

            poll_timer = gr.Timer(0.5, active=False)

            listening_checkbox.change(
                fn=toggle_listening,
                inputs=[listening_checkbox, chatbot],
                outputs=[status_box, chatbot, listening_checkbox, poll_timer],
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
                for msg_idx, msg in enumerate(self.handler.conversation_history):
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

                    elif role == "tool" and msg.get("name") not in ("speak", None):
                        tool_name = msg.get("name")
                        if tool_name == "camera":
                            # Display camera image from the base64 data in conversation history
                            try:
                                tool_content = json.loads(content) if isinstance(content, str) else content
                                if "b64_im" in tool_content:
                                    import base64

                                    import cv2

                                    # Decode base64 to numpy array
                                    img_bytes = base64.b64decode(tool_content["b64_im"])
                                    np_arr = np.frombuffer(img_bytes, np.uint8)
                                    np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                    if np_img is not None:
                                        rgb_frame = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                                        new_history.append(
                                            {
                                                "role": "assistant",
                                                "content": gr.Image(value=rgb_frame),
                                            }
                                        )
                                        # Only log once per camera image
                                        if msg_idx not in self._shown_camera_indices:
                                            self._shown_camera_indices.add(msg_idx)
                                            logger.info(f"poll_continuous_updates: Added camera image (idx={msg_idx})")
                            except Exception as e:
                                logger.warning(f"poll_continuous_updates: Failed to add camera image: {e}")
                        else:
                            # Display other tool executions with metadata
                            tool_content = msg.get("content", "{}")
                            new_history.append(
                                {
                                    "role": "assistant",
                                    "content": tool_content,
                                    "metadata": {"title": f"🛠️ Used tool {tool_name}", "status": "done"},
                                }
                            )

                return new_history if new_history else chat_history, status

            # Wire up the poll timer (created earlier, before toggle handler)
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
                        logger.info("Processing camera tool for UI display")
                        try:
                            raw_content = message.get("content", "{}")
                            tool_content = json.loads(raw_content)
                            has_b64 = "b64_im" in tool_content
                            has_worker = self.handler.deps.camera_worker is not None
                            logger.info(f"Camera conditions: has_b64_im={has_b64}, has_camera_worker={has_worker}")
                            if has_b64 and has_worker:
                                np_img = self.handler.deps.camera_worker.get_latest_frame()
                                if np_img is not None:
                                    import os
                                    import tempfile

                                    import cv2

                                    # Save to temp file for Gradio Chatbot display
                                    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir="/tmp")
                                    temp_path = temp_file.name
                                    temp_file.close()
                                    # cv2.imwrite expects BGR which is what camera provides
                                    success = cv2.imwrite(temp_path, np_img)
                                    file_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
                                    logger.info(
                                        f"Image save: success={success}, path={temp_path}, size={file_size} bytes, shape={np_img.shape}"
                                    )
                                    # Store image path - FileData will be created in sync context
                                    result["responses"].append(
                                        {
                                            "role": "assistant",
                                            "content": {"_image_path": temp_path},
                                        }
                                    )
                                    logger.info(f"Added camera image path to responses: {temp_path}")
                                else:
                                    logger.warning("camera_worker.get_latest_frame() returned None")
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse camera tool result: {e}")
                        except Exception as e:
                            logger.exception(f"Failed to add camera image to chat: {e}")
                    elif tool_name:
                        # Display other tool executions with metadata (collapsible in Gradio)
                        tool_content = message.get("content", "{}")
                        result["responses"].append(
                            {
                                "role": "assistant",
                                "content": tool_content,  # Already JSON string from handler
                                "metadata": {"title": f"🛠️ Used tool {tool_name}", "status": "done"},
                            }
                        )

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

            # Generate TTS with PARALLEL sentence generation and ORDERED playback
            # Key optimization: Generate sentences in parallel, but queue in order
            total_chunks = 0

            # Gate events ensure chunks queue in sentence order even if TTS responses arrive out of order
            queue_events = [asyncio.Event() for _ in sentences]
            queue_events[0].set()  # Sentence 0 can queue immediately

            async def generate_and_queue_sentence(idx: int, sentence: str) -> List[npt.NDArray[np.int16]]:
                """Generate TTS for one sentence and queue chunks in order."""
                nonlocal total_chunks, first_chunk_queued

                logger.debug(f"TTS sentence {idx + 1}/{len(sentences)}: '{sentence}' (PARALLEL)")
                sentence_chunks: list[npt.NDArray[np.int16]] = []
                sentence_start = time.time()

                # If gate is already open (sentence 0), stream directly to playback
                gate_is_open = queue_events[idx].is_set()

                if gate_is_open:
                    # Stream each chunk to playback as it arrives — no buffering
                    async for chunk in self.handler.tts.synthesize(sentence):
                        total_chunks += 1
                        audio_array = np.frombuffer(chunk, dtype=np.int16)
                        sentence_chunks.append(audio_array)
                        audio_chunks.append(audio_array)
                        self.playback.put_audio(audio_array)
                        self.playback.put_wobbler(chunk)
                        if not first_chunk_queued:
                            first_chunk_queued = True
                            tracker.mark("audio_playback_started")
                            logger.info(
                                "First audio chunk playing - playback started while TTS continues in background"
                            )
                else:
                    # Buffer all chunks, then wait for gate before queuing
                    raw_chunks: list[bytes] = []
                    async for chunk in self.handler.tts.synthesize(sentence):
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
                            logger.info(
                                "First audio chunk playing - playback started while TTS continues in background"
                            )

                gen_duration = time.time() - sentence_start
                if sentence_chunks:
                    total_samples = sum(len(c) for c in sentence_chunks)
                    logger.debug(
                        f"Sentence {idx + 1} generated: {len(sentence_chunks)} chunks ({total_samples} samples, {total_samples / 24000:.2f}s) in {gen_duration:.2f}s"
                    )

                # Signal next sentence can queue
                if idx + 1 < len(queue_events):
                    queue_events[idx + 1].set()

                logger.debug(f"Sentence {idx + 1} queued for playback")
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

            # Aggregate TTS cost after all sentences are synthesized
            self.handler._aggregate_cost(self.handler.tts, "TTS")

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
        self._shown_camera_indices.clear()
        return [], "History cleared"

    def launch(self, **kwargs: Any) -> None:
        """Launch Gradio interface."""
        demo = self.create_interface()
        demo.launch(**kwargs)

    def close(self) -> None:
        """Close Gradio interface and shutdown all subsystems."""
        # Stop continuous mode if active
        if self._vad_recorder and self._vad_recorder.is_active:
            self._vad_recorder.stop()

        # Shutdown playback system
        self.playback.close()
