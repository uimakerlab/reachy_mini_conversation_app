"""Gradio UI for cascade mode."""

from __future__ import annotations
import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, List

import cv2
import numpy as np
import numpy.typing as npt
import gradio as gr

from .audio_playback import AudioPlaybackSystem
from .audio_recording import ContinuousState, ContinuousVADRecorder, StreamingASRCallbacks


if TYPE_CHECKING:
    from reachy_mini import ReachyMini
    from reachy_mini_conversation_app.cascade.handler import CascadeHandler

from reachy_mini_conversation_app.cascade.asr import StreamingASRProvider
from reachy_mini_conversation_app.cascade.turn_result import TurnResult


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

        self.shutdown_event = threading.Event()

        # Create playback system (pre-warmed threads)
        self.playback = AudioPlaybackSystem(
            robot=robot,
            head_wobbler=self.handler.deps.head_wobbler,
            shutdown_event=self.shutdown_event,
            tts_sample_rate=self.handler.tts.sample_rate,
        )

        # Wire speech output so handler plays audio through Gradio's playback system
        from reachy_mini_conversation_app.cascade.speech_output import GradioSpeechOutput

        self.handler.speech_output = GradioSpeechOutput(
            tts=self.handler.tts,
            playback=self.playback,
        )

        # VAD recorder created lazily after handler.start() provides event loop
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
            assert self.handler.loop is not None
            future = asyncio.run_coroutine_threadsafe(self.handler.process_audio_streaming_start(), self.handler.loop)
            try:
                future.result(timeout=5.0)
            except Exception as e:
                logger.error(f"Failed to start streaming ASR: {e}")

        def on_chunk(chunk_wav: bytes) -> None:
            assert self.handler.loop is not None
            asyncio.run_coroutine_threadsafe(self.handler.process_audio_streaming_chunk(chunk_wav), self.handler.loop)

        return StreamingASRCallbacks(on_start=on_start, on_chunk=on_chunk)

    def _get_vad_recorder(self) -> ContinuousVADRecorder:
        """Get or create VAD recorder (lazy initialization)."""
        if self._vad_recorder is None:
            self._vad_recorder = ContinuousVADRecorder(
                sample_rate=16000,
                streaming_asr_callbacks=self._create_streaming_callbacks(),
                on_speech_captured=self._on_vad_speech_captured,
            )
        return self._vad_recorder

    def _on_vad_speech_captured(self, wav_bytes: bytes) -> None:
        """Handle speech captured by VAD recorder."""
        try:
            assert self.handler.loop is not None
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

                # Rebuild chat from turn results
                turns = self.handler.turn_results
                if not turns:
                    return chat_history, status

                new_history: list[dict[str, Any]] = []
                for turn in turns:
                    if turn.transcript:
                        new_history.append({"role": "user", "content": turn.transcript})
                    new_history.extend(self._turn_items_to_chat(turn))

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
            "error": None,
        }

        try:
            # Choose streaming or batch processing based on ASR provider
            if self._is_streaming_asr():
                logger.info("Finalizing streaming ASR session...")
                turn = await self.handler.process_audio_streaming_end()
            else:
                turn = await self.handler.process_audio_manual(audio_bytes)

            result["transcript"] = turn.transcript

            if not turn.transcript.strip():
                result["error"] = "Empty transcript"
                return result

            # Speech was already played during tool execution via speech_output.
            # Print latency summary for non-speech turns (speech path prints its own).
            if not turn.has_speak:
                from reachy_mini_conversation_app.cascade.timing import tracker

                logger.info("No speech output - printing latency summary")
                tracker.print_summary()

            result["success"] = True

        except Exception as e:
            logger.exception(f"Error in async processing: {e}")
            result["error"] = str(e)

        return result

    def _turn_items_to_chat(self, turn: TurnResult) -> list[dict[str, Any]]:
        """Convert TurnResult items to Gradio chatbot message dicts."""
        messages: list[dict[str, Any]] = []
        for item in turn.items:
            if item.kind == "speak":
                messages.append({"role": "assistant", "content": item.text})
            elif item.kind == "assistant":
                messages.append({"role": "assistant", "content": item.text})
            elif item.kind == "image":
                rgb = self._decode_jpeg_to_rgb(item.image_jpeg)
                if rgb is not None:
                    messages.append({"role": "assistant", "content": gr.Image(value=rgb)})
            elif item.kind == "tool":
                messages.append({
                    "role": "assistant",
                    "content": item.tool_content,
                    "metadata": {"title": f"🛠️ Used tool {item.tool_name}", "status": "done"},
                })
        return messages

    @staticmethod
    def _decode_jpeg_to_rgb(jpeg_bytes: bytes) -> npt.NDArray[Any] | None:
        """Decode JPEG bytes to RGB numpy array, or None on failure."""
        try:
            np_arr = np.frombuffer(jpeg_bytes, np.uint8)
            np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if np_img is None:
                return None
            return cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"Failed to decode JPEG: {e}")
            return None

    def _clear_history(self) -> tuple[List[Dict[str, Any]], str]:
        """Clear conversation history."""
        self.handler.clear_state()
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
