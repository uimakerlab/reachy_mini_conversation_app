"""Bidirectional local audio stream for robot-device audio I/O."""

import os
import time
import asyncio
import logging
import threading
from typing import Any, List, Callable, Optional

from fastrtc import AdditionalOutputs, audio_to_float32
from scipy.signal import resample

from reachy_mini import ReachyMini
from reachy_mini.media.media_manager import MediaBackend
from reachy_mini_conversation_app.utils import ensure_openai_api_key
from reachy_mini_conversation_app.config import config, persist_api_key
from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler


logger = logging.getLogger(__name__)


class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(
        self,
        handler: OpenaiRealtimeHandler,
        robot: ReachyMini,
        *,
        settings_app: Optional[Any] = None,
        instance_path: Optional[str] = None,
        on_transcript_message: Optional[Callable[[dict[str, Any]], None]] = None,
        wait_for_api_key: bool = False,
    ):
        """Initialize the stream with an OpenAI realtime handler and pipelines.

        - ``settings_app``: legacy argument kept for compatibility.
        - ``instance_path``: directory where per-instance ``.env`` should be stored.
        """
        self.handler = handler
        self._robot = robot
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task[None]] = []
        # Allow the handler to flush the player queue when appropriate.
        self.handler._clear_queue = self.clear_audio_queue
        _ = settings_app
        self._instance_path: Optional[str] = instance_path
        self._asyncio_loop = None
        self._on_transcript_message = on_transcript_message
        self._close_lock = threading.Lock()
        self._close_requested = False
        self._launch_thread_id: Optional[int] = None
        self._stopped_event = threading.Event()
        self._stopped_event.set()
        self._wait_for_api_key = wait_for_api_key

    @staticmethod
    def _has_api_key() -> bool:
        """Return True when an API key is available in config or environment."""
        key = str(getattr(config, "OPENAI_API_KEY", "") or "").strip()
        if key:
            return True

        runtime_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not runtime_key:
            return False

        try:
            config.OPENAI_API_KEY = runtime_key
        except Exception:
            pass
        return True

    def _persist_api_key(self, key: str) -> None:
        """Persist API key in the same way as the Gradio settings flow."""
        persist_api_key(
            key,
            self._instance_path,
            source="huggingface_setup",
            custom_logger=logger,
        )

    def launch(self) -> None:
        """Start the recorder/player and run the async processing loops."""
        self._launch_thread_id = threading.get_ident()
        self._stopped_event.clear()
        self._close_requested = False
        self._stop_event.clear()

        ensure_openai_api_key(
            self._instance_path,
            persist_key=self._persist_api_key,
            load_profile=True,
            logger=logger,
        )

        if not self._has_api_key():
            if not self._wait_for_api_key:
                # LocalStream has no settings UI, fail fast instead of waiting forever.
                logger.error(
                    "OPENAI_API_KEY not found. Set it via environment/.env or run with --gradio and save it in Settings."
                )
                self._stopped_event.set()
                return

            logger.warning("OPENAI_API_KEY not found. Waiting for key from Gradio Settings.")
            while not self._has_api_key():
                if self._stop_event.is_set() or self._close_requested:
                    logger.info("Stop requested while waiting for OPENAI_API_KEY.")
                    self._stopped_event.set()
                    return
                time.sleep(0.2)

        # Start media after key is set/available
        self._robot.media.start_recording()
        self._robot.media.start_playing()
        time.sleep(1)  # give some time to the pipelines to start

        async def runner() -> None:
            # Capture loop for cross-thread personality actions
            loop = asyncio.get_running_loop()
            self._asyncio_loop = loop  # type: ignore[assignment]
            self._tasks = [
                asyncio.create_task(self.handler.start_up(), name="openai-handler"),
                asyncio.create_task(self.record_loop(), name="stream-record-loop"),
                asyncio.create_task(self.play_loop(), name="stream-play-loop"),
            ]
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled during shutdown")
            finally:
                # Ensure handler connection is closed
                await self.handler.shutdown()
                self._asyncio_loop = None
                self._stopped_event.set()

        asyncio.run(runner())

    def _request_async_shutdown(self) -> None:
        """Signal loops to stop and cancel tasks (must run on stream loop thread)."""
        try:
            self._stop_event.set()
        except Exception as e:
            logger.debug("Error setting stop event: %s", e)

        for task in list(self._tasks):
            try:
                if not task.done():
                    task.cancel()
            except Exception as e:
                logger.debug("Error cancelling task %s: %s", task.get_name(), e)

    def close(self) -> None:
        """Stop the stream and underlying media pipelines.

        This method:
        - Stops audio recording and playback first
        - Sets the stop event to signal async loops to terminate
        - Cancels all pending async tasks (openai-handler, record-loop, play-loop)
        """
        with self._close_lock:
            if self._close_requested:
                return
            self._close_requested = True

        logger.info("Stopping LocalStream...")

        # Stop media pipelines FIRST before cancelling async tasks
        # This ensures clean shutdown before PortAudio cleanup
        try:
            self._robot.media.stop_recording()
        except Exception as e:
            logger.debug(f"Error stopping recording (may already be stopped): {e}")

        try:
            self._robot.media.stop_playing()
        except Exception as e:
            logger.debug(f"Error stopping playback (may already be stopped): {e}")

        # Signal async loops to stop from the stream loop thread.
        loop = self._asyncio_loop
        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(self._request_async_shutdown)
            except Exception as e:
                logger.debug("Error scheduling async shutdown on loop thread: %s", e)
                self._request_async_shutdown()
        else:
            self._request_async_shutdown()

        # If close is called from another thread, wait for async runner to stop.
        if self._launch_thread_id is not None and threading.get_ident() != self._launch_thread_id:
            if not self._stopped_event.wait(timeout=5):
                logger.warning("Timed out waiting for LocalStream shutdown")

    def clear_audio_queue(self) -> None:
        """Flush the player's appsrc to drop any queued audio immediately."""
        logger.info("User intervention: flushing player queue")
        if self._robot.media.backend == MediaBackend.GSTREAMER:
            # Directly flush gstreamer audio pipe
            self._robot.media.audio.clear_player()
        elif self._robot.media.backend == MediaBackend.DEFAULT or self._robot.media.backend == MediaBackend.DEFAULT_NO_VIDEO:
            self._robot.media.audio.clear_output_buffer()
        self.handler.output_queue = asyncio.Queue()

    async def record_loop(self) -> None:
        """Read mic frames from the recorder and forward them to the handler."""
        input_sample_rate = self._robot.media.get_input_audio_samplerate()
        logger.debug(f"Audio recording started at {input_sample_rate} Hz")

        while not self._stop_event.is_set():
            audio_frame = self._robot.media.get_audio_sample()
            if audio_frame is not None:
                await self.handler.receive((input_sample_rate, audio_frame))
            await asyncio.sleep(0)  # avoid busy loop

    async def play_loop(self) -> None:
        """Fetch outputs from the handler: log text and play audio frames."""
        while not self._stop_event.is_set():
            handler_output = await self.handler.emit()

            if isinstance(handler_output, AdditionalOutputs):
                for msg in handler_output.args:
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get("role")
                    content = msg.get("content", "")
                    metadata = msg.get("metadata")
                    if isinstance(content, str):
                        logger.info(
                            "role=%s content=%s",
                            role,
                            content if len(content) < 500 else content[:500] + "…",
                        )
                    if self._on_transcript_message is not None:
                        try:
                            transcript_role = role if isinstance(role, str) else "assistant"
                            transcript_msg: dict[str, Any] = {
                                "role": transcript_role,
                                "content": content,
                            }
                            if isinstance(metadata, dict):
                                transcript_msg["metadata"] = metadata
                            self._on_transcript_message(transcript_msg)
                        except Exception as e:
                            logger.debug("Transcript callback failed: %s", e)

            elif isinstance(handler_output, tuple):
                input_sample_rate, audio_data = handler_output
                output_sample_rate = self._robot.media.get_output_audio_samplerate()

                # Reshape if needed
                if audio_data.ndim == 2:
                    # Scipy channels last convention
                    if audio_data.shape[1] > audio_data.shape[0]:
                        audio_data = audio_data.T
                    # Multiple channels -> Mono channel
                    if audio_data.shape[1] > 1:
                        audio_data = audio_data[:, 0]

                # Cast if needed
                audio_frame = audio_to_float32(audio_data)

                # Resample if needed
                if input_sample_rate != output_sample_rate:
                    audio_frame = resample(
                        audio_frame,
                        int(len(audio_frame) * output_sample_rate / input_sample_rate),
                    )

                self._robot.media.push_audio_sample(audio_frame)

            else:
                logger.debug("Ignoring output type=%s", type(handler_output).__name__)

            await asyncio.sleep(0)  # yield to event loop
