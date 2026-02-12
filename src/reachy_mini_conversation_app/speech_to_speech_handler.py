"""
Speech-to-Speech Handler

Connects to the local speech-to-speech WebSocket server and handles:
- Audio resampling (24kHz FastRTC <-> 16kHz speech-to-speech)
- Tool call extraction from text messages
- Tool execution using dispatch_tool_call
"""

import json
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional

import numpy as np
import websockets
from fastrtc import AdditionalOutputs, AsyncStreamHandler
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    dispatch_tool_call,
)


logger = logging.getLogger(__name__)

# Audio sample rates
FASTRTC_SAMPLE_RATE: Final[Literal[24000]] = 24000
SPEECH_TO_SPEECH_SAMPLE_RATE: Final[Literal[16000]] = 16000


class SpeechToSpeechHandler(AsyncStreamHandler):
    """Handler for speech-to-speech WebSocket backend."""

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=FASTRTC_SAMPLE_RATE,
            input_sample_rate=FASTRTC_SAMPLE_RATE,
        )

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()
        self.receive_task: Optional[asyncio.Task] = None
        self._shutdown_requested: bool = False

    def copy(self) -> "SpeechToSpeechHandler":
        """Create a copy of the handler."""
        return SpeechToSpeechHandler(self.deps, self.gradio_mode, self.instance_path)

    async def start_up(self) -> None:
        """Connect to speech-to-speech WebSocket server."""
        server_url = config.SPEECH_TO_SPEECH_SERVER_URL
        logger.info(f"Connecting to speech-to-speech server at {server_url}")

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                self.websocket = await websockets.connect(server_url)
                logger.info("Connected to speech-to-speech server")

                # Start background task to receive audio and text messages
                self.receive_task = asyncio.create_task(self._receive_loop())
                return

            except Exception as e:
                logger.error(f"Failed to connect to speech-to-speech server (attempt {attempt}/{max_attempts}): {e}")
                if attempt < max_attempts:
                    delay = 2 ** (attempt - 1)  # Exponential backoff: 1s, 2s
                    logger.info(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise RuntimeError(f"Could not connect to speech-to-speech server after {max_attempts} attempts")

    async def _receive_loop(self) -> None:
        """Background task to receive audio and text messages from server."""
        try:
            async for message in self.websocket:
                if isinstance(message, bytes):
                    # Binary message: audio data (16kHz PCM int16)
                    audio_16k = np.frombuffer(message, dtype=np.int16)

                    # Resample from 16kHz to 24kHz for FastRTC
                    num_samples_24k = int(len(audio_16k) * FASTRTC_SAMPLE_RATE / SPEECH_TO_SPEECH_SAMPLE_RATE)
                    audio_24k = resample(audio_16k, num_samples_24k).astype(np.int16)

                    # Add channel dimension for FastRTC (mono -> 1xN)
                    audio_24k = audio_24k.reshape(1, -1)

                    # Queue audio for playback
                    await self.output_queue.put((FASTRTC_SAMPLE_RATE, audio_24k))

                elif isinstance(message, str):
                    # Text message: JSON with transcripts and tool calls
                    try:
                        data = json.loads(message)

                        if data.get("type") == "assistant_text":
                            text = data.get("text", "")
                            tools = data.get("tools", [])

                            # Emit transcript to UI
                            if text:
                                await self.output_queue.put(
                                    AdditionalOutputs({"role": "assistant", "content": text})
                                )

                            # Execute tools
                            for tool in tools:
                                tool_name = tool.get("name")
                                params = tool.get("parameters", {})

                                if tool_name:
                                    try:
                                        logger.info(f"Executing tool '{tool_name}' with params: {params}")
                                        result = await dispatch_tool_call(
                                            tool_name,
                                            json.dumps(params),
                                            self.deps
                                        )
                                        logger.info(f"Tool '{tool_name}' result: {result}")

                                        # Optionally emit tool result to UI
                                        await self.output_queue.put(
                                            AdditionalOutputs(
                                                {
                                                    "role": "assistant",
                                                    "content": json.dumps(result),
                                                    "metadata": {"title": f"🛠️ Used tool {tool_name}", "status": "done"},
                                                }
                                            )
                                        )
                                    except Exception as e:
                                        logger.error(f"Tool '{tool_name}' failed: {e}")

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON message: {e}")

        except asyncio.CancelledError:
            logger.info("Receive loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio from FastRTC and send to speech-to-speech server."""
        if self._shutdown_requested or not self.websocket:
            return

        sample_rate, audio_24k = frame

        # Extract audio data (remove channel dimension if present)
        if audio_24k.ndim == 2:
            audio_24k = audio_24k[0]  # Take first channel for mono

        # Ensure we have audio data
        if len(audio_24k) == 0:
            return

        # Resample from 24kHz to 16kHz for speech-to-speech
        num_samples_16k = int(len(audio_24k) * SPEECH_TO_SPEECH_SAMPLE_RATE / FASTRTC_SAMPLE_RATE)

        # Skip very small chunks that would result in invalid tensors
        if num_samples_16k < 1:
            return

        audio_16k = resample(audio_24k, num_samples_16k).astype(np.int16)

        # Ensure output is 1-dimensional array
        if audio_16k.ndim == 0:
            audio_16k = np.array([audio_16k])

        # Send as raw PCM bytes
        try:
            await self.websocket.send(audio_16k.tobytes())
        except Exception as e:
            logger.error(f"Failed to send audio to server: {e}")

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs:
        """Emit audio or additional outputs to FastRTC."""
        return await self.output_queue.get()

    async def shutdown(self) -> None:
        """Gracefully shutdown the handler."""
        self._shutdown_requested = True

        # Cancel receive task
        if self.receive_task and not self.receive_task.done():
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket connection
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("Closed connection to speech-to-speech server")
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None
