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

        # Audio buffering for VAD (requires 512 samples at 16kHz)
        self.audio_buffer = np.array([], dtype=np.int16)
        self.vad_chunk_size = 512  # VAD expects exactly 512 samples at 16kHz

    def copy(self) -> "SpeechToSpeechHandler":
        """Create a copy of the handler."""
        # Return self to share the same output_queue between all references
        return self

    async def start_up(self) -> None:
        """Connect to speech-to-speech WebSocket server."""
        logger.info("=== SpeechToSpeechHandler start_up() called ===")
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
        logger.info("Starting receive loop")
        try:
            async for message in self.websocket:
                if isinstance(message, bytes):
                    # Binary message: audio data (16kHz PCM int16)
                    audio_16k = np.frombuffer(message, dtype=np.int16)
                    logger.debug(f"Received audio: {len(audio_16k)} samples @ 16kHz")

                    # Resample from 16kHz to 24kHz for FastRTC
                    num_samples_24k = int(len(audio_16k) * FASTRTC_SAMPLE_RATE / SPEECH_TO_SPEECH_SAMPLE_RATE)
                    audio_24k = resample(audio_16k, num_samples_24k).astype(np.int16)

                    # Add channel dimension for FastRTC (mono -> 1xN)
                    audio_24k = audio_24k.reshape(1, -1)

                    # Queue audio for playback
                    logger.info(f">>> Queueing audio: {len(audio_24k[0])} samples")
                    await self.output_queue.put((FASTRTC_SAMPLE_RATE, audio_24k))
                    logger.info(f">>> Audio queued successfully")

                elif isinstance(message, str):
                    # Text message: JSON with transcripts and tool calls
                    logger.debug(f"Received text message: {message[:200]}")
                    try:
                        data = json.loads(message)
                        logger.debug(f"Parsed JSON: type={data.get('type')}, keys={list(data.keys())}")

                        if data.get("type") == "speech_started":
                            # User started speaking - stop antenna movements for visual feedback
                            if self.deps.head_wobbler is not None:
                                self.deps.head_wobbler.reset()
                            self.deps.movement_manager.set_listening(True)
                            logger.debug("User speech started - antennas stopped")

                        elif data.get("type") == "speech_stopped":
                            # User stopped speaking - resume antenna movements
                            self.deps.movement_manager.set_listening(False)
                            logger.debug("User speech stopped - antennas resumed")

                        elif data.get("type") == "assistant_text":
                            text = data.get("text", "")
                            tools = data.get("tools", [])
                            logger.info(f"Assistant text: '{text}', tools: {[t['name'] for t in tools]}")

                            # Emit transcript to UI
                            if text:
                                await self.output_queue.put(
                                    AdditionalOutputs({"role": "assistant", "content": text})
                                )
                                logger.debug(f"Emitted transcript to UI")

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
            logger.debug(f"Skipping receive: shutdown={self._shutdown_requested}, websocket={self.websocket is not None}")
            return

        sample_rate, audio_data = frame
        logger.debug(f"Received audio frame: rate={sample_rate}, shape={audio_data.shape}, samples={audio_data.shape[0] if audio_data.ndim > 0 else 0}")

        # Extract audio data (remove channel dimension if present)
        if audio_data.ndim == 2:
            # Shape is (frames, channels) - extract first channel
            audio_data = audio_data[:, 0]  # Take all frames from first channel
            logger.debug(f"Extracted mono channel: shape={audio_data.shape}")
        elif audio_data.ndim == 1:
            # Already mono
            pass
        else:
            logger.error(f"Unexpected audio shape: {audio_data.shape}")
            return

        # Ensure we have audio data
        if len(audio_data) == 0:
            logger.debug("Skipping empty audio chunk")
            return

        # Convert float audio to int16 if needed
        if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
            # Audio is in float format (-1.0 to 1.0), convert to int16 (-32768 to 32767)
            audio_data = (audio_data * 32767).astype(np.int16)

        # Check if resampling is needed
        if sample_rate == SPEECH_TO_SPEECH_SAMPLE_RATE:
            # Already at 16kHz, no resampling needed
            audio_16k = audio_data.astype(np.int16)
            logger.debug(f"Audio already at 16kHz: {len(audio_16k)} samples")
        else:
            # Resample to 16kHz for speech-to-speech
            num_samples_16k = int(len(audio_data) * SPEECH_TO_SPEECH_SAMPLE_RATE / sample_rate)

            # Skip very small chunks that would result in invalid tensors
            if num_samples_16k < 1:
                logger.debug(f"Skipping chunk too small for resampling: {num_samples_16k} samples")
                return

            audio_16k = resample(audio_data, num_samples_16k).astype(np.int16)
            logger.debug(f"Resampled audio: {len(audio_data)} samples @ {sample_rate}Hz -> {len(audio_16k)} samples @ 16kHz")

        # Ensure output is 1-dimensional array
        if audio_16k.ndim == 0:
            logger.warning("Converting 0-d tensor to 1-d array")
            audio_16k = np.array([audio_16k])

        # Buffer audio and send in multiples of 512 samples (VAD requirement)
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_16k])

        # Send multiple complete 512-sample chunks together to reduce WebSocket overhead
        num_complete_chunks = len(self.audio_buffer) // self.vad_chunk_size

        if num_complete_chunks > 0:
            samples_to_send = num_complete_chunks * self.vad_chunk_size
            audio_to_send = self.audio_buffer[:samples_to_send]
            self.audio_buffer = self.audio_buffer[samples_to_send:]

            try:
                await self.websocket.send(audio_to_send.tobytes())
                logger.debug(f"Sent {num_complete_chunks} chunks ({len(audio_to_send)} samples, {len(audio_to_send.tobytes())} bytes) to server, {len(self.audio_buffer)} samples buffered")
            except Exception as e:
                logger.error(f"Failed to send audio to server: {e}")
        else:
            logger.debug(f"Buffering audio: {len(self.audio_buffer)} samples (need {self.vad_chunk_size} to send)")

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs:
        """Emit audio or additional outputs to FastRTC."""
        logger.info("=== emit() called ===")
        try:
            logger.info("emit() waiting on queue.get()...")
            result = await self.output_queue.get()
            logger.info(f"emit() got result from queue!")
            if isinstance(result, tuple) and len(result) == 2:
                sample_rate, audio = result
                logger.info(f"Emitting audio: {audio.shape}, {len(audio[0]) if len(audio.shape) > 1 else len(audio)} samples @ {sample_rate}Hz")
            else:
                logger.info(f"Emitting AdditionalOutputs")
            logger.info("emit() returning result")
            return result
        except Exception as e:
            logger.error(f"Error in emit(): {e}", exc_info=True)
            raise

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
