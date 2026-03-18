import base64
import asyncio
import logging
from typing import Any, Dict
from fractions import Fraction

import av

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class Camera(Tool):
    """Take a picture with the camera and ask a question about it."""

    name = "camera"
    description = "Take a picture with the camera and ask a question about it."
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask about the picture",
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Take a picture with the camera and ask a question about it."""
        image_query = (kwargs.get("question") or "").strip()
        if not image_query:
            logger.warning("camera: empty question")
            return {"error": "question must be a non-empty string"}

        logger.info("Tool call: camera question=%s", image_query[:120])

        # Get frame from camera worker buffer (like main_works.py)
        if deps.camera_worker is not None:
            frame = deps.camera_worker.get_latest_frame()
            if frame is None:
                logger.error("No frame available from camera worker")
                return {"error": "No frame available"}
        else:
            logger.error("Camera worker not available")
            return {"error": "Camera worker not available"}

        # Use vision manager for processing if available
        if deps.vision_manager is not None:
            vision_result = await asyncio.to_thread(
                deps.vision_manager.processor.process_image, frame, image_query,
            )
            if isinstance(vision_result, dict) and "error" in vision_result:
                return vision_result
            return (
                {"image_description": vision_result}
                if isinstance(vision_result, str)
                else {"error": "vision returned non-string"}
            )

        rgb_frame = frame[:, :, ::-1].copy()
        video_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")

        codec = av.CodecContext.create("mjpeg", "w")
        codec.width = rgb_frame.shape[1]  # type: ignore[attr-defined]
        codec.height = rgb_frame.shape[0]  # type: ignore[attr-defined]
        codec.pix_fmt = "yuvj444p"  # type: ignore[attr-defined]
        codec.time_base = Fraction(1, 1)
        codec.options = {"qscale": "2"}

        packets = codec.encode(video_frame)  # type: ignore[attr-defined]
        packets += codec.encode(None)  # type: ignore[attr-defined]
        if not packets:
            raise RuntimeError("Failed to encode frame as JPEG")

        jpeg_bytes = b"".join(bytes(packet) for packet in packets)
        return {"b64_im": base64.b64encode(jpeg_bytes).decode("utf-8")}
