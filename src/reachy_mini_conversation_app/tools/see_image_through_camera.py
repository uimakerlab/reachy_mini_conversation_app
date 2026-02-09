"""Tool: see_image_through_camera — capture a frame and return raw JPEG for LLM vision."""

import base64
import logging
from typing import Any, Dict

import cv2

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class SeeImageThroughCamera(Tool):
    """Take a picture with the camera so you can see what's in front of you."""

    name = "see_image_through_camera"
    description = "Get a picture taken with the camera in your eyes. Use this so you can see what's in front of you."
    parameters_schema = {
        "type": "object",
        "properties": {},
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Capture a frame and return it as base64-encoded JPEG."""
        logger.info("Tool call: see_image_through_camera")

        if deps.camera_worker is None:
            logger.error("Camera worker not available")
            return {"error": "Camera worker not available"}

        frame = deps.camera_worker.get_latest_frame()
        if frame is None:
            logger.error("No frame available from camera worker")
            return {"error": "No frame available"}

        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError("Failed to encode frame as JPEG")

        b64_encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
        return {"b64_im": b64_encoded}
