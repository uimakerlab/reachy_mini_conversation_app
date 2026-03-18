"""Tests for the camera tool."""

import base64
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import reachy_mini_conversation_app.tools.camera as camera_mod
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _build_deps(frame: np.ndarray | None = None, vision_manager: Any = None) -> ToolDependencies:
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = frame
    return ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
        vision_manager=vision_manager,
    )


@pytest.mark.asyncio
async def test_camera_returns_base64_jpeg_when_no_local_vision() -> None:
    """Camera tool encodes the buffered frame for realtime vision."""
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    deps = _build_deps(frame=frame)

    with patch.object(camera_mod, "encode_jpeg", return_value=b"jpeg-bytes") as mock_encode:
        result = await camera_mod.Camera()(deps, question="What do you see?")

    assert result == {"b64_im": base64.b64encode(b"jpeg-bytes").decode("utf-8")}
    mock_encode.assert_called_once()


@pytest.mark.asyncio
async def test_camera_uses_local_vision_manager_when_available() -> None:
    """Camera tool delegates to the local vision processor when enabled."""
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    processor = MagicMock()
    processor.process_image.return_value = "a test description"
    vision_manager = MagicMock(processor=processor)
    deps = _build_deps(frame=frame, vision_manager=vision_manager)

    result = await camera_mod.Camera()(deps, question="Describe the frame")

    assert result == {"image_description": "a test description"}
    processor.process_image.assert_called_once_with(frame, "Describe the frame")
