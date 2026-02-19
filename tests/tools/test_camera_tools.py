"""Tests for see_image_through_camera and describe_camera_image tools.

Both tools depend on two external services: camera_worker (provides video frames)
and vision_manager (runs a local vision model). All tests mock these dependencies
so no hardware or model is needed. The mandatory ToolDependencies fields
(reachy_mini, movement_manager) are also mocked since camera tools never use them.

_fake_frame() returns a tiny 4x4 black BGR numpy array — just enough for
cv2.imencode to produce valid JPEG bytes without needing a real image.

SeeImageThroughCamera tests (4):
  Pipeline: get frame -> encode JPEG -> return base64.
  Each test targets one step that can fail:
  - Happy path: camera returns a frame, encoding works -> {"b64_im": ...}
  - No camera_worker (None) -> error before trying to get a frame
  - No frame (camera exists but get_latest_frame returns None) -> error
  - JPEG encode fails (cv2.imencode patched to return False) -> error dict
    (not a raised exception, which was the old behavior fixed in this PR)

DescribeCameraImage tests (6):
  Pipeline: validate question -> get frame -> call vision model -> check result type.
  More moving parts, so more failure modes:
  - Happy path: vision model returns a string -> {"description": ...}
    Note: vision_manager is a MagicMock with a hardcoded return_value, so no
    real model runs. The test just checks the tool wraps the string correctly.
  - Empty question -> fails validation before touching camera/vision
  - No camera_worker -> error
  - No vision_manager -> error (this tool *requires* vision, unlike see_image)
  - No frame (camera exists but returns None) -> error
  - Vision returns non-string (int 42) -> error message includes the type name

asyncio.to_thread doesn't need special mocking: MagicMock returns synchronously,
so to_thread just runs it in the thread pool and returns the mock's value.
"""

import base64
from unittest.mock import MagicMock

import numpy as np
import pytest

import reachy_mini_conversation_app.tools.see_image_through_camera as see_cam_mod
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.describe_camera_image import DescribeCameraImage
from reachy_mini_conversation_app.tools.see_image_through_camera import SeeImageThroughCamera


def _make_deps(camera_worker=None, vision_manager=None) -> ToolDependencies:
    """Build ToolDependencies with only camera/vision deps, rest mocked."""
    return ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_worker=camera_worker,
        vision_manager=vision_manager,
    )


def _fake_frame() -> np.ndarray:
    """Return a tiny 4x4 black BGR image for cv2.imencode."""
    return np.zeros((4, 4, 3), dtype=np.uint8)


# ---------- SeeImageThroughCamera ----------


class TestSeeImageThroughCamera:
    """Test the see_image_through_camera tool."""

    tool = SeeImageThroughCamera()

    @pytest.mark.asyncio
    async def test_returns_b64_image(self) -> None:
        """Test happy path: valid frame returns base64-encoded JPEG."""
        cam = MagicMock()
        cam.get_latest_frame.return_value = _fake_frame()
        deps = _make_deps(camera_worker=cam)

        result = await self.tool(deps)

        assert "b64_im" in result
        # Verify it's valid base64
        raw = base64.b64decode(result["b64_im"])
        assert len(raw) > 0

    @pytest.mark.asyncio
    async def test_error_when_no_camera_worker(self) -> None:
        """Test error when camera_worker is None."""
        deps = _make_deps(camera_worker=None)
        result = await self.tool(deps)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_error_when_no_frame(self) -> None:
        """Test error when camera exists but returns no frame."""
        cam = MagicMock()
        cam.get_latest_frame.return_value = None
        deps = _make_deps(camera_worker=cam)

        result = await self.tool(deps)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_error_when_jpeg_encode_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test error dict (not exception) when JPEG encoding fails."""
        cam = MagicMock()
        cam.get_latest_frame.return_value = _fake_frame()
        deps = _make_deps(camera_worker=cam)

        mock_cv2 = MagicMock()
        mock_cv2.imencode.return_value = (False, None)
        monkeypatch.setattr(see_cam_mod, "cv2", mock_cv2)

        result = await self.tool(deps)

        assert "error" in result
        assert "JPEG" in result["error"]


# ---------- DescribeCameraImage ----------


class TestDescribeCameraImage:
    """Test the describe_camera_image tool."""

    tool = DescribeCameraImage()

    @pytest.mark.asyncio
    async def test_returns_description(self) -> None:
        """Test happy path: vision model string is wrapped in description key."""
        cam = MagicMock()
        cam.get_latest_frame.return_value = _fake_frame()
        vision = MagicMock()
        vision.processor.process_image.return_value = "A small black square"
        deps = _make_deps(camera_worker=cam, vision_manager=vision)

        result = await self.tool(deps, question="What do you see?")

        assert result == {"description": "A small black square"}

    @pytest.mark.asyncio
    async def test_error_when_empty_question(self) -> None:
        """Test error when question is empty string."""
        deps = _make_deps(camera_worker=MagicMock(), vision_manager=MagicMock())
        result = await self.tool(deps, question="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_error_when_no_camera_worker(self) -> None:
        """Test error when camera_worker is None."""
        deps = _make_deps(camera_worker=None, vision_manager=MagicMock())
        result = await self.tool(deps, question="What is this?")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_error_when_no_vision_manager(self) -> None:
        """Test error when vision_manager is None."""
        deps = _make_deps(camera_worker=MagicMock(), vision_manager=None)
        result = await self.tool(deps, question="What is this?")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_error_when_no_frame(self) -> None:
        """Test error when camera exists but returns no frame."""
        cam = MagicMock()
        cam.get_latest_frame.return_value = None
        deps = _make_deps(camera_worker=cam, vision_manager=MagicMock())

        result = await self.tool(deps, question="What is this?")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_error_includes_type_for_non_string(self) -> None:
        """Test error includes type name when vision returns non-string."""
        cam = MagicMock()
        cam.get_latest_frame.return_value = _fake_frame()
        vision = MagicMock()
        vision.processor.process_image.return_value = 42
        deps = _make_deps(camera_worker=cam, vision_manager=vision)

        result = await self.tool(deps, question="What is this?")

        assert "error" in result
        assert "int" in result["error"]
