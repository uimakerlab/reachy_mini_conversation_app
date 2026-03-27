"""JPEG encoding helpers for BGR camera frames."""

from fractions import Fraction

import av
import numpy as np
from numpy.typing import NDArray


def bgr_to_rgb(frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert a BGR image with shape (H, W, 3) to RGB."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected BGR frame with shape (H, W, 3), got {frame.shape}")
    return frame[:, :, [2, 1, 0]]


def encode_jpeg(frame: NDArray[np.uint8], quality: int = 95) -> bytes:
    """Encode a BGR image as JPEG bytes."""
    clamped_quality = max(1, min(100, quality))
    qscale = 2 + round((100 - clamped_quality) * 29 / 99)

    rgb_frame = bgr_to_rgb(frame)
    video_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")

    codec = av.CodecContext.create("mjpeg", "w")
    codec.width = rgb_frame.shape[1]  # type: ignore[attr-defined]
    codec.height = rgb_frame.shape[0]  # type: ignore[attr-defined]
    codec.pix_fmt = "yuvj444p"  # type: ignore[attr-defined]
    codec.time_base = Fraction(1, 1)
    codec.options = {"qscale": str(qscale)}

    packets = codec.encode(video_frame)  # type: ignore[attr-defined]
    packets += codec.encode(None)  # type: ignore[attr-defined]
    if not packets:
        raise RuntimeError("Failed to encode frame as JPEG")

    return b"".join(bytes(packet) for packet in packets)
