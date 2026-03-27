"""Tests for head tracker process management."""

import sys
import time
import subprocess
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

import reachy_mini_conversation_app.vision.head_tracker as head_tracker_module
from reachy_mini_conversation_app.vision.head_tracker import HeadTracker


def test_head_tracker_round_trip() -> None:
    """A frame can be processed by the child process."""
    tracker = HeadTracker("stub")
    try:
        frame = np.zeros((12, 20, 3), dtype=np.uint8)
        eye_center, roll = tracker.get_head_position(frame)

        assert eye_center is not None
        assert np.allclose(eye_center, np.array([0.0, 0.0], dtype=np.float32))
        assert roll == 0.0
    finally:
        tracker.close()


def test_head_tracker_rejects_unknown_backend() -> None:
    """Unknown backends fail fast with a helpful error."""
    with pytest.raises(RuntimeError, match="Unsupported head tracker backend"):
        tracker = HeadTracker("unknown")
        tracker.close()


def test_head_tracker_close_is_idempotent() -> None:
    """close() can be called repeatedly without crashing."""
    tracker = HeadTracker("stub")
    tracker.close()
    tracker.close()


def test_head_tracker_discards_stale_reply_after_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Late replies from timed-out requests should not leak into the next frame."""
    worker_script = tmp_path / "fake_head_tracker_worker.py"
    worker_script.write_text(
        dedent(
            """
            import pickle
            import struct
            import sys
            import time

            import numpy as np

            HEADER = struct.Struct("!I")


            def _read_exact(size: int) -> bytes:
                data = bytearray()
                while len(data) < size:
                    chunk = sys.stdin.buffer.read(size - len(data))
                    if not chunk:
                        raise EOFError
                    data.extend(chunk)
                return bytes(data)


            def _receive_message():
                (size,) = HEADER.unpack(_read_exact(HEADER.size))
                return pickle.loads(_read_exact(size))


            def _send_message(payload) -> None:
                data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
                sys.stdout.buffer.write(HEADER.pack(len(data)))
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()


            _send_message(("ready", None))
            call_count = 0

            while True:
                try:
                    message = _receive_message()
                except EOFError:
                    raise SystemExit(0)

                if message[0] == "close":
                    raise SystemExit(0)

                request_id = message[1]
                call_count += 1
                if call_count == 1:
                    time.sleep(0.05)

                value = float(call_count)
                _send_message(
                    ("result", request_id, (np.array([value, value], dtype=np.float32), value))
                )
            """
        ),
        encoding="utf-8",
    )

    real_popen = subprocess.Popen

    def _spawn_fake_worker(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        return real_popen([sys.executable, str(worker_script)], **kwargs)

    monkeypatch.setattr(head_tracker_module.subprocess, "Popen", _spawn_fake_worker)

    tracker = HeadTracker("stub", request_timeout=0.01)
    try:
        frame = np.zeros((12, 20, 3), dtype=np.uint8)

        eye_center, roll = tracker.get_head_position(frame)
        assert eye_center is None
        assert roll is None

        time.sleep(0.08)

        eye_center, roll = tracker.get_head_position(frame)
        assert eye_center is not None
        assert np.allclose(eye_center, np.array([2.0, 2.0], dtype=np.float32))
        assert roll == 2.0
    finally:
        tracker.close()
