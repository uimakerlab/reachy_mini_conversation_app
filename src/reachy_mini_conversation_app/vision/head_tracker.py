"""Run head-tracking backends in a dedicated subprocess."""

from __future__ import annotations
import os
import sys
import queue
import atexit
import pickle
import struct
import logging
import threading
import subprocess
from typing import IO, Any
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

_PROCESS_START_TIMEOUT = 20.0
_REQUEST_TIMEOUT = 0.5
_SHUTDOWN_TIMEOUT = 2.0
_HEADER_STRUCT = struct.Struct("!I")


class _StubTrackerBackend:
    """Test double used by unit tests."""

    def get_head_position(self, img: NDArray[np.uint8]) -> tuple[NDArray[np.float32], float]:
        """Return a deterministic fake head position."""
        return np.zeros(2, dtype=np.float32), 0.0


def _build_tracker_backend(backend: str) -> Any:
    """Instantiate a concrete head-tracker backend."""
    if backend == "yolo":
        from reachy_mini_conversation_app.vision.yolo_face_detector import YoloFaceDetector

        return YoloFaceDetector()
    if backend == "mediapipe":
        from reachy_mini_toolbox.vision import HeadTracker

        return HeadTracker()
    if backend == "stub":
        return _StubTrackerBackend()
    raise ValueError(f"Unsupported head tracker backend: {backend}")


def _read_exact(stream: IO[bytes], size: int) -> bytes:
    """Read exactly `size` bytes or raise EOFError."""
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("Unexpected EOF while reading tracker message")
        chunks.extend(chunk)
    return bytes(chunks)


def _send_message(stream: IO[bytes], payload: Any) -> None:
    """Serialize and write a single message."""
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER_STRUCT.pack(len(data)))
    stream.write(data)
    stream.flush()


def _receive_message(stream: IO[bytes]) -> Any:
    """Read and deserialize a single message."""
    header = _read_exact(stream, _HEADER_STRUCT.size)
    (size,) = _HEADER_STRUCT.unpack(header)
    data = _read_exact(stream, size)
    return pickle.loads(data)


def _reader_loop(stream: IO[bytes], messages: "queue.Queue[tuple[str, Any]]") -> None:
    """Read responses from the child process in the background."""
    try:
        while True:
            payload = _receive_message(stream)
            messages.put(("message", payload))
    except EOFError:
        messages.put(("eof", None))
    except Exception as exc:
        messages.put(("error", repr(exc)))


def _worker_main(backend: str) -> int:
    """Run the tracker worker protocol."""
    protocol_out = sys.stdout.buffer
    sys.stdout = sys.stderr

    try:
        tracker = _build_tracker_backend(backend)
        _send_message(protocol_out, ("ready", None))
    except Exception as exc:
        _send_message(protocol_out, ("error", repr(exc)))
        return 1

    while True:
        try:
            command, payload = _receive_message(sys.stdin.buffer)
        except EOFError:
            return 0

        if command == "close":
            return 0
        if command != "frame":
            _send_message(protocol_out, ("error", f"Unknown command: {command}"))
            continue

        try:
            result = tracker.get_head_position(payload)
            _send_message(protocol_out, ("result", result))
        except Exception as exc:
            _send_message(protocol_out, ("error", repr(exc)))


class HeadTracker:
    """Proxy that runs the configured head-tracker backend out of process."""

    def __init__(self, backend: str, *, request_timeout: float = _REQUEST_TIMEOUT) -> None:
        """Start the child process and wait until the tracker is ready."""
        self.backend = backend
        self.request_timeout = request_timeout
        self._closed = False
        self._send_lock = threading.Lock()
        self._messages: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        module_path = "reachy_mini_conversation_app.vision.head_tracker"
        env = os.environ.copy()
        project_src = Path(__file__).resolve().parents[2]
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(project_src)
            if not existing_pythonpath
            else f"{project_src}{os.pathsep}{existing_pythonpath}"
        )
        env["PYTHONUNBUFFERED"] = "1"

        self._process = subprocess.Popen(
            [sys.executable, "-m", module_path, backend],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
            env=env,
        )

        if self._process.stdin is None or self._process.stdout is None:
            self.close()
            raise RuntimeError(f"Failed to create pipes for {backend} head tracker process")

        self._stdin = self._process.stdin
        self._stdout = self._process.stdout
        self._reader = threading.Thread(
            target=_reader_loop,
            args=(self._stdout, self._messages),
            daemon=True,
            name=f"{backend}-head-tracker-reader",
        )
        self._reader.start()
        atexit.register(self.close)

        status, payload = self._wait_for_message(_PROCESS_START_TIMEOUT)
        if status != "ready":
            self.close()
            raise RuntimeError(f"Failed to initialize {backend} head tracker: {payload}")

    def _wait_for_message(self, timeout: float) -> tuple[str, Any]:
        """Wait for the next child-process message."""
        try:
            event, payload = self._messages.get(timeout=timeout)
        except queue.Empty as exc:
            raise RuntimeError(f"{self.backend} head tracker timed out after {timeout:.2f}s") from exc

        if event == "message":
            if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], str):
                return payload[0], payload[1]
            raise RuntimeError(f"{self.backend} head tracker returned an invalid message: {payload!r}")
        if event == "eof":
            raise RuntimeError(f"{self.backend} head tracker exited unexpectedly")
        raise RuntimeError(f"{self.backend} head tracker reader failed: {payload}")

    def get_head_position(
        self,
        frame: NDArray[np.uint8],
    ) -> tuple[NDArray[np.float32] | None, float | None]:
        """Return the detected head position from the child process."""
        if self._closed:
            return None, None

        if self._process.poll() is not None:
            logger.error("Head tracker process for %s is not alive", self.backend)
            return None, None

        try:
            with self._send_lock:
                _send_message(self._stdin, ("frame", frame))
                status, payload = self._wait_for_message(self.request_timeout)
        except Exception as exc:
            logger.error("Head tracker %s communication failed: %s", self.backend, exc)
            return None, None

        if status == "result":
            eye_center, roll = payload
            return eye_center, roll

        logger.error("%s head tracker failed to process frame: %s", self.backend, payload)
        return None, None

    def close(self) -> None:
        """Stop the child process."""
        if self._closed:
            return

        self._closed = True

        try:
            if self._process.poll() is None:
                with self._send_lock:
                    _send_message(self._stdin, ("close", None))
        except Exception:
            pass

        try:
            self._stdin.close()
        except Exception:
            pass

        try:
            self._process.wait(timeout=_SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.warning("Force-terminating %s head tracker process", self.backend)
            self._process.terminate()
            try:
                self._process.wait(timeout=_SHUTDOWN_TIMEOUT)
            except subprocess.TimeoutExpired:
                logger.warning("Force-killing %s head tracker process", self.backend)
                self._process.kill()
                self._process.wait()

        try:
            self._stdout.close()
        except Exception:
            pass

    def __del__(self) -> None:
        """Best-effort shutdown when the proxy is garbage-collected."""
        self.close()


def main() -> int:
    """CLI entrypoint for the head-tracker worker process."""
    if len(sys.argv) != 2:
        print("usage: python -m reachy_mini_conversation_app.vision.head_tracker <backend>", file=sys.stderr)
        return 2
    return _worker_main(sys.argv[1])


if __name__ == "__main__":
    raise SystemExit(main())
