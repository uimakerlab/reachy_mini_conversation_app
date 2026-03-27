#!/usr/bin/env python3
"""
Reachy Mini Door Greeter
========================
Monitors the camera for motion via the daemon's HTTP API (no media lock),
then launches the conversation app which gets full audio/video control.

Usage:
    python greeter.py
    python greeter.py --threshold 5000  (adjust motion sensitivity)
"""

import argparse
import cv2
import numpy as np
import time
import logging
import subprocess
import sys
import signal
import os
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("greeter")


class MotionDetector:
    """Detect motion using frame differencing."""

    def __init__(self, threshold: float = 3000, cooldown: float = 30.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self.prev_frame = None
        self.last_trigger = 0

    def check(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False

        delta = cv2.absdiff(self.prev_frame, gray)
        self.prev_frame = gray

        _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
        motion_score = np.sum(thresh) / 255

        if motion_score > self.threshold:
            now = time.time()
            if now - self.last_trigger > self.cooldown:
                self.last_trigger = now
                log.info(f"Motion detected! Score: {motion_score:.0f}")
                return True
        return False


def grab_frame_from_daemon() -> np.ndarray | None:
    """Grab a JPEG snapshot from the daemon's HTTP endpoint."""
    try:
        r = httpx.get("http://localhost:8000/api/camera/snapshot", timeout=3.0)
        if r.status_code == 200:
            arr = np.frombuffer(r.content, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        pass
    return None


def grab_frame_from_sdk() -> np.ndarray | None:
    """Grab a frame using SDK with no_media (releases after each grab)."""
    try:
        from reachy_mini import ReachyMini
        with ReachyMini(media_backend="no_media") as mini:
            # Use OpenCV directly since we told SDK not to use media
            pass
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Reachy Mini Door Greeter")
    parser.add_argument("--threshold", type=float, default=3000,
                        help="Motion detection threshold (default: 3000)")
    parser.add_argument("--cooldown", type=float, default=30,
                        help="Seconds between greetings (default: 30)")
    args = parser.parse_args()

    detector = MotionDetector(threshold=args.threshold, cooldown=args.cooldown)
    conversation_proc = None

    # Use OpenCV directly on the Reachy camera USB device
    # Find the Reachy camera device index
    cam = None
    for idx in range(5):
        test = cv2.VideoCapture(idx)
        if test.isOpened():
            ret, frame = test.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                # Reachy camera is 1920x1080
                if w >= 1280:
                    cam = test
                    log.info(f"Found camera at index {idx} ({w}x{h})")
                    break
            test.release()

    if cam is None:
        log.error("Could not find Reachy camera. Is it plugged in?")
        sys.exit(1)

    def cleanup(sig=None, frame=None):
        nonlocal conversation_proc, cam
        log.info("Shutting down greeter...")
        if cam:
            cam.release()
        if conversation_proc and conversation_proc.poll() is None:
            conversation_proc.terminate()
            conversation_proc.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    log.info("=" * 50)
    log.info("  Reachy Mini Door Greeter")
    log.info(f"  Motion threshold: {args.threshold}")
    log.info(f"  Cooldown: {args.cooldown}s between greetings")
    log.info("  Watching for visitors...")
    log.info("=" * 50)

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.2)
            continue

        if detector.check(frame):
            log.info("Someone is here!")

            # Release camera so conversation app can use it
            cam.release()
            cam = None

            # Start conversation app (gets full media control)
            if conversation_proc is None or conversation_proc.poll() is not None:
                log.info("Starting conversation app...")
                conversation_proc = subprocess.Popen(
                    ["reachy-mini-conversation-app"],
                    cwd=os.path.expanduser("~/reachy_mini_conversation_demo"),
                )

            # Wait for cooldown, then reclaim camera for monitoring
            log.info(f"Conversation active. Monitoring paused for {args.cooldown}s...")
            time.sleep(args.cooldown)

            # Re-open camera for motion detection
            for idx in range(5):
                cam = cv2.VideoCapture(idx)
                if cam.isOpened():
                    ret, test_frame = cam.read()
                    if ret and test_frame is not None and test_frame.shape[1] >= 1280:
                        log.info("Camera reclaimed. Watching for visitors again...")
                        break
                    cam.release()
                    cam = None

            if cam is None:
                log.warning("Could not reclaim camera. Conversation app may still be using it.")
                log.info("Waiting for conversation app to finish...")
                if conversation_proc:
                    conversation_proc.wait()
                # Retry camera
                for idx in range(5):
                    cam = cv2.VideoCapture(idx)
                    if cam.isOpened():
                        break
                    cam.release()
                    cam = None

        time.sleep(0.2)


if __name__ == "__main__":
    main()
