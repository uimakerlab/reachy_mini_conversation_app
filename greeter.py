#!/usr/bin/env python3
"""
Reachy Mini Door Greeter
========================
Uses camera-based motion detection to spot people approaching,
then starts a conversation. Tracks token usage locally.

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

from reachy_mini import ReachyMini

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("greeter")

# --- Motion detection ---

class MotionDetector:
    """Detect motion using frame differencing on Reachy's camera."""

    def __init__(self, threshold: float = 3000, cooldown: float = 30.0):
        self.threshold = threshold      # pixel-change threshold to trigger
        self.cooldown = cooldown        # seconds between greetings
        self.prev_frame = None
        self.last_trigger = 0

    def check(self, frame: np.ndarray) -> bool:
        """Returns True if significant motion detected and cooldown has passed."""
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
                log.info(f"Motion detected! Score: {motion_score:.0f} (threshold: {self.threshold})")
                return True
            else:
                remaining = self.cooldown - (now - self.last_trigger)
                log.debug(f"Motion detected but in cooldown ({remaining:.0f}s remaining)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Reachy Mini Door Greeter")
    parser.add_argument("--threshold", type=float, default=3000, help="Motion detection threshold (default: 3000)")
    parser.add_argument("--cooldown", type=float, default=30, help="Seconds between greetings (default: 30)")
    parser.add_argument("--no-conversation", action="store_true", help="Just detect motion, don't start conversation app")
    args = parser.parse_args()

    detector = MotionDetector(threshold=args.threshold, cooldown=args.cooldown)
    conversation_proc = None

    def cleanup(sig=None, frame=None):
        nonlocal conversation_proc
        log.info("Shutting down greeter...")
        if conversation_proc and conversation_proc.poll() is None:
            conversation_proc.terminate()
            conversation_proc.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    log.info("=" * 50)
    log.info("  Reachy Mini Door Greeter")
    log.info(f"  Motion threshold: {args.threshold}")
    log.info(f"  Cooldown: {args.cooldown}s")
    log.info("=" * 50)

    with ReachyMini(media_backend="default") as mini:
        mini.enable_motors()
        time.sleep(1)
        mini.wake_up()
        time.sleep(2)
        log.info("Reachy is awake and watching the door...")

        while True:
            frame = mini.media.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            if detector.check(frame):
                log.info("Someone is here! Starting greeting...")

                # Wiggle antennas excitedly
                mini.goto_target(antennas=[0.5, -0.5], duration=0.3)
                mini.goto_target(antennas=[-0.5, 0.5], duration=0.3)
                mini.goto_target(antennas=[0, 0], duration=0.3)

                # Start conversation app if not already running
                if not args.no_conversation:
                    if conversation_proc is None or conversation_proc.poll() is not None:
                        log.info("Starting conversation app...")
                        conversation_proc = subprocess.Popen(
                            ["reachy-mini-conversation-app"],
                            cwd=os.path.expanduser("~/reachy_mini_conversation_demo"),
                        )
                    else:
                        log.info("Conversation app already running.")

            # Check at ~5fps to save CPU
            time.sleep(0.2)


if __name__ == "__main__":
    main()
