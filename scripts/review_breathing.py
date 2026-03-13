#!/usr/bin/env python3
"""Review breathing configurations live on the robot.

Cycles through selected configs, running each for a set duration with a
big banner announcing each one. Press Ctrl+C to skip to next config,
or Ctrl+C twice quickly to exit.

Usage:
    python3 review_breathing.py                    # review top picks (10s each)
    python3 review_breathing.py --all              # all configs
    python3 review_breathing.py --duration 15      # 15s per config
    python3 review_breathing.py --config curious sleepy alive_v2  # specific ones
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time

import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

# Import configs and evaluator from the benchmark script
from breathing_bench import CONFIGS, BreathingConfig, evaluate_breathing, LOOP_HZ, LOOP_PERIOD


# Configs to show by default (curated top picks for review)
DEFAULT_REVIEW = [
    "still_offset",
    "baseline_old",
    "offset_ant5",
    "offset_ant10",
    "big_z",
    "roll_big",
    "yaw_big",
    "nod_big",
    "figure8_big",
    "curious",
    "sleepy",
    "alive_v2",
    "dreamy_v2",
    "playful",
    "zen",
    "ocean",
]


def print_banner(cfg: BreathingConfig, index: int, total: int) -> None:
    """Print a big visible banner for the current config."""
    width = 60
    print("\n" + "=" * width)
    print(f"  [{index}/{total}]  {cfg.name.upper()}")
    print(f"  {cfg.description}")
    print("=" * width)

    # Print key parameters
    parts = []
    if cfg.z_amplitude > 0:
        parts.append(f"z={cfg.z_amplitude*1000:.0f}mm@{cfg.z_frequency}Hz")
    if cfg.roll_amplitude_deg > 0:
        parts.append(f"roll={cfg.roll_amplitude_deg}°@{cfg.roll_frequency}Hz")
    if cfg.pitch_amplitude_deg > 0:
        parts.append(f"pitch={cfg.pitch_amplitude_deg}°@{cfg.pitch_frequency}Hz")
    if cfg.yaw_amplitude_deg > 0:
        parts.append(f"yaw={cfg.yaw_amplitude_deg}°@{cfg.yaw_frequency}Hz")
    if cfg.antenna_amplitude_deg > 0:
        parts.append(f"ant={cfg.antenna_amplitude_deg}°@{cfg.antenna_frequency}Hz")
    if cfg.antenna_center_deg > 0:
        parts.append(f"ant_center={cfg.antenna_center_deg}°")

    if parts:
        print(f"  {' | '.join(parts)}")
    else:
        print("  (no motion)")
    print()


def run_config(robot: ReachyMini, cfg: BreathingConfig, duration: float) -> bool:
    """Run one config. Returns True if completed, False if skipped via Ctrl+C."""
    start_pose = robot.get_current_head_pose()
    _, start_ant_raw = robot.get_current_joint_positions()
    start_antennas = np.array(start_ant_raw if start_ant_raw is not None else [0.0, 0.0])

    t0 = time.monotonic()
    total_t = duration + cfg.interpolation_duration

    try:
        while True:
            now = time.monotonic()
            t = now - t0
            if t >= total_t:
                break

            head_pose, antennas = evaluate_breathing(cfg, t, start_pose, start_antennas)
            robot.set_target(head=head_pose, antennas=antennas, body_yaw=0.0)

            # Print countdown every second
            remaining = total_t - t
            if int(remaining * 10) % 10 == 0 and remaining > 0:
                sys.stdout.write(f"\r  {remaining:.0f}s remaining... (Ctrl+C to skip)  ")
                sys.stdout.flush()

            elapsed = time.monotonic() - now
            sleep_time = LOOP_PERIOD - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("\r  Done.                                      ")
        return True
    except KeyboardInterrupt:
        print("\r  Skipped.                                   ")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Review breathing configs on robot")
    parser.add_argument("--all", action="store_true", help="Review all configs")
    parser.add_argument("--config", nargs="*", help="Specific config names to review")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds per config")
    parser.add_argument("--list", action="store_true", help="List available configs")
    args = parser.parse_args()

    if args.list:
        for name, cfg in CONFIGS.items():
            marker = " *" if name in DEFAULT_REVIEW else ""
            print(f"  {name:20s}  {cfg.description}{marker}")
        return

    if args.config:
        names = args.config
    elif args.all:
        names = list(CONFIGS.keys())
    else:
        names = DEFAULT_REVIEW

    configs = []
    for name in names:
        if name not in CONFIGS:
            print(f"WARNING: Unknown config '{name}', skipping")
            continue
        configs.append(CONFIGS[name])

    if not configs:
        print("No configs to review.")
        return

    print(f"\nReviewing {len(configs)} breathing configs ({args.duration}s each)")
    print("Ctrl+C once = skip to next | Ctrl+C twice quickly = exit\n")

    with ReachyMini(media_backend="no_media") as robot:
        last_interrupt = 0
        for i, cfg in enumerate(configs, 1):
            print_banner(cfg, i, len(configs))
            time.sleep(0.5)  # Brief pause to read the banner

            completed = run_config(robot, cfg, args.duration)

            if not completed:
                # Check for double Ctrl+C (within 1.5s)
                now = time.monotonic()
                if now - last_interrupt < 1.5:
                    print("\nDouble Ctrl+C — exiting.")
                    break
                last_interrupt = now

            # Brief pause between configs
            if i < len(configs):
                time.sleep(1.0)

    print("\nReview complete.")


if __name__ == "__main__":
    main()
