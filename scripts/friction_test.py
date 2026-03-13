#!/usr/bin/env python3
"""Friction characterization: detect stick-slip at various velocities and angles.

Runs constant-velocity linear interpolations on the robot and records present
positions to detect stick-slip behavior (stall runs where the motor is stuck).

Tests:
  1. Sinusoidal baseline (one test, for reference)
  2. Linear interpolation at various speeds on antennas at different center angles
  3. Linear interpolation at various speeds on head Z

Usage:
    python3 friction_test.py                  # run all tests
    python3 friction_test.py --test antenna   # antenna tests only
    python3 friction_test.py --test head_z    # head Z tests only
    python3 friction_test.py --test sinus     # sinusoidal baseline only
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


LOOP_HZ = 50.0  # Must match daemon's internal control loop (50 Hz)
LOOP_PERIOD = 1.0 / LOOP_HZ


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

@dataclass
class FrictionTest:
    """One friction test run."""
    name: str
    description: str
    test_type: str  # "antenna" | "head_z" | "sinus"
    # For linear tests:
    center_deg: float = 0.0        # operating point (degrees) for antennas
    center_z: float = 0.0          # operating point (metres) for head Z
    amplitude_deg: float = 10.0    # total travel (degrees) for antennas
    amplitude_z: float = 0.005     # total travel (metres) for head Z
    velocity_deg_s: float = 5.0    # constant angular velocity (deg/s) for antennas
    velocity_mm_s: float = 2.0     # constant linear velocity (mm/s) for head Z
    # For sinus tests:
    sinus_freq: float = 0.1        # Hz
    duration: float = 15.0         # seconds


TESTS: list[FrictionTest] = []

# ---- Sinusoidal baseline (one test) ----
TESTS.append(FrictionTest(
    name="sinus_baseline",
    description="Sinusoidal z 5mm@0.1Hz + antenna 15deg@0.5Hz (original breathing)",
    test_type="sinus",
    sinus_freq=0.1,
    duration=20.0,
))

# ---- Antenna tests at different angles and speeds ----
# Velocities to test (deg/s)
ANT_VELOCITIES = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0]
# Center angles to test (degrees)
ANT_CENTERS = [0.0, 10.0, 90.0]
# Amplitude (half-travel each direction, degrees)
ANT_AMPLITUDE = 10.0

for center in ANT_CENTERS:
    for vel in ANT_VELOCITIES:
        TESTS.append(FrictionTest(
            name=f"ant_c{int(center)}_v{vel:.0f}",
            description=f"Antenna linear: center={center}deg, velocity={vel}deg/s, travel=±{ANT_AMPLITUDE}deg",
            test_type="antenna",
            center_deg=center,
            amplitude_deg=ANT_AMPLITUDE,
            velocity_deg_s=vel,
            duration=15.0,
        ))

# ---- Head Z tests at different speeds ----
# Velocities to test (mm/s)
Z_VELOCITIES = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
# Amplitude (half-travel each direction, metres)
Z_AMPLITUDE = 0.005  # 5mm

for vel in Z_VELOCITIES:
    TESTS.append(FrictionTest(
        name=f"z_v{vel:.1f}",
        description=f"Head Z linear: velocity={vel}mm/s, travel=±{Z_AMPLITUDE*1000:.0f}mm",
        test_type="head_z",
        amplitude_z=Z_AMPLITUDE,
        velocity_mm_s=vel,
        duration=15.0,
    ))


# ---------------------------------------------------------------------------
# Movement generators
# ---------------------------------------------------------------------------

def triangle_wave(t: float, amplitude: float, velocity: float) -> float:
    """Generate a triangle wave (constant velocity, back and forth).

    Returns position offset from 0, oscillating between -amplitude and +amplitude
    at the given constant velocity.
    """
    if velocity <= 0 or amplitude <= 0:
        return 0.0
    # Period = 4 * amplitude / velocity (go +amp, come back to 0, go -amp, come back)
    period = 4.0 * amplitude / velocity
    phase = (t % period) / period  # 0..1

    if phase < 0.25:
        # Going from 0 to +amplitude
        return amplitude * (phase / 0.25)
    elif phase < 0.75:
        # Going from +amplitude to -amplitude
        return amplitude * (1.0 - (phase - 0.25) / 0.25)  # +amp to 0 at 0.5
        # Actually need: +amp -> 0 -> -amp over 0.25..0.75
    else:
        # Going from -amplitude to 0
        return amplitude * (-1.0 + (phase - 0.75) / 0.25)

    # Cleaner implementation:


def triangle_wave_v2(t: float, amplitude: float, velocity: float) -> float:
    """Triangle wave: constant velocity, bounces between -amplitude and +amplitude."""
    if velocity <= 0 or amplitude <= 0:
        return 0.0
    period = 4.0 * amplitude / velocity
    phase = t % period
    half_period = period / 2.0

    if phase < half_period:
        # Going from -amplitude to +amplitude
        return -amplitude + velocity * phase
    else:
        # Going from +amplitude to -amplitude
        return amplitude - velocity * (phase - half_period)


# ---------------------------------------------------------------------------
# Pose decomposition
# ---------------------------------------------------------------------------

def decompose_pose(pose: np.ndarray) -> dict[str, float]:
    """Extract x, y, z, roll, pitch, yaw from 4x4."""
    from scipy.spatial.transform import Rotation as R
    x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
    rot = R.from_matrix(pose[:3, :3])
    roll_d, pitch_d, yaw_d = rot.as_euler("xyz", degrees=True)
    return {"x": x, "y": y, "z": z, "roll": roll_d, "pitch": pitch_d, "yaw": yaw_d}


# ---------------------------------------------------------------------------
# Run one test
# ---------------------------------------------------------------------------

def run_test(robot: ReachyMini, test: FrictionTest, output_dir: str) -> str:
    """Run one friction test and record data."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / f"{test.name}.csv"

    print(f"  [{test.name}] {test.description}")

    # Read initial state
    start_pose = robot.get_current_head_pose()
    _, start_ant_raw = robot.get_current_joint_positions()
    start_ant = list(start_ant_raw) if start_ant_raw is not None else [0.0, 0.0]

    fieldnames = [
        "t", "loop_dt",
        "cmd_z", "cmd_ant_l", "cmd_ant_r",
        "cmd_roll", "cmd_pitch", "cmd_yaw",
        "pres_z", "pres_ant_l", "pres_ant_r",
        "pres_roll", "pres_pitch", "pres_yaw",
        "pres_j0", "pres_j1", "pres_j2", "pres_j3", "pres_j4", "pres_j5", "pres_j6",
    ]

    # First: interpolate to starting position (1 second)
    interp_dur = 1.0
    if test.test_type == "antenna":
        target_ant_center = math.radians(test.center_deg)
        target_ant = [-target_ant_center, target_ant_center]
    else:
        target_ant = start_ant

    t0 = time.monotonic()
    while time.monotonic() - t0 < interp_dur:
        alpha = min(1.0, (time.monotonic() - t0) / interp_dur)
        ant_l = (1 - alpha) * start_ant[0] + alpha * target_ant[0]
        ant_r = (1 - alpha) * start_ant[1] + alpha * target_ant[1]
        neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        head = (1 - alpha) * start_pose + alpha * neutral
        robot.set_target(head=head, antennas=[ant_l, ant_r], body_yaw=0.0)
        time.sleep(LOOP_PERIOD)

    # Now run the actual test
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        t0 = time.monotonic()
        prev_tick = t0

        while True:
            now = time.monotonic()
            t = now - t0
            if t >= test.duration:
                break

            loop_dt = now - prev_tick
            prev_tick = now

            # Compute commanded position
            if test.test_type == "sinus":
                z_cmd = 0.005 * math.sin(2 * math.pi * 0.1 * t)
                ant_amp = math.radians(15.0)
                ant_sway = ant_amp * math.sin(2 * math.pi * 0.5 * t)
                cmd_ant = [ant_sway, -ant_sway]
                head_pose = create_head_pose(x=0, y=0, z=z_cmd, roll=0, pitch=0, yaw=0,
                                             degrees=True, mm=False)
                cmd_roll = cmd_pitch = cmd_yaw = 0.0

            elif test.test_type == "antenna":
                amp_rad = math.radians(test.amplitude_deg)
                vel_rad = math.radians(test.velocity_deg_s)
                offset = triangle_wave_v2(t, amp_rad, vel_rad)
                center_rad = math.radians(test.center_deg)
                cmd_ant = [-center_rad + offset, center_rad - offset]
                z_cmd = 0.0
                head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
                cmd_roll = cmd_pitch = cmd_yaw = 0.0

            elif test.test_type == "head_z":
                amp_m = test.amplitude_z
                vel_m = test.velocity_mm_s / 1000.0
                z_cmd = triangle_wave_v2(t, amp_m, vel_m)
                head_pose = create_head_pose(x=0, y=0, z=z_cmd, roll=0, pitch=0, yaw=0,
                                             degrees=True, mm=False)
                cmd_ant = target_ant
                cmd_roll = cmd_pitch = cmd_yaw = 0.0

            robot.set_target(head=head_pose, antennas=cmd_ant, body_yaw=0.0)

            # Read present state
            pres_pose = robot.get_current_head_pose()
            head_joints, pres_ant_raw = robot.get_current_joint_positions()
            pres_ant = list(pres_ant_raw) if pres_ant_raw is not None else [0.0, 0.0]
            head_joints = list(head_joints) if head_joints is not None else [0.0] * 7
            pres_d = decompose_pose(pres_pose)

            row = {
                "t": f"{t:.4f}",
                "loop_dt": f"{loop_dt:.6f}",
                "cmd_z": f"{z_cmd:.6f}",
                "cmd_ant_l": f"{cmd_ant[0]:.6f}",
                "cmd_ant_r": f"{cmd_ant[1]:.6f}",
                "cmd_roll": f"{cmd_roll:.4f}",
                "cmd_pitch": f"{cmd_pitch:.4f}",
                "cmd_yaw": f"{cmd_yaw:.4f}",
                "pres_z": f"{pres_d['z']:.6f}",
                "pres_ant_l": f"{pres_ant[0]:.6f}",
                "pres_ant_r": f"{pres_ant[1]:.6f}",
                "pres_roll": f"{pres_d['roll']:.4f}",
                "pres_pitch": f"{pres_d['pitch']:.4f}",
                "pres_yaw": f"{pres_d['yaw']:.4f}",
            }
            for i in range(7):
                row[f"pres_j{i}"] = f"{head_joints[i]:.6f}"
            writer.writerow(row)

            elapsed = time.monotonic() - now
            sleep_time = LOOP_PERIOD - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print(f"    -> {csv_path}")
    return str(csv_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Friction characterization tests")
    parser.add_argument("--test", choices=["antenna", "head_z", "sinus", "all"], default="all")
    parser.add_argument("--name", nargs="*", help="Run specific test(s) by name (e.g. ant_c10_v12)")
    parser.add_argument("--output", default="/tmp/friction_test")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for t in TESTS:
            print(f"  {t.name:<25} [{t.test_type}] {t.description}")
        return

    if args.name:
        selected = [t for t in TESTS if t.name in args.name]
        if not selected:
            print(f"No tests matching: {args.name}")
            print("Use --list to see available tests.")
            sys.exit(1)
    else:
        selected = [t for t in TESTS if args.test == "all" or t.test_type == args.test]
    print(f"Running {len(selected)} friction tests...")

    with ReachyMini(media_backend="no_media") as robot:
        for i, test in enumerate(selected, 1):
            print(f"\n[{i}/{len(selected)}]")
            run_test(robot, test, args.output)
            time.sleep(2)  # Settle between tests

    print(f"\nAll tests saved to {args.output}/")


if __name__ == "__main__":
    main()
