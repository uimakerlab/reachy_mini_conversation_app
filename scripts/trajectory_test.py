#!/usr/bin/env python3
"""Test different antenna trajectory shapes to minimize mechanical oscillation.

The goal: find a trajectory that moves the antenna back and forth (~30 deg/s
equivalent speed, ±10 deg amplitude, centered at 10 deg) with minimal shaking.

The shaking at high speed is primarily mechanical vibration — the antenna is a
flexible rod that rings when subjected to sudden acceleration changes (jerk).
So we test trajectories with different jerk profiles.

Usage:
    python3 trajectory_test.py                     # run all trajectory tests
    python3 trajectory_test.py --name min_jerk     # run specific test(s)
    python3 trajectory_test.py --list              # list all tests
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


LOOP_HZ = 50.0  # Must match daemon's internal control loop (50 Hz)
LOOP_PERIOD = 1.0 / LOOP_HZ

# Common parameters — keep these consistent across all trajectory tests
CENTER_DEG = 10.0       # SDK default antenna offset
AMPLITUDE_DEG = 10.0    # ±10 degrees of travel
SPEED_DEG_S = 30.0      # ~30 deg/s equivalent (target average speed)
DURATION = 15.0         # seconds per test


# ---------------------------------------------------------------------------
# Trajectory generators
# Each returns a position offset (radians) given time t.
# The offset oscillates around 0; the caller adds the center offset.
# ---------------------------------------------------------------------------

def triangle(t: float, amp: float, vel: float) -> float:
    """Constant velocity triangle wave. Sharp corners = high jerk at reversals."""
    if vel <= 0 or amp <= 0:
        return 0.0
    period = 4.0 * amp / vel
    phase = t % period
    half = period / 2.0
    if phase < half:
        return -amp + vel * phase
    else:
        return amp - vel * (phase - half)


def sinusoidal(t: float, amp: float, vel: float) -> float:
    """Pure sinusoid. Smooth everywhere, but velocity varies (zero at peaks)."""
    period = 4.0 * amp / vel  # match same period as triangle
    freq = 1.0 / period
    return amp * math.sin(2 * math.pi * freq * t)


def min_jerk(t: float, amp: float, vel: float) -> float:
    """Minimum-jerk trajectory segments stitched into a back-and-forth motion.

    Minimum-jerk profile: x(s) = 10s^3 - 15s^4 + 6s^5  where s in [0,1]
    This has zero velocity, acceleration, and jerk at both endpoints.
    """
    period = 4.0 * amp / vel
    half = period / 2.0
    phase = t % period

    if phase < half:
        s = phase / half  # 0..1 going from -amp to +amp
        profile = 10*s**3 - 15*s**4 + 6*s**5
        return -amp + 2*amp * profile
    else:
        s = (phase - half) / half  # 0..1 going from +amp to -amp
        profile = 10*s**3 - 15*s**4 + 6*s**5
        return amp - 2*amp * profile


def smoothed_triangle(t: float, amp: float, vel: float) -> float:
    """Triangle wave with cosine-blended corners. Reduces jerk at reversals."""
    period = 4.0 * amp / vel
    phase = t % period
    half = period / 2.0
    # Blend radius: how much of the corner to smooth (fraction of half-period)
    blend_frac = 0.15
    blend_time = blend_frac * half

    if phase < half:
        # Upward stroke: -amp to +amp
        if phase < blend_time:
            # Cosine blend at start (coming from downward)
            s = phase / blend_time  # 0..1
            # Blend from zero velocity to full velocity
            pos = -amp + vel * blend_time * (s - math.sin(math.pi * s) / math.pi)
        elif phase > half - blend_time:
            # Cosine blend approaching top
            linear_pos = -amp + vel * (half - blend_time)
            s = (phase - (half - blend_time)) / blend_time
            remaining = vel * blend_time
            pos = linear_pos + remaining * (s + math.sin(math.pi * s) / math.pi) / 2.0
        else:
            # Linear middle
            pos = -amp + vel * phase
    else:
        # Downward stroke: +amp to -amp
        p2 = phase - half
        if p2 < blend_time:
            s = p2 / blend_time
            pos = amp - vel * blend_time * (s - math.sin(math.pi * s) / math.pi)
        elif p2 > half - blend_time:
            linear_pos = amp - vel * (half - blend_time)
            s = (p2 - (half - blend_time)) / blend_time
            remaining = vel * blend_time
            pos = linear_pos - remaining * (s + math.sin(math.pi * s) / math.pi) / 2.0
        else:
            pos = amp - vel * p2

    return max(-amp, min(amp, pos))


def trapezoidal(t: float, amp: float, vel: float) -> float:
    """Trapezoidal velocity profile: accelerate, cruise, decelerate.

    Smoother than triangle (finite acceleration) but still has jerk at
    transitions between accel/cruise/decel phases.
    """
    period = 4.0 * amp / vel
    half = period / 2.0
    phase = t % period
    # Acceleration phase: 20% of half-period
    accel_frac = 0.20
    accel_time = accel_frac * half
    cruise_time = half - 2 * accel_time
    # Peak velocity is higher than average to cover same distance
    # distance = vel * half = 2 * (0.5 * v_peak * accel_time) + v_peak * cruise_time
    # 2*amp = v_peak * accel_time + v_peak * cruise_time = v_peak * (half - accel_time)
    v_peak = 2 * amp / (half - accel_time)
    accel = v_peak / accel_time

    def stroke(p):
        """Position within one half-period stroke (0 to 2*amp)."""
        if p < accel_time:
            return 0.5 * accel * p * p
        elif p < accel_time + cruise_time:
            d = 0.5 * accel * accel_time * accel_time
            return d + v_peak * (p - accel_time)
        else:
            p3 = p - accel_time - cruise_time
            d = 0.5 * accel * accel_time**2 + v_peak * cruise_time
            return d + v_peak * p3 - 0.5 * accel * p3 * p3

    if phase < half:
        return -amp + stroke(phase)
    else:
        return amp - stroke(phase - half)


def s_curve(t: float, amp: float, vel: float) -> float:
    """S-curve (jerk-limited) profile. Smoothest of the standard motion profiles.

    Uses a 7-segment profile with bounded jerk, acceleration, and velocity.
    Simplified here as cosine acceleration profile.
    """
    period = 4.0 * amp / vel
    half = period / 2.0
    phase = t % period

    if phase < half:
        s = phase / half  # 0..1
        # Cosine-based S-curve: position = amp * (1 - cos(pi*s)) / 2, scaled
        pos = -amp + 2*amp * (1 - math.cos(math.pi * s)) / 2
        return pos
    else:
        s = (phase - half) / half
        pos = amp - 2*amp * (1 - math.cos(math.pi * s)) / 2
        return pos


def pause_at_peaks(t: float, amp: float, vel: float) -> float:
    """Move fast, then pause briefly at each peak.

    Idea: if the antenna rings after stopping, let it settle before reversing.
    Spend 80% of half-period moving, 20% paused at the peak.
    """
    period = 4.0 * amp / vel
    half = period / 2.0
    phase = t % period
    move_frac = 0.80
    move_time = move_frac * half
    # Faster velocity to cover distance in less time
    v_fast = 2 * amp / move_time

    if phase < half:
        if phase < move_time:
            return -amp + v_fast * phase
        else:
            return amp  # pause at top
    else:
        p2 = phase - half
        if p2 < move_time:
            return amp - v_fast * p2
        else:
            return -amp  # pause at bottom


def overshoot_return(t: float, amp: float, vel: float) -> float:
    """Overshoot the target, then come back. May pre-load the spring.

    Move to amp*1.1, then settle back to amp. Idea: the overshoot might
    counteract the mechanical ringing by pulling the antenna past its target
    and letting it spring back to the right position.
    """
    period = 4.0 * amp / vel
    half = period / 2.0
    phase = t % period
    overshoot = 0.10  # 10% overshoot
    amp_over = amp * (1 + overshoot)

    if phase < half:
        s = phase / half
        if s < 0.85:
            # Move to overshoot position
            ss = s / 0.85
            return -amp + (amp + amp_over) * (10*ss**3 - 15*ss**4 + 6*ss**5)
        else:
            # Settle back from overshoot to target
            ss = (s - 0.85) / 0.15
            return amp_over - (amp_over - amp) * (10*ss**3 - 15*ss**4 + 6*ss**5)
    else:
        s = (phase - half) / half
        if s < 0.85:
            ss = s / 0.85
            return amp - (amp + amp_over) * (10*ss**3 - 15*ss**4 + 6*ss**5)
        else:
            ss = (s - 0.85) / 0.15
            return -amp_over + (amp_over - amp) * (10*ss**3 - 15*ss**4 + 6*ss**5)


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryTest:
    name: str
    description: str
    generator: object  # callable(t, amp_rad, vel_rad) -> float


TESTS: list[TrajectoryTest] = [
    TrajectoryTest("triangle", "Constant velocity triangle wave (baseline)", triangle),
    TrajectoryTest("sinusoidal", "Pure sinusoid (smooth but velocity→0 at peaks)", sinusoidal),
    TrajectoryTest("min_jerk", "Minimum-jerk segments (zero jerk at endpoints)", min_jerk),
    TrajectoryTest("smoothed_tri", "Triangle with cosine-blended corners", smoothed_triangle),
    TrajectoryTest("trapezoidal", "Trapezoidal velocity (accel/cruise/decel)", trapezoidal),
    TrajectoryTest("s_curve", "S-curve / cosine acceleration profile", s_curve),
    TrajectoryTest("pause_peaks", "Fast move + 20% pause at each peak to let it settle", pause_at_peaks),
    TrajectoryTest("overshoot", "10% overshoot then settle (pre-load the spring)", overshoot_return),
]


# ---------------------------------------------------------------------------
# Pose decomposition
# ---------------------------------------------------------------------------

def decompose_pose(pose: np.ndarray) -> dict[str, float]:
    x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
    rot = R.from_matrix(pose[:3, :3])
    roll_d, pitch_d, yaw_d = rot.as_euler("xyz", degrees=True)
    return {"x": x, "y": y, "z": z, "roll": roll_d, "pitch": pitch_d, "yaw": yaw_d}


# ---------------------------------------------------------------------------
# Run one test
# ---------------------------------------------------------------------------

def run_test(robot: ReachyMini, test: TrajectoryTest, output_dir: str) -> str:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / f"traj_{test.name}.csv"

    print(f"  [{test.name}] {test.description}")

    start_pose = robot.get_current_head_pose()
    _, start_ant_raw = robot.get_current_joint_positions()
    start_ant = list(start_ant_raw) if start_ant_raw is not None else [0.0, 0.0]

    center_rad = math.radians(CENTER_DEG)
    amp_rad = math.radians(AMPLITUDE_DEG)
    vel_rad = math.radians(SPEED_DEG_S)
    target_ant = [-center_rad, center_rad]

    fieldnames = [
        "t", "loop_dt",
        "cmd_ant_l", "cmd_ant_r",
        "pres_ant_l", "pres_ant_r",
        "pres_z", "pres_roll", "pres_pitch", "pres_yaw",
    ]

    # Interpolate to center position (1 second)
    neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:
        alpha = min(1.0, (time.monotonic() - t0))
        ant_l = (1 - alpha) * start_ant[0] + alpha * target_ant[0]
        ant_r = (1 - alpha) * start_ant[1] + alpha * target_ant[1]
        head = (1 - alpha) * start_pose + alpha * neutral
        robot.set_target(head=head, antennas=[ant_l, ant_r], body_yaw=0.0)
        time.sleep(LOOP_PERIOD)

    # Run test
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        t0 = time.monotonic()
        prev_tick = t0

        while True:
            now = time.monotonic()
            t = now - t0
            if t >= DURATION:
                break

            loop_dt = now - prev_tick
            prev_tick = now

            # Compute trajectory offset
            offset = test.generator(t, amp_rad, vel_rad)
            cmd_ant = [-center_rad + offset, center_rad - offset]

            robot.set_target(
                head=neutral,
                antennas=cmd_ant,
                body_yaw=0.0,
            )

            # Read present state
            pres_pose = robot.get_current_head_pose()
            _, pres_ant_raw = robot.get_current_joint_positions()
            pres_ant = list(pres_ant_raw) if pres_ant_raw is not None else [0.0, 0.0]
            pres_d = decompose_pose(pres_pose)

            row = {
                "t": f"{t:.4f}",
                "loop_dt": f"{loop_dt:.6f}",
                "cmd_ant_l": f"{cmd_ant[0]:.6f}",
                "cmd_ant_r": f"{cmd_ant[1]:.6f}",
                "pres_ant_l": f"{pres_ant[0]:.6f}",
                "pres_ant_r": f"{pres_ant[1]:.6f}",
                "pres_z": f"{pres_d['z']:.6f}",
                "pres_roll": f"{pres_d['roll']:.4f}",
                "pres_pitch": f"{pres_d['pitch']:.4f}",
                "pres_yaw": f"{pres_d['yaw']:.4f}",
            }
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
    parser = argparse.ArgumentParser(description="Trajectory shape experiments")
    parser.add_argument("--name", nargs="*", help="Run specific test(s) by name")
    parser.add_argument("--output", default="/tmp/trajectory_test")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for t in TESTS:
            print(f"  {t.name:<20} {t.description}")
        return

    if args.name:
        selected = [t for t in TESTS if t.name in args.name]
        if not selected:
            print(f"No tests matching: {args.name}")
            print("Use --list to see available tests.")
            sys.exit(1)
    else:
        selected = list(TESTS)

    print(f"Running {len(selected)} trajectory tests...")
    print(f"  Center: {CENTER_DEG}deg, Amplitude: ±{AMPLITUDE_DEG}deg, Speed: ~{SPEED_DEG_S}deg/s")
    print(f"  Duration: {DURATION}s each")

    with ReachyMini(media_backend="no_media") as robot:
        for i, test in enumerate(selected, 1):
            print(f"\n[{i}/{len(selected)}]")
            run_test(robot, test, args.output)
            time.sleep(2)

    print(f"\nAll tests saved to {args.output}/")


if __name__ == "__main__":
    main()
