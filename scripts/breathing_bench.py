#!/usr/bin/env python3
"""Breathing benchmark: run a breathing config and record present positions.

Runs on the robot (Wireless). Reproduces the conversation app's BreathingMove
at 60 Hz, recording commanded vs present positions to a CSV file for offline
vibration analysis.

Usage:
    python3 breathing_bench.py                    # default config
    python3 breathing_bench.py --config gentle    # named config
    python3 breathing_bench.py --duration 30      # 30s recording
    python3 breathing_bench.py --list             # list available configs
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


# ---------------------------------------------------------------------------
# Breathing configurations
# ---------------------------------------------------------------------------

@dataclass
class BreathingConfig:
    """Parameters for a breathing pattern."""
    name: str
    description: str

    # Head motion (sinusoidal offsets from neutral, in metres / degrees)
    z_amplitude: float = 0.005       # metres
    z_frequency: float = 0.1         # Hz
    x_amplitude: float = 0.0
    x_frequency: float = 0.0
    y_amplitude: float = 0.0
    y_frequency: float = 0.0
    roll_amplitude_deg: float = 0.0
    roll_frequency: float = 0.0
    pitch_amplitude_deg: float = 0.0
    pitch_frequency: float = 0.0
    yaw_amplitude_deg: float = 0.0
    yaw_frequency: float = 0.0

    # Antenna motion (sinusoidal, opposite signs for L/R)
    antenna_amplitude_deg: float = 15.0
    antenna_frequency: float = 0.5

    # Antenna center offset (SDK default: left=-10deg, right=+10deg to avoid vertical)
    # 0.0 = old behaviour (oscillate around vertical)
    # 10.0 = oscillate around the 10-degree offset position
    antenna_center_deg: float = 0.0

    # Interpolation to neutral before breathing starts
    interpolation_duration: float = 1.0  # seconds


# Registry of configs to iterate over
CONFIGS: dict[str, BreathingConfig] = {}


def register(cfg: BreathingConfig) -> BreathingConfig:
    CONFIGS[cfg.name] = cfg
    return cfg


# =========================================================================
# Round 3: 10-degree antenna offset + more head motion variety
#
# Key insight from rounds 1-2: antenna sway around vertical (0 deg) is the
# dominant vibration source. The SDK now defaults antennas to ±10 deg offset.
# All configs below use antenna_center_deg=10 (oscillate around offset).
# =========================================================================

# ---- Controls ----
register(BreathingConfig(
    name="still_offset",
    description="CONTROL: antennas at 10deg offset, no motion",
    z_amplitude=0.0, z_frequency=0.0,
    antenna_amplitude_deg=0.0, antenna_frequency=0.0,
    antenna_center_deg=10.0,
))

register(BreathingConfig(
    name="still_vertical",
    description="CONTROL: antennas at vertical (0deg), no motion",
    z_amplitude=0.0, z_frequency=0.0,
    antenna_amplitude_deg=0.0, antenna_frequency=0.0,
    antenna_center_deg=0.0,
))

register(BreathingConfig(
    name="baseline_old",
    description="OLD baseline: z 5mm + ant 15deg around vertical",
    antenna_center_deg=0.0,
))

# ---- Antenna sway around 10deg offset (amplitude sweep) ----
register(BreathingConfig(
    name="offset_ant5",
    description="Ant 5deg@0.5Hz around 10deg offset + z 5mm",
    antenna_amplitude_deg=5.0, antenna_center_deg=10.0,
))

register(BreathingConfig(
    name="offset_ant8",
    description="Ant 8deg@0.5Hz around 10deg offset + z 5mm",
    antenna_amplitude_deg=8.0, antenna_center_deg=10.0,
))

register(BreathingConfig(
    name="offset_ant10",
    description="Ant 10deg@0.5Hz around 10deg offset + z 5mm",
    antenna_amplitude_deg=10.0, antenna_center_deg=10.0,
))

register(BreathingConfig(
    name="offset_ant15",
    description="Ant 15deg@0.5Hz around 10deg offset + z 5mm (same amp as old)",
    antenna_amplitude_deg=15.0, antenna_center_deg=10.0,
))

# ---- Head motion variety (with sensible antenna at 10deg offset) ----

# Big z breathing
register(BreathingConfig(
    name="big_z",
    description="z 8mm@0.1Hz + ant 5deg around 10deg offset",
    z_amplitude=0.008, z_frequency=0.1,
    antenna_amplitude_deg=5.0, antenna_frequency=0.4, antenna_center_deg=10.0,
))

# Strong roll
register(BreathingConfig(
    name="roll_big",
    description="Roll 4deg@0.1Hz + ant 5deg around 10deg offset",
    z_amplitude=0.0,
    roll_amplitude_deg=4.0, roll_frequency=0.1,
    antenna_amplitude_deg=5.0, antenna_frequency=0.4, antenna_center_deg=10.0,
))

# Strong yaw (looking around)
register(BreathingConfig(
    name="yaw_big",
    description="Yaw 5deg@0.06Hz + z 3mm + ant 5deg around 10deg",
    z_amplitude=0.003,
    yaw_amplitude_deg=5.0, yaw_frequency=0.06,
    antenna_amplitude_deg=5.0, antenna_frequency=0.3, antenna_center_deg=10.0,
))

# Pitch nod (bigger)
register(BreathingConfig(
    name="nod_big",
    description="Pitch 3deg@0.12Hz + z 3mm + ant 5deg around 10deg",
    z_amplitude=0.003,
    pitch_amplitude_deg=3.0, pitch_frequency=0.12,
    antenna_amplitude_deg=5.0, antenna_frequency=0.3, antenna_center_deg=10.0,
))

# Figure-8 (bigger)
register(BreathingConfig(
    name="figure8_big",
    description="Yaw 3deg@0.08Hz + pitch 2deg@0.16Hz + ant 5deg around 10deg",
    z_amplitude=0.0,
    yaw_amplitude_deg=3.0, yaw_frequency=0.08,
    pitch_amplitude_deg=2.0, pitch_frequency=0.16,
    antenna_amplitude_deg=5.0, antenna_frequency=0.3, antenna_center_deg=10.0,
))

# ---- Creative combos with more presence ----

# Curious: looking around slowly with breathing
register(BreathingConfig(
    name="curious",
    description="z 4mm + yaw 4deg@0.05Hz + roll 2deg@0.11Hz + ant 5deg@0.3Hz",
    z_amplitude=0.004, z_frequency=0.1,
    yaw_amplitude_deg=4.0, yaw_frequency=0.05,
    roll_amplitude_deg=2.0, roll_frequency=0.11,
    antenna_amplitude_deg=5.0, antenna_frequency=0.3, antenna_center_deg=10.0,
))

# Sleepy: slow everything, gentle nod
register(BreathingConfig(
    name="sleepy",
    description="z 5mm@0.07Hz + pitch 2deg@0.06Hz + ant 3deg@0.15Hz",
    z_amplitude=0.005, z_frequency=0.07,
    pitch_amplitude_deg=2.0, pitch_frequency=0.06,
    antenna_amplitude_deg=3.0, antenna_frequency=0.15, antenna_center_deg=10.0,
))

# Alive (updated with offset)
register(BreathingConfig(
    name="alive_v2",
    description="z 3mm + roll 1.5deg@0.13Hz + yaw 2deg@0.07Hz + ant 4deg@0.3Hz",
    z_amplitude=0.003, z_frequency=0.1,
    roll_amplitude_deg=1.5, roll_frequency=0.13,
    yaw_amplitude_deg=2.0, yaw_frequency=0.07,
    antenna_amplitude_deg=4.0, antenna_frequency=0.3, antenna_center_deg=10.0,
))

# Dreamy (updated with offset and more motion)
register(BreathingConfig(
    name="dreamy_v2",
    description="z 4mm@0.08Hz + yaw 3deg@0.05Hz + roll 1deg@0.12Hz + ant 4deg@0.2Hz",
    z_amplitude=0.004, z_frequency=0.08,
    yaw_amplitude_deg=3.0, yaw_frequency=0.05,
    roll_amplitude_deg=1.0, roll_frequency=0.12,
    antenna_amplitude_deg=4.0, antenna_frequency=0.2, antenna_center_deg=10.0,
))

# Playful: faster movements, more energy
register(BreathingConfig(
    name="playful",
    description="z 3mm@0.15Hz + roll 2deg@0.2Hz + yaw 2deg@0.12Hz + ant 6deg@0.5Hz",
    z_amplitude=0.003, z_frequency=0.15,
    roll_amplitude_deg=2.0, roll_frequency=0.2,
    yaw_amplitude_deg=2.0, yaw_frequency=0.12,
    antenna_amplitude_deg=6.0, antenna_frequency=0.5, antenna_center_deg=10.0,
))

# Zen: minimal, very slow, meditative
register(BreathingConfig(
    name="zen",
    description="z 2mm@0.06Hz + roll 0.5deg@0.04Hz + ant 2deg@0.1Hz",
    z_amplitude=0.002, z_frequency=0.06,
    roll_amplitude_deg=0.5, roll_frequency=0.04,
    antenna_amplitude_deg=2.0, antenna_frequency=0.1, antenna_center_deg=10.0,
))

# Ocean: rolling like waves
register(BreathingConfig(
    name="ocean",
    description="z 5mm@0.08Hz + roll 3deg@0.1Hz (phase-locked wave) + ant 4deg@0.2Hz",
    z_amplitude=0.005, z_frequency=0.08,
    roll_amplitude_deg=3.0, roll_frequency=0.1,
    antenna_amplitude_deg=4.0, antenna_frequency=0.2, antenna_center_deg=10.0,
))


# ---------------------------------------------------------------------------
# Breathing evaluator (mirrors BreathingMove.evaluate)
# ---------------------------------------------------------------------------

def evaluate_breathing(cfg: BreathingConfig, t: float,
                       start_pose: np.ndarray, start_antennas: np.ndarray
                       ) -> tuple[np.ndarray, list[float]]:
    """Compute target head pose and antenna positions at time t."""
    ant_center = math.radians(cfg.antenna_center_deg)
    target_ant = np.array([-ant_center, ant_center])

    if t < cfg.interpolation_duration:
        # Interpolate from start to neutral (head) / offset (antennas)
        alpha = t / cfg.interpolation_duration
        neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        head_pose = (1.0 - alpha) * start_pose + alpha * neutral
        ant = (1.0 - alpha) * start_antennas + alpha * target_ant
        return head_pose, [ant[0], ant[1]]

    bt = t - cfg.interpolation_duration

    z = cfg.z_amplitude * math.sin(2 * math.pi * cfg.z_frequency * bt)
    x = cfg.x_amplitude * math.sin(2 * math.pi * cfg.x_frequency * bt)
    y = cfg.y_amplitude * math.sin(2 * math.pi * cfg.y_frequency * bt)
    roll = cfg.roll_amplitude_deg * math.sin(2 * math.pi * cfg.roll_frequency * bt)
    pitch = cfg.pitch_amplitude_deg * math.sin(2 * math.pi * cfg.pitch_frequency * bt)
    yaw = cfg.yaw_amplitude_deg * math.sin(2 * math.pi * cfg.yaw_frequency * bt)

    head_pose = create_head_pose(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                                 degrees=True, mm=False)

    ant_amp = math.radians(cfg.antenna_amplitude_deg)
    ant_center = math.radians(cfg.antenna_center_deg)
    ant_sway = ant_amp * math.sin(2 * math.pi * cfg.antenna_frequency * bt)
    # Left antenna: -center + sway, Right antenna: +center - sway
    # (matches SDK convention: INIT_ANTENNAS = [-0.1745, +0.1745])
    antennas = [-ant_center + ant_sway, ant_center - ant_sway]

    return head_pose, antennas


# ---------------------------------------------------------------------------
# Pose decomposition (extract x,y,z,roll,pitch,yaw from 4x4 matrix)
# ---------------------------------------------------------------------------

def decompose_pose(pose: np.ndarray) -> dict[str, float]:
    """Extract x, y, z (metres) and roll, pitch, yaw (degrees) from a 4x4 matrix."""
    from scipy.spatial.transform import Rotation as R
    x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
    rot = R.from_matrix(pose[:3, :3])
    roll_deg, pitch_deg, yaw_deg = rot.as_euler("xyz", degrees=True)
    return {"x": x, "y": y, "z": z, "roll": roll_deg, "pitch": pitch_deg, "yaw": yaw_deg}


# ---------------------------------------------------------------------------
# Main recording loop
# ---------------------------------------------------------------------------

LOOP_HZ = 50.0  # Must match daemon's internal control loop (50 Hz)
LOOP_PERIOD = 1.0 / LOOP_HZ


def run_benchmark(cfg: BreathingConfig, duration: float, output_dir: str) -> str:
    """Run one breathing config and record data. Returns output CSV path."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / f"{cfg.name}.csv"

    print(f"[{cfg.name}] Connecting to robot...")
    with ReachyMini(media_backend="no_media") as robot:
        # Read initial state
        start_pose = robot.get_current_head_pose()
        _, start_ant_raw = robot.get_current_joint_positions()
        start_antennas = np.array(start_ant_raw if start_ant_raw is not None else [0.0, 0.0])

        print(f"[{cfg.name}] Recording {duration}s at {LOOP_HZ}Hz -> {csv_path}")
        print(f"  {cfg.description}")

        fieldnames = [
            "t", "loop_dt",
            # Commanded
            "cmd_x", "cmd_y", "cmd_z", "cmd_roll", "cmd_pitch", "cmd_yaw",
            "cmd_ant_l", "cmd_ant_r",
            # Present (read back)
            "pres_x", "pres_y", "pres_z", "pres_roll", "pres_pitch", "pres_yaw",
            "pres_ant_l", "pres_ant_r",
            # Joint positions (raw)
            "pres_j0", "pres_j1", "pres_j2", "pres_j3", "pres_j4", "pres_j5", "pres_j6",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            t0 = time.monotonic()
            prev_tick = t0

            while True:
                now = time.monotonic()
                t = now - t0
                if t >= duration + cfg.interpolation_duration:
                    break

                loop_dt = now - prev_tick
                prev_tick = now

                # Compute and send command
                head_pose, antennas = evaluate_breathing(cfg, t, start_pose, start_antennas)
                robot.set_target(head=head_pose, antennas=antennas, body_yaw=0.0)

                # Read present state
                pres_pose = robot.get_current_head_pose()
                head_joints, pres_ant_raw = robot.get_current_joint_positions()
                pres_ant = pres_ant_raw if pres_ant_raw is not None else [0.0, 0.0]
                head_joints = head_joints if head_joints is not None else [0.0] * 7

                cmd_d = decompose_pose(head_pose)
                pres_d = decompose_pose(pres_pose)

                row = {
                    "t": f"{t:.4f}",
                    "loop_dt": f"{loop_dt:.6f}",
                    "cmd_x": f"{cmd_d['x']:.6f}",
                    "cmd_y": f"{cmd_d['y']:.6f}",
                    "cmd_z": f"{cmd_d['z']:.6f}",
                    "cmd_roll": f"{cmd_d['roll']:.4f}",
                    "cmd_pitch": f"{cmd_d['pitch']:.4f}",
                    "cmd_yaw": f"{cmd_d['yaw']:.4f}",
                    "cmd_ant_l": f"{antennas[0]:.6f}",
                    "cmd_ant_r": f"{antennas[1]:.6f}",
                    "pres_x": f"{pres_d['x']:.6f}",
                    "pres_y": f"{pres_d['y']:.6f}",
                    "pres_z": f"{pres_d['z']:.6f}",
                    "pres_roll": f"{pres_d['roll']:.4f}",
                    "pres_pitch": f"{pres_d['pitch']:.4f}",
                    "pres_yaw": f"{pres_d['yaw']:.4f}",
                    "pres_ant_l": f"{pres_ant[0]:.6f}",
                    "pres_ant_r": f"{pres_ant[1]:.6f}",
                }
                for i in range(7):
                    row[f"pres_j{i}"] = f"{head_joints[i]:.6f}"
                writer.writerow(row)

                # Sleep to maintain target frequency
                elapsed = time.monotonic() - now
                sleep_time = LOOP_PERIOD - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        tick_count = int((duration + cfg.interpolation_duration) * LOOP_HZ)
        actual_duration = time.monotonic() - t0
        print(f"[{cfg.name}] Done. {actual_duration:.1f}s actual, file: {csv_path}")

    return str(csv_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Breathing vibration benchmark")
    parser.add_argument("--config", default="baseline", help="Config name or 'all'")
    parser.add_argument("--duration", type=float, default=20.0, help="Recording duration (seconds)")
    parser.add_argument("--output", default="/tmp/breathing_bench", help="Output directory")
    parser.add_argument("--list", action="store_true", help="List available configs")
    args = parser.parse_args()

    if args.list:
        for name, cfg in CONFIGS.items():
            print(f"  {name:20s}  {cfg.description}")
        return

    if args.config == "all":
        configs = list(CONFIGS.values())
    else:
        if args.config not in CONFIGS:
            print(f"Unknown config '{args.config}'. Available: {', '.join(CONFIGS.keys())}")
            sys.exit(1)
        configs = [CONFIGS[args.config]]

    for cfg in configs:
        run_benchmark(cfg, args.duration, args.output)
        if len(configs) > 1:
            print("  Pausing 3s between configs...")
            time.sleep(3)

    print(f"\nAll recordings saved to {args.output}/")


if __name__ == "__main__":
    main()
