#!/usr/bin/env python3
"""Test antenna vibration using microphone audio analysis.

The motor encoders can't see antenna tip vibration because the antennas are
flexible spring-like rods. But the vibration is audible — the mic picks up
buzzing/rattling when the antennas shake.

This script:
1. Runs a trajectory test while recording audio from the robot's mic
2. Analyzes the audio for vibration energy in relevant frequency bands
3. Outputs metrics that capture actual mechanical vibration

Usage:
    python3 vibration_audio_test.py                          # s_curve test
    python3 vibration_audio_test.py --name triangle          # specific trajectory
    python3 vibration_audio_test.py --output /tmp/vib_test   # custom output dir
    python3 vibration_audio_test.py --audio-only             # just record, no trajectory
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

# Import trajectory generators from trajectory_test.py (same directory on robot)
sys.path.insert(0, str(Path(__file__).parent))
from trajectory_test import TESTS, CENTER_DEG, AMPLITUDE_DEG, SPEED_DEG_S, DURATION, LOOP_PERIOD, LOOP_HZ


# Audio config
AUDIO_DEVICE = "reachymini_audio_src"
AUDIO_RATE = 16000
AUDIO_CHANNELS = 2
AUDIO_FORMAT = "S16_LE"


def start_recording(output_path: str, duration: float) -> subprocess.Popen:
    """Start recording audio in background."""
    cmd = [
        "arecord",
        "-D", AUDIO_DEVICE,
        "-f", AUDIO_FORMAT,
        "-r", str(AUDIO_RATE),
        "-c", str(AUDIO_CHANNELS),
        "-d", str(int(duration) + 2),  # extra 2s buffer
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.3)  # let recording start
    return proc


def stop_recording(proc: subprocess.Popen) -> None:
    """Stop audio recording."""
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


def analyze_audio(wav_path: str) -> dict:
    """Analyze audio file for vibration content."""
    with wave.open(wav_path) as w:
        frames = w.readframes(w.getnframes())
        rate = w.getframerate()
        channels = w.getnchannels()

    data = np.frombuffer(frames, dtype=np.int16).astype(float)

    # Use first channel only
    if channels > 1:
        data = data[::channels]

    n = len(data)
    if n < rate:
        return {"error": "too short"}

    # Skip first 2 seconds (settling + recording start)
    skip = 2 * rate
    data = data[skip:]
    n = len(data)

    # Normalize to [-1, 1]
    data = data / 32768.0

    # Overall RMS
    rms_total = float(np.sqrt(np.mean(data**2)))

    # FFT analysis
    freqs = np.fft.rfftfreq(n, d=1.0/rate)
    fft_mag = np.abs(np.fft.rfft(data)) / n

    # Energy in frequency bands
    def band_energy(low, high):
        mask = (freqs >= low) & (freqs < high)
        return float(np.sqrt(np.mean(fft_mag[mask]**2))) if mask.any() else 0.0

    # Motor noise: ~0-80 Hz (servo PWM, gear meshing)
    motor_energy = band_energy(20, 80)

    # Vibration band: 80-500 Hz (antenna rattling, buzzing)
    vib_energy = band_energy(80, 500)

    # High-freq vibration: 500-2000 Hz (ringing, harmonics)
    hf_energy = band_energy(500, 2000)

    # Combined vibration metric: weighted sum of vibration bands
    vib_metric = vib_energy + 0.5 * hf_energy

    # Vibration-to-motor ratio (higher = more vibration relative to motor noise)
    vmr = vib_energy / max(motor_energy, 1e-10)

    # Time-domain: compute RMS in sliding windows to find peak vibration moments
    window_size = rate // 4  # 250ms windows
    n_windows = n // window_size
    window_rms = []
    for i in range(n_windows):
        chunk = data[i * window_size : (i + 1) * window_size]
        window_rms.append(float(np.sqrt(np.mean(chunk**2))))

    peak_rms = max(window_rms) if window_rms else 0.0
    min_rms = min(window_rms) if window_rms else 0.0
    rms_variance = float(np.std(window_rms)) if window_rms else 0.0

    return {
        "rms_total": rms_total,
        "motor_energy": motor_energy,
        "vib_energy": vib_energy,
        "hf_energy": hf_energy,
        "vib_metric": vib_metric,
        "vib_motor_ratio": vmr,
        "peak_rms": peak_rms,
        "min_rms": min_rms,
        "rms_variance": rms_variance,
        "duration_s": n / rate,
        "n_samples": n,
    }


def run_trajectory_with_audio(robot: ReachyMini, test_name: str, output_dir: str) -> dict:
    """Run a trajectory while recording audio."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Find the test
    test = None
    for t in TESTS:
        if t.name == test_name:
            test = t
            break
    if test is None:
        print(f"Unknown test: {test_name}")
        return {"error": f"unknown test: {test_name}"}

    wav_path = str(out_path / f"audio_{test.name}.wav")
    csv_path = str(out_path / f"traj_{test.name}.csv")

    print(f"  [{test.name}] {test.description}")

    # Get current state
    start_pose = robot.get_current_head_pose()
    _, start_ant_raw = robot.get_current_joint_positions()
    start_ant = list(start_ant_raw) if start_ant_raw is not None else [0.0, 0.0]

    center_rad = math.radians(CENTER_DEG)
    amp_rad = math.radians(AMPLITUDE_DEG)
    vel_rad = math.radians(SPEED_DEG_S)
    target_ant = [-center_rad, center_rad]
    neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)

    # Interpolate to center (1s)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:
        alpha = min(1.0, (time.monotonic() - t0))
        ant_l = (1 - alpha) * start_ant[0] + alpha * target_ant[0]
        ant_r = (1 - alpha) * start_ant[1] + alpha * target_ant[1]
        head = (1 - alpha) * start_pose + alpha * neutral
        robot.set_target(head=head, antennas=[ant_l, ant_r], body_yaw=0.0)
        time.sleep(LOOP_PERIOD)

    # Start audio recording
    print(f"    Recording audio to {wav_path}")
    audio_proc = start_recording(wav_path, DURATION)

    # Run trajectory
    fieldnames = ["t", "loop_dt", "cmd_ant_l", "cmd_ant_r", "pres_ant_l", "pres_ant_r"]
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

            offset = test.generator(t, amp_rad, vel_rad)
            cmd_ant = [-center_rad + offset, center_rad - offset]
            robot.set_target(head=neutral, antennas=cmd_ant, body_yaw=0.0)

            _, pres_ant_raw = robot.get_current_joint_positions()
            pres_ant = list(pres_ant_raw) if pres_ant_raw is not None else [0.0, 0.0]

            writer.writerow({
                "t": f"{t:.4f}",
                "loop_dt": f"{loop_dt:.6f}",
                "cmd_ant_l": f"{cmd_ant[0]:.6f}",
                "cmd_ant_r": f"{cmd_ant[1]:.6f}",
                "pres_ant_l": f"{pres_ant[0]:.6f}",
                "pres_ant_r": f"{pres_ant[1]:.6f}",
            })

            elapsed = time.monotonic() - now
            sleep_time = LOOP_PERIOD - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Stop recording
    stop_recording(audio_proc)
    print(f"    -> {csv_path}")
    print(f"    -> {wav_path}")

    # Analyze audio
    result = analyze_audio(wav_path)
    result["name"] = test.name
    result["wav_path"] = wav_path
    result["csv_path"] = csv_path
    return result


def record_silence(output_dir: str, duration: float = 5.0) -> dict:
    """Record silence (no movement) as a noise floor reference."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    wav_path = str(out_path / "audio_silence.wav")

    print("  [silence] Recording noise floor (no movement)...")
    proc = start_recording(wav_path, duration)
    time.sleep(duration + 1)
    stop_recording(proc)

    result = analyze_audio(wav_path)
    result["name"] = "silence"
    result["wav_path"] = wav_path
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Vibration audio analysis")
    parser.add_argument("--name", nargs="*", default=["s_curve"],
                        help="Trajectory test name(s)")
    parser.add_argument("--output", default="/tmp/vibration_test")
    parser.add_argument("--audio-only", action="store_true",
                        help="Just record audio (no trajectory)")
    parser.add_argument("--no-silence", action="store_true",
                        help="Skip silence recording")
    args = parser.parse_args()

    results = []

    if args.audio_only:
        result = record_silence(args.output, duration=10.0)
        results.append(result)
    else:
        # Record silence first as baseline
        if not args.no_silence:
            print("\n[0] Recording silence baseline...")
            result = record_silence(args.output)
            results.append(result)

        with ReachyMini(media_backend="no_media") as robot:
            for i, name in enumerate(args.name, 1):
                print(f"\n[{i}/{len(args.name)}]")
                result = run_trajectory_with_audio(robot, name, args.output)
                results.append(result)
                time.sleep(2)

    # Print results
    print(f"\n{'='*90}")
    print("  VIBRATION AUDIO ANALYSIS")
    print(f"{'='*90}")
    hdr = (f"{'Name':<18} {'RMS':>8} {'Motor':>8} {'Vib':>8} {'HF':>8} "
           f"{'VibMetric':>10} {'V/M Ratio':>10} {'PeakRMS':>8}")
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        if "error" in r:
            print(f"  {r.get('name', '?')}: {r['error']}")
            continue
        print(f"{r['name']:<18} {r['rms_total']:>8.5f} {r['motor_energy']:>8.6f} "
              f"{r['vib_energy']:>8.6f} {r['hf_energy']:>8.6f} "
              f"{r['vib_metric']:>10.6f} {r['vib_motor_ratio']:>10.2f} "
              f"{r['peak_rms']:>8.5f}")

    print(f"\nHigher VibMetric = more audible vibration")
    print(f"V/M Ratio = vibration energy / motor noise (higher = vibration dominates)")

    # Save CSV summary
    summary_path = Path(args.output) / "vibration_summary.csv"
    with open(summary_path, "w", newline="") as f:
        if results and "error" not in results[0]:
            writer = csv.DictWriter(f, fieldnames=[k for k in results[0] if k not in ("wav_path", "csv_path")])
            writer.writeheader()
            for r in results:
                if "error" not in r:
                    row = {k: v for k, v in r.items() if k not in ("wav_path", "csv_path")}
                    writer.writerow(row)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
