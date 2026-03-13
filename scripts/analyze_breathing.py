#!/usr/bin/env python3
"""Analyze breathing benchmark CSVs for vibration / oscillation.

Reads CSV files produced by breathing_bench.py and computes:
- Tracking error (commanded vs present) per axis
- High-frequency vibration energy (jerk / acceleration of present positions)
- RMS and peak-to-peak of residuals after removing the commanded signal
- Summary table comparing all configs

Usage:
    python3 analyze_breathing.py                              # all CSVs in breathing_results/
    python3 analyze_breathing.py breathing_results/baseline.csv  # single file
    python3 analyze_breathing.py --plot                        # generate PNG plots
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: str) -> dict[str, np.ndarray]:
    """Load a benchmark CSV into a dict of numpy arrays (column-oriented)."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV: {path}")

    data: dict[str, list[float]] = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k, v in row.items():
            data[k].append(float(v))

    return {k: np.array(v) for k, v in data.items()}


# ---------------------------------------------------------------------------
# Vibration metrics
# ---------------------------------------------------------------------------

@dataclass
class AxisMetrics:
    """Vibration metrics for a single axis."""
    axis: str
    tracking_rms: float        # RMS of (present - commanded)
    residual_rms: float        # RMS of high-pass filtered present signal
    residual_p2p: float        # Peak-to-peak of residual
    jerk_rms: float            # RMS of numerical jerk (3rd derivative)
    vibration_energy: float    # Sum of squared high-freq components (>2Hz)


def highpass_filter(signal: np.ndarray, dt: float, cutoff_hz: float = 2.0) -> np.ndarray:
    """Simple single-pole high-pass filter."""
    if len(signal) < 2 or dt <= 0:
        return np.zeros_like(signal)
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    alpha = rc / (rc + dt)
    filtered = np.zeros_like(signal)
    for i in range(1, len(signal)):
        filtered[i] = alpha * (filtered[i - 1] + signal[i] - signal[i - 1])
    return filtered


def compute_axis_metrics(
    t: np.ndarray, cmd: np.ndarray, present: np.ndarray, axis_name: str,
    skip_seconds: float = 2.0,
) -> AxisMetrics:
    """Compute vibration metrics for one axis, skipping the interpolation phase."""
    # Skip interpolation + settling
    mask = t >= skip_seconds
    if mask.sum() < 10:
        return AxisMetrics(axis_name, 0, 0, 0, 0, 0)

    t_s = t[mask]
    cmd_s = cmd[mask]
    pres_s = present[mask]

    dt = np.median(np.diff(t_s))
    if dt <= 0:
        dt = 1.0 / 60.0

    # Tracking error
    error = pres_s - cmd_s
    tracking_rms = float(np.sqrt(np.mean(error ** 2)))

    # High-pass residual (removes the slow breathing signal, keeps vibration)
    residual = highpass_filter(pres_s, dt, cutoff_hz=2.0)
    residual_rms = float(np.sqrt(np.mean(residual ** 2)))
    residual_p2p = float(np.max(residual) - np.min(residual))

    # Jerk (3rd derivative of position)
    if len(pres_s) > 3:
        vel = np.diff(pres_s) / dt
        acc = np.diff(vel) / dt
        jerk = np.diff(acc) / dt
        jerk_rms = float(np.sqrt(np.mean(jerk ** 2)))
    else:
        jerk_rms = 0.0

    # Vibration energy via FFT (>2Hz components)
    n = len(pres_s)
    if n > 10:
        fft_vals = np.fft.rfft(pres_s - np.mean(pres_s))
        freqs = np.fft.rfftfreq(n, d=dt)
        high_mask = freqs > 2.0
        vibration_energy = float(np.sum(np.abs(fft_vals[high_mask]) ** 2)) / n
    else:
        vibration_energy = 0.0

    return AxisMetrics(axis_name, tracking_rms, residual_rms, residual_p2p, jerk_rms, vibration_energy)


# ---------------------------------------------------------------------------
# Analysis per config
# ---------------------------------------------------------------------------

@dataclass
class ConfigResult:
    """Summary for one breathing config."""
    name: str
    axis_metrics: list[AxisMetrics]
    mean_loop_dt: float
    total_vibration_score: float  # aggregate across axes


def analyze_file(path: str) -> ConfigResult:
    """Analyze one CSV file."""
    data = load_csv(path)
    name = Path(path).stem
    t = data["t"]

    # Axes to analyze: position (metres) and rotation (degrees)
    axes = [
        ("x", "cmd_x", "pres_x"),
        ("y", "cmd_y", "pres_y"),
        ("z", "cmd_z", "pres_z"),
        ("roll", "cmd_roll", "pres_roll"),
        ("pitch", "cmd_pitch", "pres_pitch"),
        ("yaw", "cmd_yaw", "pres_yaw"),
        ("ant_l", "cmd_ant_l", "pres_ant_l"),
        ("ant_r", "cmd_ant_r", "pres_ant_r"),
    ]

    metrics = []
    for axis_name, cmd_key, pres_key in axes:
        m = compute_axis_metrics(t, data[cmd_key], data[pres_key], axis_name)
        metrics.append(m)

    mean_loop_dt = float(np.mean(data["loop_dt"]))

    # Aggregate vibration score: weighted sum of residual RMS
    # Weight position axes (m) more than rotation (deg) by scaling
    pos_weight = 1000.0  # convert m -> mm for comparable scale
    rot_weight = 1.0
    ant_weight = 100.0   # radians -> ~degrees scale
    score = 0.0
    for m in metrics:
        if m.axis in ("x", "y", "z"):
            score += (m.residual_rms * pos_weight) ** 2
        elif m.axis in ("ant_l", "ant_r"):
            score += (m.residual_rms * ant_weight) ** 2
        else:
            score += (m.residual_rms * rot_weight) ** 2
    score = float(np.sqrt(score))

    return ConfigResult(name, metrics, mean_loop_dt, score)


# ---------------------------------------------------------------------------
# Plotting (optional)
# ---------------------------------------------------------------------------

def plot_config(path: str, out_dir: str) -> None:
    """Generate a multi-axis plot for one CSV."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    data = load_csv(path)
    name = Path(path).stem
    t = data["t"]

    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
    fig.suptitle(f"Breathing Benchmark: {name}", fontsize=14)

    plot_pairs = [
        (axes[0, 0], "z", "cmd_z", "pres_z", "Z position (m)"),
        (axes[0, 1], "x", "cmd_x", "pres_x", "X position (m)"),
        (axes[1, 0], "roll", "cmd_roll", "pres_roll", "Roll (deg)"),
        (axes[1, 1], "pitch", "cmd_pitch", "pres_pitch", "Pitch (deg)"),
        (axes[2, 0], "yaw", "cmd_yaw", "pres_yaw", "Yaw (deg)"),
        (axes[2, 1], "y", "cmd_y", "pres_y", "Y position (m)"),
        (axes[3, 0], "ant_l", "cmd_ant_l", "pres_ant_l", "Antenna L (rad)"),
        (axes[3, 1], "ant_r", "cmd_ant_r", "pres_ant_r", "Antenna R (rad)"),
    ]

    for ax, label, cmd_key, pres_key, ylabel in plot_pairs:
        ax.plot(t, data[cmd_key], "b-", alpha=0.5, linewidth=0.8, label="commanded")
        ax.plot(t, data[pres_key], "r-", alpha=0.7, linewidth=0.8, label="present")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[3, 0].set_xlabel("Time (s)")
    axes[3, 1].set_xlabel("Time (s)")

    plt.tight_layout()
    out_path = Path(out_dir) / f"{name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze breathing benchmark data")
    parser.add_argument("files", nargs="*", help="CSV files (default: breathing_results/*.csv)")
    parser.add_argument("--plot", action="store_true", help="Generate PNG plots")
    parser.add_argument("--results-dir", default="breathing_results", help="Results directory")
    args = parser.parse_args()

    if args.files:
        csv_files = [Path(f) for f in args.files]
    else:
        results_dir = Path(args.results_dir)
        csv_files = sorted(results_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found. Run the benchmark first.")
        sys.exit(1)

    results: list[ConfigResult] = []
    for f in csv_files:
        print(f"Analyzing {f.name}...")
        try:
            r = analyze_file(str(f))
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")

        if args.plot:
            plot_config(str(f), args.results_dir)

    # Summary table
    print("\n" + "=" * 100)
    print(f"{'Config':<20} {'Vibration':>10} {'Loop dt':>10} "
          f"{'Z res_rms':>10} {'Z p2p':>10} "
          f"{'Roll res':>10} {'Pitch res':>10} {'Yaw res':>10} "
          f"{'AntL res':>10}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: x.total_vibration_score):
        z_m = next((m for m in r.axis_metrics if m.axis == "z"), None)
        roll_m = next((m for m in r.axis_metrics if m.axis == "roll"), None)
        pitch_m = next((m for m in r.axis_metrics if m.axis == "pitch"), None)
        yaw_m = next((m for m in r.axis_metrics if m.axis == "yaw"), None)
        ant_m = next((m for m in r.axis_metrics if m.axis == "ant_l"), None)

        print(f"{r.name:<20} {r.total_vibration_score:>10.4f} {r.mean_loop_dt * 1000:>8.1f}ms "
              f"{z_m.residual_rms * 1000 if z_m else 0:>9.3f}mm "
              f"{z_m.residual_p2p * 1000 if z_m else 0:>9.3f}mm "
              f"{roll_m.residual_rms if roll_m else 0:>9.4f}° "
              f"{pitch_m.residual_rms if pitch_m else 0:>9.4f}° "
              f"{yaw_m.residual_rms if yaw_m else 0:>9.4f}° "
              f"{ant_m.residual_rms if ant_m else 0:>9.6f}r")

    print("=" * 100)
    print("\nVibration score = weighted RMS of high-pass residuals (lower is better)")
    print("Columns: res_rms = high-pass (>2Hz) RMS, p2p = peak-to-peak of residual")

    # Per-axis detail for best and worst
    if len(results) >= 2:
        best = min(results, key=lambda x: x.total_vibration_score)
        worst = max(results, key=lambda x: x.total_vibration_score)
        print(f"\nBest:  {best.name} (score={best.total_vibration_score:.4f})")
        print(f"Worst: {worst.name} (score={worst.total_vibration_score:.4f})")
        if best.total_vibration_score > 0:
            ratio = worst.total_vibration_score / best.total_vibration_score
            print(f"Ratio: {ratio:.1f}x")


if __name__ == "__main__":
    main()
