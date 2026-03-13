#!/usr/bin/env python3
"""Analyze friction test CSVs: detect stick-slip, plot, and summarize.

For each test, computes:
- Number of stall ticks (present position unchanged between consecutive samples)
- Maximum consecutive stall run (in ticks and ms)
- Stall percentage (what fraction of time the motor is stuck)
- Generates per-test plots showing commanded vs present with stall regions highlighted

Usage:
    python3 analyze_friction.py                                # all CSVs
    python3 analyze_friction.py friction_results/ant_c0_v5.csv # single file
    python3 analyze_friction.py --plot                         # with plots
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np


TICK_MS = 1000.0 / 50.0  # Daemon runs at 50 Hz


def load_csv(path: str) -> dict[str, np.ndarray]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty: {path}")
    data: dict[str, list[float]] = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k, v in row.items():
            data[k].append(float(v))
    return {k: np.array(v) for k, v in data.items()}


@dataclass
class StallAnalysis:
    """Stick-slip metrics for one signal."""
    signal_name: str
    total_ticks: int
    stall_ticks: int
    max_stall_run: int      # consecutive ticks
    max_stall_ms: float
    stall_pct: float
    stall_runs: list[tuple[int, int]]  # (start_idx, length) of each run


def analyze_stalls(signal: np.ndarray, name: str, threshold: float = 1e-5,
                   skip_ticks: int = 60) -> StallAnalysis:
    """Detect stall runs in a present-position signal.

    A stall = the position doesn't change by more than `threshold` between ticks.
    Skip the first `skip_ticks` (interpolation phase).
    """
    sig = signal[skip_ticks:]
    n = len(sig)
    if n < 2:
        return StallAnalysis(name, 0, 0, 0, 0, 0, [])

    diffs = np.abs(np.diff(sig))
    is_stall = diffs < threshold

    # Find runs of stalls
    runs = []
    cur_start = -1
    cur_len = 0
    for i, stalled in enumerate(is_stall):
        if stalled:
            if cur_start < 0:
                cur_start = i + skip_ticks
            cur_len += 1
        else:
            if cur_len > 0:
                runs.append((cur_start, cur_len))
            cur_start = -1
            cur_len = 0
    if cur_len > 0:
        runs.append((cur_start, cur_len))

    total_stall = int(is_stall.sum())
    max_run = max((r[1] for r in runs), default=0)

    return StallAnalysis(
        signal_name=name,
        total_ticks=n - 1,
        stall_ticks=total_stall,
        max_stall_run=max_run,
        max_stall_ms=max_run * TICK_MS,
        stall_pct=100.0 * total_stall / max(1, n - 1),
        stall_runs=runs,
    )


def analyze_file(path: str) -> tuple[str, StallAnalysis, StallAnalysis]:
    """Analyze one CSV. Returns (name, z_analysis, ant_analysis)."""
    data = load_csv(path)
    name = Path(path).stem

    # Choose which signal to analyze based on test type
    z_thresh = 1e-5    # 10 microns
    ant_thresh = 1e-4  # 0.1 mrad ≈ 0.006 degrees

    z_analysis = analyze_stalls(data["pres_z"], "head_z", z_thresh)
    ant_analysis = analyze_stalls(data["pres_ant_l"], "ant_l", ant_thresh)

    return name, z_analysis, ant_analysis


def plot_test(path: str, out_dir: str) -> None:
    """Generate a plot for one test with stall regions highlighted."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    data = load_csv(path)
    name = Path(path).stem
    t = data["t"]

    # Determine which signal is the main one
    is_ant = "ant_c" in name
    is_z = name.startswith("z_v")
    is_sinus = "sinus" in name

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Friction Test: {name}", fontsize=13)

    # Top: main signal
    if is_ant or is_sinus:
        cmd = data["cmd_ant_l"]
        pres = data["pres_ant_l"]
        axes[0].set_ylabel("Antenna L (rad)")
        analysis = analyze_stalls(pres, "ant", 1e-4)
    else:
        cmd = data["cmd_z"]
        pres = data["pres_z"]
        axes[0].set_ylabel("Z position (m)")
        analysis = analyze_stalls(pres, "z", 1e-5)

    axes[0].plot(t, cmd, "b-", alpha=0.6, linewidth=1, label="commanded")
    axes[0].plot(t, pres, "r-", alpha=0.8, linewidth=1, label="present")

    # Highlight stall regions
    for start, length in analysis.stall_runs:
        if length >= 2:  # Only highlight runs >= 2 ticks
            end = min(start + length, len(t) - 1)
            axes[0].axvspan(t[start], t[end], alpha=0.2, color="red")

    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        f"Stalls: {analysis.stall_ticks}/{analysis.total_ticks} ticks "
        f"({analysis.stall_pct:.1f}%), max run: {analysis.max_stall_run} "
        f"({analysis.max_stall_ms:.0f}ms)"
    )

    # Bottom: velocity (numerical derivative of present)
    if len(pres) > 1:
        dt = np.diff(t)
        dt[dt == 0] = TICK_MS / 1000
        vel = np.diff(pres) / dt
        t_vel = t[:-1]
        axes[1].plot(t_vel, vel, "g-", alpha=0.7, linewidth=0.8)
        axes[1].axhline(y=0, color="k", alpha=0.3)
        axes[1].set_ylabel("Velocity")
        axes[1].set_xlabel("Time (s)")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title("Present velocity (numerical derivative)")

    plt.tight_layout()
    out_path = Path(out_dir) / f"{name}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze friction test data")
    parser.add_argument("files", nargs="*")
    parser.add_argument("--results-dir", default="friction_results")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.files:
        csv_files = [Path(f) for f in args.files]
    else:
        csv_files = sorted(Path(args.results_dir).glob("*.csv"))

    if not csv_files:
        print("No CSV files found.")
        sys.exit(1)

    results: list[tuple[str, StallAnalysis, StallAnalysis]] = []
    for f in csv_files:
        try:
            name, z_a, ant_a = analyze_file(str(f))
            results.append((name, z_a, ant_a))
            if args.plot:
                plot_test(str(f), args.results_dir)
        except Exception as e:
            print(f"  ERROR {f.name}: {e}")

    # Group by test type
    ant_tests = [(n, z, a) for n, z, a in results if "ant_c" in n]
    z_tests = [(n, z, a) for n, z, a in results if n.startswith("z_v")]
    sinus_tests = [(n, z, a) for n, z, a in results if "sinus" in n]

    def print_table(title, items, signal_idx):
        """signal_idx: 1=z_analysis, 2=ant_analysis"""
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")
        hdr = f"{'Test':<25} {'Stall%':>7} {'Stalls':>7} {'MaxRun':>7} {'MaxMs':>8} {'Total':>7}"
        print(hdr)
        print("-" * len(hdr))
        for name, z_a, ant_a in items:
            a = z_a if signal_idx == 1 else ant_a
            marker = " <<<" if a.max_stall_run > 3 else ""
            print(f"{name:<25} {a.stall_pct:>6.1f}% {a.stall_ticks:>7} "
                  f"{a.max_stall_run:>7} {a.max_stall_ms:>7.0f}ms {a.total_ticks:>7}{marker}")

    if sinus_tests:
        print_table("SINUSOIDAL BASELINE", sinus_tests, 1)
        print_table("SINUSOIDAL BASELINE (antenna)", sinus_tests, 2)

    if ant_tests:
        # Sort by center angle, then velocity
        ant_tests.sort(key=lambda x: x[0])
        print_table("ANTENNA TESTS (antenna signal)", ant_tests, 2)

    if z_tests:
        z_tests.sort(key=lambda x: x[0])
        print_table("HEAD Z TESTS (z signal)", z_tests, 1)

    # Summary: find threshold velocities
    print(f"\n{'='*80}")
    print("  STICK-SLIP THRESHOLD ANALYSIS")
    print(f"{'='*80}")
    print("(Looking for max stall run <= 3 ticks = no visible stick-slip)\n")

    for center in [0, 10, 90]:
        prefix = f"ant_c{center}_"
        group = [(n, z, a) for n, z, a in ant_tests if n.startswith(prefix)]
        if not group:
            continue
        print(f"  Antenna center={center}deg:")
        for name, _, ant_a in group:
            vel = name.split("_v")[1] if "_v" in name else "?"
            status = "OK" if ant_a.max_stall_run <= 3 else f"STICK-SLIP (max {ant_a.max_stall_run} ticks)"
            print(f"    v={vel:>5}deg/s: {status}")

    z_group = [(n, z, a) for n, z, a in z_tests]
    if z_group:
        print(f"\n  Head Z:")
        for name, z_a, _ in z_group:
            vel = name.split("_v")[1] if "_v" in name else "?"
            status = "OK" if z_a.max_stall_run <= 3 else f"STICK-SLIP (max {z_a.max_stall_run} ticks)"
            print(f"    v={vel:>5}mm/s: {status}")


if __name__ == "__main__":
    main()
