#!/usr/bin/env python3
"""Analyze trajectory test results: compare oscillation/shaking across shapes.

Metrics:
- Tracking error: RMS of (commanded - present) after removing DC offset
- Oscillation energy: high-frequency content in present position (highpass > 2Hz)
- Stall analysis: same as friction tests
- Jerk: derivative of acceleration in present position (smoothness measure)

Usage:
    python3 analyze_trajectories.py                    # all CSVs
    python3 analyze_trajectories.py --plot             # with plots
    python3 analyze_trajectories.py --results-dir DIR  # custom directory
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

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


def highpass(signal: np.ndarray, cutoff_hz: float, fs: float) -> np.ndarray:
    """Simple first-order highpass via FFT."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    fft = np.fft.rfft(signal)
    fft[freqs < cutoff_hz] = 0
    return np.fft.irfft(fft, n=n)


def analyze_one(path: str) -> dict:
    """Analyze one trajectory CSV."""
    data = load_csv(path)
    name = Path(path).stem.replace("traj_", "")

    t = data["t"]
    cmd = data["cmd_ant_l"]
    pres = data["pres_ant_l"]

    # Skip first 60 ticks (1s interpolation settling)
    skip = 60
    t = t[skip:]
    cmd = cmd[skip:]
    pres = pres[skip:]
    n = len(t)

    if n < 10:
        return {"name": name, "error": "too short"}

    # 1. Tracking error (RMS)
    error = cmd - pres
    rms_error = float(np.sqrt(np.mean(error**2)))

    # 2. High-frequency oscillation energy
    dt_mean = float(np.mean(np.diff(t)))
    fs = 1.0 / max(dt_mean, 0.001)
    hp = highpass(pres, 2.0, fs)
    osc_rms = float(np.sqrt(np.mean(hp**2)))

    # 3. Stall analysis
    diffs = np.abs(np.diff(pres))
    threshold = 1e-4  # 0.1 mrad
    is_stall = diffs < threshold
    stall_pct = 100.0 * float(is_stall.sum()) / max(1, len(is_stall))

    # Find max stall run
    max_run = 0
    cur = 0
    for s in is_stall:
        if s:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0

    # 4. Jerk (smoothness) — 3rd derivative of position
    if n > 3:
        dt = np.diff(t)
        dt[dt == 0] = TICK_MS / 1000
        vel = np.diff(pres) / dt
        if len(vel) > 1:
            dt2 = dt[:-1]
            acc = np.diff(vel) / dt2
            if len(acc) > 1:
                dt3 = dt2[:-1]
                jerk = np.diff(acc) / dt3
                rms_jerk = float(np.sqrt(np.mean(jerk**2)))
            else:
                rms_jerk = 0.0
        else:
            rms_jerk = 0.0
    else:
        rms_jerk = 0.0

    # 5. Velocity smoothness — std of velocity (lower = more constant speed)
    if n > 1:
        vel_all = np.diff(pres) / np.maximum(np.diff(t), 0.001)
        vel_std = float(np.std(vel_all))
    else:
        vel_std = 0.0

    # 6. Vibration index — acceleration sign changes per second
    #    Numerical derivative of velocity = acceleration.
    #    Count how often acceleration flips sign: high flip rate = shaky antenna.
    #    This captures the visible tip vibration that encoders can measure.
    if n > 2:
        dt = np.diff(t)
        dt[dt == 0] = TICK_MS / 1000
        vel = np.diff(pres) / dt
        if len(vel) > 1:
            acc = np.diff(vel) / dt[:-1]
            # Count sign changes in acceleration
            signs = np.sign(acc)
            # Remove zeros (ambiguous sign)
            nonzero = signs[signs != 0]
            if len(nonzero) > 1:
                sign_changes = int(np.sum(np.diff(nonzero) != 0))
                duration_s = float(t[-1] - t[0])
                sign_changes_per_s = sign_changes / max(duration_s, 0.01)
            else:
                sign_changes = 0
                sign_changes_per_s = 0.0
        else:
            sign_changes = 0
            sign_changes_per_s = 0.0
    else:
        sign_changes = 0
        sign_changes_per_s = 0.0

    return {
        "name": name,
        "rms_error": rms_error,
        "osc_rms": osc_rms,
        "stall_pct": stall_pct,
        "max_stall": max_run,
        "max_stall_ms": max_run * TICK_MS,
        "rms_jerk": rms_jerk,
        "vel_std": vel_std,
        "vib_flips": sign_changes,
        "vib_flips_per_s": sign_changes_per_s,
        "n_samples": n,
    }


def plot_test(path: str, out_dir: str) -> None:
    """Generate a comparison plot for one trajectory test."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    data = load_csv(path)
    name = Path(path).stem

    t = data["t"]
    cmd = data["cmd_ant_l"]
    pres = data["pres_ant_l"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    fig.suptitle(f"Trajectory: {name}", fontsize=13)

    # 1. Commanded vs present position
    axes[0].plot(t, cmd, "b-", alpha=0.6, linewidth=1, label="commanded")
    axes[0].plot(t, pres, "r-", alpha=0.8, linewidth=1, label="present")
    axes[0].set_ylabel("Antenna L (rad)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # 2. Tracking error
    error = cmd - pres
    axes[1].plot(t, error, "m-", alpha=0.7, linewidth=0.8)
    axes[1].axhline(y=0, color="k", alpha=0.3)
    axes[1].set_ylabel("Error (cmd - pres)")
    axes[1].grid(True, alpha=0.3)
    skip = min(50, len(error) - 1)
    rms = np.sqrt(np.mean(error[skip:]**2))
    axes[1].set_title(f"Tracking error (RMS after 1s: {rms:.4f} rad = {np.degrees(rms):.2f} deg)")

    if len(pres) > 2:
        dt = np.diff(t)
        dt[dt == 0] = TICK_MS / 1000
        vel = np.diff(pres) / dt

        # 3. Velocity
        axes[2].plot(t[:-1], vel, "g-", alpha=0.7, linewidth=0.8)
        axes[2].axhline(y=0, color="k", alpha=0.3)
        axes[2].set_ylabel("Velocity (rad/s)")
        axes[2].grid(True, alpha=0.3)

        # 4. Acceleration + sign changes (vibration index)
        if len(vel) > 1:
            acc = np.diff(vel) / dt[:-1]
            t_acc = t[:-2]
            axes[3].plot(t_acc, acc, "orange", alpha=0.6, linewidth=0.8, label="acceleration")
            axes[3].axhline(y=0, color="k", alpha=0.3)

            # Mark sign changes
            signs = np.sign(acc)
            changes = np.where(np.diff(signs[signs != 0]) != 0)[0]
            # Map back to original indices (skip zeros)
            nonzero_idx = np.where(signs != 0)[0]
            if len(changes) > 0 and len(nonzero_idx) > 0:
                change_idx = nonzero_idx[np.minimum(changes, len(nonzero_idx) - 1)]
                axes[3].plot(t_acc[change_idx], acc[change_idx], "r.", markersize=1.5, alpha=0.5)

            duration_s = float(t[-1] - t[0])
            n_flips = int(np.sum(np.diff(signs[signs != 0]) != 0)) if len(signs[signs != 0]) > 1 else 0
            flips_per_s = n_flips / max(duration_s, 0.01)
            axes[3].set_title(f"Acceleration sign changes: {n_flips} total, {flips_per_s:.1f}/s (higher = shakier)")
            axes[3].set_ylabel("Accel (rad/s²)")
            axes[3].set_xlabel("Time (s)")
            axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_dir) / f"{name}.png"
    plt.savefig(out_path, dpi=120)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze trajectory test data")
    parser.add_argument("files", nargs="*")
    parser.add_argument("--results-dir", default="/tmp/trajectory_test")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.files:
        csv_files = [Path(f) for f in args.files]
    else:
        csv_files = sorted(Path(args.results_dir).glob("traj_*.csv"))

    if not csv_files:
        print("No trajectory CSV files found.")
        sys.exit(1)

    results = []
    for f in csv_files:
        try:
            r = analyze_one(str(f))
            results.append(r)
            if args.plot:
                plot_test(str(f), args.results_dir)
        except Exception as e:
            print(f"  ERROR {f.name}: {e}")

    # Sort by oscillation energy (lower = better)
    results.sort(key=lambda r: r.get("osc_rms", 999))

    # Sort by vibration flips/s (primary metric), fall back to osc_rms
    results.sort(key=lambda r: r.get("vib_flips_per_s", 999))

    print(f"\n{'='*105}")
    print("  TRAJECTORY COMPARISON (sorted by vibration index — lower = smoother)")
    print(f"{'='*105}")
    hdr = (f"{'Rank':<5} {'Trajectory':<18} {'Vib/s':>7} {'VibTot':>7} {'OscRMS':>8} {'RMSErr':>8} "
           f"{'Stall%':>7} {'MaxMs':>7} {'Jerk':>10} {'VelStd':>8}")
    print(hdr)
    print("-" * len(hdr))

    for i, r in enumerate(results, 1):
        if "error" in r:
            print(f"  {r['name']}: {r['error']}")
            continue
        best = " *" if i == 1 else ""
        print(f"{i:<5} {r['name']:<18} {r['vib_flips_per_s']:>7.1f} {r['vib_flips']:>7} "
              f"{r['osc_rms']:>8.5f} {r['rms_error']:>8.5f} "
              f"{r['stall_pct']:>6.1f}% {r['max_stall_ms']:>6.0f}ms "
              f"{r['rms_jerk']:>10.1f} {r['vel_std']:>8.3f}{best}")

    print(f"\nVib/s = acceleration sign changes per second (lower = less shaky)")
    print(f"Lower OscRMS = less high-frequency encoder oscillation")
    print(f"Lower RMSErr = better tracking")
    print(f"Lower Stall% = less time stuck")


if __name__ == "__main__":
    main()
