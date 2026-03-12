#!/usr/bin/env python3
"""Plot control loop frequency benchmark from CSV dumped by the conversation app.

Usage:
    # Fetches CSV from robot and plots
    python scripts/plot_bench.py

    # Plot a local CSV file
    python scripts/plot_bench.py local_file.csv
"""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fetch_csv(host: str = "reachy-mini.local") -> Path:
    """SCP the benchmark CSV from the robot to a local temp file."""
    tmp = Path(tempfile.mktemp(suffix=".csv"))
    subprocess.run(
        ["scp", f"pollen@{host}:/tmp/control_loop_bench.csv", str(tmp)],
        check=True,
    )
    return tmp


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (t, freq_hz) arrays from CSV."""
    ts, freqs = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts.append(float(row["t"]))
            freqs.append(float(row["freq_hz"]))
    return np.array(ts), np.array(freqs)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean with same-size output (padded with NaN at edges)."""
    kernel = np.ones(window) / window
    padded = np.convolve(values, kernel, mode="same")
    # Edge correction
    padded[:window // 2] = np.nan
    padded[-(window // 2):] = np.nan
    return padded


def main():
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        print("Fetching CSV from robot...")
        csv_path = fetch_csv()
        print(f"  -> {csv_path}")

    t, freq = load_csv(csv_path)
    print(f"Loaded {len(t)} samples, duration: {t[-1]:.1f}s")
    print(f"  avg: {freq.mean():.1f}Hz, std: {freq.std():.1f}, min: {freq.min():.1f}Hz, max: {freq.max():.1f}Hz")

    # Rolling average (1-second window based on expected sample count)
    samples_per_sec = int(np.median(np.diff(t)) ** -1) if len(t) > 1 else 100
    window = max(1, samples_per_sec)
    smooth = rolling_mean(freq, window)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(t, freq, s=1, alpha=0.3, color="steelblue", label="Per-tick frequency")
    ax.plot(t, smooth, color="red", linewidth=1.5, label=f"Rolling avg ({window}-sample window)")
    target_hz = 60
    ax.axhline(y=target_hz, color="green", linestyle="--", linewidth=1, alpha=0.7, label=f"Target ({target_hz}Hz)")
    ax.set_xlabel("Time since start (s)")
    ax.set_ylabel("Loop frequency (Hz)")
    ax.set_title("Control Loop Frequency Benchmark")
    ax.legend()
    ax.set_ylim(0, max(120, freq.max() + 10))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = csv_path.parent / f"control_loop_bench_{timestamp}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
