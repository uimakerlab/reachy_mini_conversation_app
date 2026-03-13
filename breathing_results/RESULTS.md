# Breathing Vibration Benchmark Results

## Key Findings

1. **Antenna sway is the dominant vibration source.** 15-degree amplitude at
   0.5 Hz causes ~300x more vibration than holding still with the offset.

2. **The 10-degree antenna offset reduces baseline noise.** Even with no motion,
   the offset position (0.022) is quieter than vertical (0.031).

3. **Antenna amplitude scales roughly linearly with vibration** — regardless
   of whether centered at 0 or 10 degrees.

4. **Head rotations (roll/yaw/pitch) are nearly free** vibration-wise when
   antenna amplitude is kept small (< 5 degrees).

## Round 3 Results (10-degree antenna offset)

| Rank | Config | Score | Description |
|------|--------|-------|-------------|
| 1 | still_offset | 0.02 | CONTROL: no motion, antennas at 10deg |
| 2 | still_vertical | 0.03 | CONTROL: no motion, antennas vertical |
| 3 | **zen** | **0.25** | z 2mm@0.06Hz + roll 0.5deg + ant 2deg |
| 4 | **sleepy** | **0.49** | z 5mm@0.07Hz + pitch 2deg + ant 3deg |
| 5 | **dreamy_v2** | **0.76** | z 4mm + yaw 3deg + roll 1deg + ant 4deg |
| 6 | **ocean** | **0.79** | z 5mm + roll 3deg + ant 4deg |
| 7 | **alive_v2** | **1.09** | z 3mm + roll 1.5deg + yaw 2deg + ant 4deg |
| 8 | figure8_big | 1.35 | yaw 3deg + pitch 2deg + ant 5deg |
| 9 | yaw_big | 1.36 | yaw 5deg + z 3mm + ant 5deg |
| 10 | nod_big | 1.37 | pitch 3deg + z 3mm + ant 5deg |
| 11 | curious | 1.37 | z 4mm + yaw 4deg + roll 2deg + ant 5deg |
| 12 | roll_big | 1.77 | roll 4deg + ant 5deg |
| 13 | big_z | 1.80 | z 8mm + ant 5deg |
| 14 | offset_ant5 | 2.18 | ant 5deg (sweep point) |
| 15 | playful | 2.60 | z 3mm + roll 2deg + yaw 2deg + ant 6deg |
| 16 | offset_ant8 | 3.46 | ant 8deg (sweep point) |
| 17 | offset_ant10 | 4.31 | ant 10deg (sweep point) |
| 18 | offset_ant15 | 6.49 | ant 15deg (old amplitude, with offset) |
| 19 | baseline_old | 6.49 | OLD: ant 15deg around vertical |

## Recommendations for Review

The review script cycles through 16 curated configs (10s each):

```bash
./scripts/run_review.sh
```

Top picks to watch for:
- **zen** — barely visible motion, very peaceful, almost zero vibration
- **sleepy** — gentle nod, good for idle/waiting
- **dreamy_v2** — slow drift with character, looks contemplative
- **ocean** — rolling wave-like motion
- **alive_v2** — most organic (4 axes at different frequencies)
- **curious** — most animated while still low-vibration

## How to Reproduce

```bash
# Review configs visually on the robot (10s each)
./scripts/run_review.sh

# Run specific ones for longer
./scripts/run_review.sh --config dreamy_v2 alive_v2 --duration 20

# Run all benchmarks (20s each, ~8 minutes total)
./scripts/run_breathing_bench.sh --config all --duration 20

# Analyze + generate plots
python3 scripts/analyze_breathing.py --plot

# List all configs
./scripts/run_breathing_bench.sh --list
```

## Files

- `scripts/breathing_bench.py` - Benchmark script (runs on robot)
- `scripts/review_breathing.py` - Visual review (runs on robot, interactive)
- `scripts/run_breathing_bench.sh` - Deploy + benchmark + download results
- `scripts/run_review.sh` - Deploy + interactive review
- `scripts/analyze_breathing.py` - Vibration analysis + plots
- `breathing_results/*.csv` - Raw data
- `breathing_results/*.png` - Time-series plots
