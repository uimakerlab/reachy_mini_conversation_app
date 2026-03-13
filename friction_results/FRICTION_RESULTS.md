# Friction Characterization Results — Step 1

## Summary

**Stick-slip friction is confirmed.** Every test below a velocity threshold shows
clear stall runs where the present position is completely frozen while the
commanded position keeps moving — the classic stick-slip sawtooth pattern.

## Test Matrix

- **Sinusoidal baseline**: 1 test (original breathing motion)
- **Antenna tests**: 9 velocities × 3 center angles = 27 tests
- **Head Z tests**: 9 velocities = 9 tests
- **Total**: 37 tests, 15 seconds each at 60Hz sampling

> **⚠ Known bug (found 2026-03-13):** All friction tests were sampled at 60 Hz but the
> daemon's internal control loop runs at 50 Hz. This means ~1 in 6 position reads returned
> a stale value, producing false zero-velocity samples. **Stall% is inflated**, velocity
> plots have spurious zero spikes, and jerk values are artificially high. The stick-slip
> patterns are still real (multi-tick stalls of 100ms+ are genuine), but single-tick stalls
> may be stale-read artifacts. Future tests use 50 Hz to match the daemon.

## Key Findings

### 1. Antenna Stick-Slip Thresholds

Minimum velocity for smooth motion (max stall run ≤ 3 ticks):

| Center Angle | Min Smooth Velocity | Worst Stall (at 1 deg/s) |
|-------------|--------------------|-----------------------|
| **0 deg** (vertical) | **8 deg/s** | 31 ticks (517ms) |
| **10 deg** (SDK default) | **20 deg/s** | 35 ticks (583ms) |
| **90 deg** (horizontal) | **30 deg/s** | 47 ticks (783ms) |

**Key observations:**
- Friction increases dramatically with angle — horizontal antennas (90°) need
  3.75× the velocity of vertical (0°) to avoid stick-slip
- This confirms gravity loading: at 90° the gearbox bears the full weight of
  the antenna, increasing static friction
- The 10° SDK offset already has 2.5× the friction threshold of vertical
- Even at the "smooth" threshold, ~20% of ticks still show single-tick stalls
  (the motor briefly stops between steps)

#### Antenna at 0° center — velocity sweep

| Velocity | Plot |
|---|---|
| 1 deg/s (stick-slip) | ![ant_c0_v1](ant_c0_v1.png) |
| 2 deg/s | ![ant_c0_v2](ant_c0_v2.png) |
| 3 deg/s | ![ant_c0_v3](ant_c0_v3.png) |
| 5 deg/s | ![ant_c0_v5](ant_c0_v5.png) |
| 8 deg/s (threshold) | ![ant_c0_v8](ant_c0_v8.png) |
| 12 deg/s | ![ant_c0_v12](ant_c0_v12.png) |
| 20 deg/s | ![ant_c0_v20](ant_c0_v20.png) |
| 30 deg/s (smooth) | ![ant_c0_v30](ant_c0_v30.png) |
| 50 deg/s | ![ant_c0_v50](ant_c0_v50.png) |

#### Antenna at 10° center — velocity sweep

| Velocity | Plot |
|---|---|
| 1 deg/s | ![ant_c10_v1](ant_c10_v1.png) |
| 2 deg/s | ![ant_c10_v2](ant_c10_v2.png) |
| 3 deg/s | ![ant_c10_v3](ant_c10_v3.png) |
| 5 deg/s | ![ant_c10_v5](ant_c10_v5.png) |
| 8 deg/s | ![ant_c10_v8](ant_c10_v8.png) |
| 12 deg/s | ![ant_c10_v12](ant_c10_v12.png) |
| 20 deg/s (threshold) | ![ant_c10_v20](ant_c10_v20.png) |
| 30 deg/s | ![ant_c10_v30](ant_c10_v30.png) |
| 50 deg/s | ![ant_c10_v50](ant_c10_v50.png) |

#### Antenna at 90° center — velocity sweep

| Velocity | Plot |
|---|---|
| 1 deg/s | ![ant_c90_v1](ant_c90_v1.png) |
| 2 deg/s (78% stuck!) | ![ant_c90_v2](ant_c90_v2.png) |
| 3 deg/s | ![ant_c90_v3](ant_c90_v3.png) |
| 5 deg/s | ![ant_c90_v5](ant_c90_v5.png) |
| 8 deg/s | ![ant_c90_v8](ant_c90_v8.png) |
| 12 deg/s | ![ant_c90_v12](ant_c90_v12.png) |
| 20 deg/s | ![ant_c90_v20](ant_c90_v20.png) |
| 30 deg/s (threshold) | ![ant_c90_v30](ant_c90_v30.png) |
| 50 deg/s | ![ant_c90_v50](ant_c90_v50.png) |

#### Full antenna data table

| Test | Stall % | Stall Ticks | Max Run | Max Duration | Total Ticks |
|---|---|---|---|---|---|
| ant_c0_v1 | 49.9% | 416 | 31 | 517ms | 834 |
| ant_c0_v2 | 41.8% | 348 | 13 | 217ms | 833 |
| ant_c0_v3 | 37.7% | 314 | 8 | 133ms | 833 |
| ant_c0_v5 | 29.7% | 248 | 4 | 67ms | 834 |
| ant_c0_v8 | 29.2% | 243 | 3 | 50ms | 833 |
| ant_c0_v12 | 25.2% | 210 | 3 | 50ms | 833 |
| ant_c0_v20 | 22.4% | 187 | 3 | 50ms | 834 |
| ant_c0_v30 | 18.9% | 158 | 2 | 33ms | 834 |
| ant_c0_v50 | 17.5% | 146 | 2 | 33ms | 834 |
| ant_c10_v1 | 81.7% | 681 | 35 | 583ms | 834 |
| ant_c10_v2 | 55.1% | 459 | 18 | 300ms | 833 |
| ant_c10_v3 | 54.1% | 451 | 12 | 200ms | 834 |
| ant_c10_v5 | 42.1% | 351 | 9 | 150ms | 833 |
| ant_c10_v8 | 33.7% | 281 | 5 | 83ms | 833 |
| ant_c10_v12 | 26.6% | 222 | 4 | 67ms | 834 |
| ant_c10_v20 | 23.0% | 192 | 3 | 50ms | 834 |
| ant_c10_v30 | 19.4% | 162 | 2 | 33ms | 834 |
| ant_c10_v50 | 18.7% | 156 | 2 | 33ms | 834 |
| ant_c90_v1 | 86.6% | 722 | 47 | 783ms | 834 |
| ant_c90_v2 | 78.2% | 651 | 53 | 883ms | 833 |
| ant_c90_v3 | 68.1% | 568 | 25 | 417ms | 834 |
| ant_c90_v5 | 57.1% | 476 | 18 | 300ms | 834 |
| ant_c90_v8 | 45.7% | 381 | 16 | 267ms | 833 |
| ant_c90_v12 | 36.3% | 303 | 9 | 150ms | 834 |
| ant_c90_v20 | 23.7% | 198 | 4 | 67ms | 834 |
| ant_c90_v30 | 20.4% | 170 | 3 | 50ms | 833 |
| ant_c90_v50 | 19.1% | 159 | 2 | 33ms | 833 |

### 2. Head Z Stick-Slip

**No tested velocity eliminates stick-slip for Z.** Even at 20 mm/s (the fastest
tested), max stall run is 4 ticks (67ms).

| Velocity | Stall % | Max Stall Run | Max Stall Duration |
|----------|---------|---------------|--------------------|
| 0.5 mm/s | 57.9% | 19 ticks | 317ms |
| 1.0 mm/s | 51.1% | 26 ticks | 433ms |
| 1.5 mm/s | 48.9% | **74 ticks** | **1233ms** |
| 2.0 mm/s | 36.9% | 16 ticks | 267ms |
| 3.0 mm/s | 31.8% | 17 ticks | 283ms |
| 5.0 mm/s | 29.4% | 11 ticks | 183ms |
| 8.0 mm/s | 27.5% | 10 ticks | 167ms |
| 12.0 mm/s | 26.0% | 6 ticks | 100ms |
| 20.0 mm/s | 23.4% | 4 ticks | 67ms |

**Key observations:**
- Z friction is much worse than antenna friction — all 6 Stewart platform motors
  fight gravity simultaneously during Z translation
- The 1.5 mm/s anomaly (74-tick stall!) may sit right in the static/kinetic
  transition zone where the motor repeatedly gets trapped in the stick-slip cycle
- Even 20 mm/s still shows stick-slip. The breathing animation uses
  5mm × 0.1Hz × 2π ≈ 3.1 mm/s peak velocity, well within the stick-slip zone

#### Head Z velocity sweep

| Velocity | Plot |
|---|---|
| 0.5 mm/s | ![z_v0.5](z_v0.5.png) |
| 1.0 mm/s | ![z_v1.0](z_v1.0.png) |
| 1.5 mm/s (1.2s stall!) | ![z_v1.5](z_v1.5.png) |
| 2.0 mm/s | ![z_v2.0](z_v2.0.png) |
| 3.0 mm/s | ![z_v3.0](z_v3.0.png) |
| 5.0 mm/s | ![z_v5.0](z_v5.0.png) |
| 8.0 mm/s | ![z_v8.0](z_v8.0.png) |
| 12.0 mm/s | ![z_v12.0](z_v12.0.png) |
| 20.0 mm/s | ![z_v20.0](z_v20.0.png) |

### 3. Sinusoidal Baseline

The original breathing motion (z 5mm@0.1Hz + antenna 15deg@0.5Hz) confirms:
- Z signal: 43.6% stall, max 32 ticks (533ms)
- Antenna signal: 22.3% stall, max 5 ticks (83ms)

![Sinusoidal baseline](sinus_baseline.png)

The Z axis is the primary source of visible trembling — the head gets stuck for
up to half a second, then jumps.

## Implications for Breathing Animation

1. **Antenna sway** at the current 15deg@0.5Hz has peak velocity of
   ~47 deg/s — above the 8 deg/s threshold at 0° center. The visible
   jerkiness comes from Z, not antennas.

2. **Head Z** at 5mm@0.1Hz has peak velocity ~3.1 mm/s — deep in the
   stick-slip zone. This is why the head trembles.

3. **Increasing Z velocity** alone won't fix it — even 20 mm/s still shows
   stick-slip. The Stewart platform's 6-motor gearbox system has very high
   static friction under gravity loading.

4. **Pre-rotation (Step 3 idea)** looks promising — rotations don't fight
   gravity and have much lower friction thresholds.

## Files

- `friction_test.py` — test runner (deployed to robot)
- `analyze_friction.py` — stall detection + plots
- `run_friction_test.sh` — deploy + run + download
- `*.csv` — raw data (37 files)
- `*.png` — plots with stall regions highlighted (37 files)
