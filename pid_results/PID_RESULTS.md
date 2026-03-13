# Antenna PID Tuning Results

## Test Setup
- **Trajectory**: s_curve (cosine acceleration profile — best from Phase 2a)
- **Parameters**: Center=10°, Amplitude=±10°, Speed=~30°/s, Duration=15s
- **Motor**: Dynamixel XL330-M288 (antennas only, IDs 17 & 18)
- **PID range**: 0–16,383 per register. Conversion: KPP=P/128, KPD=D/16, KPI=I/65536
- **Robot default**: P=200 I=0 D=0 (factory default for XL330 is P=400 I=0 D=0)
- **Date**: 2026-03-13

## Results (sorted by OscRMS — lower = less vibration)

| Rank | P | I | D | OscRMS | RMSErr | Stall% | MaxStall | Jerk | VelStd | Notes |
|------|------|---|------|---------|---------|--------|----------|--------|--------|-------|
| **1** | **1000** | 0 | **1000** | **0.00736** | 0.04141 | 20.7% | 67ms | 3819 | 0.696 | **BEST — 10% less vibration than default** |
| 2 | 1000 | 0 | 500 | 0.00761 | 0.03712 | 20.1% | 50ms | 4316 | 0.734 | Forum-recommended for XL330 |
| 3 | 400 | 0 | 200 | 0.00766 | 0.04078 | 20.9% | 50ms | 3754 | 0.693 | |
| 4 | 600 | 0 | 300 | 0.00776 | 0.03906 | 20.5% | 33ms | 3610 | 0.704 | |
| 5 | 200 | 0 | 0 | 0.00793 | 0.04151 | 21.9% | 50ms | 3793 | 0.701 | Default (restore run) |
| 6 | 300 | 0 | 200 | 0.00797 | 0.04423 | 21.3% | 67ms | 3601 | 0.680 | |
| 7 | 200 | 0 | 0 | 0.00816 | 0.04055 | 20.4% | 50ms | 3813 | 0.698 | Default (initial baseline) |
| 8 | 400 | 0 | 500 | 0.00829 | 0.04782 | 19.6% | 50ms | 3162 | 0.664 | |
| 9 | 800 | 0 | 800 | 0.00834 | 0.04251 | 19.8% | 50ms | 3619 | 0.694 | |
| 10 | 1500 | 0 | 1000 | 0.00863 | 0.03689 | 21.1% | 50ms | 4701 | 0.755 | Best tracking but more vibration |
| 11 | 1000 | 0 | 2000 | 0.00883 | 0.04899 | 20.5% | 50ms | 4372 | 0.729 | Over-damped |
| 12 | 400 | 0 | 0 | 0.00961 | 0.03722 | 22.0% | 50ms | 5535 | 0.835 | Factory default P, no D |
| 13 | 200 | 0 | 500 | 0.00966 | 0.06726 | 19.9% | 67ms | 2901 | 0.636 | Too much D for low P — sluggish |
| 14 | 2000 | 0 | 2000 | 0.00986 | 0.03906 | 19.8% | 33ms | 6937 | 0.879 | High gains — shaky |
| 15 | 800 | 0 | 0 | 0.01168 | 0.03389 | 20.3% | 33ms | 6200 | 0.981 | Worst vibration — high P, no damping |

## Key Findings (Encoder-Based — SUPERSEDED)

The results above used OscRMS (high-frequency energy in encoder position) as the primary
metric. This measures **motor shaft** oscillation but **misses antenna tip vibration**.

The antennas are flexible spring-like rods. The real vibration happens at the tip, excited
by aggressive motor corrections. Higher PID gains make the motor smoother (lower OscRMS)
but the antenna tip shakes MORE because each correction impulse rings the spring.

## Corrected Analysis: Vibration Index (50 Hz, 2026-03-13)

New metric: **acceleration sign changes per second (Vib/s)**. Numerical double derivative
of position → count how often acceleration flips sign. Higher = shakier.

Retested at 50 Hz (matching daemon) with s_curve and triangle trajectories:

| PID | s_curve Vib/s | triangle Vib/s | s_curve OscRMS | RMSErr |
|---|---|---|---|---|
| **P=200 D=0 (default)** | **26.8** | **25.4** | 0.00597 | 0.04187 |
| P=1000 D=1000 | 32.4 | 35.4 | 0.00531 | 0.04137 |

**P=1000 D=1000 is 21-40% shakier** despite having better encoder metrics.

## Recommendation (REVISED)

**Keep the default P=200 I=0 D=0 for antennas.** Higher gains make the motor shaft
smoother but excite more tip vibration in the flexible antennas. The tracking difference
is negligible (0.04187 vs 0.04137 rad ≈ 0.03° difference).

The vibration problem is primarily mechanical (antenna flexibility), not a PID tuning
issue. Solutions should focus on trajectory shaping rather than motor control.

## Expressive Movement Presets (from user observations)

*(To be filled in — user noted some shaky configs could be used as expressive animations: "scared robot" and "fly impersonation")*

## Known Bug: 60 Hz Sampling vs 50 Hz Daemon (FIXED 2026-03-13)

**All results above were collected with a 60 Hz control loop, but the daemon's internal
loop runs at 50 Hz.** This means ~1 in 6 position reads returned a stale value from the
previous tick, producing false zero-velocity samples in the data.

**Impact on metrics:**
- **Stall%** is inflated — many "stalls" are actually stale reads, not real motor stalls
- **Velocity plots** show spurious exact-zero spikes that are artifacts, not real stops
- **Jerk** is inflated — the zero-velocity artifacts create artificial acceleration spikes
- **OscRMS** may be slightly affected but is computed from position (not velocity), so the
  impact is smaller — the relative ranking of PID configs is likely still valid
- **RMSErr** is unaffected (compares commanded vs present position, both sampled at same time)

**Fix:** `LOOP_HZ` changed from 60 to 50 in `trajectory_test.py` and `TICK_MS` updated
in `analyze_trajectories.py`. All future tests will sample at 50 Hz to match the daemon.

**Action needed:** The PID ranking above should be re-validated with 50 Hz data. The
relative ordering is probably correct but the absolute Stall%, Jerk, and VelStd numbers
will change. The earlier friction test results (`friction_results/`) were also collected
at 60 Hz and have the same stale-read artifact.

## Procedure

Each test:
1. Modify `/venvs/mini_daemon/.../hardware_config.yaml` (antenna PID only)
2. Restart daemon (`systemctl restart reachy-mini-daemon`)
3. Enable motors (`POST /api/motors/set_mode/enabled`)
4. Wake up (`POST /api/move/play/wake_up`), wait 5s
5. Run s_curve trajectory test (15s, 50Hz sampling)
6. Download and analyze results

Script: `scripts/pid_test.sh P I D`
Analysis: `scripts/analyze_trajectories.py --results-dir DIR --plot`
