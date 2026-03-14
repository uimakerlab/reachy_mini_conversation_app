# Why Does My Robot's Head Tremble? A Deep Dive Into Stick-Slip Friction

## The Symptom

Reachy Mini has a breathing animation — a subtle idle motion that makes the robot
feel alive when it's not talking or doing anything else. The head gently bobs up
and down (5mm at 0.1Hz) while the antennas sway (15 degrees at 0.5Hz).

The trajectory we command is perfectly smooth. A clean sinusoid. But what comes
out on the physical robot is jerky, trembling, almost shivering. It looks like
the robot is anxious, not calm.

Why?

## The Obvious (Wrong) Hypothesis: Control Frequency

The first instinct is to blame the control loop. We send position commands at
100Hz. Maybe the motors are stepping between discrete positions? Maybe 100Hz
isn't fast enough for smooth motion?

This is a reasonable first guess. At 100Hz, each tick is 10ms. If the motor moves
in discrete steps every 10ms, you might expect a 100Hz vibration. But here's the
thing — we measured the actual motor positions at 50Hz and found something
completely different.

The motors aren't stepping too fast. **They're not moving at all.**

## What We Actually Measured

We recorded the commanded position (what we tell the motors to do) and the
present position (what the motors actually report) at 50Hz — matching the
daemon's internal control loop. Then we counted **stall events** — any time the
present position stays unchanged for two or more consecutive ticks, meaning the
motor has stopped and static friction has engaged.

If the problem were control frequency, we'd expect brief, regular pauses. Instead,
we found stall runs of **30, 50, even 66 consecutive ticks**. That's the motor
being completely stuck for over a second while the commanded position keeps
moving away from it.

Here's what that looks like. This is the antenna at 1 deg/s constant velocity,
centered at 0 degrees. Blue is commanded, red is present. Red shaded regions are
stalls:

![Antenna at 1 deg/s — classic stick-slip sawtooth](../friction_results_50hz/ant_c0_v1.png)

The present position (red) is a staircase. The motor gets stuck, the commanded
position pulls ahead, and then suddenly the motor breaks free and jumps to catch
up. Then it gets stuck again. This is textbook **stick-slip friction**.

And here's what smooth motion looks like, same antenna but at 30 deg/s:

![Antenna at 30 deg/s — smooth tracking](../friction_results_50hz/ant_c0_v30.png)

Night and day. At higher velocity, the motor tracks the commanded position
smoothly with only brief pauses at direction reversals.

## The Physics: Stick-Slip Friction

Every motor in the robot drives through a gearbox. Gearboxes have friction. But
friction isn't a single number — it's a function of velocity, and it has a nasty
discontinuity at zero:

<!-- TODO: Replace with real friction model diagrams showing:
     1. Coulomb model (simplest: constant friction, sign change at v=0)
     2. Coulomb + viscous (adds linear velocity term)
     3. Stribeck model (static friction peak at v=0, drops to kinetic, then viscous rise)
     4. Load-dependent model (friction threshold scales with torque load)
-->

The key feature is the **discontinuity at zero velocity**. Static friction
(also called stiction) is the maximum friction force — the peak is right at
zero velocity. Once the motor starts moving, friction *drops* to the lower
kinetic (dynamic) friction level. At higher velocities, viscous friction adds a
component proportional to speed, but this is smooth and predictable.

The critical asymmetry is between static and kinetic: starting to move is harder
than continuing to move. This creates a vicious cycle:

1. **STICK**: The commanded position moves slowly. The PID controller computes
   position error → converts to voltage → motor produces torque. But when the
   commanded velocity is low, the position error stays small, the torque stays
   small, and the torque is *below the static friction threshold*. The motor
   stays stuck. Error accumulates.

2. **SLIP**: The commanded position keeps moving. Error grows. Eventually the PID
   output exceeds the static friction threshold. The motor breaks free. But now
   kinetic friction is *lower* than static friction, so the motor accelerates
   faster than expected. It overshoots slightly, the error reverses, and the
   motor stops again.

3. **Repeat**: The position trace becomes a sawtooth — flat segments (stuck)
   interrupted by sudden jumps (breaking free).

### Why Slow Movements Are Worst

For fast movements, the motor never stops. The PID error is always large enough
to exceed static friction. The motor stays in the kinetic regime where friction
is smooth and proportional to velocity.

For slow movements — like a gentle breathing animation — the commanded position
moves so slowly that the PID controller can "catch up." The error drops below the
static friction threshold. The motor stops. Then it needs to overcome static
friction *again* to resume moving.

**The critical insight**: increasing the control frequency does NOT fix this. If
you go from 100Hz to 1000Hz, the motor reaches its target *faster* and gets
stuck *sooner*. The problem is velocity-dependent, not frequency-dependent.

### Gravity Makes It Worse (Load-Dependent Friction)

Reachy Mini's head sits on a Stewart platform — 6 linear actuators arranged in
parallel that can position the head in 6 degrees of freedom (x, y, z, roll,
pitch, yaw). When the head moves up (Z+), *all 6 motors fight gravity*. The
gearboxes are under torque load even at rest, which pushes the gear teeth harder
against each other, increasing friction.

This is known as **load-dependent friction** — the static friction threshold
increases with the torque load on the gearbox. Under gravity, the gear teeth are
pressed harder against each other, and the force needed to start them moving
grows accordingly. Duclusaud, Passault et al. recently published an excellent
treatment of extended friction models for servo actuators, including
load-dependent and Stribeck effects, along with an open-source identification
tool called [BAM (Better Actuator Models)](https://github.com/Rhoban/bam)
[1].

This means the static friction threshold for Z translation is much higher than
for rotations (where roughly half the motors push up and half push down,
partially canceling the gravity load).

### The Stribeck Effect

In the Stribeck friction model, the friction force is highest at zero velocity
(static friction) and then drops as velocity increases, eventually reaching the
lower kinetic friction level. There is no velocity that produces *more* friction
than being at rest — but the transition from static to kinetic creates a narrow
velocity band where the motor is most likely to get trapped in the stick-slip
cycle. Once the motor is moving fast enough to stay firmly in the kinetic regime,
friction becomes smooth and predictable.

## The Experiments

### Phase 1: Friction Characterization (constant-velocity tests)

To verify all of this, we designed a systematic characterization. Instead of
sinusoidal breathing motions (where velocity constantly changes), we used
**constant-velocity linear interpolation** — triangle waves that move back and
forth at a fixed speed. This lets us cleanly measure the friction threshold as a
function of velocity.

#### Test Matrix

- **Antenna tests**: 9 velocities (1, 2, 3, 5, 8, 12, 20, 30, 50 deg/s) × 3
  center angles (0°, 10°, 90°) = 27 tests
- **Head Z tests**: 9 velocities (0.5, 1, 1.5, 2, 3, 5, 8, 12, 20 mm/s) = 9
  tests
- **Sinusoidal baseline**: 1 test (original breathing motion for reference)
- 15 seconds each, 50Hz sampling (matching daemon control loop)

#### Antenna Results: Stall Events vs Speed

We counted stall events — each time the motor stops (two consecutive identical
position readings), that's one event where static friction engaged and will
cause a jerk when it breaks free.

![Antenna stall events vs speed](../friction_results_50hz/stall_events_antenna.png)

Key observations:
- **Stalls never reach zero.** Even at 50 deg/s, there are still 16-26 stall
  events per 15 seconds. The gearbox friction causes micro-stalls at every speed.
- **Minimum stalls around 20-30 deg/s** for all angles. The sweet spot for the
  default 10° offset is 30 deg/s (21 stall events).
- **Gravity loading is dramatic.** At 3 deg/s: 111 events at 0° vs 146 at 10°
  vs 116 at 90°. The load-dependent friction effect is clearly visible.
- **Slow speeds are worst.** The 2-5 deg/s range has the most stall events
  (120-146 per 15s), because the motor repeatedly enters and exits the
  stick-slip zone.

Some representative raw traces:

![Antenna at 10° center, 3 deg/s — 146 stall events](../friction_results_50hz/ant_c10_v3.png)
![Antenna at 10° center, 12 deg/s — 87 stall events](../friction_results_50hz/ant_c10_v12.png)
![Antenna at 10° center, 30 deg/s — 21 stall events](../friction_results_50hz/ant_c10_v30.png)

#### Head Z Results: Friction At All Speeds

![Head Z stall events vs speed](../friction_results_50hz/stall_events_head_z.png)

The head Z shows a minimum around 5 mm/s (25 events) but stalls increase again
at higher speeds. The shape of this curve is not fully explained yet — we would
expect a monotonic decrease with speed in a simple friction model. The increase
at high speeds may be related to direction reversals becoming more frequent and
violent at higher velocities, or to the complex multi-actuator dynamics of the
Stewart platform.

| Velocity | Stall Events | Stall % | Max Stall Run |
|---|---|---|---|
| 0.5 mm/s | 72 | 38.6% | 66 ticks (1320ms) |
| 1.0 mm/s | 58 | 23.5% | 30 ticks (600ms) |
| 1.5 mm/s | 59 | 17.1% | 12 ticks (240ms) |
| 2.0 mm/s | 50 | 12.9% | 13 ticks (260ms) |
| 3.0 mm/s | 33 | 6.5% | 8 ticks (160ms) |
| **5.0 mm/s** | **25** | **9.6%** | **15 ticks (300ms)** |
| 8.0 mm/s | 35 | 7.8% | 5 ticks (100ms) |
| 12.0 mm/s | 36 | 6.5% | 6 ticks (120ms) |
| 20.0 mm/s | 50 | 7.9% | 5 ticks (100ms) |

Some representative raw traces:

![Head Z at 1.0 mm/s — 58 stall events](../friction_results_50hz/z_v1.0.png)
![Head Z at 5.0 mm/s — 25 stall events](../friction_results_50hz/z_v5.0.png)
![Head Z at 20.0 mm/s — 50 stall events](../friction_results_50hz/z_v20.0.png)

### Phase 2a: Trajectory Shape Optimization

With the friction characterization done, we tried different trajectory shapes to
see if we could reduce stalls by avoiding zero-velocity moments. We tested 8
shapes, all at 30 deg/s antenna speed:

| Rank | Trajectory | Vib/s | OscRMS | Stall% | Description |
|---|---|---|---|---|---|
| 1 | overshoot | 24.7 | 0.01698 | 9.6% | 10% overshoot then settle |
| 2 | smoothed_tri | 25.2 | 0.01527 | 7.0% | Triangle with cosine-blended corners |
| 3 | s_curve | 25.2 | 0.00560 | 6.4% | Cosine acceleration profile |
| 4 | min_jerk | 26.2 | 0.00990 | 9.9% | Minimum-jerk segments |
| 5 | sinusoidal | 26.2 | 0.01509 | 6.7% | Pure sinusoid |
| 6 | pause_peaks | 26.7 | 0.01244 | 6.0% | Fast move + pause at peaks |
| 7 | trapezoidal | 26.9 | 0.00959 | 8.2% | Trapezoidal velocity |
| 8 | triangle | 27.2 | 0.01493 | 3.4% | Constant velocity baseline |

The differences are modest — trajectory shape alone doesn't solve the friction
problem. The stall events are dominated by direction reversals where velocity
passes through zero, regardless of the shape.

### Phase 2b: PID Tuning

We tested 12 PID configurations on the antenna motors (Dynamixel XL330-M288).
The robot's default is P=200 I=0 D=0 (well below the factory default of P=400).

We found that higher PID gains (P=1000 D=1000) made the **encoder metrics look
better** — lower OscRMS, fewer stall ticks — but the **antennas visibly shook
more**. The antennas are flexible spring-like metallic rods with very low inertia.
They ring like springs when the motor makes sharp corrections.

This is a crucial finding: **encoder-based metrics don't capture antenna tip
vibration.** The motor shaft can be tracking perfectly while the antenna tip
oscillates wildly. Higher PID gains make the motor more aggressive, which excites
the antenna spring more at each correction.

We attempted to measure the vibration acoustically using the robot's microphone,
but ambient noise dominated the signal. We also computed a "vibration index"
(acceleration sign changes per second) from the encoder data, which showed the
right trend (P=1000 D=1000 was 21-40% shakier) but the magnitude was too small
to match the visually dramatic difference.

**Conclusion: keep the default PID (P=200 D=0) for antennas.** The problem is
mechanical (antenna flexibility), not a PID tuning issue.

Full PID results in `pid_results/PID_RESULTS.md`.

### A Critical Bug: 60Hz vs 50Hz Sampling

During this investigation, we discovered that all our test scripts were sampling
at 60 Hz while the daemon's internal control loop runs at 50 Hz. This meant
~1 in 6 position reads returned a stale value from the previous tick, producing
false zero-velocity samples.

**Impact**: Stall percentages were inflated by ~3x (e.g., 20% reported as stalls
were actually stale reads, not real motor stops). All velocity plots had spurious
zero spikes. Jerk values were artificially high.

**Fix**: Changed all scripts to 50 Hz. The corrected data is in
`friction_results_50hz/` and `trajectory_results_50hz/`. The original 60Hz data
is preserved in `friction_results/` for reference.

## What Now?

The stall event analysis shows that:

1. **Zero stalls is unreachable** at any tested speed — the gearbox always has
   micro-stalls.
2. **Antenna sweet spot: ~30 deg/s** at the default 10° offset (21 stall events
   per 15s).
3. **Head Z sweet spot: ~5 mm/s** (25 events), but the curve is non-monotonic
   and not yet fully explained.
4. **PID tuning helps the encoder but hurts the antenna tips** — the problem is
   mechanical, not control.
5. **Trajectory shape gives marginal improvements** — the stalls come from
   direction reversals, not the trajectory between them.

Open questions:
- Can pre-rotation (starting with a small roll/pitch before Z translation) break
  static friction and reduce head Z stalls?
- Can we find a combined trajectory that keeps the motor moving continuously
  (e.g., circular motion in roll+Z space) to avoid zero-velocity crossings?
- Can the non-monotonic head Z curve be explained by reversal dynamics?

*This investigation is ongoing. We'll update this post as we test these
approaches and find out what works.*

---

*All experiments run on Reachy Mini by Pollen Robotics. Test scripts, raw data,
and analysis code available in the repository.*

## References

[1] M. Duclusaud, G. Passault, V. Padois, O. Ly, "Extended Friction Models for
the Physics Simulation of Servo Actuators," IEEE International Conference on
Robotics and Automation (ICRA), 2025.
[arXiv:2410.08650](https://arxiv.org/abs/2410.08650) —
Code: [github.com/Rhoban/bam](https://github.com/Rhoban/bam) —
[PDF](passault2024_friction_models.pdf)
