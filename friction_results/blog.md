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
thing — we measured the actual motor positions at 60Hz and found something
completely different.

The motors aren't stepping too fast. **They're not moving at all.**

## What We Actually Measured

We recorded the commanded position (what we tell the motors to do) and the
present position (what the motors actually report) at 60Hz. Then we looked for
**stall runs** — consecutive ticks where the present position doesn't change.

If the problem were control frequency, we'd expect stall runs of exactly 1 tick
(16.7ms at 60Hz). The motor would update every tick, maybe sometimes skip one.

Instead, we found stall runs of **30, 50, even 74 consecutive ticks**. That's
the motor being completely stuck for over a second while the commanded position
keeps moving away from it.

Here's what that looks like. This is the antenna at 1 deg/s constant velocity,
centered at 0 degrees. Blue is commanded, red is present. Red shaded regions are
stalls:

![Antenna at 1 deg/s — classic stick-slip sawtooth](ant_c0_v1.png)

The present position (red) is a staircase. The motor gets stuck, the commanded
position pulls ahead, and then suddenly the motor breaks free and jumps to catch
up. Then it gets stuck again. This is textbook **stick-slip friction**.

And here's what smooth motion looks like, same antenna but at 30 deg/s:

![Antenna at 30 deg/s — smooth tracking](ant_c0_v30.png)

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

![Friction models — placeholder for real diagrams](TODO_friction_models.png)

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

The 1.5 mm/s head Z test showed a 74-tick (1.2 second) continuous stall — worse
than both slower and faster tests — suggesting this velocity sits right in the
transition zone where the motor repeatedly crosses the static/kinetic boundary.

## The Experiments

To verify all of this, we designed a systematic characterization. Instead of
sinusoidal breathing motions (where velocity constantly changes), we used
**constant-velocity linear interpolation** — triangle waves that move back and
forth at a fixed speed. This lets us cleanly measure the friction threshold as a
function of velocity.

### Test Matrix

- **Antenna tests**: 9 velocities (1, 2, 3, 5, 8, 12, 20, 30, 50 deg/s) × 3
  center angles (0°, 10°, 90°) = 27 tests
- **Head Z tests**: 9 velocities (0.5, 1, 1.5, 2, 3, 5, 8, 12, 20 mm/s) = 9
  tests
- **Sinusoidal baseline**: 1 test (original breathing motion for reference)
- 15 seconds each, 60Hz sampling, recording commanded and present positions

### Antenna Results: Friction Scales With Gravity

The minimum velocity for smooth motion (no stall runs longer than 3 ticks)
depends dramatically on the operating angle:

| Center Angle | Min Smooth Velocity | Physics |
|---|---|---|
| 0° (vertical) | 8 deg/s | Antenna hanging straight down, minimal gravity load on gearbox |
| 10° (SDK default offset) | 20 deg/s | Slight angle, moderate gravity loading |
| 90° (horizontal) | 30 deg/s | Full antenna weight on gearbox, maximum friction |

Horizontal antennas need nearly 4× the velocity of vertical ones. This directly
confirms gravity loading as a major factor in friction.

Here's what stick-slip looks like at 90° center angle — the motor is stuck 78%
of the time:

![Antenna at 90° center, 2 deg/s — 78% stuck](ant_c90_v2.png)

Compare with the same velocity at 0° center — still bad, but much less severe
(50% stuck):

![Antenna at 0° center, 1 deg/s — 50% stuck](ant_c0_v1.png)

### Head Z Results: Friction Is Severe At All Tested Speeds

This is the bad news. Even at 20 mm/s — far faster than any reasonable breathing
animation — the head Z axis still shows stick-slip:

| Velocity | Stall % | Max Stall | Duration |
|---|---|---|---|
| 0.5 mm/s | 57.9% | 19 ticks | 317ms |
| 1.0 mm/s | 51.1% | 26 ticks | 433ms |
| 1.5 mm/s | 48.9% | **74 ticks** | **1233ms** |
| 2.0 mm/s | 36.9% | 16 ticks | 267ms |
| 3.0 mm/s | 31.8% | 17 ticks | 283ms |
| 5.0 mm/s | 29.4% | 11 ticks | 183ms |
| 8.0 mm/s | 27.5% | 10 ticks | 167ms |
| 12.0 mm/s | 26.0% | 6 ticks | 100ms |
| 20.0 mm/s | 23.4% | 4 ticks | 67ms |

Note the 1.5 mm/s anomaly — a 1.2-second stall, worse than both 0.5 and 2.0
mm/s. This is likely the Stribeck effect: we're sitting right at the
friction-velocity peak.

Here's that 1.5 mm/s test. The head is completely stuck for over a second while
the command moves 2mm away:

![Head Z at 1.5 mm/s — stuck for 1.2 seconds](z_v1.5.png)

The breathing animation's peak Z velocity is about 3.1 mm/s (5mm amplitude ×
0.1Hz × 2π), and its *average* velocity is much lower. It spends most of its
time deep in the stick-slip zone.

### The Baseline: Confirming the Breathing Is Affected

The original breathing motion shows exactly what we predicted:
- **Z signal**: 43.6% stall, max 32 ticks (533ms) — the head trembles
- **Antenna signal**: 22.3% stall, max 5 ticks (83ms) — antennas are smoother
  because they move faster (47 deg/s peak)

The antenna sway is fast enough to stay mostly in the kinetic friction regime.
The Z motion is far too slow. This is why the *head* trembles but the *antennas*
look relatively smooth.

## What Now?

We're exploring two approaches:

### Approach 1: Avoid Low Velocities

If we can keep all movements above the stick-slip threshold, the motors stay in
the kinetic regime and motion is smooth. This would mean replacing smooth
sinusoidal trajectories (whose velocity passes through zero at every peak) with
trajectories that maintain a minimum velocity.

The challenge: for head Z, even 20 mm/s still shows stick-slip. The threshold
may be impractically high for a gentle breathing animation.

### Approach 2: Pre-Rotation to Break Static Friction

This is the more promising idea. Instead of fighting gravity head-on with a pure
Z translation, we start each movement phase with a small rotation (roll or
pitch). Rotations don't change gravitational potential energy — they don't fight
gravity — so their friction threshold is much lower. Once the motors are moving
in the kinetic regime, we add Z translation while they're already "warmed up."

Think of it like unsticking a jar lid: you don't just pull harder, you twist
first.

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
