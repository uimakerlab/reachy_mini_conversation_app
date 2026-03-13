# Stick-Slip Friction Analysis for Reachy Mini Breathing Animation

## The Problem

The breathing animation produces visible jerky/trembling motion even though the
commanded trajectory is smooth. This is not caused by control frequency,
quantization, or software — it's a mechanical phenomenon.

## Root Cause: Stick-Slip Friction in Gearboxes

### The Physics

Each motor+gearbox has a friction characteristic with a discontinuity at zero
velocity:

```
Torque
  ^
  |     /--- viscous (proportional to velocity)
  |    /
  |---*       <-- static friction threshold (Coulomb friction)
  |   |
  |   |  (Stribeck dip here in real systems)
--+---+-----------> Velocity
  |   |
  |---*       <-- static friction in reverse direction
  |    \
  |     \--- viscous
```

**Static friction** is the minimum force required to start moving from rest.
Once overcome, dynamic (kinetic) friction is lower — the motor "breaks free"
and accelerates. This creates the stick-slip cycle:

1. **STICK**: The commanded position moves slowly. The PID controller computes a
   small error → small voltage → small torque. But the torque is below the static
   friction threshold. The motor stays stuck. Error accumulates.

2. **SLIP**: Error grows until PID output exceeds static friction. The motor
   breaks free, overshoots slightly (kinetic friction < static friction), then
   stops again.

3. **Repeat**: This creates a sawtooth pattern in the present position — the
   motor alternates between being stuck and jumping forward.

### Why Slow Movements Are Worst

For fast movements, the motor never stops — it's always in the kinetic friction
regime. The PID error is always large enough to overcome static friction.

For slow movements, the commanded position moves so slowly that the PID can
"catch up" and the error drops below the static friction threshold. The motor
stops. Then it has to overcome static friction again → jerk.

**Critical insight**: Higher control frequency (100Hz vs 10Hz) does NOT fix
this. It just means the motor reaches the target faster and gets stuck sooner.
The problem is velocity-dependent, not frequency-dependent.

### Additional Factors

1. **Gravity loading**: When moving the head up (Z+), all 6 Stewart platform
   motors fight gravity. The gearboxes are under torque load even at rest, which
   increases the friction (the loaded gear teeth press harder against each other).
   This makes the static friction threshold higher for Z-up movements.

2. **Stribeck effect**: Friction is highest at zero velocity (static friction)
   and drops as velocity increases toward the kinetic friction level. The
   transition zone creates a narrow velocity band where the motor is most
   prone to getting trapped in the stick-slip cycle.

3. **Gearbox backlash**: When the motor reverses direction, it first has to
   take up the backlash (dead zone) in the gearbox. During this time, the output
   doesn't move at all even though the motor shaft is turning.

## Experimental Evidence

From our breathing benchmark data, we measured **present position stall runs**
(consecutive ticks where the present position doesn't change):

- Even the "baseline" breathing (z 5mm @ 0.1Hz) shows Z stalls up to 99 ticks
  (1.6 seconds stuck at 60Hz sampling)
- The slowest configs (zen, slow) show Z stalls of 200+ ticks (3+ seconds)
- This confirms the motor is genuinely stuck, not just changing slowly

## Proposed Solutions

### Idea 1: Avoid Low Velocities (Minimum Velocity Threshold)

**Hypothesis**: There exists a minimum velocity V_min below which stick-slip
occurs. If we keep all movements above V_min, the motors stay in the kinetic
friction regime and motion is smooth.

**Implications**:
- Instead of smooth sinusoidal ramps (velocity → 0 at peaks), use linear
  interpolation with constant velocity
- The breathing animation would need to move fast enough that the motors never
  stop during the trajectory
- May need to accept "less smooth" trajectories to avoid stick-slip

**Experimental plan** (Step 1 — characterize):
- Linear interpolation at various constant speeds on antennas and head Z
- Test at different operating angles (different gravity/friction loading)
- Measure where stick-slip appears/disappears
- Find V_min for each motor group

**Experimental plan** (Step 2 — find usable velocity):
- Sweep velocities from very slow to fast
- Find the lowest velocity with no multi-tick stalls
- Verify it looks acceptable for a breathing animation

### Idea 2: Pre-Rotation to Break Static Friction

**Hypothesis**: Starting a movement with a small rotation (roll/pitch) before
or simultaneously with Z translation can "pre-break" the static friction in
the Stewart platform motors.

**Reasoning**:
- A pure rotation around the center of mass doesn't change gravitational
  potential energy — only friction forces oppose the motion
- In a rotation, roughly half the motors push "up" and half push "down",
  so the gravity loading on each motor is lower than in pure Z translation
- The static friction threshold is therefore lower for rotations
- Once the motors are moving (kinetic regime), adding Z translation is easier

**Experimental plan** (Step 3 — for later):
- Design movements: small roll/pitch → then Z translation
- Compare stick-slip behavior vs pure Z translation at same Z velocity
- Measure if the rotation "kick" reduces the stall count
- If successful, design breathing animations that naturally incorporate
  this pattern (e.g., slight roll → Z up, slight roll back → Z down)

### Idea 3: PID Tuning

**Hypothesis**: The current motor controllers likely use only P (proportional)
gain. The micro-vibrations and overshoot during stick-slip breakaway could be
reduced with better PID tuning — specifically adding D (derivative) gain to
dampen oscillations, and possibly I (integral) gain to overcome static friction
faster.

**Reasoning**:
- P-only control: when the motor breaks free from static friction, the full
  proportional error drives it hard → overshoot → stop → stuck again
- Adding D: the derivative term resists sudden velocity changes, which would
  dampen the "break free and overshoot" phase
- Adding I: integral accumulates error over time, which could help push through
  static friction sooner (smaller stall runs), though risk of windup

**Experimental plan** (Step 4 — explore PID):
- Investigate how to read/set PID parameters via the SDK
- Run the same constant-velocity tests with different PID configurations
- Measure whether D gain reduces the amplitude of stick-slip jumps
- Measure whether I gain reduces stall run duration

### Note on Antenna Mechanical Vibration

The antennas have very low inertia and are mechanically flexible (long metallic
rods). At high velocities, the acceleration at direction reversals can excite
mechanical resonance — the antenna acts like a loaded spring that rings after
being released. Some of the oscillation visible in the present position at high
speeds (e.g., ant_c0_v30, ant_c0_v50) may be physical vibration of the antenna
structure, not friction effects. This limits how fast we can move antennas as a
workaround for stick-slip.
