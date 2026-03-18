import { BaseGenerator } from "../types";
import {
  type Mat4,
  mat4Identity,
  mat4Clone,
  createHeadPoseMat4,
  linearPoseInterpolation,
} from "../mat4";

/**
 * Breathing generator matching Python's BreathingMove exactly:
 *
 * Phase 1: SLERP interpolation from start pose to neutral (1s)
 * Phase 2: continuous z-axis breathing (5mm, 0.1Hz)
 *          + opposite-direction antenna sway (15deg, 0.5Hz)
 */

const CFG = {
  INTERPOLATION_DURATION: 1.0,
  BREATHING_Z_AMPLITUDE: 0.005, // 5mm
  BREATHING_FREQUENCY: 0.1,     // Hz (6 breaths per minute)
  ANTENNA_SWAY_AMPLITUDE: 15 * Math.PI / 180, // 15 degrees
  ANTENNA_FREQUENCY: 0.5,       // Hz
};

export class BreathingGenerator extends BaseGenerator {
  private time = 0;
  private startPose: Mat4 = mat4Identity();
  private neutralPose: Mat4 = mat4Identity();
  private antennaStartLeft = 0;
  private antennaStartRight = 0;
  private currentAntennas: [number, number] = [0, 0];

  /**
   * Start breathing from a given initial pose, interpolating to neutral
   * then oscillating. Matches Python BreathingMove(interpolation_start_pose, ...).
   */
  startFrom(startPose: Mat4, startAntennas: [number, number]): void {
    this.active = true;
    this.time = 0;
    this.startPose = mat4Clone(startPose);
    this.neutralPose = mat4Identity();
    this.antennaStartLeft = startAntennas[0];
    this.antennaStartRight = startAntennas[1];
    this.currentAntennas = [...startAntennas];
  }

  /** Simplified start without a known pose (falls back to neutral start). */
  start(): void {
    this.startFrom(mat4Identity(), [0, 0]);
  }

  stop(): void {
    this.active = false;
  }

  stopImmediate(): void {
    this.active = false;
    this.time = 0;
  }

  update(dt: number): void {
    if (!this.active) return;
    this.time += dt;

    if (this.time < CFG.INTERPOLATION_DURATION) {
      // Phase 1: interpolate antennas linearly toward neutral
      const t = this.time / CFG.INTERPOLATION_DURATION;
      this.currentAntennas[0] = this.antennaStartLeft * (1 - t);
      this.currentAntennas[1] = this.antennaStartRight * (1 - t);
    } else {
      // Phase 2: antenna sway (opposite directions)
      const breathTime = this.time - CFG.INTERPOLATION_DURATION;
      const sway = CFG.ANTENNA_SWAY_AMPLITUDE *
        Math.sin(2 * Math.PI * CFG.ANTENNA_FREQUENCY * breathTime);
      this.currentAntennas[0] = sway;
      this.currentAntennas[1] = -sway;
    }
  }

  getMat4(): Mat4 {
    if (!this.active) return mat4Identity();

    if (this.time < CFG.INTERPOLATION_DURATION) {
      // Phase 1: SLERP from start pose to neutral
      const t = this.time / CFG.INTERPOLATION_DURATION;
      return linearPoseInterpolation(this.startPose, this.neutralPose, t);
    }

    // Phase 2: z-axis breathing only
    const breathTime = this.time - CFG.INTERPOLATION_DURATION;
    const z = CFG.BREATHING_Z_AMPLITUDE *
      Math.sin(2 * Math.PI * CFG.BREATHING_FREQUENCY * breathTime);
    return createHeadPoseMat4(0, 0, z, 0, 0, 0);
  }

  getAntennas(): [number, number] {
    return [...this.currentAntennas] as [number, number];
  }

  reset(): void {
    super.reset();
    this.time = 0;
    this.startPose = mat4Identity();
    this.antennaStartLeft = 0;
    this.antennaStartRight = 0;
    this.currentAntennas = [0, 0];
  }
}
