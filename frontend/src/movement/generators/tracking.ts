import { type HeadPose, createNeutralPose, BaseGenerator, smoothStep } from "../types";
import { type Mat4, createHeadPoseMat4, mat4Identity } from "../mat4";

const CFG = {
  SMOOTHING: 0.05,
  MAX_YAW_DEG: 25,
  MAX_PITCH_DEG: 15,
  FADE_DURATION: 0.8,
};

export class TrackingGenerator extends BaseGenerator {
  private targetYaw = 0;
  private targetPitch = 0;
  private currentYaw = 0;
  private currentPitch = 0;
  private hasTarget = false;
  private fadeAmount = 0;

  setTarget(x: number, y: number): void {
    this.active = true;
    this.hasTarget = true;
    this.targetYaw = -(x - 0.5) * 2 * CFG.MAX_YAW_DEG * (Math.PI / 180);
    this.targetPitch = (y - 0.5) * 2 * CFG.MAX_PITCH_DEG * (Math.PI / 180);
  }

  clearTarget(): void {
    this.hasTarget = false;
  }

  update(dt: number): void {
    if (this.hasTarget) {
      this.fadeAmount = Math.min(1, this.fadeAmount + dt * 4);
    } else {
      this.fadeAmount = Math.max(0, this.fadeAmount - dt / CFG.FADE_DURATION);
      this.targetYaw = 0;
      this.targetPitch = 0;
      if (this.fadeAmount <= 0.001) { this.active = false; this.currentYaw = 0; this.currentPitch = 0; return; }
    }
    const s = smoothStep(CFG.SMOOTHING, dt);
    this.currentYaw += (this.targetYaw - this.currentYaw) * s;
    this.currentPitch += (this.targetPitch - this.currentPitch) * s;
  }

  getMat4(): Mat4 {
    if (!this.active) return mat4Identity();
    return createHeadPoseMat4(0, 0, 0, 0, this.currentPitch * this.fadeAmount, this.currentYaw * this.fadeAmount);
  }

  getPose(): HeadPose {
    if (!this.active) return createNeutralPose();
    return { x: 0, y: 0, z: 0, roll: 0, pitch: this.currentPitch * this.fadeAmount, yaw: this.currentYaw * this.fadeAmount };
  }

  reset(): void {
    super.reset();
    this.targetYaw = 0;
    this.targetPitch = 0;
    this.currentYaw = 0;
    this.currentPitch = 0;
    this.hasTarget = false;
    this.fadeAmount = 0;
  }
}
