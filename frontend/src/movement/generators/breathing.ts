import { type HeadPose, createNeutralPose, BaseGenerator } from "../types";

const CFG = {
  FREQ_PITCH: 0.15,
  FREQ_ROLL: 0.1,
  FREQ_YAW: 0.08,
  FREQ_Z: 0.15,
  AMP_PITCH_DEG: 2.0,
  AMP_ROLL_DEG: 0.8,
  AMP_YAW_DEG: 1.5,
  AMP_Z_MM: 2.0,
  PHASE_PITCH: 0,
  PHASE_ROLL: Math.PI / 3,
  PHASE_YAW: Math.PI / 2,
  PHASE_Z: 0,
  FADE_DURATION: 0.5,
};

export class BreathingGenerator extends BaseGenerator {
  private time = 0;
  private fadeAmount = 0;
  private targetFade = 0;
  private phaseOffset = Math.random() * Math.PI * 2;

  start(): void {
    this.active = true;
    this.targetFade = 1;
  }

  stop(): void {
    this.targetFade = 0;
  }

  stopImmediate(): void {
    this.active = false;
    this.targetFade = 0;
    this.fadeAmount = 0;
  }

  update(dt: number): void {
    if (!this.active && this.fadeAmount <= 0) return;
    this.time += dt;
    const speed = 1 / CFG.FADE_DURATION;
    if (this.fadeAmount < this.targetFade) this.fadeAmount = Math.min(this.targetFade, this.fadeAmount + speed * dt);
    else if (this.fadeAmount > this.targetFade) this.fadeAmount = Math.max(this.targetFade, this.fadeAmount - speed * dt);
    if (this.fadeAmount <= 0 && this.targetFade <= 0) this.active = false;
  }

  getPose(): HeadPose {
    if (this.fadeAmount <= 0) return createNeutralPose();
    const t = this.time;
    const p = this.phaseOffset;
    const d = Math.PI / 180;
    return {
      x: 0,
      y: 0,
      z: Math.sin(2 * Math.PI * CFG.FREQ_Z * t + CFG.PHASE_Z + p) * (CFG.AMP_Z_MM / 1000) * this.fadeAmount,
      roll: Math.sin(2 * Math.PI * CFG.FREQ_ROLL * t + CFG.PHASE_ROLL + p) * CFG.AMP_ROLL_DEG * d * this.fadeAmount,
      pitch: Math.sin(2 * Math.PI * CFG.FREQ_PITCH * t + CFG.PHASE_PITCH + p) * CFG.AMP_PITCH_DEG * d * this.fadeAmount,
      yaw: Math.sin(2 * Math.PI * CFG.FREQ_YAW * t + CFG.PHASE_YAW + p) * CFG.AMP_YAW_DEG * d * this.fadeAmount,
    };
  }

  reset(): void {
    super.reset();
    this.time = 0;
    this.fadeAmount = 0;
    this.targetFade = 0;
  }
}
