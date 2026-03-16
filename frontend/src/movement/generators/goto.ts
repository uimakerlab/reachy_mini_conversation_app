import { type HeadPose, createNeutralPose, clonePose, lerpPose, minJerkPose, BaseGenerator, HEAD_PRESETS } from "../types";

export class GotoGenerator extends BaseGenerator {
  private startPose: HeadPose = createNeutralPose();
  private targetPose: HeadPose = createNeutralPose();
  private currentPose: HeadPose = createNeutralPose();
  private duration = 1.0;
  private elapsed = 0;
  private interpolation: "linear" | "minjerk" = "minjerk";
  private onComplete: (() => void) | null = null;

  goto(
    target: HeadPose | string,
    opts: { from?: HeadPose; duration?: number; interpolation?: "linear" | "minjerk"; onComplete?: () => void } = {},
  ): void {
    let tp: HeadPose;
    if (typeof target === "string") {
      tp = clonePose(HEAD_PRESETS[target] ?? HEAD_PRESETS.front);
    } else {
      tp = clonePose(target);
    }
    this.startPose = opts.from ? clonePose(opts.from) : clonePose(this.currentPose);
    this.targetPose = tp;
    this.duration = opts.duration ?? 1.0;
    this.interpolation = opts.interpolation ?? "minjerk";
    this.onComplete = opts.onComplete ?? null;
    this.elapsed = 0;
    this.active = true;
  }

  stop(): void {
    this.active = false;
    this.elapsed = 0;
  }

  update(dt: number): void {
    if (!this.active) return;
    this.elapsed += dt;
    const t = Math.min(1, this.elapsed / this.duration);
    this.currentPose = this.interpolation === "minjerk" ? minJerkPose(this.startPose, this.targetPose, t) : lerpPose(this.startPose, this.targetPose, t);
    if (t >= 1) {
      this.active = false;
      this.currentPose = clonePose(this.targetPose);
      const cb = this.onComplete;
      this.onComplete = null;
      cb?.();
    }
  }

  getPose(): HeadPose {
    return clonePose(this.currentPose);
  }

  getProgress(): number {
    return this.active ? Math.min(1, this.elapsed / this.duration) : 1;
  }

  reset(): void {
    super.reset();
    this.startPose = createNeutralPose();
    this.targetPose = createNeutralPose();
    this.currentPose = createNeutralPose();
    this.elapsed = 0;
    this.onComplete = null;
  }
}
