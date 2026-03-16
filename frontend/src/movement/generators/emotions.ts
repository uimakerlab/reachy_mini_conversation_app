import {
  type HeadPose,
  createPose,
  clonePose,
  createNeutralPose,
  lerpPose,
  BaseGenerator,
  JOINT_LIMITS,
} from "../types";

interface EmotionPoseData {
  head: { pitch: number; yaw: number; roll: number };
  antennas: [number, number];
}

interface EmotionPoseFull {
  head: HeadPose;
  antennas: [number, number];
}

const EMOTION_POSES: Record<string, EmotionPoseData> = {
  neutral: { head: { pitch: 0, yaw: 0, roll: 0 }, antennas: [0, 0] },
  joy: { head: { pitch: -0.3, yaw: 0.08, roll: 0 }, antennas: [-0.7, 0.7] },
  curious: { head: { pitch: -0.32, yaw: 0.08, roll: 0 }, antennas: [-1.1, 0.4] },
  surprise: { head: { pitch: 0.06, yaw: 0.1, roll: 0 }, antennas: [-0.5, 0.2] },
  sadness: { head: { pitch: 0.5, yaw: 0, roll: 0 }, antennas: [-2.7, 2.7] },
  anger: { head: { pitch: 0.49, yaw: 0, roll: 0 }, antennas: [-2.2, 2.1] },
};

export const EMOTION_NAMES = Object.keys(EMOTION_POSES);

export const EMOTION_COORDS: Record<string, { valence: number; arousal: number }> = {
  neutral: { valence: 0, arousal: 0 },
  joy: { valence: 0.8, arousal: 0.6 },
  curious: { valence: 0.4, arousal: 0.4 },
  surprise: { valence: 0.1, arousal: 0.9 },
  anger: { valence: -0.8, arousal: 0.7 },
  sadness: { valence: -0.7, arousal: -0.5 },
};

const IDW_POWER = 3;

function cloneEP(ep: EmotionPoseData): EmotionPoseFull {
  return { head: createPose(ep.head), antennas: [...ep.antennas] as [number, number] };
}

function neutralEP(): EmotionPoseFull {
  return cloneEP(EMOTION_POSES.neutral);
}

function lerpEP(from: EmotionPoseFull, to: EmotionPoseFull, t: number): EmotionPoseFull {
  const c = Math.max(0, Math.min(1, t));
  const { min: aMin, max: aMax } = JOINT_LIMITS.antenna;
  return {
    head: lerpPose(from.head, to.head, c),
    antennas: [
      Math.max(aMin, Math.min(aMax, from.antennas[0] + (to.antennas[0] - from.antennas[0]) * c)),
      Math.max(aMin, Math.min(aMax, from.antennas[1] + (to.antennas[1] - from.antennas[1]) * c)),
    ],
  };
}

function getPoseForCoords(v: number, a: number): EmotionPoseFull {
  let totalW = 0;
  const entries: { w: number; pose: EmotionPoseData }[] = [];
  for (const [name, coords] of Object.entries(EMOTION_COORDS)) {
    const pose = EMOTION_POSES[name];
    if (!pose) continue;
    const dist = Math.sqrt((v - coords.valence) ** 2 + (a - coords.arousal) ** 2);
    if (dist < 0.001) return cloneEP(pose);
    const w = 1 / Math.pow(dist, IDW_POWER);
    entries.push({ w, pose });
    totalW += w;
  }
  let pitch = 0, yaw = 0, roll = 0, antL = 0, antR = 0;
  for (const { w, pose } of entries) {
    const nw = w / totalW;
    pitch += pose.head.pitch * nw;
    yaw += pose.head.yaw * nw;
    roll += (pose.head.roll ?? 0) * nw;
    antL += pose.antennas[0] * nw;
    antR += pose.antennas[1] * nw;
  }
  const { min: aMin, max: aMax } = JOINT_LIMITS.antenna;
  return {
    head: createPose({ pitch, yaw, roll }),
    antennas: [Math.max(aMin, Math.min(aMax, antL)), Math.max(aMin, Math.min(aMax, antR))],
  };
}

export class EmotionsGenerator extends BaseGenerator {
  private targetPose = neutralEP();
  private currentPose = neutralEP();
  private currentEmotionName: string | null = null;
  private progress = 1;
  private arrivalSpeed = 2.5;
  private returnSpeed = 0.5;
  private decayIdleSeconds = 4;
  private idleTime = 0;
  private isDecaying = false;

  setEmotion(name: string, intensity = 1): boolean {
    const pose = EMOTION_POSES[name];
    if (!pose) return false;
    if (name === "neutral") { this.clear(); return true; }
    const full = cloneEP(pose);
    const neutral = neutralEP();
    this.targetPose = intensity >= 1 ? full : lerpEP(neutral, full, intensity);
    this.progress = 0;
    this.currentEmotionName = name;
    this.active = true;
    this.idleTime = 0;
    this.isDecaying = false;
    return true;
  }

  setCoords(valence: number, arousal: number): boolean {
    const v = Math.max(-1, Math.min(1, valence));
    const a = Math.max(-1, Math.min(1, arousal));
    this.targetPose = getPoseForCoords(v, a);
    let closest = "neutral";
    let minD = Infinity;
    for (const [n, c] of Object.entries(EMOTION_COORDS)) {
      const d = Math.sqrt((c.valence - v) ** 2 + (c.arousal - a) ** 2);
      if (d < minD) { minD = d; closest = n; }
    }
    this.progress = 0;
    this.currentEmotionName = closest;
    this.active = true;
    this.idleTime = 0;
    this.isDecaying = false;
    return true;
  }

  clear(): void {
    this.targetPose = neutralEP();
    this.progress = 0;
    this.currentEmotionName = "neutral";
    this.isDecaying = true;
  }

  update(dt: number): void {
    this.idleTime += dt;
    if (!this.isDecaying && this.idleTime >= this.decayIdleSeconds) this.clear();
    if (this.progress < 1) {
      const speed = this.isDecaying ? this.returnSpeed : this.arrivalSpeed;
      this.progress = Math.min(1, this.progress + speed * dt);
      const eased = 1 - Math.pow(1 - this.progress, 2);
      this.currentPose = lerpEP(this.currentPose, this.targetPose, eased);
    }
    if (this.isDecaying && this.progress >= 0.98) {
      this.active = false;
      this.currentEmotionName = null;
      this.isDecaying = false;
      this.progress = 1;
    }
  }

  getPose(): HeadPose {
    return clonePose(this.currentPose.head);
  }

  getAntennas(): [number, number] {
    return [...this.currentPose.antennas] as [number, number];
  }

  getCurrentEmotion(): string | null {
    return this.active ? this.currentEmotionName : null;
  }

  getState() {
    return { emotion: this.getCurrentEmotion(), progress: this.progress, isDecaying: this.isDecaying, active: this.active };
  }

  stop(): void { this.reset(); }

  reset(): void {
    super.reset();
    this.targetPose = neutralEP();
    this.currentPose = neutralEP();
    this.currentEmotionName = null;
    this.progress = 1;
    this.idleTime = 0;
    this.isDecaying = false;
  }
}
