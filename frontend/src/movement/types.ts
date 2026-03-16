export interface HeadPose {
  x: number; // meters
  y: number;
  z: number;
  roll: number; // radians
  pitch: number;
  yaw: number;
}

export interface FullBodyPose {
  head: HeadPose;
  antennas: [number, number]; // [left, right] radians
  bodyYaw: number;
}

export const CONFIG = {
  LOOP_FREQUENCY: 100,
  DEFAULT_DAEMON_URL: "",
  IDLE_DELAY_MS: 500,
  WS_RECONNECT_DELAY: 1000,
} as const;

export const JOINT_LIMITS = {
  pitch: { min: -0.52, max: 0.52 },
  yaw: { min: -0.7, max: 0.7 },
  roll: { min: -0.26, max: 0.26 },
  x: { min: -0.02, max: 0.02 },
  y: { min: -0.02, max: 0.02 },
  z: { min: -0.02, max: 0.02 },
  antenna: { min: (-160 * Math.PI) / 180, max: (160 * Math.PI) / 180 },
} as const;

export function createNeutralPose(): HeadPose {
  return { x: 0, y: 0, z: 0, roll: 0, pitch: 0, yaw: 0 };
}

export function createPose(
  values: Partial<HeadPose> = {},
  opts: { degrees?: boolean; mm?: boolean } = {},
): HeadPose {
  const d = opts.degrees ? Math.PI / 180 : 1;
  const m = opts.mm ? 0.001 : 1;
  return {
    x: (values.x ?? 0) * m,
    y: (values.y ?? 0) * m,
    z: (values.z ?? 0) * m,
    roll: (values.roll ?? 0) * d,
    pitch: (values.pitch ?? 0) * d,
    yaw: (values.yaw ?? 0) * d,
  };
}

export function clonePose(p: HeadPose): HeadPose {
  return { ...p };
}

export function addPoses(a: HeadPose, b: HeadPose): HeadPose {
  return {
    x: a.x + b.x,
    y: a.y + b.y,
    z: a.z + b.z,
    roll: a.roll + b.roll,
    pitch: a.pitch + b.pitch,
    yaw: a.yaw + b.yaw,
  };
}

export function scalePose(p: HeadPose, f: number): HeadPose {
  return { x: p.x * f, y: p.y * f, z: p.z * f, roll: p.roll * f, pitch: p.pitch * f, yaw: p.yaw * f };
}

export function lerpPose(from: HeadPose, to: HeadPose, t: number): HeadPose {
  const c = Math.max(0, Math.min(1, t));
  return {
    x: from.x + (to.x - from.x) * c,
    y: from.y + (to.y - from.y) * c,
    z: from.z + (to.z - from.z) * c,
    roll: from.roll + (to.roll - from.roll) * c,
    pitch: from.pitch + (to.pitch - from.pitch) * c,
    yaw: from.yaw + (to.yaw - from.yaw) * c,
  };
}

export function minJerk(t: number): number {
  const c = Math.max(0, Math.min(1, t));
  return 10 * c ** 3 - 15 * c ** 4 + 6 * c ** 5;
}

export function minJerkPose(from: HeadPose, to: HeadPose, t: number): HeadPose {
  return lerpPose(from, to, minJerk(t));
}

export function smoothStep(factor: number, dt: number, refRate = 100): number {
  return 1 - Math.pow(1 - factor, dt * refRate);
}

export function clampPose(p: HeadPose): HeadPose {
  const L = JOINT_LIMITS;
  return {
    x: Math.max(L.x.min, Math.min(L.x.max, p.x)),
    y: Math.max(L.y.min, Math.min(L.y.max, p.y)),
    z: Math.max(L.z.min, Math.min(L.z.max, p.z)),
    roll: Math.max(L.roll.min, Math.min(L.roll.max, p.roll)),
    pitch: Math.max(L.pitch.min, Math.min(L.pitch.max, p.pitch)),
    yaw: Math.max(L.yaw.min, Math.min(L.yaw.max, p.yaw)),
  };
}

export function clampAntennas(a: [number, number]): [number, number] {
  const L = JOINT_LIMITS.antenna;
  return [Math.max(L.min, Math.min(L.max, a[0])), Math.max(L.min, Math.min(L.max, a[1]))];
}

export function createFullBodyPose(): FullBodyPose {
  return { head: createNeutralPose(), antennas: [0, 0], bodyYaw: 0 };
}

export function addFullBodyPoses(a: FullBodyPose, b: FullBodyPose): FullBodyPose {
  return {
    head: addPoses(a.head, b.head),
    antennas: [a.antennas[0] + b.antennas[0], a.antennas[1] + b.antennas[1]],
    bodyYaw: a.bodyYaw + b.bodyYaw,
  };
}

export const HEAD_PRESETS: Record<string, HeadPose> = {
  front: createPose(),
  left: createPose({ yaw: 40 }, { degrees: true }),
  right: createPose({ yaw: -40 }, { degrees: true }),
  up: createPose({ pitch: -30 }, { degrees: true }),
  down: createPose({ pitch: 30 }, { degrees: true }),
  upLeft: createPose({ pitch: -20, yaw: 30 }, { degrees: true }),
  upRight: createPose({ pitch: -20, yaw: -30 }, { degrees: true }),
  downLeft: createPose({ pitch: 20, yaw: 30 }, { degrees: true }),
  downRight: createPose({ pitch: 20, yaw: -30 }, { degrees: true }),
};

export abstract class BaseGenerator {
  active = false;
  abstract getPose(): HeadPose;
  abstract update(dt: number): void;
  reset(): void {
    this.active = false;
  }
  isActive(): boolean {
    return this.active;
  }
}
