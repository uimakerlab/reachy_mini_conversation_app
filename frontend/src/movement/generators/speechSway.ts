import { type HeadPose, createNeutralPose, BaseGenerator } from "../types";
import { type Mat4, createHeadPoseMat4, mat4Identity } from "../mat4";

const CFG = {
  SAMPLE_RATE: 16000,
  HOP_MS: 10,
  FRAME_MS: 20,
  MASTER: 2.5,
  VAD_DB_ON: -35.0,
  VAD_DB_OFF: -45.0,
  VAD_ATTACK_MS: 40,
  VAD_RELEASE_MS: 250,
  ENV_FOLLOW_GAIN: 0.65,
  FREQ_PITCH: 2.2,
  FREQ_YAW: 0.6,
  FREQ_ROLL: 1.3,
  FREQ_X: 0.35,
  FREQ_Y: 0.45,
  FREQ_Z: 0.25,
  AMP_PITCH_DEG: 4.5,
  AMP_YAW_DEG: 7.5,
  AMP_ROLL_DEG: 2.25,
  AMP_X_MM: 4.5,
  AMP_Y_MM: 3.75,
  AMP_Z_MM: 2.25,
  DB_LOW: -46.0,
  DB_HIGH: -18.0,
  LOUDNESS_GAMMA: 0.9,
  SWAY_ATTACK_MS: 50,
  SWAY_RELEASE_MS: 250,
  SENS_DB_OFFSET: 4.0,
};

const HOP = Math.floor(CFG.SAMPLE_RATE * CFG.HOP_MS / 1000);
const FRAME = Math.floor(CFG.SAMPLE_RATE * CFG.FRAME_MS / 1000);
const ATTACK_FR = Math.max(1, Math.floor(CFG.VAD_ATTACK_MS / CFG.HOP_MS));
const RELEASE_FR = Math.max(1, Math.floor(CFG.VAD_RELEASE_MS / CFG.HOP_MS));
const SWAY_ATTACK_FR = Math.max(1, Math.floor(CFG.SWAY_ATTACK_MS / CFG.HOP_MS));
const SWAY_RELEASE_FR = Math.max(1, Math.floor(CFG.SWAY_RELEASE_MS / CFG.HOP_MS));

function rmsDbfs(samples: Float32Array): number {
  if (samples.length === 0) return -100;
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return 20 * Math.log10(Math.sqrt(sum / samples.length + 1e-12) + 1e-12);
}

function loudnessGain(db: number): number {
  let t = (db + CFG.SENS_DB_OFFSET - CFG.DB_LOW) / (CFG.DB_HIGH - CFG.DB_LOW);
  t = Math.max(0, Math.min(1, t));
  return CFG.LOUDNESS_GAMMA !== 1.0 ? Math.pow(t, CFG.LOUDNESS_GAMMA) : t;
}

function resampleLinear(data: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate || data.length === 0) return data;
  const ratio = toRate / fromRate;
  const newLen = Math.floor(data.length * ratio);
  const out = new Float32Array(newLen);
  for (let i = 0; i < newLen; i++) {
    const src = i / ratio;
    const idx = Math.floor(src);
    const frac = src - idx;
    out[i] = data[Math.min(idx, data.length - 1)] * (1 - frac) + data[Math.min(idx + 1, data.length - 1)] * frac;
  }
  return out;
}

/**
 * Seeded PRNG matching numpy's default_rng for deterministic phase offsets.
 * Uses a simple mulberry32 algorithm - only needs uniform [0,1) values.
 */
function seededRandom(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const RNG_SEED = 7;

export class SpeechSwayGenerator extends BaseGenerator {
  private ringBuffer = new Float32Array(2 * CFG.SAMPLE_RATE);
  private ringWritePos = 0;
  private ringReadPos = 0;
  private ringLength = 0;
  private vadOn = false;
  private vadAbove = 0;
  private vadBelow = 0;
  private swayEnv = 0;
  private swayUp = 0;
  private swayDown = 0;
  private time = 0;

  // Deterministic phases matching Python's rng_seed=7
  private phasePitch: number;
  private phaseYaw: number;
  private phaseRoll: number;
  private phaseX: number;
  private phaseY: number;
  private phaseZ: number;

  private outputQueue: HeadPose[] = [];
  private currentPose: HeadPose = createNeutralPose();

  constructor() {
    super();
    const rng = seededRandom(RNG_SEED);
    this.phasePitch = rng() * Math.PI * 2;
    this.phaseYaw = rng() * Math.PI * 2;
    this.phaseRoll = rng() * Math.PI * 2;
    this.phaseX = rng() * Math.PI * 2;
    this.phaseY = rng() * Math.PI * 2;
    this.phaseZ = rng() * Math.PI * 2;
  }

  feedSamples(audioData: Float32Array, sampleRate = 24000): void {
    if (!audioData || audioData.length === 0) return;
    this.active = true;
    const data = sampleRate !== CFG.SAMPLE_RATE ? resampleLinear(audioData, sampleRate, CFG.SAMPLE_RATE) : audioData;
    const bufLen = this.ringBuffer.length;
    for (let i = 0; i < data.length; i++) {
      this.ringBuffer[this.ringWritePos] = data[i];
      this.ringWritePos = (this.ringWritePos + 1) % bufLen;
    }
    this.ringLength = Math.min(this.ringLength + data.length, bufLen);
    this._process();
  }

  private _readFromRing(count: number, offset = 0): Float32Array {
    const bufLen = this.ringBuffer.length;
    const out = new Float32Array(count);
    const start = (this.ringReadPos + offset) % bufLen;
    for (let i = 0; i < count; i++) out[i] = this.ringBuffer[(start + i) % bufLen];
    return out;
  }

  private _consumeFromRing(count: number): void {
    this.ringReadPos = (this.ringReadPos + count) % this.ringBuffer.length;
    this.ringLength -= count;
  }

  private _process(): void {
    while (this.ringLength >= FRAME) {
      const frame = this._readFromRing(FRAME, this.ringLength - FRAME);
      const db = rmsDbfs(frame);

      if (db >= CFG.VAD_DB_ON) { this.vadAbove++; this.vadBelow = 0; if (!this.vadOn && this.vadAbove >= ATTACK_FR) this.vadOn = true; }
      else if (db <= CFG.VAD_DB_OFF) { this.vadBelow++; this.vadAbove = 0; if (this.vadOn && this.vadBelow >= RELEASE_FR) this.vadOn = false; }

      if (this.vadOn) { this.swayUp = Math.min(SWAY_ATTACK_FR, this.swayUp + 1); this.swayDown = 0; }
      else { this.swayDown = Math.min(SWAY_RELEASE_FR, this.swayDown + 1); this.swayUp = 0; }

      const target = this.vadOn ? this.swayUp / SWAY_ATTACK_FR : 1 - this.swayDown / SWAY_RELEASE_FR;
      this.swayEnv += CFG.ENV_FOLLOW_GAIN * (target - this.swayEnv);
      this.swayEnv = Math.max(0, Math.min(1, this.swayEnv));

      const loud = loudnessGain(db) * CFG.MASTER;
      const env = this.swayEnv;
      this.time += CFG.HOP_MS / 1000;
      const t = this.time;
      const d = Math.PI / 180;

      this.outputQueue.push({
        x: (CFG.AMP_X_MM / 1000) * loud * env * Math.sin(2 * Math.PI * CFG.FREQ_X * t + this.phaseX),
        y: (CFG.AMP_Y_MM / 1000) * loud * env * Math.sin(2 * Math.PI * CFG.FREQ_Y * t + this.phaseY),
        z: (CFG.AMP_Z_MM / 1000) * loud * env * Math.sin(2 * Math.PI * CFG.FREQ_Z * t + this.phaseZ),
        roll: CFG.AMP_ROLL_DEG * d * loud * env * Math.sin(2 * Math.PI * CFG.FREQ_ROLL * t + this.phaseRoll),
        pitch: CFG.AMP_PITCH_DEG * d * loud * env * Math.sin(2 * Math.PI * CFG.FREQ_PITCH * t + this.phasePitch),
        yaw: CFG.AMP_YAW_DEG * d * loud * env * Math.sin(2 * Math.PI * CFG.FREQ_YAW * t + this.phaseYaw),
      });

      this._consumeFromRing(HOP);
    }
    if (this.outputQueue.length > 100) this.outputQueue = this.outputQueue.slice(-100);
  }

  update(_dt: number): void {
    // Pop one result per tick (100Hz = 10ms, matching HOP_MS).
    // When queue is empty, hold the last pose - no decay.
    // Python's HeadWobbler holds offsets until reset(); the SwayRollRT's
    // own envelope follower handles fade-out via VAD release.
    if (this.outputQueue.length > 0) {
      this.currentPose = this.outputQueue.shift()!;
    }
  }

  getMat4(): Mat4 {
    const p = this.currentPose;
    if (p.x === 0 && p.y === 0 && p.z === 0 && p.roll === 0 && p.pitch === 0 && p.yaw === 0) {
      return mat4Identity();
    }
    return createHeadPoseMat4(p.x, p.y, p.z, p.roll, p.pitch, p.yaw);
  }

  getPose(): HeadPose {
    return { ...this.currentPose };
  }

  getQueueSize(): number {
    return this.outputQueue.length;
  }

  reset(): void {
    super.reset();
    this.ringWritePos = 0;
    this.ringReadPos = 0;
    this.ringLength = 0;
    this.vadOn = false;
    this.vadAbove = 0;
    this.vadBelow = 0;
    this.swayEnv = 0;
    this.swayUp = 0;
    this.swayDown = 0;
    this.time = 0;
    this.outputQueue = [];
    this.currentPose = createNeutralPose();
  }
}
