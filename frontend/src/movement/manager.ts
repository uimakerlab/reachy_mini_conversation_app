import { CONFIG, type HeadPose, clampPose, clampAntennas, smoothStep } from "./types";
import { PoseComposer } from "./composer";
import * as connection from "./connection";
import { BreathingGenerator } from "./generators/breathing";
import { SpeechSwayGenerator } from "./generators/speechSway";
import { GotoGenerator } from "./generators/goto";
import { EmotionsGenerator } from "./generators/emotions";
import { TrackingGenerator } from "./generators/tracking";

const ANTENNA_SPEAKING = 0.4; // radians (~23 deg) when bot is speaking
const ANTENNA_SMOOTHING = 0.14;

export class MovementManager {
  private daemonUrl: string;
  private idleDelay: number;
  private composer = new PoseComposer();
  private gotoGen = new GotoGenerator();
  private breathGen = new BreathingGenerator();
  private swayGen = new SpeechSwayGenerator();
  private emotGen = new EmotionsGenerator();
  private trackGen = new TrackingGenerator();
  private targetAntennas: [number, number] = [0, 0];
  private currentAntennas: [number, number] = [0, 0];
  private running = false;
  private connected = false;
  private listening = false;
  private speaking = false;
  private lastActivityTime = 0;
  private lastLoopTime = 0;
  private loopInterval: ReturnType<typeof setInterval> | null = null;

  onConnect: (() => void) | null = null;
  onDisconnect: (() => void) | null = null;

  constructor(opts: { daemonUrl?: string; idleDelay?: number } = {}) {
    this.daemonUrl = opts.daemonUrl ?? CONFIG.DEFAULT_DAEMON_URL;
    this.idleDelay = opts.idleDelay ?? CONFIG.IDLE_DELAY_MS;
  }

  async start(): Promise<boolean> {
    if (this.running) return true;
    connection.configure({
      daemonUrl: this.daemonUrl,
      onConnect: () => { this.connected = true; this.onConnect?.(); },
      onDisconnect: () => { this.connected = false; this.onDisconnect?.(); },
    });
    await connection.connect();
    this.running = true;
    this.lastLoopTime = performance.now();
    this.loopInterval = setInterval(() => this._tick(), 1000 / CONFIG.LOOP_FREQUENCY);
    this._markActivity();
    return true;
  }

  stop(): void {
    this.running = false;
    if (this.loopInterval) { clearInterval(this.loopInterval); this.loopInterval = null; }
    this.breathGen.stopImmediate();
    this.swayGen.reset();
    this.gotoGen.stop();
    this.emotGen.stop();
    this.trackGen.reset();
    this.targetAntennas = [0, 0];
    this.currentAntennas = [0, 0];
    this.speaking = false;
    connection.disconnect();
  }

  private _tick(): void {
    const now = performance.now();
    const dt = (now - this.lastLoopTime) / 1000;
    this.lastLoopTime = now;

    this.gotoGen.update(dt);
    this.trackGen.update(dt);
    this.emotGen.update(dt);
    this._updateIdle(now);
    this.breathGen.update(dt);
    this.swayGen.update(dt);

    this.composer.setPrimary(this.gotoGen.getPose());

    if (this.breathGen.isActive() && !this.gotoGen.isActive()) this.composer.setSecondary("breathing", this.breathGen.getPose());
    else this.composer.removeSecondary("breathing");

    if (this.trackGen.isActive()) this.composer.setSecondary("tracking", this.trackGen.getPose());
    else this.composer.removeSecondary("tracking");

    if (this.swayGen.isActive()) this.composer.setSecondary("speechSway", this.swayGen.getPose());
    else this.composer.removeSecondary("speechSway");

    if (this.emotGen.isActive()) this.composer.setSecondary("emotions", this.emotGen.getPose());
    else this.composer.removeSecondary("emotions");

    // Antennas: driven by explicit speaking/emotion state (not swayGen analysis)
    if (this.emotGen.isActive()) {
      const a = this.emotGen.getAntennas();
      this.targetAntennas[0] = a[0];
      this.targetAntennas[1] = a[1];
    } else if (this.speaking) {
      this.targetAntennas[0] = ANTENNA_SPEAKING;
      this.targetAntennas[1] = ANTENNA_SPEAKING;
    } else if (this.listening) {
      this.targetAntennas[0] = this.currentAntennas[0];
      this.targetAntennas[1] = this.currentAntennas[1];
    } else {
      this.targetAntennas[0] = 0;
      this.targetAntennas[1] = 0;
    }

    const s = smoothStep(ANTENNA_SMOOTHING, dt);
    this.currentAntennas[0] += (this.targetAntennas[0] - this.currentAntennas[0]) * s;
    this.currentAntennas[1] += (this.targetAntennas[1] - this.currentAntennas[1]) * s;

    const safePose = clampPose(this.composer.compose());
    const safeAnt = clampAntennas(this.currentAntennas);
    connection.sendFullBodyPose({ head: safePose, antennas: safeAnt, bodyYaw: 0 });
  }

  private _updateIdle(now: number): void {
    if (this.gotoGen.isActive()) { this.breathGen.stop(); }
    else if (now - this.lastActivityTime > this.idleDelay && !this.breathGen.isActive()) { this.breathGen.start(); }
  }

  private _markActivity(): void { this.lastActivityTime = performance.now(); }

  allowBreathingNow(): void {
    if (!this.running || this.gotoGen.isActive()) return;
    this.lastActivityTime = performance.now() - (this.idleDelay + 100);
  }

  // -- Public API --

  setSpeaking(v: boolean): void { this.speaking = v; }

  goto(target: HeadPose | string, opts: { duration?: number; onComplete?: () => void } = {}): void {
    this._markActivity();
    this.breathGen.stop();
    this.gotoGen.goto(target, opts);
  }

  lookAt(direction: string, duration = 1): void { this.goto(direction, { duration }); }
  lookFront(duration = 0.8): void { this.goto("front", { duration }); }

  async nod(count = 2, speed = 0.25): Promise<void> {
    this._markActivity();
    for (let i = 0; i < count; i++) {
      await this._gotoWait("down", speed);
      await this._gotoWait("front", speed);
    }
  }

  async shake(count = 2, speed = 0.25): Promise<void> {
    this._markActivity();
    for (let i = 0; i < count; i++) {
      await this._gotoWait("left", speed);
      await this._gotoWait("right", speed);
    }
    await this._gotoWait("front", speed);
  }

  private _gotoWait(target: string, duration: number): Promise<void> {
    return new Promise((resolve) => this.gotoGen.goto(target, { duration, onComplete: resolve }));
  }

  feedSpeechAudio(samples: Float32Array, sampleRate = 24000): void {
    this._markActivity();
    this.swayGen.feedSamples(samples, sampleRate);
  }

  stopSpeechSway(): void { this.swayGen.reset(); }

  setEmotion(name: string, intensity = 1): boolean { return this.emotGen.setEmotion(name, intensity); }
  clearEmotion(): void { this.emotGen.clear(); }
  getCurrentEmotion(): string | null { return this.emotGen.getCurrentEmotion(); }

  setTrackingTarget(x: number, y: number): void { this.trackGen.setTarget(x, y); }
  clearTrackingTarget(): void { this.trackGen.clearTarget(); }

  setListening(v: boolean): void { this.listening = v; }
  getIsConnected(): boolean { return this.connected; }
  getIsRunning(): boolean { return this.running; }
}
