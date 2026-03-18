import { CONFIG, type HeadPose, clampAntennas } from "./types";
import { type Mat4, mat4Identity, mat4Clone, mat4ToEuler } from "./mat4";
import { PoseComposer } from "./composer";
import * as connection from "./connection";
import { BreathingGenerator } from "./generators/breathing";
import { SpeechSwayGenerator } from "./generators/speechSway";
import { GotoGenerator } from "./generators/goto";
import { EmotionsGenerator } from "./generators/emotions";
import { TrackingGenerator } from "./generators/tracking";

const ANTENNA_SPEAKING = 0.4; // radians (~23 deg) when bot is speaking

// Python-matching constants
const ANTENNA_BLEND_DURATION = 0.4;   // seconds to blend antennas back after listening
const LISTENING_DEBOUNCE_S = 0.15;    // seconds between listening state changes

export class MovementManager {
  private daemonUrl: string;
  private idleDelay: number;
  private composer = new PoseComposer();
  private gotoGen = new GotoGenerator();
  private breathGen = new BreathingGenerator();
  private swayGen = new SpeechSwayGenerator();
  private emotGen = new EmotionsGenerator();
  private trackGen = new TrackingGenerator();

  // Antenna state (linear blend matching Python)
  private lastCommandedAntennas: [number, number] = [0, 0];
  private listeningAntennas: [number, number] = [0, 0];
  private antennaUnfreezeBlend = 1.0;
  private lastListeningBlendTime = 0;

  // Last commanded head pose (for breathing start interpolation)
  private lastCommandedHead: Mat4 = mat4Identity();

  private running = false;
  private connected = false;
  private listening = false;
  private speaking = false;
  private lastActivityTime = 0;
  private lastLoopTime = 0;
  private lastListeningToggleTime = 0;
  private breathingActive = false;
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
    const now = performance.now();
    this.lastLoopTime = now;
    this.lastListeningBlendTime = now;
    this.lastListeningToggleTime = now;
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
    this.lastCommandedAntennas = [0, 0];
    this.listeningAntennas = [0, 0];
    this.antennaUnfreezeBlend = 1.0;
    this.speaking = false;
    this.breathingActive = false;
    connection.disconnect();
  }

  private _tick(): void {
    const now = performance.now();
    const dt = (now - this.lastLoopTime) / 1000;
    this.lastLoopTime = now;

    // 1) Update generators
    this.gotoGen.update(dt);
    this.trackGen.update(dt);
    this.emotGen.update(dt);
    this._updateIdle(now);
    this.breathGen.update(dt);
    this.swayGen.update(dt);

    // 2) Build primary pose (goto takes priority, then breathing)
    this.composer.setPrimary(this.gotoGen.getMat4());

    // 3) Apply secondary offsets via matrix composition
    if (this.breathGen.isActive() && !this.gotoGen.isActive()) {
      this.composer.setSecondary("breathing", this.breathGen.getMat4());
    } else {
      this.composer.removeSecondary("breathing");
    }

    if (this.trackGen.isActive()) {
      this.composer.setSecondary("tracking", this.trackGen.getMat4());
    } else {
      this.composer.removeSecondary("tracking");
    }

    if (this.swayGen.isActive()) {
      this.composer.setSecondary("speechSway", this.swayGen.getMat4());
    } else {
      this.composer.removeSecondary("speechSway");
    }

    if (this.emotGen.isActive()) {
      this.composer.setSecondary("emotions", this.emotGen.getMat4());
    } else {
      this.composer.removeSecondary("emotions");
    }

    // 4) Compose the final head pose (Mat4)
    const composedHead = this.composer.compose();

    // 5) Calculate target antennas (emotions > speaking > breathing > neutral)
    let targetAntennas: [number, number];
    if (this.emotGen.isActive()) {
      targetAntennas = this.emotGen.getAntennas();
    } else if (this.breathGen.isActive() && !this.gotoGen.isActive()) {
      targetAntennas = this.breathGen.getAntennas();
    } else if (this.speaking) {
      targetAntennas = [ANTENNA_SPEAKING, ANTENNA_SPEAKING];
    } else {
      targetAntennas = [0, 0];
    }

    // 6) Apply listening antenna freeze/blend (linear, matching Python)
    const antennasCmd = this._calculateBlendedAntennas(targetAntennas, now);

    // 7) Send to daemon
    const safeAnt = clampAntennas(antennasCmd);
    connection.sendFullBodyPose({ head: composedHead, antennas: safeAnt, bodyYaw: 0 });

    // 8) Store last commanded state for future reference
    this.lastCommandedHead = mat4Clone(composedHead);
    this.lastCommandedAntennas = [...safeAnt] as [number, number];
  }

  /**
   * Linear antenna blending matching Python's _calculate_blended_antennas.
   * When listening: freeze antennas at snapshot.
   * When unfreezing: linear blend over ANTENNA_BLEND_DURATION.
   */
  private _calculateBlendedAntennas(
    targetAntennas: [number, number],
    nowMs: number,
  ): [number, number] {
    const now = nowMs;
    const lastUpdate = this.lastListeningBlendTime;
    this.lastListeningBlendTime = now;

    if (this.listening) {
      // Freeze: return the snapshot taken when listening started
      this.antennaUnfreezeBlend = 0;
      return [...this.listeningAntennas] as [number, number];
    }

    // Not listening: blend from frozen position back to target
    const dtSec = Math.max(0, (now - lastUpdate) / 1000);
    let blend = this.antennaUnfreezeBlend;
    if (ANTENNA_BLEND_DURATION <= 0) {
      blend = 1.0;
    } else {
      blend = Math.min(1.0, blend + dtSec / ANTENNA_BLEND_DURATION);
    }
    this.antennaUnfreezeBlend = blend;

    const result: [number, number] = [
      this.listeningAntennas[0] * (1 - blend) + targetAntennas[0] * blend,
      this.listeningAntennas[1] * (1 - blend) + targetAntennas[1] * blend,
    ];

    // Once fully blended, sync the frozen reference
    if (blend >= 1.0) {
      this.listeningAntennas = [...targetAntennas] as [number, number];
    }

    return result;
  }

  private _updateIdle(now: number): void {
    if (this.gotoGen.isActive()) {
      // Active goto cancels breathing
      if (this.breathingActive) {
        this.breathGen.stop();
        this.breathingActive = false;
      }
    } else if (
      !this.breathingActive &&
      !this.listening &&
      now - this.lastActivityTime > this.idleDelay
    ) {
      // Start breathing with interpolation from current pose
      this.breathingActive = true;
      this.breathGen.startFrom(
        mat4Clone(this.lastCommandedHead),
        [...this.lastCommandedAntennas] as [number, number],
      );
      this._markActivity();
    }

    // If a new move interrupted breathing, cancel it
    if (this.breathGen.isActive() && this.gotoGen.isActive()) {
      this.breathGen.stop();
      this.breathingActive = false;
    }

    // Non-breathing move cancels the breathing flag
    if (this.gotoGen.isActive()) {
      this.breathingActive = false;
    }
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
    this.breathingActive = false;
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

  /**
   * Set listening state with debounce matching Python's 0.15s threshold.
   * When listening starts: freeze antennas, suppress breathing.
   * When listening stops: begin blend-back.
   */
  setListening(v: boolean): void {
    if (this.listening === v) return;

    const now = performance.now();
    if ((now - this.lastListeningToggleTime) / 1000 < LISTENING_DEBOUNCE_S) return;
    this.lastListeningToggleTime = now;

    this.listening = v;
    this.lastListeningBlendTime = now;

    if (v) {
      // Freeze: snapshot current commanded antennas
      this.listeningAntennas = [...this.lastCommandedAntennas] as [number, number];
      this.antennaUnfreezeBlend = 0;
    } else {
      // Unfreeze: start blending from frozen position
      this.antennaUnfreezeBlend = 0;
    }

    this._markActivity();
  }

  getIsConnected(): boolean { return this.connected; }
  getIsRunning(): boolean { return this.running; }

  getLastCommandedEuler(): { x: number; y: number; z: number; roll: number; pitch: number; yaw: number } {
    return mat4ToEuler(this.lastCommandedHead);
  }
}
