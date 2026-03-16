import { type VoiceEventBus } from "./eventBus";

const COOLDOWN_MS = 800;

/**
 * Mutes the mic track during model speech to prevent echo feedback.
 * Listens to tts:start / tts:done / tts:stop events from the bus.
 */
export class AudioGate {
  private bus: VoiceEventBus;
  private _shouldListen = true;
  private _cooldownTimer: ReturnType<typeof setTimeout> | null = null;
  private _unsubs: (() => void)[] = [];
  private _micTrack: MediaStreamTrack | null = null;

  get shouldListen() {
    return this._shouldListen;
  }

  constructor(bus: VoiceEventBus) {
    this.bus = bus;
    this._unsubs.push(
      bus.on("tts:start", () => this._mute()),
      bus.on("tts:done", () => this._startCooldown()),
      bus.on("tts:stop", () => this._unmute()),
    );
  }

  setMicTrack(track: MediaStreamTrack | null) {
    this._micTrack = track;
    if (track) track.enabled = this._shouldListen;
  }

  private _mute() {
    if (this._cooldownTimer) {
      clearTimeout(this._cooldownTimer);
      this._cooldownTimer = null;
    }
    this._shouldListen = false;
    if (this._micTrack) this._micTrack.enabled = false;
    this.bus.emit("gate:muted", {});
  }

  private _unmute() {
    if (this._cooldownTimer) {
      clearTimeout(this._cooldownTimer);
      this._cooldownTimer = null;
    }
    this._shouldListen = true;
    if (this._micTrack) this._micTrack.enabled = true;
    this.bus.emit("gate:unmuted", {});
  }

  private _startCooldown() {
    this._cooldownTimer = setTimeout(() => {
      this._cooldownTimer = null;
      this._unmute();
    }, COOLDOWN_MS);
  }

  dispose() {
    this._unsubs.forEach((fn) => fn());
    if (this._cooldownTimer) clearTimeout(this._cooldownTimer);
  }
}
