import { type VoiceEventBus } from "./eventBus";

const GRACE_PERIOD_MS = 500;

/**
 * Detects user speech during model output and triggers an interrupt.
 * In Realtime WebRTC mode, barge-in sends response.cancel via the adapter.
 */
export class BargeIn {
  private bus: VoiceEventBus;
  private _monitoring = false;
  private _graceStart = 0;
  private _onTrigger: (() => void) | null = null;
  private _unsubs: (() => void)[] = [];

  constructor(bus: VoiceEventBus, onTrigger: () => void) {
    this.bus = bus;
    this._onTrigger = onTrigger;

    this._unsubs.push(
      bus.on("tts:start", () => {
        this._monitoring = true;
        this._graceStart = Date.now();
      }),
      bus.on("tts:done", () => {
        this._monitoring = false;
      }),
      bus.on("tts:stop", () => {
        this._monitoring = false;
      }),
      bus.on("vad:start", () => {
        if (this._monitoring && Date.now() - this._graceStart > GRACE_PERIOD_MS) {
          this._doTrigger();
        }
      }),
    );
  }

  private _doTrigger() {
    this._monitoring = false;
    this.bus.emit("bargein:trigger", {});
    this._onTrigger?.();
  }

  dispose() {
    this._unsubs.forEach((fn) => fn());
    this._onTrigger = null;
  }
}
