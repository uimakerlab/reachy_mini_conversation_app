/**
 * Streams PCM int16 24 kHz audio received from the backend WebSocket.
 *
 * Uses AudioBufferSourceNode scheduling for gapless playback.
 * The AudioContext handles upsampling from 24 kHz to the device's native rate.
 */

const INPUT_RATE = 24000;
const SILENCE_TIMEOUT_MS = 300;

export class PcmPlayer {
  private ctx: AudioContext;
  private gainNode: GainNode;
  private nextTime = 0;
  private sources: AudioBufferSourceNode[] = [];
  private onAudioStart: (() => void) | null = null;
  private onAudioEnd: (() => void) | null = null;
  private isPlaying = false;
  private silenceTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(opts?: { onAudioStart?: () => void; onAudioEnd?: () => void }) {
    this.ctx = new AudioContext();
    this.gainNode = this.ctx.createGain();
    this.gainNode.connect(this.ctx.destination);
    this.onAudioStart = opts?.onAudioStart ?? null;
    this.onAudioEnd = opts?.onAudioEnd ?? null;
  }

  async resume(): Promise<void> {
    if (this.ctx.state === "suspended") await this.ctx.resume();
  }

  /** Push a PCM int16 chunk (24 kHz mono) for playback. */
  pushChunk(pcmBytes: ArrayBuffer): void {
    const int16 = new Int16Array(pcmBytes);
    if (int16.length === 0) return;

    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

    const buffer = this.ctx.createBuffer(1, float32.length, INPUT_RATE);
    buffer.copyToChannel(float32, 0);

    const source = this.ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.gainNode);

    const now = this.ctx.currentTime;
    const startTime = Math.max(now, this.nextTime);
    source.start(startTime);
    this.nextTime = startTime + buffer.duration;

    this.sources.push(source);
    source.onended = () => {
      const idx = this.sources.indexOf(source);
      if (idx >= 0) this.sources.splice(idx, 1);
    };

    if (!this.isPlaying) {
      this.isPlaying = true;
      this.onAudioStart?.();
    }
    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    this.silenceTimer = setTimeout(() => {
      this.isPlaying = false;
      this.onAudioEnd?.();
    }, SILENCE_TIMEOUT_MS);
  }

  /** Immediately stop all scheduled audio (used on interrupt). */
  clear(): void {
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {
        /* already stopped */
      }
    }
    this.sources = [];
    this.nextTime = 0;
    if (this.silenceTimer) {
      clearTimeout(this.silenceTimer);
      this.silenceTimer = null;
    }
    if (this.isPlaying) {
      this.isPlaying = false;
      this.onAudioEnd?.();
    }
  }

  stop(): void {
    this.clear();
    this.ctx.close();
  }
}
