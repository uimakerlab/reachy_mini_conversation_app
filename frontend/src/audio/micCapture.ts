/**
 * Captures microphone audio, resamples to 24 kHz PCM int16, and
 * delivers chunks via a callback for WebSocket transmission.
 */

const PROCESSOR_CODE = `
class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = [];
    this._chunkSamples = 960;
  }
  process(inputs) {
    const ch = inputs[0]?.[0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) this._buf.push(ch[i]);
    while (this._buf.length >= this._chunkSamples) {
      this.port.postMessage(new Float32Array(this._buf.splice(0, this._chunkSamples)));
    }
    return true;
  }
}
registerProcessor('mic-capture', MicCaptureProcessor);
`;

const TARGET_RATE = 24000;

function resampleToInt16(
  input: Float32Array,
  fromRate: number,
  toRate: number,
): Int16Array {
  if (fromRate === toRate) {
    const out = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      out[i] = Math.max(-32768, Math.min(32767, Math.round(input[i] * 32767)));
    }
    return out;
  }
  const ratio = toRate / fromRate;
  const outLen = Math.round(input.length * ratio);
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcIdx = i / ratio;
    const lo = Math.floor(srcIdx);
    const hi = Math.min(lo + 1, input.length - 1);
    const frac = srcIdx - lo;
    const sample = input[lo] * (1 - frac) + input[hi] * frac;
    out[i] = Math.max(-32768, Math.min(32767, Math.round(sample * 32767)));
  }
  return out;
}

export class MicCapture {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private worklet: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private onChunk: ((pcm: ArrayBuffer) => void) | null = null;

  async start(onChunk: (pcmInt16: ArrayBuffer) => void): Promise<MediaStream> {
    this.onChunk = onChunk;

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    this.stream = stream;

    const ctx = new AudioContext();
    this.ctx = ctx;

    const blobUrl = URL.createObjectURL(
      new Blob([PROCESSOR_CODE], { type: "application/javascript" }),
    );
    await ctx.audioWorklet.addModule(blobUrl);
    URL.revokeObjectURL(blobUrl);

    const source = ctx.createMediaStreamSource(stream);
    this.source = source;

    const worklet = new AudioWorkletNode(ctx, "mic-capture");
    this.worklet = worklet;

    const nativeRate = ctx.sampleRate;
    worklet.port.onmessage = (e: MessageEvent<Float32Array>) => {
      const int16 = resampleToInt16(e.data, nativeRate, TARGET_RATE);
      this.onChunk?.(int16.buffer);
    };

    source.connect(worklet);
    return stream;
  }

  setMuted(muted: boolean): void {
    this.stream?.getAudioTracks().forEach((t) => {
      t.enabled = !muted;
    });
  }

  isMuted(): boolean {
    const track = this.stream?.getAudioTracks()[0];
    return track ? !track.enabled : false;
  }

  getStream(): MediaStream | null {
    return this.stream;
  }

  stop(): void {
    this.worklet?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ctx?.close();
    this.worklet = null;
    this.source = null;
    this.stream = null;
    this.ctx = null;
    this.onChunk = null;
  }
}
