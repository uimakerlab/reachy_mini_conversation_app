/**
 * WebRTC session for bidirectional audio with the fastrtc backend.
 *
 * Handles SDP signaling, microphone capture, and remote audio playback.
 * Browser AEC works natively because both mic and speaker go through
 * the same RTCPeerConnection, which the browser optimizes for echo cancellation.
 */

import { voiceEventBus } from "../voice/eventBus";

const SIGNALING_PATH = "/webrtc/offer";
const SILENCE_THRESHOLD = 0.005;
const SILENCE_TIMEOUT_MS = 400;

export class WebRTCSession {
  private pc: RTCPeerConnection | null = null;
  private localStream: MediaStream | null = null;
  private remoteAudio: HTMLAudioElement | null = null;
  private ttsAnalyser: AnalyserNode | null = null;
  private ttsCtx: AudioContext | null = null;
  private ttsRafId: number | null = null;
  private isSpeaking = false;
  private silenceTimer: ReturnType<typeof setTimeout> | null = null;
  readonly webrtcId: string;

  constructor(webrtcId: string) {
    this.webrtcId = webrtcId;
  }

  /**
   * Establish WebRTC connection with the server.
   * Returns the local MediaStream (for the mic visualizer in AudioControls).
   */
  async connect(baseUrl: string): Promise<MediaStream> {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: { exact: true },
        noiseSuppression: { exact: true },
        autoGainControl: { exact: true },
        sampleRate: { ideal: 24000 },
        sampleSize: { ideal: 16 },
        channelCount: { exact: 1 },
      },
    });
    this.localStream = stream;

    const pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });
    this.pc = pc;

    stream.getAudioTracks().forEach((track) => pc.addTrack(track, stream));

    // fastrtc requires a data channel to unblock the handler's start_up()
    pc.createDataChannel("chat");

    pc.ontrack = (event) => {
      const remoteStream = event.streams[0];
      if (!remoteStream) return;

      const audio = document.createElement("audio");
      audio.srcObject = remoteStream;
      audio.autoplay = true;
      this.remoteAudio = audio;

      this._monitorRemoteAudio(remoteStream);
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    await this._waitForIceGathering(pc);

    const signalUrl = `${baseUrl.replace(/\/$/, "")}${SIGNALING_PATH}`;
    const resp = await fetch(signalUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: pc.localDescription!.sdp,
        type: pc.localDescription!.type,
        webrtc_id: this.webrtcId,
      }),
    });

    const answer = await resp.json();

    if (!resp.ok || answer.status === "failed") {
      const detail = answer?.meta?.error ?? `HTTP ${resp.status}`;
      throw new Error(`WebRTC signaling failed: ${detail}`);
    }

    await pc.setRemoteDescription(
      new RTCSessionDescription({ sdp: answer.sdp, type: answer.type }),
    );

    return stream;
  }

  disconnect(): void {
    this._stopTtsMonitor();

    if (this.remoteAudio) {
      this.remoteAudio.srcObject = null;
      this.remoteAudio = null;
    }

    this.localStream?.getTracks().forEach((t) => t.stop());
    this.localStream = null;

    this.pc?.close();
    this.pc = null;
  }

  setMuted(muted: boolean): void {
    this.localStream?.getAudioTracks().forEach((t) => {
      t.enabled = !muted;
    });
  }

  isMutedState(): boolean {
    const track = this.localStream?.getAudioTracks()[0];
    return track ? !track.enabled : false;
  }

  getLocalStream(): MediaStream | null {
    return this.localStream;
  }

  /**
   * Monitor the remote audio stream level to emit tts:start / tts:done events.
   * This replaces the PcmPlayer's onAudioStart / onAudioEnd callbacks.
   */
  private _monitorRemoteAudio(stream: MediaStream): void {
    try {
      const ctx = new AudioContext();
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      this.ttsCtx = ctx;
      this.ttsAnalyser = analyser;

      const data = new Float32Array(analyser.fftSize);

      const tick = () => {
        if (!this.ttsAnalyser) return;
        analyser.getFloatTimeDomainData(data);

        let rms = 0;
        for (let i = 0; i < data.length; i++) rms += data[i] * data[i];
        rms = Math.sqrt(rms / data.length);

        if (rms > SILENCE_THRESHOLD) {
          if (this.silenceTimer) {
            clearTimeout(this.silenceTimer);
            this.silenceTimer = null;
          }
          if (!this.isSpeaking) {
            this.isSpeaking = true;
            voiceEventBus.emit("tts:start", {});
          }
        } else if (this.isSpeaking) {
          if (!this.silenceTimer) {
            this.silenceTimer = setTimeout(() => {
              this.isSpeaking = false;
              this.silenceTimer = null;
              voiceEventBus.emit("bot:audio_silent", {});
            }, SILENCE_TIMEOUT_MS);
          }
        }

        this.ttsRafId = requestAnimationFrame(tick);
      };

      this.ttsRafId = requestAnimationFrame(tick);
    } catch {
      // AudioContext unavailable
    }
  }

  private _stopTtsMonitor(): void {
    if (this.ttsRafId !== null) {
      cancelAnimationFrame(this.ttsRafId);
      this.ttsRafId = null;
    }
    if (this.silenceTimer) {
      clearTimeout(this.silenceTimer);
      this.silenceTimer = null;
    }
    this.ttsAnalyser = null;
    this.ttsCtx?.close();
    this.ttsCtx = null;
    this.isSpeaking = false;
  }

  private _waitForIceGathering(pc: RTCPeerConnection): Promise<void> {
    if (pc.iceGatheringState === "complete") return Promise.resolve();

    return new Promise((resolve) => {
      const timeout = setTimeout(resolve, 2000);
      pc.onicegatheringstatechange = () => {
        if (pc.iceGatheringState === "complete") {
          clearTimeout(timeout);
          resolve();
        }
      };
    });
  }
}
