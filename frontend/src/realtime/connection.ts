import type { SessionConfig } from "./session";

export interface RealtimeConnection {
  pc: RTCPeerConnection;
  dc: RTCDataChannel;
  audioEl: HTMLAudioElement;
  close: () => void;
}

const DEFAULT_MODEL = "gpt-4o-realtime-preview-2024-12-17";
const DEBUG = import.meta.env.DEV;

/**
 * Establish a WebRTC connection to the OpenAI Realtime API.
 *
 * Flow:
 * 1. POST /v1/realtime/sessions  -> ephemeral client_secret
 * 2. Create RTCPeerConnection + mic + DataChannel
 * 3. Wait for ICE gathering to complete
 * 4. POST /v1/realtime?model=...  with SDP offer -> SDP answer
 */
export async function connectRealtime(
  apiKey: string,
  sessionConfig: Partial<SessionConfig> = {},
): Promise<RealtimeConnection> {
  const model = sessionConfig.model ?? DEFAULT_MODEL;

  // Step 1: Get an ephemeral token
  if (DEBUG) console.log("[Realtime] Creating ephemeral session...");
  const sessionRes = await fetch("https://api.openai.com/v1/realtime/sessions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      voice: sessionConfig.voice ?? "cedar",
      instructions: sessionConfig.instructions ?? "",
      input_audio_transcription: { model: "whisper-1" },
      turn_detection: {
        type: "server_vad",
        threshold: 0.7,
        silence_duration_ms: 800,
        prefix_padding_ms: 300,
        create_response: true,
      },
      tools: (sessionConfig.tools ?? []).map((t) => ({
        type: "function",
        name: t.name,
        description: t.description,
        parameters: t.parameters,
      })),
    }),
  });

  if (!sessionRes.ok) {
    const text = await sessionRes.text();
    throw new Error(`Failed to create session (${sessionRes.status}): ${text}`);
  }

  const sessionData = await sessionRes.json();
  const ephemeralKey: string = sessionData.client_secret?.value;
  if (!ephemeralKey) {
    throw new Error("No ephemeral key in session response");
  }
  if (DEBUG) console.log("[Realtime] Got ephemeral key");

  // Step 2: Create peer connection
  const pc = new RTCPeerConnection();

  const audioEl = document.createElement("audio");
  audioEl.autoplay = true;
  audioEl.style.display = "none";
  document.body.appendChild(audioEl);
  pc.ontrack = (e) => {
    if (DEBUG) console.log("[Realtime] Got remote track:", e.track.kind);
    audioEl.srcObject = e.streams[0];
    audioEl.play().catch(() => {});
  };

  // Local mic
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  pc.addTrack(stream.getAudioTracks()[0], stream);

  // DataChannel
  const dc = pc.createDataChannel("oai-events");

  // Step 3: Create offer and wait for ICE gathering
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  await new Promise<void>((resolve) => {
    if (pc.iceGatheringState === "complete") {
      resolve();
      return;
    }
    pc.onicegatheringstatechange = () => {
      if (pc.iceGatheringState === "complete") resolve();
    };
    setTimeout(resolve, 3000);
  });

  // Step 4: SDP exchange
  const localSdp = pc.localDescription?.sdp;
  if (!localSdp) {
    pc.close();
    stream.getTracks().forEach((t) => t.stop());
    audioEl.remove();
    throw new Error("No local SDP after ICE gathering");
  }

  const sdpRes = await fetch(
    `https://api.openai.com/v1/realtime?model=${encodeURIComponent(model)}`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${ephemeralKey}`,
        "Content-Type": "application/sdp",
      },
      body: localSdp,
    },
  );

  if (!sdpRes.ok) {
    pc.close();
    stream.getTracks().forEach((t) => t.stop());
    audioEl.srcObject = null;
    audioEl.remove();
    const text = await sdpRes.text();
    throw new Error(`SDP exchange failed (${sdpRes.status}): ${text}`);
  }

  const answerSdp = await sdpRes.text();
  await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });
  if (DEBUG) console.log("[Realtime] WebRTC connected");

  const close = () => {
    dc.close();
    pc.close();
    stream.getTracks().forEach((t) => t.stop());
    audioEl.pause();
    audioEl.srcObject = null;
    audioEl.remove();
  };

  return { pc, dc, audioEl, close };
}
