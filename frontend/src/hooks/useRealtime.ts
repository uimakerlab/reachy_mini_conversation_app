import { useState, useRef, useCallback, useEffect } from "react";
import { connectRealtime, type RealtimeConnection } from "../realtime/connection";
import { RealtimeAdapter } from "../realtime/adapter";
import { voiceEventBus } from "../voice/eventBus";
import { AudioGate } from "../voice/audioGate";
import { BargeIn } from "../voice/bargeIn";
import { executeTool, getToolDefinitions, configureTools } from "../tools";
import { buildInstructions, resolveVoice, resolveEnabledTools } from "../config/prompts";
import { MovementManager } from "../movement/manager";
import type { AppSettings } from "../config/settings";
import type { useChat } from "./useChat";

export type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

const DEBUG = import.meta.env.DEV;

interface CleanupRefs {
  swayInterval: ReturnType<typeof setInterval>;
  audioCtx: AudioContext;
  eventUnsubs: (() => void)[];
}

export function useRealtime(
  settings: AppSettings,
  chat: ReturnType<typeof useChat>,
) {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [error, setError] = useState<string | null>(null);
  const [robotConnected, setRobotConnected] = useState(false);
  const [isMuted, setIsMuted] = useState(false);

  const connRef = useRef<RealtimeConnection | null>(null);
  const adapterRef = useRef<RealtimeAdapter | null>(null);
  const gateRef = useRef<AudioGate | null>(null);
  const bargeinRef = useRef<BargeIn | null>(null);
  const mgrRef = useRef<MovementManager | null>(null);
  const cleanupRef = useRef<CleanupRefs | null>(null);
  const micTrackRef = useRef<MediaStreamTrack | null>(null);

  // Stabilize chat handlers so `connect` callback doesn't recreate on every render
  const chatRef = useRef(chat);
  chatRef.current = chat;

  const connect = useCallback(async () => {
    if (DEBUG) console.log("[useRealtime] connect()");
    if (connRef.current) return;
    if (!settings.openaiApiKey) {
      setError("No API key configured");
      setStatus("error");
      return;
    }

    setStatus("connecting");
    setError(null);
    chatRef.current.clear();

    const eventUnsubs: (() => void)[] = [];

    try {
      const mgr = new MovementManager({ daemonUrl: settings.daemonUrl });
      mgr.onConnect = () => setRobotConnected(true);
      mgr.onDisconnect = () => setRobotConnected(false);
      await mgr.start();
      mgrRef.current = mgr;

      configureTools({ manager: mgr, daemonUrl: settings.daemonUrl });

      const cp = settings.customProfiles;
      const enabledTools = resolveEnabledTools(settings.profileId, settings.customEnabledTools, cp);

      const conn = await connectRealtime(settings.openaiApiKey, {
        voice: resolveVoice(settings.profileId, settings.voice, cp),
        instructions: buildInstructions(settings.profileId, settings.customInstructions, cp),
        tools: getToolDefinitions(enabledTools),
      });
      connRef.current = conn;

      const gate = new AudioGate(voiceEventBus);
      const micTrack = conn.pc.getSenders().find((s) => s.track?.kind === "audio")?.track ?? null;
      micTrackRef.current = micTrack;
      if (micTrack) gate.setMicTrack(micTrack);
      gateRef.current = gate;

      // Movement events (tracked for cleanup)
      eventUnsubs.push(
        voiceEventBus.on("gate:muted", () => mgr.setListening(false)),
        voiceEventBus.on("gate:unmuted", () => mgr.setListening(true)),
        voiceEventBus.on("tts:start", () => mgr.setSpeaking(true)),
        voiceEventBus.on("tts:done", () => { mgr.setSpeaking(false); mgr.stopSpeechSway(); mgr.allowBreathingNow(); }),
        voiceEventBus.on("tts:stop", () => { mgr.setSpeaking(false); mgr.stopSpeechSway(); }),
      );

      // Speech sway (head movement) from model audio output
      const remoteStream = conn.audioEl.srcObject as MediaStream | null;
      let swayInterval: ReturnType<typeof setInterval> | undefined;
      const audioCtx = new AudioContext();
      await audioCtx.resume();

      if (remoteStream) {
        const source = audioCtx.createMediaStreamSource(remoteStream);
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);
        const buffer = new Float32Array(analyser.fftSize);
        swayInterval = setInterval(() => {
          if (!mgr.getIsRunning()) return;
          analyser.getFloatTimeDomainData(buffer);
          mgr.feedSpeechAudio(buffer.slice(0, 480), audioCtx.sampleRate);
        }, 10);
      }

      // Wait for DataChannel
      await new Promise<void>((resolve, reject) => {
        if (conn.dc.readyState === "open") { resolve(); return; }
        conn.dc.onopen = () => resolve();
        conn.dc.onerror = () => reject(new Error("DataChannel failed"));
        setTimeout(() => reject(new Error("DataChannel timeout")), 15000);
      });

      // Adapter
      const adapter = new RealtimeAdapter({
        dc: conn.dc,
        eventBus: voiceEventBus,
        executeTool,
        onUserSpeechStarted: () => chatRef.current.reserveUserMessage(),
        onUserTranscript: (text, final) => chatRef.current.handleUserTranscript(text, final),
        onAssistantTranscript: (text, final) => chatRef.current.handleAssistantTranscript(text, final),
        onError: (err) => setError(err.message),
      });
      adapterRef.current = adapter;

      // Tool results in chat
      eventUnsubs.push(
        voiceEventBus.on("tool:result", ({ name, result }) => {
          chatRef.current.addToolMessage(name, typeof result === "string" ? result : JSON.stringify(result));
        }),
      );

      configureTools({ adapter });

      // Barge-in
      const bargein = new BargeIn(voiceEventBus, () => adapter.cancelResponse());
      bargeinRef.current = bargein;

      cleanupRef.current = {
        swayInterval: swayInterval ?? setInterval(() => {}, 1e9),
        audioCtx,
        eventUnsubs,
      };

      setStatus("connected");
    } catch (err) {
      if (DEBUG) console.error("[useRealtime] connect error:", err);
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setStatus("error");
      eventUnsubs.forEach((fn) => fn());
      connRef.current?.close();
      connRef.current = null;
      mgrRef.current?.stop();
      mgrRef.current = null;
    }
  }, [settings.openaiApiKey, settings.voice, settings.profileId, settings.customInstructions, settings.daemonUrl]);

  const disconnect = useCallback(() => {
    adapterRef.current?.dispose();
    adapterRef.current = null;
    gateRef.current?.dispose();
    gateRef.current = null;
    bargeinRef.current?.dispose();
    bargeinRef.current = null;
    micTrackRef.current = null;

    if (cleanupRef.current) {
      clearInterval(cleanupRef.current.swayInterval);
      cleanupRef.current.audioCtx.close();
      cleanupRef.current.eventUnsubs.forEach((fn) => fn());
      cleanupRef.current = null;
    }

    connRef.current?.close();
    connRef.current = null;
    mgrRef.current?.stop();
    mgrRef.current = null;
    configureTools({ manager: null, adapter: null });
    setStatus("disconnected");
    setRobotConnected(false);
    setIsMuted(false);
  }, []);

  const toggleMute = useCallback(() => {
    const track = micTrackRef.current;
    if (!track) return;
    const next = !track.enabled;
    track.enabled = next;
    setIsMuted(!next);
  }, []);

  const cancelResponse = useCallback(() => {
    adapterRef.current?.cancelResponse();
  }, []);

  useEffect(() => () => disconnect(), [disconnect]);

  const getLocalStream = useCallback((): MediaStream | null => {
    const senders = connRef.current?.pc.getSenders();
    const audioSender = senders?.find((s) => s.track?.kind === "audio");
    if (audioSender?.track) {
      return new MediaStream([audioSender.track]);
    }
    return null;
  }, []);

  return { status, error, robotConnected, isMuted, connect, disconnect, toggleMute, cancelResponse, getManager: () => mgrRef.current, getLocalStream };
}
