/**
 * WebSocket-based conversation hook.
 *
 * Replaces useRealtime (direct WebRTC to OpenAI) with a WebSocket connection
 * to the Python backend which manages OpenAI, movements, tools, and camera.
 *
 * Protocol (matching web_ui.py):
 *   Client -> Server:  binary = PCM int16 mono 24 kHz
 *                       text   = JSON control messages
 *   Server -> Client:  binary = PCM int16 mono 24 kHz
 *                       text   = JSON { type: "transcript"|"tool"|"image"|"interrupt"|"state" }
 */

import { useState, useRef, useCallback, useEffect } from "react";
import { MicCapture } from "../audio/micCapture";
import { PcmPlayer } from "../audio/pcmPlayer";
import { voiceEventBus } from "../voice/eventBus";
import type { AppSettings } from "../config/settings";
import type { useChat } from "./useChat";

export type ConnectionStatus =
  | "disconnected"
  | "connecting"
  | "connected"
  | "error";

const DEBUG = import.meta.env.DEV;

function getWsUrl(settings: AppSettings): string {
  if (settings.daemonUrl) {
    const base = settings.daemonUrl.replace(/\/$/, "");
    return `${base.replace(/^http/, "ws")}/ws`;
  }
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}

export function useConversation(
  settings: AppSettings,
  chat: ReturnType<typeof useChat>,
) {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [error, setError] = useState<string | null>(null);
  const [isMuted, setIsMuted] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const micRef = useRef<MicCapture | null>(null);
  const playerRef = useRef<PcmPlayer | null>(null);
  const cancelledRef = useRef(false);
  const pausedRef = useRef(false);

  const chatRef = useRef(chat);
  chatRef.current = chat;

  const connect = useCallback(async () => {
    if (DEBUG) console.log("[useConversation] connect()");
    if (wsRef.current) return;

    cancelledRef.current = false;
    setStatus("connecting");
    setError(null);
    chatRef.current.clear();

    try {
      // Audio playback
      const player = new PcmPlayer({
        onAudioStart: () => voiceEventBus.emit("tts:start", {}),
        onAudioEnd: () => voiceEventBus.emit("tts:done", {}),
      });
      await player.resume();
      playerRef.current = player;

      // Mic capture
      const mic = new MicCapture();
      const stream = await mic.start((pcm) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(pcm);
        }
      });
      micRef.current = mic;

      if (cancelledRef.current) {
        mic.stop();
        player.stop();
        return;
      }

      // WebSocket to backend
      const wsUrl = getWsUrl(settings);
      if (DEBUG) console.log("[useConversation] connecting to", wsUrl);

      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      await new Promise<void>((resolve, reject) => {
        const timer = setTimeout(
          () => reject(new Error("WebSocket connection timeout")),
          10000,
        );
        ws.onopen = () => {
          clearTimeout(timer);
          resolve();
        };
        ws.onerror = () => {
          clearTimeout(timer);
          reject(new Error("WebSocket connection failed"));
        };
      });

      if (cancelledRef.current) {
        ws.close();
        mic.stop();
        player.stop();
        return;
      }

      ws.onmessage = (e: MessageEvent) => {
        if (pausedRef.current) return;
        if (e.data instanceof ArrayBuffer) {
          player.pushChunk(e.data);
          return;
        }
        if (typeof e.data === "string") {
          try {
            const msg = JSON.parse(e.data);
            handleServerMessage(msg);
          } catch {
            /* ignore malformed JSON */
          }
        }
      };

      ws.onclose = () => {
        if (DEBUG) console.log("[useConversation] ws closed");
        cleanup();
        setStatus("disconnected");
      };

      ws.onerror = () => {
        setError("WebSocket error");
        cleanup();
        setStatus("error");
      };

      // Unused local stream ref for AudioControls visualizer
      void stream;

      setStatus("connected");
    } catch (err) {
      if (cancelledRef.current) return;
      if (DEBUG) console.error("[useConversation] connect error:", err);
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setStatus("error");
      cleanup();
    }
  }, [settings.daemonUrl]);

  function handleServerMessage(msg: Record<string, unknown>) {
    const type = msg.type as string;

    switch (type) {
      case "interrupt":
        playerRef.current?.clear();
        voiceEventBus.emit("tts:stop", {});
        break;

      case "state": {
        const state = msg.state as string;
        if (state === "vad_start") {
          chatRef.current.reserveUserMessage();
          voiceEventBus.emit("vad:start", {});
        } else if (state === "vad_end") {
          voiceEventBus.emit("vad:end", {});
        }
        break;
      }

      case "transcript": {
        const role = msg.role as string;
        const content = (msg.content as string) ?? "";
        if (role === "user") {
          chatRef.current.handleUserTranscript(content, true);
        } else if (role === "user_partial") {
          chatRef.current.handleUserTranscript(content, false);
        } else if (role === "assistant") {
          chatRef.current.handleAssistantTranscript(content, true);
        }
        break;
      }

      case "tool": {
        const title = (msg.title as string) ?? "";
        const content = (msg.content as string) ?? "";
        chatRef.current.addToolMessage(title, content);
        break;
      }

      case "image": {
        const dataUrl = msg.data as string;
        if (dataUrl) chatRef.current.attachImageToLastTool(dataUrl);
        break;
      }

      default:
        if (DEBUG) console.log("[useConversation] unknown message type:", type);
    }
  }

  function cleanup() {
    wsRef.current?.close();
    wsRef.current = null;
    micRef.current?.stop();
    micRef.current = null;
    playerRef.current?.stop();
    playerRef.current = null;
    setIsMuted(false);
    setIsPaused(false);
  }

  const disconnect = useCallback(() => {
    cancelledRef.current = true;
    cleanup();
    setStatus("disconnected");
  }, []);

  const toggleMute = useCallback(() => {
    const mic = micRef.current;
    if (!mic) return;
    const wasMuted = mic.isMuted();
    mic.setMuted(!wasMuted);
    setIsMuted(!wasMuted);
  }, []);

  const [isPaused, setIsPaused] = useState(false);

  const togglePause = useCallback(() => {
    setIsPaused((prev) => {
      const next = !prev;
      pausedRef.current = next;
      const mic = micRef.current;
      if (next) {
        mic?.setMuted(true);
        setIsMuted(true);
        playerRef.current?.clear();
        voiceEventBus.emit("tts:stop", {});
      } else {
        mic?.setMuted(false);
        setIsMuted(false);
      }
      return next;
    });
  }, []);

  useEffect(() => () => { cancelledRef.current = true; cleanup(); }, []);

  const getLocalStream = useCallback(
    (): MediaStream | null => micRef.current?.getStream() ?? null,
    [],
  );

  return {
    status,
    error,
    isMuted,
    isPaused,
    connect,
    disconnect,
    toggleMute,
    togglePause,
    getLocalStream,
  };
}
