import { useState, useEffect, useRef } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import Tooltip from "@mui/material/Tooltip";
import { alpha } from "@mui/material/styles";
import MicIcon from "@mui/icons-material/Mic";
import MicOffIcon from "@mui/icons-material/MicOff";
import PlayArrowRoundedIcon from "@mui/icons-material/PlayArrowRounded";
import VolumeUpIcon from "@mui/icons-material/VolumeUp";
import StopIcon from "@mui/icons-material/Stop";
import type { ConnectionStatus } from "../hooks/useRealtime";
import { voiceEventBus } from "../voice/eventBus";

type AgentState = "idle" | "connecting" | "listening" | "hearing" | "processing" | "speaking";

const STATE_STYLES: Record<AgentState, { color: string; label: string }> = {
  idle: { color: "#64748b", label: "Tap to start" },
  connecting: { color: "#f59e0b", label: "Connecting..." },
  listening: { color: "#3b82f6", label: "Listening" },
  hearing: { color: "#6366f1", label: "Listening..." },
  processing: { color: "#f59e0b", label: "Thinking..." },
  speaking: { color: "#10b981", label: "Speaking..." },
};

const NUM_BARS = 5;
const EMPTY_BANDS = new Array(NUM_BARS).fill(0) as number[];

// Log-spaced bin edges for 5 bands (fftSize 1024 -> 512 bins, ~47 Hz/bin at 48 kHz)
const BAND_EDGES = [4, 8, 16, 32, 64, 128];

const LOG1P_10 = Math.log1p(10);
const compress = (v: number) => Math.log1p(v * 10) / LOG1P_10;

const EMA_ATTACK = 0.7;
const EMA_RELEASE = 0.25;

function useAudioLevel(
  isActive: boolean,
  getStream: (() => MediaStream | null) | undefined,
): { level: number; bands: number[] } {
  const [level, setLevel] = useState(0);
  const [bands, setBands] = useState<number[]>(EMPTY_BANDS);
  const rafRef = useRef<number | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const smoothedRef = useRef<number[]>(new Array(NUM_BARS).fill(0));

  useEffect(() => {
    if (!isActive) {
      setLevel(0);
      setBands(EMPTY_BANDS);
      smoothedRef.current = new Array(NUM_BARS).fill(0);
      return;
    }
    let cancelled = false;

    const setup = () => {
      if (cancelled) return;
      const stream = getStream?.();
      if (!stream) return;

      try {
        const ctx = new AudioContext();
        if (ctx.state === "suspended") ctx.resume();
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 1024;
        analyser.smoothingTimeConstant = 0.4;
        const source = ctx.createMediaStreamSource(stream);
        source.connect(analyser);
        ctxRef.current = ctx;
        sourceRef.current = source;
        const data = new Uint8Array(analyser.frequencyBinCount);
        const sm = smoothedRef.current;

        const tick = () => {
          if (cancelled) return;
          analyser.getByteFrequencyData(data);

          let sum = 0;
          for (let i = 0; i < data.length; i++) sum += data[i];
          setLevel(sum / data.length / 255);

          const out = new Array(NUM_BARS) as number[];
          for (let b = 0; b < NUM_BARS; b++) {
            const lo = BAND_EDGES[b];
            const hi = BAND_EDGES[b + 1];
            let bandSum = 0;
            for (let j = lo; j < hi; j++) bandSum += data[j];
            const raw = compress(bandSum / (hi - lo) / 255);
            const alpha = raw > sm[b] ? EMA_ATTACK : EMA_RELEASE;
            sm[b] += alpha * (raw - sm[b]);
            out[b] = Math.min(1, sm[b]);
          }

          setBands(out);
          rafRef.current = requestAnimationFrame(tick);
        };
        rafRef.current = requestAnimationFrame(tick);
      } catch { /* audio context unavailable */ }
    };

    const timer = setTimeout(setup, 100);

    return () => {
      cancelled = true;
      clearTimeout(timer);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      sourceRef.current?.disconnect();
      ctxRef.current?.close();
      ctxRef.current = null;
      sourceRef.current = null;
    };
  }, [isActive, getStream]);

  return { level, bands };
}

interface Props {
  status: ConnectionStatus;
  isMuted: boolean;
  onConnect: () => void;
  onDisconnect: () => void;
  onToggleMute: () => void;
  onCancelResponse: () => void;
  getLocalStream?: () => MediaStream | null;
}

export default function AudioControls({
  status, isMuted, onConnect, onDisconnect, onToggleMute, onCancelResponse, getLocalStream,
}: Props) {
  const isConnected = status === "connected";
  const isConnecting = status === "connecting";

  const [isSpeaking, setIsSpeaking] = useState(false);
  const [vadActive, setVadActive] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const serverDoneRef = useRef(false);

  useEffect(() => {
    if (!isConnected) {
      serverDoneRef.current = false;
      setIsSpeaking(false);
      setVadActive(false);
      setIsProcessing(false);
    }
  }, [isConnected]);

  useEffect(() => {
    const unsubs = [
      voiceEventBus.on("tts:start", () => {
        serverDoneRef.current = false;
        setIsSpeaking(true);
        setIsProcessing(false);
      }),
      voiceEventBus.on("tts:done", () => {
        serverDoneRef.current = true;
        setIsProcessing(false);
      }),
      voiceEventBus.on("tts:stop", () => {
        serverDoneRef.current = false;
        setIsSpeaking(false);
        setIsProcessing(false);
      }),
      voiceEventBus.on("bot:audio_silent", () => {
        if (serverDoneRef.current) {
          serverDoneRef.current = false;
          setIsSpeaking(false);
        }
      }),
      voiceEventBus.on("vad:start", () => { setVadActive(true); setIsProcessing(false); }),
      voiceEventBus.on("vad:end", () => { setVadActive(false); setIsProcessing(true); }),
    ];
    return () => unsubs.forEach((fn) => fn());
  }, []);

  const { level: audioLevel, bands } = useAudioLevel(isConnected, getLocalStream);

  const agentState: AgentState = (() => {
    if (!isConnected && !isConnecting) return "idle";
    if (isConnecting) return "connecting";
    if (isSpeaking) return "speaking";
    if (vadActive) return "hearing";
    if (isProcessing) return "processing";
    return "listening";
  })();

  const style = STATE_STYLES[agentState];
  const isHearing = agentState === "hearing";
  const isAudioReactive = agentState === "listening" || isHearing;
  const boosted = Math.min(1, Math.pow(audioLevel * 2.5, 0.7));
  const glowSize = 20;
  const ringOpacity = agentState === "idle" ? 0.2 : 0.4;
  const ringScale = 1;

  const handleClick = () => {
    if (isConnected) onDisconnect();
    else if (!isConnecting) onConnect();
  };

  const ORB_SIZE = 56;
  const RING_SIZE = 76;
  const ORB_AREA = RING_SIZE + 20;
  const showControls = isConnected;

  const sideButtonSx = {
    width: 36,
    height: 36,
    opacity: showControls ? 1 : 0,
    transform: showControls ? "scale(1)" : "scale(0.5)",
    transition: "opacity 0.25s ease, transform 0.25s ease, background-color 0.2s ease, color 0.2s ease",
    pointerEvents: showControls ? "auto" as const : "none" as const,
  };

  return (
    <Box
      sx={{
        py: 1.5,
        pb: 2,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 0.5,
        borderTop: 1,
        borderColor: "divider",
        bgcolor: "background.paper",
      }}
    >
      {/* Main row: [Mic] -- ORB -- [Stop] */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 1.5,
        }}
      >
        {/* Left: Mic */}
        <Tooltip title={isMuted ? "Unmute mic" : "Mute mic"} arrow>
          <IconButton
            onClick={onToggleMute}
            aria-label={isMuted ? "Unmute microphone" : "Mute microphone"}
            sx={{
              ...sideButtonSx,
              color: isMuted ? "error.main" : "text.secondary",
              bgcolor: isMuted ? (t) => alpha(t.palette.error.main, 0.1) : "transparent",
              "&:hover": {
                bgcolor: isMuted
                  ? (t) => alpha(t.palette.error.main, 0.18)
                  : "action.hover",
              },
            }}
          >
            {isMuted ? <MicOffIcon fontSize="small" /> : <MicIcon fontSize="small" />}
          </IconButton>
        </Tooltip>

        {/* Center: Orb */}
        <Box sx={{ position: "relative", width: ORB_AREA, height: ORB_AREA, flexShrink: 0 }}>
          <Box
            sx={{
              position: "absolute",
              top: "50%",
              left: "50%",
              width: RING_SIZE,
              height: RING_SIZE,
              borderRadius: "50%",
              border: "1.5px solid",
              borderColor: alpha(style.color, 0.3),
              opacity: ringOpacity,
              transform: `translate(-50%, -50%) scale(${ringScale})`,
              transition: isHearing
                ? "transform 0.08s ease-out, border-color 0.5s ease, opacity 0.08s ease-out"
                : "transform 0.3s ease-out, border-color 0.5s ease, opacity 0.3s ease",
              animation:
                !isAudioReactive && agentState !== "idle" && agentState !== "processing"
                  ? "ctrlPulse 2s ease-in-out infinite"
                  : "none",
              "@keyframes ctrlPulse": {
                "0%, 100%": { transform: "translate(-50%, -50%) scale(1)", opacity: 0.4 },
                "50%": { transform: "translate(-50%, -50%) scale(1.06)", opacity: 0.15 },
              },
            }}
          />

          {(agentState === "speaking" || isHearing) && (
            <Box
              sx={{
                position: "absolute",
                top: "50%",
                left: "50%",
                width: RING_SIZE + 16,
                height: RING_SIZE + 16,
                borderRadius: "50%",
                border: "1px solid",
                borderColor: alpha(style.color, 0.2),
                opacity: isHearing ? 0.1 + boosted * 0.35 : 0.25,
                transform: isHearing
                  ? `translate(-50%, -50%) scale(${1 + boosted * 0.12})`
                  : "translate(-50%, -50%)",
                transition: isHearing
                  ? "transform 0.08s ease-out, opacity 0.08s ease-out"
                  : "none",
                animation: !isHearing
                  ? "ctrlPulseOuter 2.5s ease-in-out infinite"
                  : "none",
                "@keyframes ctrlPulseOuter": {
                  "0%, 100%": { transform: "translate(-50%, -50%) scale(1)", opacity: 0.25 },
                  "50%": { transform: "translate(-50%, -50%) scale(1.08)", opacity: 0.05 },
                },
              }}
            />
          )}

          <Box
            onClick={handleClick}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); handleClick(); } }}
            aria-label={isConnected ? "Disconnect" : "Connect"}
            sx={{
              position: "absolute",
              top: "50%",
              left: "50%",
              width: ORB_SIZE,
              height: ORB_SIZE,
              borderRadius: "50%",
              background: `radial-gradient(circle at 40% 35%, ${alpha(style.color, 0.25)}, ${alpha(style.color, 0.1)} 60%, ${alpha(style.color, 0.05)})`,
              border: `2px solid ${alpha(style.color, 0.35)}`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              cursor: isConnecting ? "wait" : "pointer",
              zIndex: 1,
              transform: "translate(-50%, -50%)",
              boxShadow: `0 0 ${glowSize}px ${alpha(style.color, 0.2)}, inset 0 0 20px ${alpha(style.color, 0.15)}`,
              transition: "all 0.5s ease",
              animation: !isAudioReactive
                ? agentState === "speaking"
                  ? "ctrlBreathing 1.5s ease-in-out infinite"
                  : (agentState === "connecting" || agentState === "processing")
                    ? "ctrlThinking 1s ease-in-out infinite"
                    : "none"
                : "none",
              "&:hover": { filter: "brightness(1.2)" },
              "&:active": { filter: "brightness(0.85)" },
              "&:focus-visible": {
                outline: `2px solid ${style.color}`,
                outlineOffset: 4,
              },
              "@keyframes ctrlBreathing": {
                "0%, 100%": { transform: "translate(-50%, -50%) scale(1)" },
                "50%": { transform: "translate(-50%, -50%) scale(1.05)" },
              },
              "@keyframes ctrlThinking": {
                "0%, 100%": { transform: "translate(-50%, -50%) scale(1)", opacity: 1 },
                "50%": { transform: "translate(-50%, -50%) scale(0.96)", opacity: 0.7 },
              },
            }}
          >
            {agentState === "listening" || agentState === "hearing" ? (
              <Box sx={{ display: "flex", alignItems: "center", gap: "2.5px", height: 28 }}>
                {bands.map((b, i) => {
                  const minH = 3;
                  const maxH = 26;
                  const h = minH + b * (maxH - minH);
                  return (
                    <Box
                      key={i}
                      sx={{
                        width: 3.5,
                        height: h,
                        minHeight: minH,
                        borderRadius: 2,
                        bgcolor: style.color,
                        opacity: 0.65 + b * 0.35,
                      }}
                    />
                  );
                })}
              </Box>
            ) : agentState === "speaking" ? (
              <VolumeUpIcon sx={{ fontSize: 26, color: style.color, opacity: 0.9 }} />
            ) : agentState === "connecting" || agentState === "processing" ? (
              <Box
                sx={{
                  width: 22,
                  height: 22,
                  border: "2.5px solid",
                  borderColor: style.color,
                  borderRightColor: "transparent",
                  borderRadius: "50%",
                  animation: "ctrlSpin 1s linear infinite",
                  opacity: 0.8,
                  "@keyframes ctrlSpin": {
                    "0%": { transform: "rotate(0deg)" },
                    "100%": { transform: "rotate(360deg)" },
                  },
                }}
              />
            ) : (
              <PlayArrowRoundedIcon sx={{ fontSize: 28, color: style.color, opacity: 0.6 }} />
            )}
          </Box>
        </Box>

        {/* Right: Stop */}
        {(() => {
          const canStop = isSpeaking || isProcessing;
          return (
            <Tooltip title="Stop response" arrow>
              <IconButton
                onClick={onCancelResponse}
                disabled={!canStop}
                aria-label="Stop AI response"
                sx={{
                  ...sideButtonSx,
                  color: canStop ? "warning.main" : "text.disabled",
                  bgcolor: canStop ? (t) => alpha(t.palette.warning.main, 0.1) : "transparent",
                  "&:hover": {
                    bgcolor: canStop
                      ? (t) => alpha(t.palette.warning.main, 0.18)
                      : "action.hover",
                  },
                }}
              >
                <StopIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          );
        })()}
      </Box>

      {/* State label */}
      <Typography
        variant="caption"
        sx={{
          color: agentState === "idle" ? "text.secondary" : style.color,
          fontWeight: 500,
          letterSpacing: "0.03em",
          opacity: 0.8,
          transition: "color 0.5s ease",
          fontSize: "0.72rem",
        }}
      >
        {isMuted && isConnected ? "Muted" : style.label}
      </Typography>
    </Box>
  );
}
