import { useState, useEffect, useRef, useCallback } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { useTheme, alpha } from "@mui/material/styles";
import MicIcon from "@mui/icons-material/Mic";
import MicOffIcon from "@mui/icons-material/MicOff";
import VolumeUpIcon from "@mui/icons-material/VolumeUp";
import type { ConnectionStatus } from "../hooks/useRealtime";
import { voiceEventBus } from "../voice/eventBus";

type AgentState = "idle" | "listening" | "hearing" | "thinking" | "speaking";

function useAudioLevel(isActive: boolean): number {
  const [level, setLevel] = useState(0);
  const rafRef = useRef<number | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataRef = useRef<Uint8Array | null>(null);

  useEffect(() => {
    if (!isActive) {
      setLevel(0);
      return;
    }

    let cancelled = false;

    const setup = async (attempt = 0) => {
      if (cancelled) return;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        const ctx = new AudioContext();
        if (ctx.state === "suspended") await ctx.resume();
        const analyser = ctx.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.5;
        const source = ctx.createMediaStreamSource(stream);
        source.connect(analyser);

        ctxRef.current = ctx;
        sourceRef.current = source;
        analyserRef.current = analyser;
        dataRef.current = new Uint8Array(analyser.frequencyBinCount);

        const tick = () => {
          if (cancelled) return;
          analyser.getByteFrequencyData(dataRef.current!);
          const arr = dataRef.current!;
          let sum = 0;
          for (let i = 0; i < arr.length; i++) sum += arr[i];
          setLevel(sum / arr.length / 255);
          rafRef.current = requestAnimationFrame(tick);
        };
        rafRef.current = requestAnimationFrame(tick);
      } catch {
        if (!cancelled && attempt < 5) {
          setTimeout(() => setup(attempt + 1), 500 * (attempt + 1));
        }
      }
    };

    setup();

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      sourceRef.current?.disconnect();
      ctxRef.current?.close();
      ctxRef.current = null;
      sourceRef.current = null;
      analyserRef.current = null;
    };
  }, [isActive]);

  return level;
}

interface Props {
  status: ConnectionStatus;
  onConnect: () => void;
  onDisconnect: () => void;
}

export default function AudioOrb({ status, onConnect, onDisconnect }: Props) {
  const theme = useTheme();
  const isConnected = status === "connected";
  const isConnecting = status === "connecting";

  const [isSpeaking, setIsSpeaking] = useState(false);
  const [vadActive, setVadActive] = useState(false);

  useEffect(() => {
    const unsubs = [
      voiceEventBus.on("tts:start", () => setIsSpeaking(true)),
      voiceEventBus.on("tts:done", () => setIsSpeaking(false)),
      voiceEventBus.on("tts:stop", () => setIsSpeaking(false)),
      voiceEventBus.on("vad:start", () => setVadActive(true)),
      voiceEventBus.on("vad:end", () => setVadActive(false)),
    ];
    return () => unsubs.forEach((fn) => fn());
  }, []);

  const audioLevel = useAudioLevel(isConnected);

  const getAgentState = useCallback((): AgentState => {
    if (!isConnected && !isConnecting) return "idle";
    if (isConnecting) return "thinking";
    if (isSpeaking) return "speaking";
    if (vadActive) return "hearing";
    if (isConnected) return "listening";
    return "idle";
  }, [isConnected, isConnecting, isSpeaking, vadActive]);

  const agentState = getAgentState();

  const glowStrength = 0.2;
  const ringStrength = 0.4;
  const stateStyles: Record<AgentState, { color: string; glow: string; ringColor: string; label: string }> = {
    idle: {
      color: "#64748b",
      glow: alpha("#64748b", glowStrength * 0.75),
      ringColor: alpha("#64748b", ringStrength * 0.75),
      label: "Ready",
    },
    listening: {
      color: "#3b82f6",
      glow: alpha("#3b82f6", glowStrength),
      ringColor: alpha("#3b82f6", ringStrength),
      label: "Listening...",
    },
    hearing: {
      color: "#6366f1",
      glow: alpha("#6366f1", glowStrength * 1.3),
      ringColor: alpha("#6366f1", ringStrength * 1.3),
      label: "Listening...",
    },
    thinking: {
      color: "#f59e0b",
      glow: alpha("#f59e0b", glowStrength * 1.25),
      ringColor: alpha("#f59e0b", ringStrength * 1.25),
      label: "Connecting...",
    },
    speaking: {
      color: "#10b981",
      glow: alpha("#10b981", glowStrength * 1.25),
      ringColor: alpha("#10b981", ringStrength * 1.25),
      label: "Speaking...",
    },
  };

  const style = stateStyles[agentState];

  const isHearing = agentState === "hearing";
  const isAudioReactive = agentState === "listening" || isHearing;
  const boostedLevel = Math.min(1, Math.pow(audioLevel * 2.5, 0.7));
  const reactiveScale = isHearing ? 1 + boostedLevel * 0.18 : isAudioReactive ? 1 + boostedLevel * 0.08 : 1;
  const reactiveGlow = isHearing ? 40 + boostedLevel * 100 : isAudioReactive ? 40 + boostedLevel * 30 : 60;
  const reactiveRingScale = isHearing ? 1 + boostedLevel * 0.15 : isAudioReactive ? 1 + boostedLevel * 0.05 : 1;
  const reactiveRingOpacity = isHearing
    ? 0.3 + boostedLevel * 0.5
    : isAudioReactive
      ? 0.15 + boostedLevel * 0.2
      : agentState === "idle"
        ? 0.3
        : 0.6;

  const handleOrbClick = () => {
    if (isConnected) {
      onDisconnect();
    } else if (!isConnecting) {
      onConnect();
    }
  };

  return (
    <Box
      sx={{
        flex: 1,
        width: "100%",
        position: "relative",
        overflow: "hidden",
        cursor: "default",
        userSelect: "none",
      }}
    >
      {/* Circle group */}
      <Box
        sx={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: 300,
          height: 300,
        }}
      >
        {/* Outer pulsing ring */}
        <Box
          sx={{
            position: "absolute",
            top: "50%",
            left: "50%",
            width: 220,
            height: 220,
            borderRadius: "50%",
            border: "2px solid",
            borderColor: style.ringColor,
            opacity: reactiveRingOpacity,
            transform: `translate(-50%, -50%) scale(${reactiveRingScale})`,
            transition: isHearing
              ? "transform 0.08s ease-out, border-color 0.5s ease, opacity 0.08s ease-out"
              : "transform 0.3s ease-out, border-color 0.5s ease, opacity 0.3s ease",
            animation:
              !isAudioReactive && agentState !== "idle"
                ? "orbPulseRing 2s ease-in-out infinite"
                : "none",
            "@keyframes orbPulseRing": {
              "0%, 100%": { transform: "translate(-50%, -50%) scale(1)", opacity: 0.6 },
              "50%": { transform: "translate(-50%, -50%) scale(1.08)", opacity: 0.2 },
            },
          }}
        />

        {/* Secondary ring */}
        {(agentState === "speaking" || isHearing) && (
          <Box
            sx={{
              position: "absolute",
              top: "50%",
              left: "50%",
              width: 260,
              height: 260,
              borderRadius: "50%",
              border: "1px solid",
              borderColor: style.ringColor,
              opacity: isHearing ? 0.1 + boostedLevel * 0.4 : 0.3,
              transform: isHearing
                ? `translate(-50%, -50%) scale(${1 + boostedLevel * 0.2})`
                : "translate(-50%, -50%)",
              transition: isHearing ? "transform 0.08s ease-out, opacity 0.08s ease-out" : "none",
              animation: !isHearing ? "orbPulseOuter 2.5s ease-in-out infinite" : "none",
              "@keyframes orbPulseOuter": {
                "0%, 100%": { transform: "translate(-50%, -50%) scale(1)", opacity: 0.3 },
                "50%": { transform: "translate(-50%, -50%) scale(1.12)", opacity: 0.05 },
              },
            }}
          />
        )}

        {/* Third ring (loud voice) */}
        {isHearing && boostedLevel > 0.1 && (
          <Box
            sx={{
              position: "absolute",
              top: "50%",
              left: "50%",
              width: 300,
              height: 300,
              borderRadius: "50%",
              border: "1px solid",
              borderColor: style.ringColor,
              opacity: boostedLevel * 0.35,
              transform: `translate(-50%, -50%) scale(${1 + boostedLevel * 0.15})`,
              transition: "transform 0.08s ease-out, opacity 0.08s ease-out",
            }}
          />
        )}

        {/* Main circle */}
        <Box
          onClick={handleOrbClick}
          sx={{
            position: "absolute",
            top: "50%",
            left: "50%",
            width: 160,
            height: 160,
            borderRadius: "50%",
            background: `radial-gradient(circle at 40% 35%, ${alpha(style.color, 0.27)}, ${alpha(style.color, 0.13)} 60%, ${alpha(style.color, 0.07)})`,
            border: `2px solid ${alpha(style.color, 0.4)}`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            zIndex: 1,
            transform: `translate(-50%, -50%) scale(${reactiveScale})`,
            boxShadow: `0 0 ${reactiveGlow}px ${style.glow}, inset 0 0 40px ${style.glow}`,
            transition: isHearing
              ? "transform 0.08s ease-out, box-shadow 0.08s ease-out"
              : isAudioReactive
                ? "transform 0.3s ease-out, box-shadow 0.3s ease-out"
                : "all 0.5s ease",
            animation: !isAudioReactive
              ? agentState === "speaking"
                ? "orbBreathing 1.5s ease-in-out infinite"
                : agentState === "thinking"
                  ? "orbThinking 1s ease-in-out infinite"
                  : "none"
              : "none",
            "&:hover": { filter: "brightness(1.15)" },
            "&:active": { filter: "brightness(0.9)" },
            "@keyframes orbBreathing": {
              "0%, 100%": { transform: "translate(-50%, -50%) scale(1)" },
              "50%": { transform: "translate(-50%, -50%) scale(1.06)" },
            },
            "@keyframes orbThinking": {
              "0%, 100%": { transform: "translate(-50%, -50%) scale(1)", opacity: 1 },
              "50%": { transform: "translate(-50%, -50%) scale(0.97)", opacity: 0.8 },
            },
          }}
        >
          {agentState === "speaking" ? (
            <VolumeUpIcon sx={{ fontSize: 48, color: style.color, opacity: 0.9 }} />
          ) : agentState === "thinking" ? (
            <Box
              sx={{
                width: 32,
                height: 32,
                border: "3px solid",
                borderColor: style.color,
                borderRightColor: "transparent",
                borderRadius: "50%",
                animation: "orbSpin 1s linear infinite",
                opacity: 0.8,
                "@keyframes orbSpin": {
                  "0%": { transform: "rotate(0deg)" },
                  "100%": { transform: "rotate(360deg)" },
                },
              }}
            />
          ) : isConnected ? (
            <MicIcon sx={{ fontSize: 48, color: style.color, opacity: 0.9 }} />
          ) : (
            <MicOffIcon sx={{ fontSize: 48, color: style.color, opacity: 0.5 }} />
          )}
        </Box>
      </Box>

      {/* State label below circle */}
      <Box
        sx={{
          position: "absolute",
          top: "calc(50% + 120px)",
          left: "50%",
          transform: "translateX(-50%)",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 1,
          pointerEvents: "none",
        }}
      >
        <Typography
          variant="body2"
          sx={{
            color: style.color,
            fontWeight: 500,
            letterSpacing: "0.05em",
            opacity: 0.8,
            transition: "color 0.5s ease",
            fontSize: "0.85rem",
          }}
        >
          {style.label}
        </Typography>

        {agentState === "idle" && (
          <Typography
            variant="caption"
            sx={{ color: "text.secondary", opacity: 0.5, mt: 0.5 }}
          >
            Tap the circle to start
          </Typography>
        )}
      </Box>
    </Box>
  );
}
