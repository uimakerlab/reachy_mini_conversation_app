import Box from "@mui/material/Box";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import CameraAltIcon from "@mui/icons-material/CameraAlt";
import CircularProgress from "@mui/material/CircularProgress";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import { alpha, useTheme } from "@mui/material/styles";
import PersonOutlineIcon from "@mui/icons-material/PersonOutline";
import BuildRoundedIcon from "@mui/icons-material/BuildRounded";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import type { ChatMessage } from "../hooks/useChat";

function timeAgo(ts: number): string {
  const seconds = Math.floor((Date.now() - ts) / 1000);
  if (seconds < 10) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  return `${Math.floor(minutes / 60)}h ago`;
}

function cleanToolName(raw?: string): string {
  if (!raw) return "tool";
  return raw.replace(/^🛠️\s*Used tool\s*/i, "").trim() || "tool";
}

function formatToolResult(content: string): { label: string; entries: [string, string][] } | null {
  try {
    const parsed = JSON.parse(content);
    if (parsed.b64_im) return { label: "Photo captured", entries: [] };
    if (parsed.image_description) return { label: parsed.image_description, entries: [] };
    if (parsed.error) return { label: `Error: ${parsed.error}`, entries: [] };

    const entries = Object.entries(parsed)
      .filter(([, v]) => typeof v === "string" || typeof v === "number" || typeof v === "boolean")
      .map(([k, v]) => [k.replace(/_/g, " "), String(v)] as [string, string])
      .slice(0, 4);

    if (entries.length > 0) return { label: "", entries };
    return { label: content.slice(0, 120), entries: [] };
  } catch {
    return { label: content.slice(0, 120), entries: [] };
  }
}

interface BubbleProps {
  msg: ChatMessage;
  botAvatar?: string | null;
  botName?: string | null;
}

function BotAvatar({ botAvatar, botName, color }: { botAvatar?: string | null; botName?: string | null; color: string }) {
  return (
    <Box
      sx={{
        width: 24,
        height: 24,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
      }}
    >
      {botAvatar ? (
        <img src={botAvatar} alt="" style={{ width: 24, height: 24, objectFit: "contain" }} />
      ) : (
        <Typography
          sx={{ fontSize: "0.82rem", fontWeight: 700, color, lineHeight: 1, userSelect: "none", opacity: 0.8 }}
        >
          {(botName ?? "R").charAt(0).toUpperCase()}
        </Typography>
      )}
    </Box>
  );
}

function ToolContent({ msg }: { msg: ChatMessage }) {
  const theme = useTheme();
  const isRunning = msg.toolStatus === "running";
  const toolColor = isRunning ? "#f59e0b" : "#8b5cf6";
  const hasImage = !!msg.imageUrl;
  const name = cleanToolName(msg.toolName);
  const result = formatToolResult(msg.content);
  const isCamera = name.includes("camera");
  const IconComponent = isCamera ? CameraAltIcon : BuildRoundedIcon;

  return (
    <Box
      sx={{
        borderRadius: "16px 16px 16px 4px",
        bgcolor: alpha(theme.palette.background.paper, 0.8),
        border: `1px solid ${alpha(toolColor, 0.3)}`,
        overflow: "hidden",
        backdropFilter: "blur(8px)",
      }}
    >
      {/* Header */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, px: 2, pt: 1.25, pb: hasImage || (result && result.entries.length > 0) ? 0.75 : 1.25 }}>
        <Box
          sx={{
            width: 22,
            height: 22,
            borderRadius: "6px",
            bgcolor: alpha(toolColor, 0.12),
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          <IconComponent sx={{ fontSize: 13, color: toolColor }} />
        </Box>
        <Typography sx={{ fontSize: "0.75rem", fontWeight: 600, color: toolColor, flex: 1 }}>
          {name}
        </Typography>
        {isRunning ? (
          <CircularProgress size={12} thickness={5} sx={{ color: toolColor }} />
        ) : (
          <CheckCircleOutlineIcon sx={{ fontSize: 14, color: toolColor, opacity: 0.6 }} />
        )}
      </Box>

      {/* Result entries */}
      {result && result.entries.length > 0 && (
        <Box sx={{ px: 2, pb: 1.25, pt: 0.25 }}>
          {result.entries.map(([key, value]) => (
            <Box key={key} sx={{ display: "flex", gap: 1, py: 0.2 }}>
              <Typography sx={{ fontSize: "0.7rem", color: "text.secondary", opacity: 0.6, minWidth: 50, textTransform: "capitalize" }}>
                {key}
              </Typography>
              <Typography sx={{ fontSize: "0.7rem", color: "text.secondary" }}>
                {value}
              </Typography>
            </Box>
          ))}
        </Box>
      )}

      {/* Fallback label */}
      {result && result.label && !hasImage && (
        <Box sx={{ px: 2, pb: 1.25, pt: 0.25 }}>
          <Typography sx={{ fontSize: "0.72rem", color: "text.secondary", lineHeight: 1.5 }}>
            {result.label}
          </Typography>
        </Box>
      )}

      {/* Camera image */}
      {hasImage && (
        <Box sx={{ px: 1.5, pb: 1.5 }}>
          <Box
            component="img"
            src={msg.imageUrl}
            alt="Camera capture"
            sx={{ width: "100%", display: "block", borderRadius: 2 }}
          />
        </Box>
      )}
    </Box>
  );
}

function ErrorContent({ msg }: { msg: ChatMessage }) {
  const theme = useTheme();
  const errorColor = theme.palette.error.main;

  return (
    <Box
      sx={{
        borderRadius: "16px 16px 16px 4px",
        bgcolor: alpha(errorColor, 0.06),
        border: `1px solid ${alpha(errorColor, 0.2)}`,
        px: 2,
        py: 1.25,
        display: "flex",
        alignItems: "flex-start",
        gap: 1,
      }}
    >
      <ErrorOutlineIcon sx={{ fontSize: 15, color: errorColor, mt: 0.15, flexShrink: 0 }} />
      <Typography sx={{ fontSize: "0.75rem", color: errorColor, lineHeight: 1.5 }}>
        {msg.content}
      </Typography>
    </Box>
  );
}

export default function MessageBubble({ msg, botAvatar, botName }: BubbleProps) {
  const theme = useTheme();
  const isUser = msg.role === "user";
  const isTool = msg.role === "tool";
  const isError = msg.role === "error";
  const avatarColor = isUser ? theme.palette.primary.main : "#10b981";

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: isUser ? "row-reverse" : "row",
        alignItems: "flex-end",
        gap: 1,
        mb: 1.5,
        animation: "bubbleIn 0.3s ease-out",
        "@keyframes bubbleIn": {
          "0%": { opacity: 0, transform: "translateY(8px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
      }}
    >
      {/* Avatar */}
      {isUser ? (
        <Box sx={{ width: 24, height: 24, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
          <PersonOutlineIcon sx={{ fontSize: 20, color: avatarColor, opacity: 0.7 }} />
        </Box>
      ) : (
        <BotAvatar botAvatar={botAvatar} botName={botName} color={avatarColor} />
      )}

      {/* Bubble + timestamp */}
      <Box sx={{ display: "flex", flexDirection: "column", alignItems: isUser ? "flex-end" : "flex-start", maxWidth: (isTool || isError) ? "85%" : "75%", minWidth: 0, flex: (isTool || isError) ? 1 : undefined }}>
        {isTool ? (
          <ToolContent msg={msg} />
        ) : isError ? (
          <ErrorContent msg={msg} />
        ) : (
          <Paper
            elevation={0}
            sx={{
              px: 2,
              py: 1.25,
              bgcolor: isUser
                ? alpha(theme.palette.primary.main, 0.15)
                : alpha(theme.palette.background.paper, 0.8),
              border: `1px solid ${isUser
                ? alpha(theme.palette.primary.main, 0.25)
                : theme.palette.divider}`,
              borderRadius: isUser ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
              opacity: msg.partial ? 0.6 : 1,
              transition: "opacity 0.2s ease",
              backdropFilter: "blur(8px)",
              overflow: "hidden",
            }}
          >
            <Typography
              variant="body2"
              sx={{
                whiteSpace: "pre-wrap",
                lineHeight: 1.6,
                fontSize: "0.84rem",
                fontStyle: msg.partial ? "italic" : "normal",
                color: isUser
                  ? (theme.palette.mode === "dark" ? theme.palette.primary.light : theme.palette.primary.dark)
                  : "text.primary",
              }}
            >
              {msg.content}
            </Typography>
          </Paper>
        )}

        {/* Timestamp */}
        {!msg.partial && (
          <Typography
            sx={{
              fontSize: "0.62rem",
              color: "text.secondary",
              opacity: 0.5,
              mt: 0.4,
              px: 0.5,
            }}
          >
            {timeAgo(msg.ts)}
          </Typography>
        )}
      </Box>
    </Box>
  );
}
