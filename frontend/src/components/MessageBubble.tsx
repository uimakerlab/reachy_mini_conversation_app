import Box from "@mui/material/Box";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";
import { alpha, useTheme } from "@mui/material/styles";
import PersonOutlineIcon from "@mui/icons-material/PersonOutline";
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

function ToolBubble({ msg }: { msg: ChatMessage }) {
  const toolColor = "#8b5cf6";

  return (
    <Box
      sx={{
        mx: 1.5,
        my: 1.5,
        animation: "bubbleIn 0.3s ease-out",
        "@keyframes bubbleIn": {
          "0%": { opacity: 0, transform: "translateY(8px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
      }}
    >
      <Box
        sx={{
          display: "flex",
          alignItems: "flex-start",
          gap: 1.25,
          px: 2,
          py: 1.25,
          borderRadius: 2,
          bgcolor: alpha(toolColor, 0.06),
          border: `1px solid ${alpha(toolColor, 0.15)}`,
        }}
      >
        <CheckCircleOutlineIcon sx={{ fontSize: 16, color: toolColor, flexShrink: 0, mt: 0.1 }} />
        <Box sx={{ minWidth: 0, flex: 1 }}>
          <Typography
            sx={{
              fontSize: "0.72rem",
              fontFamily: "monospace",
              fontWeight: 600,
              color: toolColor,
              mb: 0.25,
            }}
          >
            {msg.toolName ?? "tool"}
          </Typography>
          <Typography
            sx={{
              fontSize: "0.72rem",
              color: "text.secondary",
              lineHeight: 1.5,
              wordBreak: "break-word",
            }}
          >
            {msg.content.slice(0, 200)}
          </Typography>
        </Box>
      </Box>
    </Box>
  );
}

interface BubbleProps {
  msg: ChatMessage;
  botAvatar?: string | null;
  botName?: string | null;
}

export default function MessageBubble({ msg, botAvatar, botName }: BubbleProps) {
  const theme = useTheme();
  const isUser = msg.role === "user";
  const isTool = msg.role === "tool";

  if (isTool) return <ToolBubble msg={msg} />;

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
        {isUser ? (
          <PersonOutlineIcon sx={{ fontSize: 20, color: avatarColor, opacity: 0.7 }} />
        ) : botAvatar ? (
          <img src={botAvatar} alt="" style={{ width: 24, height: 24, objectFit: "contain" }} />
        ) : (
          <Typography
            sx={{
              fontSize: "0.82rem",
              fontWeight: 700,
              color: avatarColor,
              lineHeight: 1,
              userSelect: "none",
              opacity: 0.8,
            }}
          >
            {(botName ?? "R").charAt(0).toUpperCase()}
          </Typography>
        )}
      </Box>

      {/* Bubble + timestamp */}
      <Box sx={{ display: "flex", flexDirection: "column", alignItems: isUser ? "flex-end" : "flex-start", maxWidth: "75%", minWidth: 0 }}>
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
          }}
        >
          <Typography
            variant="body2"
            sx={{
              whiteSpace: "pre-wrap",
              lineHeight: 1.6,
              fontSize: "0.84rem",
              fontStyle: msg.partial ? "italic" : "normal",
              color: isUser ? theme.palette.primary.light : "text.primary",
            }}
          >
            {msg.content}
          </Typography>
        </Paper>

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
