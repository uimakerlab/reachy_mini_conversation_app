import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import { alpha, useTheme } from "@mui/material/styles";
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutline";
import MicIcon from "@mui/icons-material/Mic";

interface Props {
  isConnected: boolean;
}

export default function EmptyChat({ isConnected }: Props) {
  const theme = useTheme();
  const accentColor = isConnected ? "#10b981" : theme.palette.text.secondary;

  return (
    <Box
      sx={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 2,
        py: 10,
        animation: "fadeIn 0.5s ease-out",
        "@keyframes fadeIn": {
          "0%": { opacity: 0 },
          "100%": { opacity: 1 },
        },
      }}
    >
      <Box
        sx={{
          width: 64,
          height: 64,
          borderRadius: "50%",
          bgcolor: alpha(accentColor, 0.08),
          border: `2px solid ${alpha(accentColor, 0.15)}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          animation: isConnected ? "gentlePulse 2.5s ease-in-out infinite" : "none",
          "@keyframes gentlePulse": {
            "0%, 100%": { transform: "scale(1)", opacity: 1 },
            "50%": { transform: "scale(1.05)", opacity: 0.7 },
          },
        }}
      >
        {isConnected ? (
          <MicIcon sx={{ fontSize: 28, color: accentColor, opacity: 0.7 }} />
        ) : (
          <ChatBubbleOutlineIcon sx={{ fontSize: 28, color: accentColor, opacity: 0.4 }} />
        )}
      </Box>

      <Box sx={{ textAlign: "center" }}>
        <Typography
          variant="body1"
          sx={{
            fontWeight: 600,
            fontSize: "0.95rem",
            color: isConnected ? accentColor : "text.secondary",
            opacity: isConnected ? 0.9 : 0.6,
            mb: 0.5,
          }}
        >
          {isConnected ? "Listening..." : "No conversation yet"}
        </Typography>
        <Typography
          variant="body2"
          sx={{
            color: "text.secondary",
            opacity: 0.45,
            fontSize: "0.8rem",
            maxWidth: 260,
          }}
        >
          {isConnected
            ? "Start talking - your conversation will appear here"
            : "Tap the mic button below to start"}
        </Typography>
      </Box>
    </Box>
  );
}
