import { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import Fade from "@mui/material/Fade";
import LockOutlinedIcon from "@mui/icons-material/LockOutlined";

interface Props {
  onSave: (key: string) => void;
}

export default function ApiKeySetup({ onSave }: Props) {
  const [keyInput, setKeyInput] = useState("");

  const handleSave = () => {
    const key = keyInput.trim();
    if (key) onSave(key);
  };

  return (
    <Box
      sx={{
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        bgcolor: "background.default",
        px: 3,
      }}
    >
      <Fade in timeout={600}>
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            maxWidth: 400,
            width: "100%",
          }}
        >
          <Box
            sx={{
              width: 160,
              height: 200,
              mb: 3,
              opacity: 0.9,
              filter: "drop-shadow(0 8px 24px rgba(0,0,0,0.12))",
              animation: "float 4s ease-in-out infinite",
              "@keyframes float": {
                "0%, 100%": { transform: "translateY(0)" },
                "50%": { transform: "translateY(-8px)" },
              },
            }}
          >
            <img
              src="/reachy-head.svg"
              alt="Reachy Mini"
              style={{ width: "100%", height: "100%", objectFit: "contain" }}
            />
          </Box>

          <Typography
            variant="h4"
            sx={{
              fontWeight: 800,
              textAlign: "center",
              letterSpacing: "-0.02em",
              fontSize: { xs: "1.5rem", sm: "1.75rem" },
            }}
          >
            Reachy Conversation App
          </Typography>

          <Typography
            color="text.secondary"
            sx={{
              mt: 1,
              mb: 4,
              textAlign: "center",
              lineHeight: 1.6,
              fontSize: "0.95rem",
            }}
          >
            Enter your OpenAI API key to start chatting.
          </Typography>

          <Box
            sx={{
              width: "100%",
              p: 3,
              borderRadius: 3,
              bgcolor: "background.paper",
              border: 1,
              borderColor: "divider",
            }}
          >
            <TextField
              fullWidth
              size="small"
              type="password"
              placeholder="sk-proj-..."
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSave()}
              sx={{
                "& .MuiOutlinedInput-root": {
                  borderRadius: 2,
                  fontSize: "0.9rem",
                },
              }}
            />
            <Button
              fullWidth
              variant="contained"
              onClick={handleSave}
              disabled={!keyInput.trim()}
              disableElevation
              sx={{
                mt: 2,
                py: 1.2,
                borderRadius: 2,
                fontWeight: 600,
                textTransform: "none",
                fontSize: "0.95rem",
              }}
            >
              Continue
            </Button>
          </Box>

          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 0.75,
              mt: 2.5,
              color: "text.disabled",
            }}
          >
            <LockOutlinedIcon sx={{ fontSize: 14 }} />
            <Typography variant="caption" sx={{ fontSize: "0.75rem" }}>
              Stored locally in your browser - never sent to our servers
            </Typography>
          </Box>
        </Box>
      </Fade>
    </Box>
  );
}
