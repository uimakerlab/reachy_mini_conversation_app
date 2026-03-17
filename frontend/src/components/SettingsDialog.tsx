import { useState } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import IconButton from "@mui/material/IconButton";
import InputAdornment from "@mui/material/InputAdornment";
import CloseIcon from "@mui/icons-material/Close";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline";
import VisibilityIcon from "@mui/icons-material/Visibility";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import KeyOutlinedIcon from "@mui/icons-material/KeyOutlined";
import RouterOutlinedIcon from "@mui/icons-material/RouterOutlined";
import type { AppSettings } from "../config/settings";

interface Props {
  open: boolean;
  onClose: () => void;
  settings: AppSettings;
  onUpdate: (patch: Partial<AppSettings>) => void;
  hasKey: boolean;
}

function SectionCard({
  icon,
  title,
  status,
  children,
}: {
  icon: React.ReactNode;
  title: string;
  status?: { ok: boolean; label: string };
  children: React.ReactNode;
}) {
  return (
    <Box
      sx={{
        border: 2,
        borderColor: status?.ok ? "success.main" : "divider",
        borderRadius: 2.5,
        p: 2.5,
        transition: "border-color 0.2s ease",
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
        <Box sx={{ color: "text.secondary", display: "flex" }}>{icon}</Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 700, flex: 1 }}>
          {title}
        </Typography>
        {status && (
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 0.5,
              color: status.ok ? "success.main" : "text.disabled",
            }}
          >
            {status.ok ? (
              <CheckCircleOutlineIcon sx={{ fontSize: 16 }} />
            ) : (
              <ErrorOutlineIcon sx={{ fontSize: 16 }} />
            )}
            <Typography variant="caption" sx={{ fontSize: "0.7rem", fontWeight: 600 }}>
              {status.label}
            </Typography>
          </Box>
        )}
      </Box>
      {children}
    </Box>
  );
}

export default function SettingsDialog({ open, onClose, settings, onUpdate, hasKey }: Props) {
  const [keyInput, setKeyInput] = useState("");
  const [showKey, setShowKey] = useState(false);

  const handleSaveKey = () => {
    const key = keyInput.trim();
    if (!key) return;
    onUpdate({ openaiApiKey: key });
    setKeyInput("");
    setShowKey(false);
  };

  const handleClearKey = () => {
    onUpdate({ openaiApiKey: "" });
    setKeyInput("");
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xs" fullWidth>
      <DialogTitle component="div" sx={{ display: "flex", alignItems: "center", pr: 6 }}>
        <Typography variant="h6" sx={{ fontWeight: 700, fontSize: "1rem", flex: 1 }}>
          Settings
        </Typography>
        <IconButton onClick={onClose} sx={{ position: "absolute", right: 12, top: 12 }}>
          <CloseIcon />
        </IconButton>
      </DialogTitle>

      <DialogContent sx={{ display: "flex", flexDirection: "column", gap: 2, pb: 3 }}>
        <SectionCard
          icon={<KeyOutlinedIcon sx={{ fontSize: 20 }} />}
          title="OpenAI API Key"
          status={hasKey ? { ok: true, label: "Configured" } : { ok: false, label: "Missing" }}
        >
          <Box sx={{ display: "flex", gap: 1 }}>
            <TextField
              fullWidth
              size="small"
              type={showKey ? "text" : "password"}
              placeholder={hasKey ? "Enter new key to update" : "sk-proj-..."}
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSaveKey()}
              slotProps={{
                input: {
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        size="small"
                        onClick={() => setShowKey((v) => !v)}
                        edge="end"
                        sx={{ mr: -0.5 }}
                      >
                        {showKey ? (
                          <VisibilityOffIcon sx={{ fontSize: 18 }} />
                        ) : (
                          <VisibilityIcon sx={{ fontSize: 18 }} />
                        )}
                      </IconButton>
                    </InputAdornment>
                  ),
                },
              }}
            />
            <Button
              variant="outlined"
              onClick={handleSaveKey}
              disabled={!keyInput.trim()}
              sx={{ minWidth: 72 }}
            >
              {hasKey ? "Update" : "Save"}
            </Button>
          </Box>
          {hasKey && (
            <Box sx={{ display: "flex", justifyContent: "flex-end", mt: 1 }}>
              <Button size="small" color="error" variant="text" onClick={handleClearKey}>
                Remove key
              </Button>
            </Box>
          )}
        </SectionCard>

        <SectionCard
          icon={<RouterOutlinedIcon sx={{ fontSize: 20 }} />}
          title="Robot Daemon"
          status={
            settings.daemonUrl
              ? { ok: true, label: settings.daemonUrl }
              : { ok: true, label: "Same origin" }
          }
        >
          <TextField
            fullWidth
            size="small"
            placeholder="http://reachy-mini.local:8000"
            value={settings.daemonUrl}
            onChange={(e) => onUpdate({ daemonUrl: e.target.value })}
          />
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ mt: 1, display: "block", lineHeight: 1.4 }}
          >
            Leave empty to use the same origin. Set to the robot IP when running the frontend locally.
          </Typography>
        </SectionCard>
      </DialogContent>
    </Dialog>
  );
}
