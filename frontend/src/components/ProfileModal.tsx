import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import Divider from "@mui/material/Divider";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogActions from "@mui/material/DialogActions";
import IconButton from "@mui/material/IconButton";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import DirectionsRunIcon from "@mui/icons-material/DirectionsRun";
import EmojiEmotionsOutlinedIcon from "@mui/icons-material/EmojiEmotionsOutlined";
import CameraAltOutlinedIcon from "@mui/icons-material/CameraAltOutlined";
import VolumeUpOutlinedIcon from "@mui/icons-material/VolumeUpOutlined";
import HandymanOutlinedIcon from "@mui/icons-material/HandymanOutlined";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import SaveOutlinedIcon from "@mui/icons-material/SaveOutlined";
import type { BuiltinProfile, ProfileAvatar } from "../config/builtinProfiles";
import { ALL_TOOLS } from "../config/builtinProfiles";
import type { CustomProfile } from "../config/settings";

interface VoiceMeta {
  id: string;
  label: string;
  tone: string;
}

const VOICES: VoiceMeta[] = [
  { id: "alloy", label: "Alloy", tone: "Neutral, balanced" },
  { id: "ash", label: "Ash", tone: "Soft, conversational" },
  { id: "ballad", label: "Ballad", tone: "Warm, expressive" },
  { id: "cedar", label: "Cedar", tone: "Calm, friendly" },
  { id: "coral", label: "Coral", tone: "Bright, clear" },
  { id: "echo", label: "Echo", tone: "Deep, resonant" },
  { id: "fable", label: "Fable", tone: "Gentle, narrative" },
  { id: "onyx", label: "Onyx", tone: "Deep, authoritative" },
  { id: "nova", label: "Nova", tone: "Warm, natural" },
  { id: "sage", label: "Sage", tone: "Wise, measured" },
  { id: "shimmer", label: "Shimmer", tone: "Light, energetic" },
  { id: "verse", label: "Verse", tone: "Versatile, dynamic" },
];

interface ToolMeta {
  id: string;
  label: string;
  description: string;
}

interface ToolCategory {
  key: string;
  label: string;
  icon: React.ReactNode;
  tools: ToolMeta[];
}

const TOOL_CATEGORIES: ToolCategory[] = [
  {
    key: "movement",
    label: "Movement",
    icon: <DirectionsRunIcon fontSize="small" />,
    tools: [
      { id: "dance", label: "Dance", description: "Trigger a dance sequence" },
      { id: "stop_dance", label: "Stop Dance", description: "Stop current dance" },
      { id: "toggle_head_tracking", label: "Head Tracking", description: "Follow a person with the head" },
      { id: "move_head", label: "Move Head", description: "Point head in a direction" },
      { id: "nod", label: "Nod", description: "Nod yes" },
      { id: "shake_head", label: "Shake", description: "Shake head no" },
      { id: "wake_up", label: "Wake Up", description: "Wake robot from sleep" },
      { id: "go_to_sleep", label: "Sleep", description: "Put robot to sleep" },
    ],
  },
  {
    key: "expression",
    label: "Expression",
    icon: <EmojiEmotionsOutlinedIcon fontSize="small" />,
    tools: [
      { id: "play_emotion", label: "Play Emotion", description: "Display an emotion animation" },
      { id: "stop_emotion", label: "Stop Emotion", description: "Stop current emotion" },
    ],
  },
  {
    key: "perception",
    label: "Perception",
    icon: <CameraAltOutlinedIcon fontSize="small" />,
    tools: [
      { id: "take_photo", label: "Camera", description: "Capture a photo to see the environment" },
    ],
  },
  {
    key: "audio",
    label: "Audio",
    icon: <VolumeUpOutlinedIcon fontSize="small" />,
    tools: [
      { id: "set_volume", label: "Set Volume", description: "Adjust speaker volume" },
      { id: "get_volume", label: "Get Volume", description: "Check current volume level" },
    ],
  },
  {
    key: "utility",
    label: "Utility",
    icon: <HandymanOutlinedIcon fontSize="small" />,
    tools: [
      { id: "do_nothing", label: "Do Nothing", description: "Explicit no-op action" },
      { id: "web_search", label: "Web Search", description: "Search the web for information" },
    ],
  },
];

function VoiceSelector({ value, onChange }: { value: string; onChange?: (id: string) => void }) {
  return (
    <Box
      sx={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
        gap: 1,
      }}
    >
      {VOICES.map((v) => {
        const selected = v.id === value;
        return (
          <Box
            key={v.id}
            onClick={onChange ? () => onChange(v.id) : undefined}
            sx={{
              position: "relative",
              px: 1.5,
              py: 1.25,
              borderRadius: 2,
              border: 2,
              borderColor: selected ? "primary.main" : "divider",
              bgcolor: selected ? "action.selected" : "transparent",
              opacity: selected ? 1 : 0.35,
              cursor: onChange ? "pointer" : "default",
              transition: "all 0.15s ease",
              ...(onChange && {
                "&:hover": {
                  borderColor: selected ? "primary.main" : "text.disabled",
                  bgcolor: selected ? "action.selected" : "action.hover",
                },
              }),
            }}
          >
            {selected && (
              <CheckCircleIcon
                sx={{
                  position: "absolute",
                  top: 6,
                  right: 6,
                  fontSize: 16,
                  color: "primary.main",
                }}
              />
            )}
            <Typography
              variant="body2"
              sx={{
                fontWeight: 600,
                fontSize: "0.8rem",
                color: selected ? "primary.main" : "text.primary",
              }}
            >
              {v.label}
            </Typography>
            <Typography
              variant="caption"
              sx={{
                fontSize: "0.68rem",
                color: selected ? "primary.main" : "text.secondary",
                opacity: selected ? 0.7 : 1,
                lineHeight: 1.3,
              }}
            >
              {v.tone}
            </Typography>
          </Box>
        );
      })}
    </Box>
  );
}

interface BuiltinModalProps {
  open: boolean;
  profile: BuiltinProfile;
  onClose: () => void;
  onDuplicate: () => void;
}

export interface CustomModalSaveData {
  id?: string;
  name: string;
  instructions: string;
  voice: string;
  enabledTools: string[];
}

interface CustomModalProps {
  open: boolean;
  editing?: CustomProfile;
  onClose: () => void;
  onSave: (data: CustomModalSaveData) => void;
  onDelete?: () => void;
}

function ToolCard({ tool, enabled, onClick }: { tool: ToolMeta; enabled: boolean; onClick?: () => void }) {
  return (
    <Box
      key={tool.id}
      onClick={onClick}
      sx={{
        px: 1.5,
        py: 1,
        borderRadius: 2,
        border: 2,
        borderColor: enabled ? "primary.main" : "divider",
        bgcolor: enabled ? "action.selected" : "transparent",
        opacity: enabled ? 1 : onClick ? 0.55 : 0.35,
        cursor: onClick ? "pointer" : "default",
        transition: "all 0.15s ease",
        ...(onClick && {
          "&:hover": {
            opacity: 1,
            borderColor: enabled ? "primary.main" : "text.disabled",
            bgcolor: enabled ? "action.selected" : "action.hover",
          },
        }),
      }}
    >
      <Typography
        variant="body2"
        sx={{ fontWeight: 600, fontSize: "0.78rem", color: enabled ? "primary.main" : "text.primary" }}
      >
        {tool.label}
      </Typography>
      <Typography
        variant="caption"
        sx={{ fontSize: "0.66rem", color: enabled ? "primary.main" : "text.secondary", opacity: enabled ? 0.7 : 1, lineHeight: 1.3 }}
      >
        {tool.description}
      </Typography>
    </Box>
  );
}

function AvatarBadge({ avatar, size = 96 }: { avatar: ProfileAvatar; size?: number }) {
  return (
    <Box
      sx={{
        width: size,
        height: size,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: 0,
      }}
    >
      <img
        src={avatar.image}
        alt=""
        style={{ width: "100%", height: "100%", objectFit: "contain" }}
      />
    </Box>
  );
}

interface BuiltinViewerProps {
  profile: BuiltinProfile;
  onBack: () => void;
  onDuplicate: () => void;
}

export function BuiltinProfileViewer({
  profile,
  onBack,
  onDuplicate,
}: BuiltinViewerProps) {
  return (
    <>
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 1.5,
          px: 2.5,
          py: 2,
          borderBottom: 1,
          borderColor: "divider",
          flexShrink: 0,
        }}
      >
        <IconButton size="small" onClick={onBack} sx={{ mr: -0.5 }}>
          <ArrowBackIcon sx={{ fontSize: 20 }} />
        </IconButton>
        <AvatarBadge avatar={profile.avatar} size={40} />
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 700, fontSize: "0.95rem", lineHeight: 1.2 }}>
            {profile.name}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {profile.description}
          </Typography>
        </Box>
        <Button
          startIcon={<ContentCopyIcon />}
          onClick={onDuplicate}
          variant="outlined"
          size="small"
          sx={{ flexShrink: 0 }}
        >
          Duplicate
        </Button>
      </Box>

      {/* Body */}
      <Box sx={{ flex: 1, overflow: "auto", p: 2.5, display: "flex", flexDirection: "column", gap: 3 }}>
        <Box>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
            Instructions
          </Typography>
          <Box
            sx={{
              p: 2,
              bgcolor: "action.hover",
              borderRadius: 2,
              whiteSpace: "pre-wrap",
              fontFamily: "monospace",
              fontSize: "0.8rem",
              lineHeight: 1.6,
              color: "text.secondary",
            }}
          >
            {profile.instructions}
          </Box>
        </Box>

        <Divider />

        <Box>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1.5 }}>
            Voice
          </Typography>
          <VoiceSelector value={profile.voice} />
        </Box>

        <Divider />

        <Box>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1.5 }}>
            Available Tools
          </Typography>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
            {TOOL_CATEGORIES.map((cat) => {
              const enabledInCat = cat.tools.filter((t) => profile.enabledTools.includes(t.id));
              if (enabledInCat.length === 0) return null;
              return (
                <Box key={cat.key}>
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 0.5, color: "text.secondary" }}>
                    {cat.icon}
                    <Typography variant="caption" sx={{ fontWeight: 600, fontSize: "0.72rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      {cat.label}
                    </Typography>
                  </Box>
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1, pl: 3 }}>
                    {cat.tools.map((t) => (
                      <ToolCard key={t.id} tool={t} enabled={profile.enabledTools.includes(t.id)} />
                    ))}
                  </Box>
                </Box>
              );
            })}
          </Box>
        </Box>
      </Box>

    </>
  );
}

interface EditorProps {
  editing?: CustomProfile;
  onBack: () => void;
  onSave: (data: CustomModalSaveData) => void;
  onDelete?: () => void;
}

export function CustomProfileEditor({
  editing,
  onBack,
  onSave,
  onDelete,
}: EditorProps) {
  const [name, setName] = useState(editing?.name ?? "");
  const [instructions, setInstructions] = useState(editing?.instructions ?? "");
  const [voice, setVoice] = useState(editing?.voice ?? "cedar");
  const [tools, setTools] = useState<string[]>(editing?.enabledTools ?? [...ALL_TOOLS]);
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => {
    setName(editing?.name ?? "");
    setInstructions(editing?.instructions ?? "");
    setVoice(editing?.voice ?? "cedar");
    setTools(editing?.enabledTools ?? [...ALL_TOOLS]);
  }, [editing]);

  const toggleTool = (t: string) => {
    setTools((prev) =>
      prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t],
    );
  };

  const handleSave = () => {
    const trimmedName = name.trim() || "Untitled";
    onSave({
      id: editing?.id,
      name: trimmedName,
      instructions,
      voice,
      enabledTools: tools,
    });
  };

  const isEdit = !!editing;
  const title = isEdit ? `Edit - ${editing!.name}` : "New Custom Profile";
  const subtitle = isEdit ? "Modify this profile" : "Create a new personality";

  return (
    <>
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 1.5,
          px: 2.5,
          py: 2,
          borderBottom: 1,
          borderColor: "divider",
          flexShrink: 0,
        }}
      >
        <IconButton size="small" onClick={onBack} sx={{ mr: -0.5 }}>
          <ArrowBackIcon sx={{ fontSize: 20 }} />
        </IconButton>
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 700, fontSize: "0.95rem", lineHeight: 1.2 }}>
            {title}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {subtitle}
          </Typography>
        </Box>
        {isEdit && onDelete && (
          <Button
            size="small"
            variant="outlined"
            color="error"
            endIcon={<DeleteOutlineIcon sx={{ fontSize: 16 }} />}
            onClick={() => setConfirmDelete(true)}
            sx={{ flexShrink: 0, fontWeight: 600, fontSize: "0.8rem" }}
          >
            Delete
          </Button>
        )}
        <Button
          variant="contained"
          size="small"
          endIcon={<SaveOutlinedIcon sx={{ fontSize: 16 }} />}
          onClick={handleSave}
          disabled={!name.trim()}
          disableElevation
          sx={{ fontWeight: 600, fontSize: "0.8rem", flexShrink: 0 }}
        >
          {isEdit ? "Save" : "Create"}
        </Button>
      </Box>

      {/* Body */}
      <Box sx={{ flex: 1, overflow: "auto", p: 2.5, display: "flex", flexDirection: "column", gap: 3 }}>
        <Box>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
            Name
          </Typography>
          <TextField
            fullWidth
            size="small"
            placeholder="e.g. My Pirate Bot"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </Box>

        <Box>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
            Instructions
          </Typography>
          <TextField
            fullWidth
            multiline
            minRows={6}
            maxRows={14}
            size="small"
            placeholder="Write your custom personality instructions..."
            value={instructions}
            onChange={(e) => setInstructions(e.target.value)}
            sx={{ "& .MuiOutlinedInput-root": { fontFamily: "monospace", fontSize: "0.8rem" } }}
          />
        </Box>

        <Divider />

        <Box>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1.5 }}>
            Voice
          </Typography>
          <VoiceSelector value={voice} onChange={setVoice} />
        </Box>

        <Divider />

        <Box>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1.5 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
              Available Tools
            </Typography>
            <Box sx={{ display: "flex", gap: 0.5 }}>
              <Button size="small" sx={{ fontSize: "0.7rem", minWidth: 0, px: 1 }} onClick={() => setTools([...ALL_TOOLS])}>
                All
              </Button>
              <Button size="small" sx={{ fontSize: "0.7rem", minWidth: 0, px: 1 }} onClick={() => setTools([])}>
                None
              </Button>
            </Box>
          </Box>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {TOOL_CATEGORIES.map((cat) => {
              const catToolIds = cat.tools.map((t) => t.id);
              const enabledCount = catToolIds.filter((id) => tools.includes(id)).length;
              const allEnabled = enabledCount === catToolIds.length;

              return (
                <Box key={cat.key}>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 0.75,
                      mb: 0.75,
                    }}
                  >
                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, color: "text.secondary" }}>
                      {cat.icon}
                    </Box>
                    <Typography variant="caption" sx={{ fontWeight: 600, fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "text.secondary", flex: 1 }}>
                      {cat.label}
                    </Typography>
                    <Box sx={{ display: "flex", gap: 0.25 }}>
                      <Button
                        size="small"
                        sx={{ fontSize: "0.65rem", minWidth: 0, px: 0.75, py: 0, color: allEnabled ? "primary.main" : "text.disabled" }}
                        onClick={() => setTools((prev) => [...new Set([...prev, ...catToolIds])])}
                      >
                        All
                      </Button>
                      <Button
                        size="small"
                        sx={{ fontSize: "0.65rem", minWidth: 0, px: 0.75, py: 0, color: enabledCount === 0 ? "primary.main" : "text.disabled" }}
                        onClick={() => setTools((prev) => prev.filter((id) => !catToolIds.includes(id)))}
                      >
                        None
                      </Button>
                    </Box>
                  </Box>
                  <Box sx={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 1 }}>
                    {cat.tools.map((t) => (
                      <ToolCard key={t.id} tool={t} enabled={tools.includes(t.id)} onClick={() => toggleTool(t.id)} />
                    ))}
                  </Box>
                </Box>
              );
            })}
          </Box>
        </Box>
      </Box>

      {/* Delete confirmation dialog */}
      <Dialog open={confirmDelete} onClose={() => setConfirmDelete(false)} maxWidth="xs">
        <DialogTitle sx={{ fontWeight: 700, fontSize: "1rem" }}>
          Delete this profile?
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary">
            "{editing?.name}" will be permanently deleted. This cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions sx={{ px: 3, pb: 2 }}>
          <Button onClick={() => setConfirmDelete(false)} color="inherit" size="small">
            Cancel
          </Button>
          <Button
            onClick={() => { setConfirmDelete(false); onDelete?.(); }}
            color="error"
            variant="contained"
            size="small"
            disableElevation
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
