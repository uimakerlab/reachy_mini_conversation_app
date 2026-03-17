import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Card from "@mui/material/Card";
import CardActionArea from "@mui/material/CardActionArea";
import IconButton from "@mui/material/IconButton";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import EditOutlinedIcon from "@mui/icons-material/EditOutlined";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import AddIcon from "@mui/icons-material/Add";
import type { ProfileAvatar } from "../config/builtinProfiles";

interface Props {
  name: string;
  description: string;
  avatar: ProfileAvatar;
  selected: boolean;
  onSelect: () => void;
  onDetails: () => void;
}

const cardSx = (selected: boolean) => ({
  position: "relative" as const,
  display: "flex",
  flexDirection: "column" as const,
  borderColor: selected ? "primary.main" : "divider",
  borderWidth: selected ? 2 : 1,
  transition: "all 0.2s ease",
  "&:hover": {
    borderColor: selected ? "primary.main" : "text.secondary",
    "& .details-btn": { opacity: 1 },
  },
});

export function ProfileCard({ name, description, avatar, selected, onSelect, onDetails }: Props) {
  return (
    <Card variant="outlined" sx={cardSx(selected)}>
      <CardActionArea
        onClick={onSelect}
        sx={{ p: 2, display: "flex", flexDirection: "column", alignItems: "center", gap: 1.5, flex: 1 }}
      >
        <Box
          sx={{
            width: 88,
            height: 88,
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
        <Box sx={{ textAlign: "center", minHeight: 48 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, lineHeight: 1.3 }}>
            {name}
          </Typography>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ display: "block", mt: 0.25, lineHeight: 1.3 }}
          >
            {description}
          </Typography>
        </Box>
      </CardActionArea>

      {selected && (
        <CheckCircleIcon
          color="primary"
          sx={{ position: "absolute", top: 8, left: 8, fontSize: 18 }}
        />
      )}

      <IconButton
        className="details-btn"
        size="small"
        onClick={(e) => { e.stopPropagation(); onDetails(); }}
        sx={{
          position: "absolute",
          top: 6,
          right: 6,
          opacity: selected ? 0.7 : 0,
          transition: "opacity 0.15s ease",
          bgcolor: "background.paper",
          "&:hover": { bgcolor: "action.hover", opacity: 1 },
        }}
      >
        <InfoOutlinedIcon sx={{ fontSize: 16 }} />
      </IconButton>
    </Card>
  );
}

interface SavedCustomProps {
  name: string;
  selected: boolean;
  onSelect: () => void;
  onEdit: () => void;
  onDelete: () => void;
}

export function SavedCustomCard({ name, selected, onSelect, onEdit, onDelete }: SavedCustomProps) {
  const initial = name.trim() ? name.trim()[0].toUpperCase() : "?";
  return (
    <Card variant="outlined" sx={cardSx(selected)}>
      <CardActionArea
        onClick={onSelect}
        sx={{ p: 2, display: "flex", flexDirection: "column", alignItems: "center", gap: 1.5, flex: 1 }}
      >
        <Box
          sx={{
            width: 64,
            height: 64,
            borderRadius: "50%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            border: "2px solid",
            borderColor: "primary.main",
            color: "primary.main",
            fontSize: "1.3rem",
            fontWeight: 700,
            bgcolor: "primary.50",
          }}
        >
          {initial}
        </Box>
        <Box sx={{ textAlign: "center", minHeight: 48 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, lineHeight: 1.3 }}>
            {name}
          </Typography>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ display: "block", mt: 0.25, lineHeight: 1.3 }}
          >
            Custom profile
          </Typography>
        </Box>
      </CardActionArea>

      {selected && (
        <CheckCircleIcon
          color="primary"
          sx={{ position: "absolute", top: 8, left: 8, fontSize: 18 }}
        />
      )}

      <Box
        className="details-btn"
        sx={{
          position: "absolute",
          top: 6,
          right: 6,
          display: "flex",
          flexDirection: "column",
          gap: 0.25,
          opacity: selected ? 0.7 : 0,
          transition: "opacity 0.15s ease",
        }}
      >
        <IconButton
          size="small"
          onClick={(e) => { e.stopPropagation(); onEdit(); }}
          sx={{ bgcolor: "background.paper", "&:hover": { bgcolor: "action.hover", opacity: 1 } }}
        >
          <EditOutlinedIcon sx={{ fontSize: 16 }} />
        </IconButton>
        <IconButton
          size="small"
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          sx={{ bgcolor: "background.paper", "&:hover": { bgcolor: "error.main", color: "error.contrastText", opacity: 1 } }}
        >
          <DeleteOutlineIcon sx={{ fontSize: 16 }} />
        </IconButton>
      </Box>
    </Card>
  );
}

interface NewCustomProps {
  onSelect: () => void;
}

export function NewCustomCard({ onSelect }: NewCustomProps) {
  return (
    <Card
      variant="outlined"
      sx={{
        position: "relative",
        borderColor: "divider",
        borderWidth: 1,
        borderStyle: "dashed",
        transition: "all 0.2s ease",
        display: "flex",
        flexDirection: "column",
        "&:hover": { borderColor: "text.secondary" },
      }}
    >
      <CardActionArea
        onClick={onSelect}
        sx={{ p: 2, display: "flex", flexDirection: "column", alignItems: "center", gap: 1.5, flex: 1 }}
      >
        <Box
          sx={{
            width: 64,
            height: 64,
            borderRadius: "50%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            border: "2px dashed",
            borderColor: "text.secondary",
            color: "text.secondary",
          }}
        >
          <AddIcon />
        </Box>
        <Box sx={{ textAlign: "center", minHeight: 48 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, lineHeight: 1.3 }}>
            New Profile
          </Typography>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ display: "block", mt: 0.25, lineHeight: 1.3 }}
          >
            Create a personality
          </Typography>
        </Box>
      </CardActionArea>
    </Card>
  );
}
