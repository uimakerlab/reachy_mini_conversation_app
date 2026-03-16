import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import ButtonBase from "@mui/material/ButtonBase";
import Fade from "@mui/material/Fade";
import AddIcon from "@mui/icons-material/Add";
import { BUILTIN_PROFILES } from "../config/builtinProfiles";
import type { BuiltinProfile } from "../config/builtinProfiles";

interface Props {
  onSelect: (profile: BuiltinProfile) => void;
  onCreateCustom: () => void;
  onSkip: () => void;
}

export default function WelcomeScreen({ onSelect, onCreateCustom, onSkip }: Props) {
  return (
    <Box
      sx={{
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        bgcolor: "background.default",
        overflow: "auto",
      }}
    >
      {/* Hero */}
      <Fade in timeout={600}>
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            pt: { xs: 4, sm: 6 },
            pb: 2,
            px: 3,
            flexShrink: 0,
          }}
        >
          <img
            src="/avatars/default.svg"
            alt="Reachy Mini"
            style={{ width: 120, height: 120, objectFit: "contain" }}
          />
          <Typography
            variant="h4"
            sx={{
              fontWeight: 800,
              mt: 2,
              textAlign: "center",
              fontSize: { xs: "1.5rem", sm: "1.75rem" },
            }}
          >
            Hey! I'm Reachy Mini.
          </Typography>
          <Typography
            variant="body1"
            color="text.secondary"
            sx={{ mt: 1, textAlign: "center", maxWidth: 400 }}
          >
            Pick a personality and let's chat.
          </Typography>
        </Box>
      </Fade>

      {/* Personality grid */}
      <Fade in timeout={900}>
        <Box
          sx={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            px: { xs: 2, sm: 4 },
            pb: 4,
          }}
        >
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))",
              gap: 1.5,
              maxWidth: 720,
              width: "100%",
            }}
          >
            {BUILTIN_PROFILES.map((p) => (
              <PersonalityTile key={p.id} profile={p} onSelect={() => onSelect(p)} />
            ))}
            <CreateCustomTile onSelect={onCreateCustom} />
          </Box>

          <ButtonBase
            onClick={onSkip}
            sx={{
              mt: 4,
              mb: 2,
              px: 2,
              py: 1,
              borderRadius: 2,
              color: "text.secondary",
              fontSize: "0.85rem",
              "&:hover": { color: "text.primary", bgcolor: "action.hover" },
              transition: "all 0.15s ease",
            }}
          >
            Just start chatting &rarr;
          </ButtonBase>
        </Box>
      </Fade>
    </Box>
  );
}

function PersonalityTile({
  profile,
  onSelect,
}: {
  profile: BuiltinProfile;
  onSelect: () => void;
}) {
  const { gradient } = profile.avatar;

  return (
    <ButtonBase
      onClick={onSelect}
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 1.5,
        p: 2,
        borderRadius: 1,
        border: 1,
        borderColor: "divider",
        bgcolor: "background.paper",
        transition: "all 0.2s ease",
        "&:hover": {
          borderColor: "primary.main",
          transform: "translateY(-2px)",
          boxShadow: `0 4px 20px ${gradient[0]}22`,
        },
      }}
    >
      <Box sx={{ width: 88, height: 88, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
        <img
          src={profile.avatar.image}
          alt=""
          style={{ width: "100%", height: "100%", objectFit: "contain" }}
        />
      </Box>
      <Box sx={{ textAlign: "center", minHeight: 48 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 600, lineHeight: 1.3 }}>
          {profile.name}
        </Typography>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: "block", mt: 0.25, lineHeight: 1.3 }}
        >
          {profile.description}
        </Typography>
      </Box>
    </ButtonBase>
  );
}

function CreateCustomTile({ onSelect }: { onSelect: () => void }) {
  return (
    <ButtonBase
      onClick={onSelect}
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 1.5,
        p: 2,
        borderRadius: 1,
        border: 1,
        borderStyle: "dashed",
        borderColor: "divider",
        bgcolor: "background.paper",
        transition: "all 0.2s ease",
        "&:hover": {
          borderColor: "primary.main",
          transform: "translateY(-2px)",
        },
      }}
    >
      <Box
        sx={{
          width: 64,
          height: 64,
          my: "12px",
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
          Create Your Own
        </Typography>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: "block", mt: 0.25, lineHeight: 1.3 }}
        >
          Write a custom personality
        </Typography>
      </Box>
    </ButtonBase>
  );
}
