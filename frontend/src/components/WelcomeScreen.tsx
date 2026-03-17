import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import ButtonBase from "@mui/material/ButtonBase";
import Fade from "@mui/material/Fade";
import { keyframes } from "@mui/material/styles";
import AddIcon from "@mui/icons-material/Add";
import SvgIcon from "@mui/material/SvgIcon";
import { useTheme } from "@mui/material/styles";

function DicePairIcon({ diceBg, ...props }: React.ComponentProps<typeof SvgIcon> & { diceBg?: string }) {
  const fill = diceBg ?? "none";
  return (
    <SvgIcon {...props} viewBox="0 0 32 24">
      {/* Left die */}
      <g transform="rotate(-10 9 12)">
        <rect x="2" y="5" width="14" height="14" rx="3" fill={fill} stroke="currentColor" strokeWidth="1.3" />
        <circle cx="6" cy="9" r="1.1" fill="currentColor" />
        <circle cx="12" cy="9" r="1.1" fill="currentColor" />
        <circle cx="6" cy="15" r="1.1" fill="currentColor" />
        <circle cx="12" cy="15" r="1.1" fill="currentColor" />
      </g>
      {/* Right die */}
      <g transform="rotate(10 23 12)">
        <rect x="16" y="5" width="14" height="14" rx="3" fill={fill} stroke="currentColor" strokeWidth="1.3" />
        <circle cx="20" cy="9" r="1.1" fill="currentColor" />
        <circle cx="23" cy="12" r="1.1" fill="currentColor" />
        <circle cx="26" cy="15" r="1.1" fill="currentColor" />
      </g>
    </SvgIcon>
  );
}
import { BUILTIN_PROFILES } from "../config/builtinProfiles";
import type { BuiltinProfile } from "../config/builtinProfiles";
import ReachiesCarousel from "./ReachiesCarousel";

const pulseGlow = keyframes`
  0%, 100% { opacity: 0.3; transform: translate(-50%, -50%) scale(1); }
  50% { opacity: 0.45; transform: translate(-50%, -50%) scale(1.05); }
`;

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
      {/* Hero with carousel */}
      <Fade in timeout={600}>
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            pt: { xs: 5, sm: 7 },
            pb: { xs: 4, sm: 5 },
            px: 3,
            flexShrink: 0,
          }}
        >
          <Box sx={{ position: "relative", mb: 1 }}>
            <Box
              sx={{
                position: "absolute",
                top: "50%",
                left: "50%",
                width: 100,
                height: 100,
                borderRadius: "50%",
                bgcolor: "primary.main",
                filter: "blur(50px)",
                animation: `${pulseGlow} 4s ease-in-out infinite`,
                pointerEvents: "none",
              }}
            />
            <ReachiesCarousel width={150} height={150} interval={1800} />
          </Box>
          <Typography
            variant="h4"
            sx={{
              fontWeight: 800,
              mt: 1.5,
              textAlign: "center",
              fontSize: { xs: "1.6rem", sm: "1.85rem" },
            }}
          >
            Hey! I'm{" "}
            <Box component="span" sx={{ color: "primary.main" }}>
              Reachy Mini
            </Box>
            .
          </Typography>
          <Typography
            variant="body1"
            sx={{
              mt: 1,
              textAlign: "center",
              maxWidth: 400,
              color: "text.secondary",
              fontSize: "1rem",
            }}
          >
            Pick a <strong>personality</strong> and let's <strong>chat</strong>.
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
            <CreateCustomTile onSelect={onCreateCustom} />
            <RandomTile onSelect={() => {
              const pick = BUILTIN_PROFILES[Math.floor(Math.random() * BUILTIN_PROFILES.length)];
              onSelect(pick);
            }} />
            {BUILTIN_PROFILES.map((p) => (
              <PersonalityTile key={p.id} profile={p} onSelect={() => onSelect(p)} />
            ))}
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

function RandomTile({ onSelect }: { onSelect: () => void }) {
  const theme = useTheme();
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
        },
      }}
    >
      <Box sx={{ width: 88, height: 88, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
        <DicePairIcon diceBg={theme.palette.background.paper} sx={{ fontSize: 56, color: "text.secondary" }} />
      </Box>
      <Box sx={{ textAlign: "center", minHeight: 48 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 600, lineHeight: 1.3 }}>
          Surprise Me
        </Typography>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: "block", mt: 0.25, lineHeight: 1.3 }}
        >
          Random personality
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
          Custom personality
        </Typography>
      </Box>
    </ButtonBase>
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

