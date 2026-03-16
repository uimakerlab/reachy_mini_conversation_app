import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import Box from "@mui/material/Box";
import Alert from "@mui/material/Alert";
import Snackbar from "@mui/material/Snackbar";
import Header from "./components/Header";
import ChatPanel from "./components/ChatPanel";
import AudioControls from "./components/AudioControls";
import ProfileDrawer from "./components/ProfileDrawer";
import SettingsDialog from "./components/SettingsDialog";
import WelcomeScreen from "./components/WelcomeScreen";
import ApiKeySetup from "./components/ApiKeySetup";
import { useSettings } from "./hooks/useSettings";
import { useChat } from "./hooks/useChat";
import { useRealtime } from "./hooks/useRealtime";
import { getBuiltinProfile } from "./config/builtinProfiles";
import type { BuiltinProfile } from "./config/builtinProfiles";

export default function App() {
  const { settings, update, hasKey } = useSettings();
  const chat = useChat();
  const { status, error, robotConnected, connect, disconnect, getLocalStream } = useRealtime(settings, chat);

  const [toastError, setToastError] = useState<string | null>(null);

  useEffect(() => {
    if (error) setToastError(error);
  }, [error]);

  const [profilesOpen, setProfilesOpen] = useState(false);
  const [profilesInitialCreate, setProfilesInitialCreate] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const isConnected = status === "connected";
  const isConnecting = status === "connecting";

  const activeBuiltin = useMemo(
    () => getBuiltinProfile(settings.profileId) ?? null,
    [settings.profileId],
  );
  const activeCustom = useMemo(
    () => settings.customProfiles.find((p) => p.id === settings.profileId) ?? null,
    [settings.profileId, settings.customProfiles],
  );

  const openProfiles = useCallback(() => { setProfilesInitialCreate(false); setProfilesOpen(true); }, []);
  const closeProfiles = useCallback(() => { setProfilesOpen(false); setProfilesInitialCreate(false); }, []);
  const openSettings = useCallback(() => setSettingsOpen(true), []);
  const closeSettings = useCallback(() => setSettingsOpen(false), []);

  // Auto-connect after profile selection from welcome screen
  const pendingConnectRef = useRef(false);

  useEffect(() => {
    if (pendingConnectRef.current && settings.onboardingDone && status === "disconnected") {
      pendingConnectRef.current = false;
      connect();
    }
  }, [settings.onboardingDone, settings.profileId, status, connect]);

  const handleWelcomeSelect = useCallback((profile: BuiltinProfile) => {
    pendingConnectRef.current = true;
    update({
      profileId: profile.id,
      voice: profile.voice,
      customInstructions: "",
      onboardingDone: true,
    });
  }, [update]);

  const handleWelcomeSkip = useCallback(() => {
    update({ onboardingDone: true });
  }, [update]);

  const handleWelcomeCreateCustom = useCallback(() => {
    update({ onboardingDone: true });
    setProfilesInitialCreate(true);
    setProfilesOpen(true);
  }, [update]);

  if (!hasKey) {
    return <ApiKeySetup onSave={(key) => update({ openaiApiKey: key })} />;
  }

  if (!settings.onboardingDone) {
    return (
      <WelcomeScreen
        onSelect={handleWelcomeSelect}
        onCreateCustom={handleWelcomeCreateCustom}
        onSkip={handleWelcomeSkip}
      />
    );
  }

  return (
    <Box
      sx={{
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        bgcolor: "background.default",
      }}
    >
      <Header
        builtinProfile={activeBuiltin}
        customProfile={activeCustom}
        onOpenProfiles={openProfiles}
        onOpenSettings={openSettings}
        isConnected={isConnected}
        isConnecting={isConnecting}
        robotConnected={robotConnected}
      />

      <ChatPanel messages={chat.messages} isConnected={isConnected} />
      <AudioControls status={status} onConnect={connect} onDisconnect={disconnect} getLocalStream={getLocalStream} />

      <Snackbar
        open={!!toastError}
        autoHideDuration={8000}
        onClose={() => setToastError(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
        sx={{
          "& .MuiSnackbarContent-root": { p: 0, bgcolor: "transparent", boxShadow: "none" },
        }}
      >
        <Alert
          severity="error"
          variant="filled"
          onClose={() => setToastError(null)}
          sx={{
            maxWidth: 420,
            backdropFilter: "blur(16px)",
            bgcolor: "rgba(211, 47, 47, 0.85)",
            borderRadius: 3,
            boxShadow: "0 8px 32px rgba(0,0,0,0.3)",
            fontSize: "0.82rem",
          }}
        >
          {toastError}
        </Alert>
      </Snackbar>

      <ProfileDrawer
        open={profilesOpen}
        onClose={closeProfiles}
        settings={settings}
        onUpdate={update}
        initialCreate={profilesInitialCreate}
      />

      <SettingsDialog
        open={settingsOpen}
        onClose={closeSettings}
        settings={settings}
        onUpdate={update}
        hasKey={hasKey}
      />
    </Box>
  );
}
