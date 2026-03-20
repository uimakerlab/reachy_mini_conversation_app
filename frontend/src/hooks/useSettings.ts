import { useState, useEffect, useCallback, useRef } from "react";
import {
  getSettings,
  updateSettings,
  fetchConfigFromBackend,
  type AppSettings,
} from "../config/settings";

export function useSettings() {
  const [settings, setSettings] = useState<AppSettings>(getSettings);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  useEffect(() => {
    let mounted = true;
    fetchConfigFromBackend().then((backendConfig) => {
      if (!mounted) return;
      const patch: Partial<AppSettings> = {};
      if (backendConfig.openaiApiKey && !settingsRef.current.openaiApiKey) {
        patch.openaiApiKey = backendConfig.openaiApiKey;
      }
      if (backendConfig.devMode) {
        patch.onboardingDone = false;
      }
      if (Object.keys(patch).length > 0) {
        const updated = updateSettings(patch);
        setSettings(updated);
      }
    });
    return () => { mounted = false; };
  }, []);

  const update = useCallback((patch: Partial<AppSettings>) => {
    const updated = updateSettings(patch);
    setSettings(updated);
  }, []);

  return { settings, update, hasKey: Boolean(settings.openaiApiKey) };
}
