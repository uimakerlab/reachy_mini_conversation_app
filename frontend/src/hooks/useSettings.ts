import { useState, useEffect, useCallback, useRef } from "react";
import {
  getSettings,
  updateSettings,
  fetchConfigFromBackend,
  type AppSettings,
} from "../config/settings";

export function useSettings() {
  const [settings, setSettings] = useState<AppSettings>(() => {
    const s = getSettings();
    if (import.meta.env.DEV) {
      const reset = updateSettings({ onboardingDone: false });
      return reset;
    }
    return s;
  });
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  useEffect(() => {
    let mounted = true;
    fetchConfigFromBackend().then((backendConfig) => {
      if (!mounted) return;
      if (backendConfig.openaiApiKey && !settingsRef.current.openaiApiKey) {
        const updated = updateSettings({ openaiApiKey: backendConfig.openaiApiKey });
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
