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
    fetchConfigFromBackend().then((backendConfig) => {
      if (backendConfig.openaiApiKey && !settingsRef.current.openaiApiKey) {
        const updated = updateSettings({ openaiApiKey: backendConfig.openaiApiKey });
        setSettings(updated);
      }
    });
  }, []);

  const update = useCallback((patch: Partial<AppSettings>) => {
    const updated = updateSettings(patch);
    setSettings(updated);
  }, []);

  const clearApiKey = useCallback(() => {
    update({ openaiApiKey: "" });
  }, [update]);

  return { settings, update, clearApiKey, hasKey: Boolean(settings.openaiApiKey) };
}
