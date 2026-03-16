const STORAGE_KEY = "reachy_conversation_settings";

export interface CustomProfile {
  id: string;
  name: string;
  instructions: string;
  voice: string;
  enabledTools: string[];
}

export interface AppSettings {
  openaiApiKey: string;
  voice: string;
  daemonUrl: string;
  profileId: string;
  customInstructions: string;
  customEnabledTools: string[];
  customProfiles: CustomProfile[];
  onboardingDone: boolean;
}

const DEFAULTS: AppSettings = {
  openaiApiKey: "",
  voice: "cedar",
  daemonUrl: "",
  profileId: "default",
  customInstructions: "",
  customEnabledTools: [],
  customProfiles: [],
  onboardingDone: false,
};

function load(): AppSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      // TODO: remove onboardingDone override once onboarding testing is done
      return { ...DEFAULTS, ...parsed, customProfiles: parsed.customProfiles ?? [], onboardingDone: false };
    }
  } catch { /* ignore */ }
  return { ...DEFAULTS };
}

function save(settings: AppSettings): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

export function getSettings(): AppSettings {
  return load();
}

export function updateSettings(patch: Partial<AppSettings>): AppSettings {
  const s = { ...load(), ...patch };
  save(s);
  return s;
}

export async function fetchConfigFromBackend(): Promise<Partial<AppSettings>> {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) return {};
    const data = await res.json();
    if (typeof data.openai_api_key === "string" && data.openai_api_key) {
      return { openaiApiKey: data.openai_api_key };
    }
  } catch { /* backend not available */ }
  return {};
}
