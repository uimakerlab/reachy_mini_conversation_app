import { getBuiltinProfile, BUILTIN_PROFILES } from "./builtinProfiles";
import type { CustomProfile } from "./settings";

function findCustom(profileId: string, customProfiles?: CustomProfile[]): CustomProfile | undefined {
  return customProfiles?.find((p) => p.id === profileId);
}

const BASE_TEMPLATE = `## IDENTITY
You are a Reachy Mini robot assistant.

## PERSONALITY
- {name}
- {personality}

## RESPONSE RULES
- Keep responses short: 1-3 sentences maximum.
- Be helpful first, personality second.
- Keep responses under 30 words when possible.
- Switch languages only if the user explicitly asks.

## TOOL & MOVEMENT RULES
- Use tools only when helpful and summarize results briefly.
- Use the camera for real visuals only - never invent details.
- The head can move (left/right/up/down/front).
- Enable head tracking when looking at a person; disable otherwise.
- After a tool returns, explain the result briefly with personality.`;

function wrapCustomInstructions(name: string, personality: string): string {
  return BASE_TEMPLATE
    .replace("{name}", name)
    .replace("{personality}", personality.trim() || "Friendly and helpful assistant.");
}

export function buildInstructions(
  profileId: string,
  customInstructions?: string,
  customProfiles?: CustomProfile[],
): string {
  const builtin = getBuiltinProfile(profileId);
  if (builtin) return builtin.instructions;
  const custom = findCustom(profileId, customProfiles);
  if (custom) return wrapCustomInstructions(custom.name, custom.instructions);
  if (customInstructions?.trim()) return customInstructions.trim();
  return BUILTIN_PROFILES[0].instructions;
}

export function resolveVoice(
  profileId: string,
  customVoice?: string,
  customProfiles?: CustomProfile[],
): string {
  const builtin = getBuiltinProfile(profileId);
  if (builtin) return builtin.voice;
  const custom = findCustom(profileId, customProfiles);
  if (custom) return custom.voice;
  if (customVoice?.trim()) return customVoice.trim();
  return "cedar";
}

export function resolveEnabledTools(
  profileId: string,
  customEnabledTools?: string[],
  customProfiles?: CustomProfile[],
): string[] | undefined {
  const builtin = getBuiltinProfile(profileId);
  if (builtin) return builtin.enabledTools;
  const custom = findCustom(profileId, customProfiles);
  if (custom) return custom.enabledTools;
  if (customEnabledTools && customEnabledTools.length > 0) return customEnabledTools;
  return undefined;
}
