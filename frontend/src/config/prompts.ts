import { getBuiltinProfile, BUILTIN_PROFILES } from "./builtinProfiles";
import type { CustomProfile } from "./settings";

function findCustom(profileId: string, customProfiles?: CustomProfile[]): CustomProfile | undefined {
  return customProfiles?.find((p) => p.id === profileId);
}

export function buildInstructions(
  profileId: string,
  customInstructions?: string,
  customProfiles?: CustomProfile[],
): string {
  const builtin = getBuiltinProfile(profileId);
  if (builtin) return builtin.instructions;
  const custom = findCustom(profileId, customProfiles);
  if (custom) return custom.instructions;
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
