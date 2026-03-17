export interface ToolDefinition {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export type ToolExecutor = (
  name: string,
  args: Record<string, unknown>,
) => Promise<string | Record<string, unknown>>;

import * as robot from "./robot";
import * as volume from "./volume";
import * as search from "./search";
import * as moveHead from "./moveHead";
import * as headTracking from "./headTracking";
import * as camera from "./camera";
import * as dance from "./dance";
import * as doNothing from "./doNothing";
import * as stopDance from "./stopDance";
import * as playEmotion from "./playEmotion";
import * as stopEmotion from "./stopEmotion";
import { setDaemonUrl } from "./daemon";
import type { MovementManager } from "../movement/manager";
import type { RealtimeAdapter } from "../realtime/adapter";

interface ToolModule {
  definitions: ToolDefinition[];
  execute: (name: string, args: Record<string, unknown>) => Promise<string>;
}

const modules: ToolModule[] = [
  robot, volume, search, moveHead, headTracking,
  camera, dance, doNothing, stopDance, playEmotion, stopEmotion,
];

const nameToModule = new Map<string, ToolModule>();
for (const mod of modules) {
  for (const def of mod.definitions) {
    nameToModule.set(def.name, mod);
  }
}

/**
 * Get tool definitions, optionally filtered by a list of enabled tool names.
 */
export function getToolDefinitions(enabledTools?: string[]): ToolDefinition[] {
  const all = modules.flatMap((m) => m.definitions);
  if (!enabledTools) return all;
  const enabled = new Set(enabledTools);
  return all.filter((d) => enabled.has(d.name));
}

export async function executeTool(
  name: string,
  args: Record<string, unknown>,
): Promise<string | Record<string, unknown>> {
  const mod = nameToModule.get(name);
  if (!mod) throw new Error(`Unknown tool: ${name}`);
  return mod.execute(name, args);
}

export function configureTools(opts: {
  manager?: MovementManager | null;
  adapter?: RealtimeAdapter | null;
  daemonUrl?: string;
}): void {
  if (opts.manager !== undefined) {
    robot.setManager(opts.manager);
    moveHead.setManager(opts.manager);
    headTracking.setManager(opts.manager);
    dance.setManager(opts.manager);
    stopDance.setManager(opts.manager);
    playEmotion.setManager(opts.manager);
    stopEmotion.setManager(opts.manager);
  }
  if (opts.adapter !== undefined) camera.setAdapter(opts.adapter);
  if (opts.daemonUrl !== undefined) setDaemonUrl(opts.daemonUrl);
}
