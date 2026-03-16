import type { ToolDefinition } from ".";
import type { MovementManager } from "../movement/manager";

let _manager: MovementManager | null = null;

export function setManager(mgr: MovementManager | null): void {
  _manager = mgr;
}

export const definitions: ToolDefinition[] = [
  {
    name: "stop_emotion",
    description: "Stop the current emotion animation.",
    parameters: {
      type: "object",
      properties: {},
    },
  },
];

export async function execute(
  name: string,
  _args: Record<string, unknown>,
): Promise<string> {
  if (name !== "stop_emotion") throw new Error(`Unknown tool: ${name}`);
  if (!_manager) return "No robot connected";
  _manager.clearEmotion();
  return "Emotion stopped";
}
