import type { ToolDefinition } from ".";
import type { MovementManager } from "../movement/manager";

let _mgr: MovementManager | null = null;
export function setManager(mgr: MovementManager | null): void { _mgr = mgr; }

export const definitions: ToolDefinition[] = [
  {
    name: "toggle_head_tracking",
    description: "Toggle face tracking on or off. When enabled, the robot follows detected faces.",
    parameters: {
      type: "object",
      properties: {
        enabled: { type: "boolean", description: "Enable (true) or disable (false) face tracking" },
      },
      required: ["enabled"],
    },
  },
];

export async function execute(
  name: string,
  args: Record<string, unknown>,
): Promise<string> {
  if (name !== "toggle_head_tracking") throw new Error(`Unknown tool: ${name}`);
  if (!_mgr) return "Movement manager not available";
  const enabled = args.enabled as boolean;
  if (!enabled) _mgr.clearTrackingTarget();
  return enabled ? "Head tracking enabled" : "Head tracking disabled";
}
