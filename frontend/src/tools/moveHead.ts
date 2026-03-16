import type { ToolDefinition } from ".";
import type { MovementManager } from "../movement/manager";

let _mgr: MovementManager | null = null;
export function setManager(mgr: MovementManager | null): void { _mgr = mgr; }

export const definitions: ToolDefinition[] = [
  {
    name: "move_head",
    description:
      "Move the robot's head to a direction. Available: front, left, right, up, down, upLeft, upRight, downLeft, downRight",
    parameters: {
      type: "object",
      properties: {
        direction: { type: "string", description: "Target direction" },
        duration: { type: "number", description: "Duration in seconds (default 1)" },
      },
      required: ["direction"],
    },
  },
];

export async function execute(
  name: string,
  args: Record<string, unknown>,
): Promise<string> {
  if (name !== "move_head") throw new Error(`Unknown tool: ${name}`);
  if (!_mgr) return "Movement manager not available";
  const dir = (args.direction as string) ?? "front";
  const dur = (args.duration as number) ?? 1;
  _mgr.lookAt(dir, dur);
  return `Moving head to "${dir}"`;
}
