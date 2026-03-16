import type { ToolDefinition } from ".";
import { getDaemonBase } from "./daemon";

export const definitions: ToolDefinition[] = [
  {
    name: "set_volume",
    description:
      "Set the robot speaker volume (0-100). 30 = quiet, 50 = normal, 80 = loud.",
    parameters: {
      type: "object",
      properties: { volume: { type: "integer", description: "Volume level 0-100" } },
      required: ["volume"],
    },
  },
  {
    name: "get_volume",
    description: "Get the current robot speaker volume level.",
    parameters: { type: "object", properties: {} },
  },
];

export async function execute(
  name: string,
  args: Record<string, unknown>,
): Promise<string> {
  const base = getDaemonBase();
  switch (name) {
    case "set_volume": {
      const vol = Math.max(0, Math.min(100, Math.round(args.volume as number)));
      const res = await fetch(`${base}/api/volume/set`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ volume: vol }),
      });
      if (!res.ok) throw new Error(`Daemon returned ${res.status}`);
      const data = await res.json();
      return `Volume set to ${data.volume}%`;
    }
    case "get_volume": {
      const res = await fetch(`${base}/api/volume/current`);
      if (!res.ok) throw new Error(`Daemon returned ${res.status}`);
      const data = await res.json();
      return `Current volume is ${data.volume}%`;
    }
    default:
      throw new Error(`Unknown volume tool: ${name}`);
  }
}
