import type { ToolDefinition } from ".";
import type { RealtimeAdapter } from "../realtime/adapter";

let _adapter: RealtimeAdapter | null = null;

export function setAdapter(adapter: RealtimeAdapter | null): void { _adapter = adapter; }

export const definitions: ToolDefinition[] = [
  {
    name: "take_photo",
    description:
      "Take a photo with the robot's camera and describe what you see. Use when asked to look at something or describe the surroundings.",
    parameters: { type: "object", properties: {} },
  },
];

async function fetchSnapshot(): Promise<string | null> {
  try {
    const res = await fetch("/api/camera/snapshot");
    if (!res.ok) return null;
    const data = await res.json();
    return data.b64 ?? null;
  } catch {
    return null;
  }
}

export async function execute(
  name: string,
  _args: Record<string, unknown>,
): Promise<string> {
  if (name !== "take_photo") throw new Error(`Unknown tool: ${name}`);
  const frame = await fetchSnapshot();
  if (!frame) return "Camera not available";
  if (_adapter) {
    _adapter.sendImage(frame);
    return "Photo taken and sent for analysis. Describe what you see in the image.";
  }
  return "Photo captured but no connection to send it.";
}
