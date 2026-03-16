import type { ToolDefinition } from ".";
import type { MovementManager } from "../movement/manager";

let _manager: MovementManager | null = null;

export function setManager(mgr: MovementManager | null): void {
  _manager = mgr;
}

const AVAILABLE_EMOTIONS = ["happy", "sad", "surprised", "confused", "angry", "excited"];

export const definitions: ToolDefinition[] = [
  {
    name: "play_emotion",
    description: `Play an emotion animation. Available: ${AVAILABLE_EMOTIONS.join(", ")}`,
    parameters: {
      type: "object",
      properties: {
        emotion: {
          type: "string",
          description: `The emotion to express. One of: ${AVAILABLE_EMOTIONS.join(", ")}`,
        },
        intensity: {
          type: "number",
          description: "Intensity from 0.0 to 1.0 (default 1.0)",
        },
      },
      required: ["emotion"],
    },
  },
];

export async function execute(
  name: string,
  args: Record<string, unknown>,
): Promise<string> {
  if (name !== "play_emotion") throw new Error(`Unknown tool: ${name}`);
  if (!_manager) return "No robot connected";
  const emotion = (args.emotion as string) ?? "happy";
  const intensity = Math.max(0, Math.min(1, (args.intensity as number) ?? 1));
  const ok = _manager.setEmotion(emotion, intensity);
  return ok ? `Playing emotion: ${emotion}` : `Unknown emotion: ${emotion}`;
}
