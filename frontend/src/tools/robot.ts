import type { ToolDefinition } from ".";
import type { MovementManager } from "../movement/manager";

let _mgr: MovementManager | null = null;

export function setManager(mgr: MovementManager | null): void {
  _mgr = mgr;
}

export const definitions: ToolDefinition[] = [
  {
    name: "play_emotion",
    description:
      "Express an emotion through head movements and antennas. Available: joy, sadness, surprise, anger, curious",
    parameters: {
      type: "object",
      properties: { emotion: { type: "string", description: "Emotion name" } },
      required: ["emotion"],
    },
  },
  {
    name: "nod",
    description: "Nod head to express agreement (yes gesture)",
    parameters: {
      type: "object",
      properties: { count: { type: "integer", description: "Number of nods (default 2)" } },
    },
  },
  {
    name: "shake_head",
    description: "Shake head to express disagreement (no gesture)",
    parameters: {
      type: "object",
      properties: { count: { type: "integer", description: "Number of shakes (default 2)" } },
    },
  },
  {
    name: "wake_up",
    description: "Wake up the robot from sleep mode",
    parameters: { type: "object", properties: {} },
  },
  {
    name: "go_to_sleep",
    description: "Put the robot to sleep (power saving mode)",
    parameters: { type: "object", properties: {} },
  },
];

export async function execute(
  name: string,
  args: Record<string, unknown>,
): Promise<string> {
  if (!_mgr) return "Movement manager not available";

  switch (name) {
    case "play_emotion": {
      const emotion = (args.emotion as string) ?? "curious";
      const ok = _mgr.setEmotion(emotion);
      return ok ? `Expressing "${emotion}"` : `Unknown emotion "${emotion}"`;
    }
    case "nod":
      await _mgr.nod((args.count as number) ?? 2);
      return "Nodded in agreement";
    case "shake_head":
      await _mgr.shake((args.count as number) ?? 2);
      return "Shook head in disagreement";
    case "wake_up":
      _mgr.lookAt("up", 0.5);
      await new Promise((r) => setTimeout(r, 600));
      _mgr.lookFront(0.8);
      return "Woke up and ready!";
    case "go_to_sleep":
      _mgr.lookAt("down", 1.5);
      return "Going to sleep... zzz";
    default:
      throw new Error(`Unknown robot tool: ${name}`);
  }
}
