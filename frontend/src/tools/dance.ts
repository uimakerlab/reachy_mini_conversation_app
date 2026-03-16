import type { ToolDefinition } from ".";
import type { MovementManager } from "../movement/manager";

let _mgr: MovementManager | null = null;
export function setManager(mgr: MovementManager | null): void { _mgr = mgr; }

export const definitions: ToolDefinition[] = [
  {
    name: "dance",
    description:
      "Make the robot perform a dance sequence. Available: nod_groove, head_bop, sway_roll, excited_bounce",
    parameters: {
      type: "object",
      properties: {
        style: { type: "string", description: "Dance style name" },
      },
      required: ["style"],
    },
  },
];

async function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

async function nodGroove(mgr: MovementManager): Promise<void> {
  for (let i = 0; i < 4; i++) {
    mgr.goto("down", { duration: 0.2 });
    await sleep(250);
    mgr.goto("front", { duration: 0.2 });
    await sleep(250);
  }
}

async function headBop(mgr: MovementManager): Promise<void> {
  const dirs = ["left", "right", "left", "right", "up", "down", "front"] as const;
  for (const d of dirs) {
    mgr.goto(d, { duration: 0.25 });
    await sleep(350);
  }
}

async function swayRoll(mgr: MovementManager): Promise<void> {
  const seq = ["upLeft", "downRight", "upRight", "downLeft", "front"] as const;
  for (const d of seq) {
    mgr.goto(d, { duration: 0.35 });
    await sleep(450);
  }
}

async function excitedBounce(mgr: MovementManager): Promise<void> {
  mgr.setEmotion("joy");
  for (let i = 0; i < 6; i++) {
    mgr.goto(i % 2 === 0 ? "up" : "front", { duration: 0.15 });
    await sleep(200);
  }
  mgr.goto("front", { duration: 0.3 });
}

const DANCES: Record<string, (mgr: MovementManager) => Promise<void>> = {
  nod_groove: nodGroove,
  head_bop: headBop,
  sway_roll: swayRoll,
  excited_bounce: excitedBounce,
};

export async function execute(
  name: string,
  args: Record<string, unknown>,
): Promise<string> {
  if (name !== "dance") throw new Error(`Unknown tool: ${name}`);
  if (!_mgr) return "Movement manager not available";
  const style = (args.style as string) ?? "nod_groove";
  const fn = DANCES[style] ?? DANCES.nod_groove;
  await fn(_mgr);
  return `Performed "${style}" dance!`;
}
