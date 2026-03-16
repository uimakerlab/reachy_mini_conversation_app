import type { ToolDefinition } from ".";

export const definitions: ToolDefinition[] = [
  {
    name: "do_nothing",
    description:
      "Choose to do nothing -- stay still and silent. Use when you want to be contemplative or just chill.",
    parameters: {
      type: "object",
      properties: {
        reason: {
          type: "string",
          description: "Optional reason for doing nothing (e.g., 'contemplating existence', 'saving energy')",
        },
      },
      required: [],
    },
  },
];

export async function execute(
  name: string,
  args: Record<string, unknown>,
): Promise<string> {
  if (name !== "do_nothing") throw new Error(`Unknown tool: ${name}`);
  const reason = (args.reason as string) || "just chilling";
  return `Doing nothing: ${reason}`;
}
