import type { ToolDefinition } from ".";
import { getDaemonBase } from "./daemon";

export const definitions: ToolDefinition[] = [
  {
    name: "web_search",
    description:
      "Search the web for current information. Use for recent news, facts, weather, sports results, etc.",
    parameters: {
      type: "object",
      properties: { query: { type: "string", description: "The search query" } },
      required: ["query"],
    },
  },
];

export async function execute(
  name: string,
  args: Record<string, unknown>,
): Promise<string> {
  if (name !== "web_search") throw new Error(`Unknown search tool: ${name}`);
  const query = args.query as string;
  const params = new URLSearchParams({ q: query });
  const res = await fetch(`${getDaemonBase()}/api/search?${params}`);
  if (!res.ok) throw new Error(`Search proxy returned ${res.status}`);
  const data = await res.json();
  const results = (data.results as { title: string; snippet: string }[]) ?? [];
  if (results.length === 0) return `No results found for "${query}".`;
  return results
    .slice(0, 5)
    .map((r, i) => `${i + 1}. ${r.title}\n   ${r.snippet}`)
    .join("\n\n");
}
