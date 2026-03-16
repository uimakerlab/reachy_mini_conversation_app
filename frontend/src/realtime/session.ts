import type { ToolDefinition } from "../tools";

export interface SessionConfig {
  model: string;
  voice: string;
  instructions: string;
  tools: ToolDefinition[];
  transcriptionModel?: string;
}
