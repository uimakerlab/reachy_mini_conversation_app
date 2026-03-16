import type { ToolDefinition } from ".";
import type { RealtimeAdapter } from "../realtime/adapter";

let _adapter: RealtimeAdapter | null = null;
let _videoEl: HTMLVideoElement | null = null;

export function setAdapter(adapter: RealtimeAdapter | null): void { _adapter = adapter; }
export function setVideoElement(el: HTMLVideoElement | null): void { _videoEl = el; }

export const definitions: ToolDefinition[] = [
  {
    name: "take_photo",
    description:
      "Take a photo with the robot's camera and describe what you see. Use when asked to look at something or describe the surroundings.",
    parameters: { type: "object", properties: {} },
  },
];

function captureFrame(): string | null {
  if (!_videoEl || _videoEl.videoWidth === 0) return null;
  const canvas = document.createElement("canvas");
  canvas.width = Math.min(_videoEl.videoWidth, 640);
  canvas.height = Math.round(canvas.width * (_videoEl.videoHeight / _videoEl.videoWidth));
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(_videoEl, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.7).split(",")[1];
}

export async function execute(
  name: string,
  _args: Record<string, unknown>,
): Promise<string> {
  if (name !== "take_photo") throw new Error(`Unknown tool: ${name}`);
  const frame = captureFrame();
  if (!frame) return "Camera not available";
  if (_adapter) {
    _adapter.sendImage(frame);
    return "Photo taken and sent for analysis. Describe what you see in the image.";
  }
  return "Photo captured but no connection to send it.";
}
