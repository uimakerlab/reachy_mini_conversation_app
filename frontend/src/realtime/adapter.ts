import type { VoiceEventBus } from "../voice/eventBus";
import type { ToolExecutor } from "../tools";

interface RealtimeServerEvent {
  type: string;
  [key: string]: unknown;
}

export interface RealtimeAdapterOptions {
  dc: RTCDataChannel;
  eventBus: VoiceEventBus;
  executeTool: ToolExecutor;
  onUserSpeechStarted?: () => void;
  onUserTranscript?: (text: string, final: boolean) => void;
  onAssistantTranscript?: (text: string, final: boolean) => void;
  onError?: (err: Error) => void;
}

/**
 * Bridges the OpenAI Realtime DataChannel events to the app's VoiceEventBus
 * and handles tool call execution.
 */
export class RealtimeAdapter {
  private dc: RTCDataChannel;
  private bus: VoiceEventBus;
  private executeTool: ToolExecutor;
  private onUserSpeechStarted: () => void;
  private onUserTranscript: (text: string, final: boolean) => void;
  private onAssistantTranscript: (text: string, final: boolean) => void;
  private onError: (err: Error) => void;
  private _disposed = false;
  private _responseActive = false;

  private _pendingToolCalls = new Map<
    string,
    { callId: string; name: string; args: string }
  >();

  private _assistantTranscript = "";
  private _userTranscript = "";

  constructor(opts: RealtimeAdapterOptions) {
    this.dc = opts.dc;
    this.bus = opts.eventBus;
    this.executeTool = opts.executeTool;
    this.onUserSpeechStarted = opts.onUserSpeechStarted ?? (() => {});
    this.onUserTranscript = opts.onUserTranscript ?? (() => {});
    this.onAssistantTranscript = opts.onAssistantTranscript ?? (() => {});
    this.onError = opts.onError ?? console.error;

    this.dc.addEventListener("message", this._onMessage);
  }

  dispose() {
    this._disposed = true;
    this.dc.removeEventListener("message", this._onMessage);
  }

  send(event: Record<string, unknown>) {
    if (this.dc.readyState === "open") {
      this.dc.send(JSON.stringify(event));
    }
  }

  sendText(text: string) {
    this.send({
      type: "conversation.item.create",
      item: {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text }],
      },
    });
    this.send({ type: "response.create" });
  }

  sendImage(base64: string, mimeType = "image/jpeg") {
    this.send({
      type: "conversation.item.create",
      item: {
        type: "message",
        role: "user",
        content: [
          { type: "input_image", image_url: `data:${mimeType};base64,${base64}` },
        ],
      },
    });
  }

  cancelResponse() {
    if (this._responseActive) {
      this.send({ type: "response.cancel" });
    }
    this.bus.emit("tts:stop", {});
    this._responseActive = false;
  }

  updateSession(update: Record<string, unknown>) {
    this.send({ type: "session.update", session: update });
  }

  private _onMessage = (e: MessageEvent) => {
    if (this._disposed) return;
    let event: RealtimeServerEvent;
    try {
      event = JSON.parse(e.data);
    } catch {
      return;
    }
    this._handleEvent(event);
  };

  private _str(val: unknown): string {
    return typeof val === "string" ? val : "";
  }

  private async _handleEvent(event: RealtimeServerEvent) {
    switch (event.type) {
      case "response.created":
        this._assistantTranscript = "";
        if (!this._responseActive) {
          this._responseActive = true;
          this.bus.emit("tts:start", {});
        }
        break;

      case "response.done": {
        this._responseActive = false;
        this.bus.emit("tts:done", {});
        const resp = event.response as Record<string, unknown> | undefined;
        if (resp?.status === "failed") {
          const details = resp.status_details as Record<string, unknown> | undefined;
          const err = details?.error as Record<string, unknown> | undefined;
          const msg = (err?.message as string) ?? "Response failed";
          console.error("[Adapter] Response failed:", err);
          this.onError(new Error(msg));
        }
        break;
      }

      case "response.audio_transcript.delta":
        this._assistantTranscript += this._str(event.delta);
        this.onAssistantTranscript(this._assistantTranscript, false);
        break;

      case "response.audio_transcript.done":
        this._assistantTranscript = "";
        this.onAssistantTranscript(this._str(event.transcript), true);
        break;

      case "conversation.item.input_audio_transcription.delta":
        this._userTranscript += this._str(event.delta);
        this.onUserTranscript(this._userTranscript, false);
        break;

      case "conversation.item.input_audio_transcription.completed": {
        this._userTranscript = "";
        const transcript = this._str(event.transcript);
        this.onUserTranscript(transcript, true);
        this.bus.emit("stt:complete", { text: transcript });
        break;
      }

      case "response.function_call_arguments.delta": {
        const callId = this._str(event.call_id);
        const existing = this._pendingToolCalls.get(callId);
        if (existing) {
          existing.args += this._str(event.delta);
        } else {
          this._pendingToolCalls.set(callId, {
            callId,
            name: this._str(event.name),
            args: this._str(event.delta),
          });
        }
        break;
      }

      case "response.function_call_arguments.done": {
        const callId = this._str(event.call_id);
        const pending = this._pendingToolCalls.get(callId);
        const name = pending?.name || this._str(event.name);
        const argsStr = pending?.args ?? this._str(event.arguments);
        this._pendingToolCalls.delete(callId);

        let args: Record<string, unknown> = {};
        try { args = JSON.parse(argsStr); } catch { /* empty args */ }

        this.bus.emit("tool:call", { name, args });

        try {
          const result = await this.executeTool(name, args);
          this.send({
            type: "conversation.item.create",
            item: {
              type: "function_call_output",
              call_id: callId,
              output: typeof result === "string" ? result : JSON.stringify(result),
            },
          });
          this.send({ type: "response.create" });
          this.bus.emit("tool:result", { name, result });
        } catch (err) {
          const errMsg = err instanceof Error ? err.message : String(err);
          this.send({
            type: "conversation.item.create",
            item: {
              type: "function_call_output",
              call_id: callId,
              output: JSON.stringify({ error: errMsg }),
            },
          });
          this.send({ type: "response.create" });
        }
        break;
      }

      case "error": {
        const errObj = event.error as Record<string, unknown> | undefined;
        if (errObj?.code === "response_cancel_not_active") break;
        this.onError(new Error(JSON.stringify(event.error)));
        break;
      }

      case "input_audio_buffer.speech_started":
        this._userTranscript = "";
        this.onUserSpeechStarted();
        this.bus.emit("vad:start", {});
        break;

      case "input_audio_buffer.speech_stopped":
        this.bus.emit("vad:end", {});
        break;

      default:
        break;
    }
  }
}
