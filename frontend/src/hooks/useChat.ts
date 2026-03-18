import { useState, useCallback, useRef } from "react";

export interface ChatMessage {
  id: number;
  role: "user" | "assistant" | "tool";
  content: string;
  partial?: boolean;
  toolName?: string;
  imageUrl?: string;
  ts: number;
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const userPartial = useRef<number | null>(null);
  const asstPartial = useRef<number | null>(null);
  const nextId = useRef(0);

  const clear = useCallback(() => {
    setMessages([]);
    userPartial.current = null;
    asstPartial.current = null;
  }, []);

  const addMessage = useCallback((msg: Omit<ChatMessage, "id" | "ts">): number => {
    const id = ++nextId.current;
    setMessages((prev) => [...prev, { ...msg, id, ts: Date.now() }]);
    return id;
  }, []);

  const reserveUserMessage = useCallback(() => {
    // Clean up stale placeholder from a previous speech that never got a transcript
    if (userPartial.current !== null) {
      const staleId = userPartial.current;
      setMessages((prev) => prev.filter((m) => m.id !== staleId));
      userPartial.current = null;
    }
    userPartial.current = addMessage({ role: "user", content: "...", partial: true });
  }, [addMessage]);

  const handleUserTranscript = useCallback(
    (text: string, final: boolean) => {
      if (final) {
        if (userPartial.current !== null) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === userPartial.current ? { ...m, content: text, partial: false } : m,
            ),
          );
          userPartial.current = null;
        } else {
          addMessage({ role: "user", content: text });
        }
      } else {
        if (userPartial.current !== null) {
          setMessages((prev) =>
            prev.map((m) => (m.id === userPartial.current ? { ...m, content: text } : m)),
          );
        } else {
          userPartial.current = addMessage({ role: "user", content: text, partial: true });
        }
      }
    },
    [addMessage],
  );

  const handleAssistantTranscript = useCallback(
    (text: string, final: boolean) => {
      if (final) {
        if (asstPartial.current !== null) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === asstPartial.current ? { ...m, content: text, partial: false } : m,
            ),
          );
          asstPartial.current = null;
        } else {
          addMessage({ role: "assistant", content: text });
        }
      } else {
        if (asstPartial.current !== null) {
          setMessages((prev) =>
            prev.map((m) => (m.id === asstPartial.current ? { ...m, content: text } : m)),
          );
        } else {
          asstPartial.current = addMessage({
            role: "assistant",
            content: text,
            partial: true,
          });
        }
      }
    },
    [addMessage],
  );

  const addToolMessage = useCallback(
    (toolName: string, result: string) => {
      addMessage({ role: "tool", content: result, toolName });
    },
    [addMessage],
  );

  const attachImageToLastTool = useCallback(
    (dataUrl: string) => {
      setMessages((prev) => {
        for (let i = prev.length - 1; i >= 0; i--) {
          if (prev[i].role === "tool" && prev[i].toolName?.includes("camera")) {
            const updated = [...prev];
            updated[i] = { ...updated[i], imageUrl: dataUrl };
            return updated;
          }
        }
        return prev;
      });
    },
    [],
  );

  return {
    messages,
    clear,
    addMessage,
    reserveUserMessage,
    handleUserTranscript,
    handleAssistantTranscript,
    addToolMessage,
    attachImageToLastTool,
  };
}
