import { useRef, useEffect } from "react";
import Box from "@mui/material/Box";
import type { ChatMessage } from "../hooks/useChat";
import MessageBubble from "./MessageBubble";
import EmptyChat from "./EmptyChat";

interface Props {
  messages: ChatMessage[];
  isConnected: boolean;
  botAvatar?: string | null;
  botName?: string | null;
}

export default function ChatPanel({ messages, isConnected, botAvatar, botName }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    // Only auto-scroll if already near the bottom (avoid hijacking manual scroll)
    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (isNearBottom) {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    }
  }, [messages]);

  return (
    <Box
      ref={containerRef}
      sx={{
        flex: 1,
        overflow: "auto",
        display: "flex",
        flexDirection: "column",
        "&::-webkit-scrollbar": { width: 6 },
        "&::-webkit-scrollbar-track": { bgcolor: "transparent" },
        "&::-webkit-scrollbar-thumb": {
          bgcolor: "divider",
          borderRadius: 3,
          "&:hover": { bgcolor: "action.disabled" },
        },
      }}
    >
      {messages.length === 0 ? (
        <EmptyChat isConnected={isConnected} />
      ) : (
        <Box sx={{ maxWidth: 640, mx: "auto", width: "100%", px: 2, py: 2 }}>
          {messages.map((msg) => <MessageBubble key={msg.id} msg={msg} botAvatar={botAvatar} botName={botName} />)}
        </Box>
      )}
    </Box>
  );
}
