"""Structured result returned by CascadeHandler after each conversation turn."""

from __future__ import annotations
from dataclasses import field, dataclass


@dataclass
class TurnItem:
    """One displayable item from a conversation turn.

    kind: "speak" | "image" | "tool" | "assistant"
    """

    kind: str
    text: str = ""
    image_jpeg: bytes = b""
    tool_name: str = ""
    tool_content: str = ""


@dataclass
class TurnResult:
    """Everything the UI needs to render one conversation turn."""

    transcript: str = ""
    items: list[TurnItem] = field(default_factory=list)
    cost: float = 0.0  # ASR + LLM cost (not TTS)

    @property
    def speak_text(self) -> str:
        """All speak items joined with '. '."""
        return ". ".join(i.text for i in self.items if i.kind == "speak")

    @property
    def has_speak(self) -> bool:
        """Whether any speak item exists."""
        return any(i.kind == "speak" for i in self.items)
