"""Three-tier memory manager for Reachy Mini conversation app.

Tiers:
  1. Conversation logs  — automatic JSONL, one file per calendar day
  2. Active memory       — markdown facts, prompt-injected, capped at ~1500 tokens
  3. Archived memory     — overflow facts, one file per archival event
"""

from __future__ import annotations

import json
import uuid
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Token estimation: GPT-4 family averages ~4 chars/token for English.
# We use 3.5 to be conservative (safe side = archive sooner).
_CHARS_PER_TOKEN = 3.5
ACTIVE_MEMORY_TOKEN_CAP = 1500
ARCHIVE_FRACTION = 1 / 3  # oldest third gets archived when over cap


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


class MemoryManager:
    """Manages all three tiers of persistent memory.

    Thread-safe: all public methods that touch active memory acquire self._lock.
    Conversation log appends are atomic on Linux for small writes.
    """

    def __init__(self, data_dir: Path) -> None:
        self._lock = threading.Lock()
        self._data_dir = data_dir
        self._memory_dir = data_dir / "memory"
        self._active_path = self._memory_dir / "active_memory.md"
        self._archive_dir = self._memory_dir / "archive"
        self._logs_dir = self._memory_dir / "logs"
        self._session_id = str(uuid.uuid4())
        self._ensure_dirs()
        logger.info(
            "MemoryManager initialized: data_dir=%s, session_id=%s",
            data_dir,
            self._session_id,
        )

    def _ensure_dirs(self) -> None:
        for d in (self._memory_dir, self._archive_dir, self._logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def new_session(self) -> None:
        """Rotate session ID when a new realtime session starts."""
        with self._lock:
            self._session_id = str(uuid.uuid4())
        logger.info("MemoryManager new session: %s", self._session_id)

    # ------------------------------------------------------------------
    # Tier 1: Conversation logging
    # ------------------------------------------------------------------

    def _log_path_for_today(self) -> Path:
        return self._logs_dir / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"

    def _append_log(self, record: dict) -> None:
        """Append a JSON record to today's log file."""
        try:
            line = json.dumps(record, ensure_ascii=False)
            with open(self._log_path_for_today(), "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            logger.warning("Failed to write conversation log: %s", e)

    def log_turn(self, role: str, content: str) -> None:
        """Log a user or assistant transcript turn."""
        if not content or not content.strip():
            return
        self._append_log(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "session_id": self._session_id,
                "role": role,
                "content": content.strip(),
            }
        )

    def log_tool_call(
        self, tool_name: str, args: dict[str, Any] | None = None, result: dict[str, Any] | None = None
    ) -> None:
        """Log a completed tool call."""
        self._append_log(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "session_id": self._session_id,
                "role": "tool",
                "tool_name": tool_name,
                "args": args or {},
                "result": result or {},
            }
        )

    # ------------------------------------------------------------------
    # Tier 2: Active memory (read / write / prompt injection)
    # ------------------------------------------------------------------

    def _read_active_lines(self) -> list[str]:
        """Return non-empty lines from active_memory.md. Lock must be held."""
        if not self._active_path.exists():
            return []
        try:
            return [ln for ln in self._active_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except OSError:
            return []

    def _write_active_lines(self, lines: list[str]) -> None:
        """Write lines to active_memory.md. Lock must be held."""
        try:
            self._active_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to write active memory: %s", e)

    def save_memory(self, fact: str) -> dict:
        """Add a fact to active memory, archiving oldest entries if over token cap.

        Returns a result dict suitable for tool return values.
        """
        fact = fact.strip()
        if not fact:
            return {"error": "fact must be a non-empty string"}

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_ref = self._log_path_for_today().name
        entry = f"- [{date_str}] {fact} (ref: {log_ref})"

        with self._lock:
            lines = self._read_active_lines()
            lines.append(entry)

            while len(lines) > 1 and _estimate_tokens("\n".join(lines)) > ACTIVE_MEMORY_TOKEN_CAP:
                prev_len = len(lines)
                lines = self._archive_oldest(lines)
                if len(lines) >= prev_len:
                    break  # archive failed or made no progress

            self._write_active_lines(lines)

        logger.info("Memory saved: %s", fact[:80])
        return {"status": "saved", "fact": fact}

    def _archive_oldest(self, lines: list[str]) -> list[str]:
        """Move the oldest 1/3 of lines to an archive file. Lock must be held."""
        n_archive = max(1, len(lines) // 3)
        to_archive = lines[:n_archive]
        to_keep = lines[n_archive:]

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")
        archive_file = self._archive_dir / f"{ts}.md"
        header = f"# Archived memory — {ts}\n\n"
        try:
            archive_file.write_text(header + "\n".join(to_archive) + "\n", encoding="utf-8")
            logger.info("Archived %d memory entries to %s", n_archive, archive_file.name)
        except OSError as e:
            logger.warning("Failed to write archive: %s", e)
            return lines  # keep everything if archive fails

        breadcrumb = f"- [Archived: {n_archive} older memories moved to archive/{archive_file.name} — use recall_memory to access]"
        return [breadcrumb] + to_keep

    # ------------------------------------------------------------------
    # Tier 3: Recall (search active + archives)
    # ------------------------------------------------------------------

    def recall_memory(self, query: str) -> dict:
        """Search active memory and archives for entries matching a query.

        Simple case-insensitive substring search — replaceable with vector search later.
        """
        with self._lock:
            active_lines = self._read_active_lines()

        if not query or not query.strip():
            return {
                "active_matches": active_lines,
                "archive_matches": [],
                "total_found": len(active_lines),
            }

        q = query.strip().lower()

        active_matches = [ln for ln in active_lines if q in ln.lower()]

        archive_matches: list[str] = []
        try:
            for archive_file in sorted(self._archive_dir.glob("*.md"), reverse=True):
                try:
                    text = archive_file.read_text(encoding="utf-8")
                    for ln in text.splitlines():
                        if ln.strip() and not ln.startswith("#") and "[Archived:" not in ln and q in ln.lower():
                            archive_matches.append(f"[{archive_file.stem}] {ln.strip()}")
                    if len(archive_matches) >= 5:
                        break
                except OSError:
                    continue
        except OSError:
            pass

        return {
            "active_matches": active_matches,
            "archive_matches": archive_matches,
            "total_found": len(active_matches) + len(archive_matches),
        }

    # ------------------------------------------------------------------
    # Prompt injection
    # ------------------------------------------------------------------

    def get_memory_block(self) -> str:
        """Return the formatted memory block for system prompt injection.

        Returns an empty string if active memory is empty.
        """
        with self._lock:
            lines = self._read_active_lines()

        if not lines:
            return ""

        return (
            "\n\n## MEMORY\n"
            "The following facts were saved from previous conversations. "
            "Use them to personalize responses. You can save new memories with "
            "save_memory and search older context with recall_memory.\n\n"
            + "\n".join(lines)
        )
