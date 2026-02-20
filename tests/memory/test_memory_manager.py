"""Tests for MemoryManager — all three tiers."""

import json
import pytest
from pathlib import Path

from reachy_mini_conversation_app.memory.memory_manager import (
    MemoryManager,
    ACTIVE_MEMORY_TOKEN_CAP,
    _estimate_tokens,
)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def manager(data_dir: Path) -> MemoryManager:
    return MemoryManager(data_dir)


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------


class TestInit:
    def test_creates_directories(self, manager: MemoryManager, data_dir: Path) -> None:
        assert (data_dir / "memory").is_dir()
        assert (data_dir / "memory" / "archive").is_dir()
        assert (data_dir / "memory" / "logs").is_dir()

    def test_active_memory_empty_on_fresh_start(self, manager: MemoryManager) -> None:
        assert manager.get_memory_block() == ""

    def test_new_session_rotates_id(self, manager: MemoryManager) -> None:
        old_id = manager._session_id
        manager.new_session()
        assert manager._session_id != old_id


# ------------------------------------------------------------------
# Tier 1: Conversation logging
# ------------------------------------------------------------------


class TestConversationLogging:
    def test_log_turn_creates_jsonl(self, manager: MemoryManager, data_dir: Path) -> None:
        manager.log_turn("user", "Hello there!")
        log_files = list((data_dir / "memory" / "logs").glob("*.jsonl"))
        assert len(log_files) == 1

        with open(log_files[0]) as f:
            record = json.loads(f.readline())
        assert record["role"] == "user"
        assert record["content"] == "Hello there!"
        assert "ts" in record
        assert "session_id" in record

    def test_log_turn_appends(self, manager: MemoryManager, data_dir: Path) -> None:
        manager.log_turn("user", "First")
        manager.log_turn("assistant", "Second")

        log_files = list((data_dir / "memory" / "logs").glob("*.jsonl"))
        with open(log_files[0]) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["role"] == "user"
        assert json.loads(lines[1])["role"] == "assistant"

    def test_log_turn_ignores_empty(self, manager: MemoryManager, data_dir: Path) -> None:
        manager.log_turn("user", "")
        manager.log_turn("user", "   ")
        log_files = list((data_dir / "memory" / "logs").glob("*.jsonl"))
        assert len(log_files) == 0

    def test_log_tool_call(self, manager: MemoryManager, data_dir: Path) -> None:
        manager.log_tool_call("dance", args={"name": "happy"}, result={"status": "queued"})
        log_files = list((data_dir / "memory" / "logs").glob("*.jsonl"))
        assert len(log_files) == 1

        with open(log_files[0]) as f:
            record = json.loads(f.readline())
        assert record["role"] == "tool"
        assert record["tool_name"] == "dance"
        assert record["args"] == {"name": "happy"}
        assert record["result"] == {"status": "queued"}

    def test_session_id_in_logs(self, manager: MemoryManager, data_dir: Path) -> None:
        manager.log_turn("user", "Before new session")
        manager.new_session()
        manager.log_turn("user", "After new session")

        log_files = list((data_dir / "memory" / "logs").glob("*.jsonl"))
        with open(log_files[0]) as f:
            lines = f.readlines()
        r1 = json.loads(lines[0])
        r2 = json.loads(lines[1])
        assert r1["session_id"] != r2["session_id"]


# ------------------------------------------------------------------
# Tier 2: Active memory
# ------------------------------------------------------------------


class TestActiveMemory:
    def test_save_memory(self, manager: MemoryManager) -> None:
        result = manager.save_memory("User's name is Alice")
        assert result["status"] == "saved"
        assert "Alice" in result["fact"]

    def test_save_memory_appears_in_block(self, manager: MemoryManager) -> None:
        manager.save_memory("User's name is Alice")
        block = manager.get_memory_block()
        assert "Alice" in block
        assert "## MEMORY" in block

    def test_save_memory_has_date_and_ref(self, manager: MemoryManager) -> None:
        manager.save_memory("User's name is Alice")
        block = manager.get_memory_block()
        assert "(ref:" in block
        assert ".jsonl)" in block

    def test_save_memory_rejects_empty(self, manager: MemoryManager) -> None:
        result = manager.save_memory("")
        assert "error" in result

    def test_save_memory_rejects_whitespace(self, manager: MemoryManager) -> None:
        result = manager.save_memory("   ")
        assert "error" in result

    def test_multiple_saves(self, manager: MemoryManager) -> None:
        manager.save_memory("Fact one")
        manager.save_memory("Fact two")
        block = manager.get_memory_block()
        assert "Fact one" in block
        assert "Fact two" in block


# ------------------------------------------------------------------
# Tier 2 → 3: Archival
# ------------------------------------------------------------------


class TestArchival:
    def _fill_memory(self, manager: MemoryManager, n: int) -> None:
        """Fill memory with n entries, each ~50 chars ≈ 14 tokens."""
        for i in range(n):
            manager.save_memory(f"Memory fact number {i:04d} with some extra padding text here")

    def test_archival_triggered_when_over_cap(self, manager: MemoryManager, data_dir: Path) -> None:
        # Each entry is ~80 chars ≈ 23 tokens. 1500/23 ≈ 65 entries to fill.
        self._fill_memory(manager, 80)

        archive_files = list((data_dir / "memory" / "archive").glob("*.md"))
        assert len(archive_files) > 0

    def test_breadcrumb_left_after_archival(self, manager: MemoryManager) -> None:
        self._fill_memory(manager, 80)

        block = manager.get_memory_block()
        assert "[Archived:" in block
        assert "recall_memory" in block

    def test_archived_entries_in_file(self, manager: MemoryManager, data_dir: Path) -> None:
        self._fill_memory(manager, 80)

        archive_files = list((data_dir / "memory" / "archive").glob("*.md"))
        content = archive_files[0].read_text()
        assert "# Archived memory" in content
        assert "Memory fact number" in content

    def test_active_memory_stays_under_cap_after_archival(self, manager: MemoryManager) -> None:
        self._fill_memory(manager, 80)

        block = manager.get_memory_block()
        tokens = _estimate_tokens(block)
        # Allow some headroom: the block includes the header text
        assert tokens < ACTIVE_MEMORY_TOKEN_CAP * 1.5


# ------------------------------------------------------------------
# Tier 3: Recall
# ------------------------------------------------------------------


class TestRecall:
    def test_recall_active_match(self, manager: MemoryManager) -> None:
        manager.save_memory("User's name is Bob")
        result = manager.recall_memory("Bob")
        assert result["total_found"] >= 1
        assert any("Bob" in m for m in result["active_matches"])

    def test_recall_no_match(self, manager: MemoryManager) -> None:
        manager.save_memory("User's name is Bob")
        result = manager.recall_memory("nonexistent_query_xyz")
        assert result["total_found"] == 0

    def test_recall_case_insensitive(self, manager: MemoryManager) -> None:
        manager.save_memory("User likes Python programming")
        result = manager.recall_memory("python")
        assert result["total_found"] >= 1

    def test_recall_searches_archives(self, manager: MemoryManager) -> None:
        # Fill memory so archival happens, then search for an archived fact
        manager.save_memory("Special fact about zebras")
        # Fill with generic facts to push zebras to archive
        for i in range(80):
            manager.save_memory(f"Generic filler fact number {i:04d} with extra padding text here")

        result = manager.recall_memory("zebras")
        assert result["total_found"] >= 1
        # Should be in archives since it was the oldest
        assert len(result["archive_matches"]) >= 1

    def test_recall_empty_query_returns_all_active(self, manager: MemoryManager) -> None:
        manager.save_memory("Fact A")
        manager.save_memory("Fact B")
        result = manager.recall_memory("")
        assert any("Fact A" in m for m in result["active_matches"])
        assert any("Fact B" in m for m in result["active_matches"])


# ------------------------------------------------------------------
# Prompt injection
# ------------------------------------------------------------------


class TestPromptInjection:
    def test_empty_memory_returns_empty_string(self, manager: MemoryManager) -> None:
        assert manager.get_memory_block() == ""

    def test_memory_block_format(self, manager: MemoryManager) -> None:
        manager.save_memory("User's name is Alice")
        block = manager.get_memory_block()
        assert block.startswith("\n\n## MEMORY\n")
        assert "save_memory" in block
        assert "recall_memory" in block
        assert "Alice" in block


# ------------------------------------------------------------------
# Token estimation
# ------------------------------------------------------------------


class TestTokenEstimation:
    def test_estimate_tokens_short(self) -> None:
        # "hello" = 5 chars ≈ 1-2 tokens
        assert _estimate_tokens("hello") >= 1

    def test_estimate_tokens_long(self) -> None:
        text = "a" * 3500  # 3500 chars ≈ 1000 tokens
        tokens = _estimate_tokens(text)
        assert 900 <= tokens <= 1100

    def test_estimate_tokens_empty(self) -> None:
        assert _estimate_tokens("") == 1  # minimum 1
