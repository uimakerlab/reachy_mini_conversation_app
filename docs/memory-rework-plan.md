# Memory System Rework Plan

## Context

The memory system works (active memory is injected, save_memory saves facts, logs are written) but has two problems:
1. `recall_memory` searches archived memory entries instead of conversation logs — it should read session logs so the LLM can access detailed conversation history behind a memory
2. Log files are verbose JSONL with session_id repeated on every line — should be human-readable, one file per session

## Changes

### 1. Per-session human-readable log files

**Current:** One JSONL file per day, session_id on every line.
```json
{"ts": "2026-03-26T16:26:54.457712+00:00", "session_id": "f41872da-...", "role": "user", "content": "Hi"}
```

**New:** One plain-text file per session, named `YYYY-MM-DD_HH-MM.log`.
```
--- session 2026-03-26 16:28 UTC ---

16:28:12 user: Hi, my name is Rémi
16:28:15 assistant: Nice to meet you! I'll remember that.
16:28:16 tool: save_memory({"fact": "User's name is Rémi"}) -> {"status": "saved"}
```

**Collision handling:** If `2026-03-26_16-28.log` exists, try `2026-03-26_16-28_2.log`, `_3`, etc.

**File in `memory_manager.py`:**
- Remove `_log_path_for_today()`. Replace with `_session_log_path` attribute set in `new_session()` and `__init__`.
- `new_session()` picks a filename based on current time, writes the header line, stores path.
- `_append_log()` writes a plain text line (`HH:MM:SS role: content`).
- `log_turn()` and `log_tool_call()` adapted to produce plain text.

### 2. Simpler active memory entry format

**Current:**
```
- [2026-03-26] Rémi plays Caro-Kann (ref: 2026-03-26.jsonl)
```

**New:**
```
Rémi plays Caro-Kann (2026-03-26_16-28.log)
```

No bullet, no brackets around date. The log filename itself contains the date. Shorter = more facts fit in the 1500-token cap.

**File in `memory_manager.py`:**
- `save_memory()`: change entry format from `- [{date}] {fact} (ref: {log_ref})` to `{fact} ({session_log_name})`
- `_read_active_lines()`: unchanged (reads non-empty lines)
- `_archive_oldest()`: unchanged (archives oldest lines regardless of format)

### 3. `recall_memory` reads session logs

**Current:** Substring search across archived memory entries. Returns matching lines.

**New:** Takes a log filename. Returns the entire file content. If file not found, returns list of available log files.

**Logic:**
```
recall_memory(log_ref):
  if log_ref is empty or blank:
    return list of all .log filenames in logs/ dir (newest first)

  path = logs_dir / log_ref
  if not path.exists():
    return error + list of available filenames

  return full file content as a string
```

No size check, no windowing — always return the entire file. Worst case: a very long session (say 2 hours, ~240 lines at 2 lines/min) is ~12,000 chars = ~3,400 tokens. This fits comfortably in the ~12K conversation context budget. If a session were absurdly long (4+ hours non-stop), the Realtime API's auto-truncation would drop older conversation items to make room, but our session log + memory would be preserved in system instructions. In practice, sessions are bounded by the robot's battery, user attention span, and the fact that the app restarts between sessions.

**File in `tools/recall_memory.py`:**
- Rename parameter from `query` to `log_ref`
- Update description (see below)
- The description IS the guide for the LLM — no separate prompt needed

**New tool description (this is what the LLM sees as tool guidance):**
```
Read the conversation log from a past session to recall detailed context.
Each memory in your MEMORY block has a filename in parentheses (e.g. '2026-03-26_16-28.log') — pass that filename here to read the full conversation from that session.
If you don't know which file to look for, call with an empty string to list available session logs.
Before calling, tell the user you're checking your memory (e.g. 'Let me think back...' or 'That rings a bell, one moment...').
If the file isn't found, let the user know you couldn't retrieve that specific conversation.
```

**File in `prompts/default_prompt.txt`:**
- Remove the `## MEMORY RULES` section entirely. The tool descriptions are sufficient and profile-independent.

### 4. Update tests

**File in `tests/memory/test_memory_manager.py`:**

Tests to update:
- `TestConversationLogging`: verify `.log` file created, plain-text format, session header present
- `TestConversationLogging::test_log_turn_appends`: check `HH:MM:SS role: content` format
- `TestConversationLogging::test_log_tool_call`: check `HH:MM:SS tool: name(args) -> result` format
- `TestConversationLogging::test_session_id_in_logs`: remove or replace (session_id no longer on each line — it's in the header)
- `TestActiveMemory::test_save_memory_has_date_and_ref`: update expected format
- `TestRecall`: rewrite entirely:
  - `test_recall_returns_session_log`: save a memory, call recall with the log ref, verify full log content returned
  - `test_recall_file_not_found`: call with nonexistent filename, verify error + available files list
  - `test_recall_empty_lists_files`: call with empty string, verify list of log filenames returned
- Remove: `test_recall_active_match`, `test_recall_no_match`, `test_recall_case_insensitive`, `test_recall_searches_archives`, `test_recall_empty_query_returns_all_active` (these tested the old substring-search behavior)

Tests that stay unchanged:
- `TestInit`: directory creation, fresh start
- `TestArchival`: archival trigger, breadcrumb, archived entries, cap enforcement
- `TestPromptInjection`: memory block format (update expected line format)
- `TestTokenEstimation`: pure math, unchanged

### 5. Update design doc

**File in `docs/memory-system-design.md`:**
- Update log format section
- Update recall_memory tool description
- Update data directory layout (`.log` files instead of `.jsonl`)
- Note: recall now reads logs, not archives

## What stays the same

- `active_memory.md` injection into system prompt at session start
- `save_memory` tool interface (LLM passes a `fact` string)
- Archival system (oldest 1/3 evicted when over cap, breadcrumb left)
- Archive files still exist on disk (for evicted entries) but are not searched by recall
- `ACTIVE_MEMORY_TOKEN_CAP` at 1500
- Memory enabled/disabled config
- Data directory path (`~/.reachy_mini/data/`)

## Commits (small, logical)

1. Switch to per-session human-readable log files
2. Simplify active memory entry format
3. Rewrite recall_memory to read session logs by ref
4. Update tests
5. Update design doc
