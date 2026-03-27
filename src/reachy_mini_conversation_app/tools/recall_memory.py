"""System tool: recall_memory — read a session log to recall detailed context."""

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class RecallMemory(Tool):
    """Read a past session log to recall detailed conversation context."""

    name = "recall_memory"
    description = (
        "Read the conversation log from a past session to recall detailed context. "
        "Each memory in your MEMORY block has a filename in parentheses (e.g. '2026-03-26_16-28.log') "
        "— pass that filename here to read the full conversation from that session. "
        "If you don't know which file to look for, call with an empty string to list available session logs. "
        "Before calling, tell the user you're checking your memory (e.g. 'Let me think back...' or 'That rings a bell, one moment...'). "
        "If the file isn't found, let the user know you couldn't retrieve that specific conversation."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "log_ref": {
                "type": "string",
                "description": "The session log filename from a memory entry (e.g. '2026-03-26_16-28.log'). Empty string to list available logs.",
            },
        },
        "required": ["log_ref"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        if deps.memory_manager is None:
            return {"status": "memory_disabled"}

        log_ref = kwargs.get("log_ref") or ""
        logger.info("Tool call: recall_memory log_ref=%r", log_ref)
        return deps.memory_manager.recall_memory(log_ref)
