"""System tool: recall_memory — search active and archived memory."""

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class RecallMemory(Tool):
    """Search long-term memory for facts matching a query."""

    name = "recall_memory"
    description = (
        "Search long-term memory (active and archived) for stored information. "
        "Use when you need to look up older context that may have been archived "
        "from the MEMORY block, or when the user asks about past conversations."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A keyword or phrase to search for in memory (e.g. 'user name', 'favorite topic').",
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        if deps.memory_manager is None:
            return {"status": "memory_disabled"}

        query = kwargs.get("query") or ""
        logger.info("Tool call: recall_memory query=%r", query)
        return deps.memory_manager.recall_memory(query)
