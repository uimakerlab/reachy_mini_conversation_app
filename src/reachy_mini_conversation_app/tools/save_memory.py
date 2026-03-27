"""System tool: save_memory — persist a fact to long-term memory."""

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)


class SaveMemory(Tool):
    """Save an important fact or observation to long-term memory."""

    name = "save_memory"
    description = (
        "Save an information to long-term memory so you remember it across sessions. "
        "Use when the user shares something worth remembering: their name, preferences, "
        "interests, or discussion topics. Save anything that would help you have a more personalized and engaging conversation in the future. "
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": (
                    "A concise statement to remember, written in third person. "
                    "Examples: 'User's name is Alice', 'User prefers brief answers', "
                    "'We discussed how Reachy's camera works'."
                ),
            },
        },
        "required": ["fact"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        if deps.memory_manager is None:
            return {"status": "memory_disabled"}

        fact = kwargs.get("fact") or ""
        logger.info("Tool call: save_memory fact=%r", fact[:80])
        return deps.memory_manager.save_memory(fact)
