"""Task cancel tool - cancel running background tasks."""

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.background_tool_manager import ToolState, BackgroundToolManager


logger = logging.getLogger(__name__)


class ToolCancel(Tool):
    """Cancel a running background task."""

    name = "tool_cancel"
    description = (
        "Cancel a running background task. "
        "Use this when the user wants to stop a task that's running in the background. "
        "Requires confirmation before cancelling."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The task ID to cancel",
            },
            "confirmed": {
                "type": "boolean",
                "description": "Must be true to confirm cancellation. Always ask the user for confirmation first.",
            },
        },
        "required": ["task_id", "confirmed"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Cancel a background task."""
        tool_id = kwargs.get("tool_id", "")
        confirmed = kwargs.get("confirmed", False)
        tool_manager: BackgroundToolManager | None = kwargs.get("tool_manager")

        if tool_manager is None:
            return {"error": "Tool manager is required."}

        logger.info(f"Tool call: task_cancel tool_id={tool_id} confirmed={confirmed}")

        if not tool_id:
            return {"error": "Tool ID is required."}

        tool = tool_manager.get_tool(tool_id)

        if not tool:
            return {"error": f"Tool {tool_id} not found."}

        # Check if task is still running
        if tool.status != ToolState.RUNNING:
            return {
                "status": f"{tool.status.value}",
                "message": f"Tool '{tool.tool_name}' is not running (status: {tool.status.value}).",
                "tool_id": tool_id,
            }

        # Require confirmation
        if not confirmed:
            return {
                "status": "confirmation_required",
                "message": f"Are you sure you want to cancel the tool '{tool.tool_name}'?",
                "tool_id": tool_id,
                "tool_name": tool.tool_name,
                "hint": "Set confirmed=true after user approval to proceed with cancellation.",
            }

        # Cancel the task
        if await tool_manager.cancel_tool(tool_id):
            return {
                "status": "cancelled",
                "message": f"Tool '{tool.tool_name}' has been cancelled.",
                "tool_id": tool_id,
                "tool_name": tool.tool_name,
            }
        else:
            return {
                "error": f"Could not cancel tool {tool_id}. It may have already completed.",
            }
