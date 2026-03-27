"""Web search tool for Reachy Mini conversation app."""

import logging
import os
from typing import Any, Dict

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


class WebSearch(Tool):
    """Search the web and return a summary of results."""

    name = "web_search"
    description = (
        "Search the web for current information. Use this when the user asks about "
        "news, weather, facts, or anything that requires up-to-date information."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute web search via OpenAI responses API with web search tool."""
        query = kwargs.get("query", "")
        if not query:
            return {"status": "error", "message": "No search query provided"}

        logger.info(f"Web search: {query}")

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/responses",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4.1-mini",
                        "tools": [{"type": "web_search_preview"}],
                        "input": f"Search the web and provide a concise summary: {query}",
                    },
                )
                response.raise_for_status()
                data = response.json()

                # Extract text from response output
                result_text = ""
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for content in item.get("content", []):
                            if content.get("type") == "output_text":
                                result_text = content.get("text", "")

                if result_text:
                    return {"status": "success", "result": result_text}
                else:
                    return {"status": "error", "message": "No results found"}

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {"status": "error", "message": str(e)}
