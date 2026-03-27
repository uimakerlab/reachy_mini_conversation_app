"""Token usage tracker — logs all OpenAI API usage to a local JSON file."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)

USAGE_FILE = Path.home() / "reachy_token_usage.json"


def load_usage() -> Dict:
    if USAGE_FILE.exists():
        with open(USAGE_FILE) as f:
            return json.load(f)
    return {"total_input_tokens": 0, "total_output_tokens": 0, "total_cost_usd": 0.0, "sessions": []}


def save_usage(data: Dict):
    with open(USAGE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def log_tokens(input_tokens: int, output_tokens: int, model: str = "gpt-realtime"):
    """Call this to log token usage."""
    usage = load_usage()
    usage["total_input_tokens"] += input_tokens
    usage["total_output_tokens"] += output_tokens

    # Rough cost estimates (gpt-4o-realtime pricing)
    input_cost = input_tokens * 0.005 / 1000
    output_cost = output_tokens * 0.020 / 1000
    usage["total_cost_usd"] += input_cost + output_cost

    usage["sessions"].append({
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(input_cost + output_cost, 4),
    })

    save_usage(usage)


class TokenUsage(Tool):
    """Report current token usage statistics."""

    name = "token_usage"
    description = "Report how many OpenAI API tokens have been used and the estimated cost."
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        usage = load_usage()
        return {
            "status": "success",
            "total_input_tokens": usage["total_input_tokens"],
            "total_output_tokens": usage["total_output_tokens"],
            "total_cost_usd": round(usage["total_cost_usd"], 4),
            "session_count": len(usage["sessions"]),
        }
