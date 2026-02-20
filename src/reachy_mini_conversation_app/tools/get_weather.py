"""Long wait tool for testing background task manager."""
import asyncio
import logging
from typing import Any, Dict
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)

class GetWeather(Tool):
    """Fetch the current weather forecast from satellite data."""

    name = "get_weather"
    description = "Fetch the current weather forecast for a given location. Use this when someone asks about the weather, temperature, or if it's going to rain. YOU MUST USE THIS TOOL WHEN THE USER ASKS ABOUT THE WEATHER."
    parameters_schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city or location to get weather for (e.g., 'Paris', 'New York', 'Tokyo')",
            },
        },
        "required": ["location"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Fetch weather data (simulated with 2 min delay for testing)."""
        location = kwargs.get("location", "Paris")
        await asyncio.sleep(30)
        if location.lower() == "paris":
            return {"location": location, "temperature": "18°C", "condition": "Partly cloudy", "humidity": "65%"}
        return {"location": location, "temperature": "15°C", "condition": "Sunny", "humidity": "50%"}
