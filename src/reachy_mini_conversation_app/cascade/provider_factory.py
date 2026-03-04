"""Factory functions for initializing cascade providers (ASR, LLM, TTS, transcript analysis)."""

from __future__ import annotations
import logging
import importlib
from typing import Any, Dict, List

from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.cascade.asr import ASRProvider
from reachy_mini_conversation_app.cascade.llm import LLMProvider
from reachy_mini_conversation_app.cascade.tts import TTSProvider
from reachy_mini_conversation_app.cascade.config import get_config
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.cascade.transcript_analysis import (
    NoOpTranscriptManager,
    TranscriptAnalysisManager,
)


logger = logging.getLogger(__name__)


CASCADE_EXTRA_INSTRUCTIONS = """\n\n**IMPORTANT:**
## SPEAKING TO THE USER
- To talk to the user, you *MUST* use the 'speak' tool, there is no other way to generate speech.
- When you want to say something, always use the 'speak' tool, even for short acknowledgments like "OK" or "Sure".

## ISSUING SEVERAL TOOLS IN ONE RESPONSE
- You can always issue several tools in one response if needed.
- You can combine the 'speak' tool with other tools in the same response.
- Do not hesitate to use multiple tools if the situation requires it, especially for complex tasks.
"""


def convert_tool_specs_to_chat_format(realtime_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert tool specs from Realtime API format to Chat Completions API format."""
    chat_specs = []
    for spec in realtime_specs:
        if spec["type"] == "function":
            chat_spec = {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            }
            chat_specs.append(chat_spec)
    return chat_specs


def init_provider(provider_type: str, extra_kwargs: Dict[str, Any] | None = None) -> Any:
    """Initialize a provider (ASR/LLM/TTS) from cascade.yaml config.

    Args:
        provider_type: One of "asr", "llm", "tts"
        extra_kwargs: Additional kwargs to pass to provider constructor

    Returns:
        Initialized provider instance

    """
    config = get_config()

    # All API keys that any provider might need
    api_key_map = {
        "OPENAI_API_KEY": config.OPENAI_API_KEY,
        "DEEPGRAM_API_KEY": config.DEEPGRAM_API_KEY,
        "GEMINI_API_KEY": config.GEMINI_API_KEY,
        "ELEVENLABS_API_KEY": config.ELEVENLABS_API_KEY,
    }

    # Get provider name, info, and settings using dynamic attribute access
    name = getattr(config, f"{provider_type}_provider")
    info = getattr(config, f"get_{provider_type}_provider_info")(name)
    kwargs = getattr(config, f"get_{provider_type}_settings")(name)

    # Add API key (validated at config load time)
    requires = info["requires"]
    if len(requires) == 1:
        kwargs["api_key"] = api_key_map[requires[0]]
    elif requires:
        raise ValueError(f"Multi-key providers not supported: {requires}")

    # Merge extra kwargs if provided
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    # Dynamic import and instantiate
    module = importlib.import_module(f"reachy_mini_conversation_app.cascade.{provider_type}.{info['module']}")
    ProviderClass = getattr(module, info["class"])

    # Log with provider-specific details
    extra_info = f", streaming={info['streaming']}" if "streaming" in info else ""
    logger.info(f"Initializing {provider_type.upper()}: {name} (location={info['location']}{extra_info})")

    return ProviderClass(**kwargs)


def init_asr_provider() -> ASRProvider:
    """Initialize ASR provider from cascade.yaml config."""
    return init_provider("asr")  # type: ignore[no-any-return]


def init_llm_provider() -> LLMProvider:
    """Initialize LLM provider from cascade.yaml config."""
    cascade_instructions = get_session_instructions() + CASCADE_EXTRA_INSTRUCTIONS
    return init_provider("llm", {"system_instructions": cascade_instructions})  # type: ignore[no-any-return]


def init_tts_provider() -> TTSProvider:
    """Initialize TTS provider from cascade.yaml config."""
    return init_provider("tts")  # type: ignore[no-any-return]


def init_transcript_analysis(deps: ToolDependencies) -> TranscriptAnalysisManager | NoOpTranscriptManager:
    """Initialize transcript analysis from profile reactions."""
    from reachy_mini_conversation_app.cascade.transcript_analysis import get_profile_reactions

    reactions = get_profile_reactions()
    if not reactions:
        logger.info("No profile reactions configured, transcript analysis disabled")
        return NoOpTranscriptManager()

    return TranscriptAnalysisManager(reactions=reactions, deps=deps)
