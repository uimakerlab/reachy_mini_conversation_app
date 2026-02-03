"""Cascade-specific configuration, loaded only when cascade mode is active."""

import os
import logging
from typing import Any, Dict
from pathlib import Path

import yaml  # type: ignore[import-untyped]


logger = logging.getLogger(__name__)

# Suppress verbose debug logging from numba (used by librosa for audio resampling)
logging.getLogger("numba").setLevel(logging.WARNING)


def _load_cascade_config() -> Dict[str, Any]:
    """Load cascade configuration from YAML file."""
    config_file = Path("cascade.yaml")

    if not config_file.exists():
        raise RuntimeError(
            "cascade.yaml not found. Please create it:\nThe cascade.yaml file configures ASR, LLM, and TTS providers."
        )

    try:
        with open(config_file) as f:
            config: Dict[str, Any] = yaml.safe_load(f)

        # Validate required top-level keys
        required_keys = ["asr", "llm", "tts"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise RuntimeError(f"cascade.yaml is missing required keys: {', '.join(missing_keys)}\n")

        logger.info("Cascade configuration loaded from cascade.yaml")
        return config

    except yaml.YAMLError as e:
        raise RuntimeError(f"Invalid YAML syntax in cascade.yaml:\n{e}\nPlease check the file for syntax errors.")


class CascadeConfig:
    """Configuration class for cascade pipeline mode."""

    def __init__(self) -> None:
        """Initialize cascade configuration by loading cascade.yaml."""
        # Load API keys from environment
        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        # Load cascade.yaml
        _cascade = _load_cascade_config()

        # ASR configuration
        self.CASCADE_ASR_PROVIDER = _cascade["asr"]["provider"]
        self.PARAKEET_MODEL = _cascade["asr"]["parakeet"]["model"]
        self.PARAKEET_PRECISION = _cascade["asr"]["parakeet"]["precision"]
        self.PARAKEET_STREAMING_CONTEXT = tuple(
            _cascade["asr"].get("parakeet_streaming", {}).get("context_size", [256, 256])
        )
        self.PARAKEET_STREAMING_DEPTH = _cascade["asr"].get("parakeet_streaming", {}).get("depth")
        self.DEEPGRAM_MODEL = _cascade["asr"].get("deepgram_streaming", {}).get("model", "nova-2")

        # LLM configuration
        self.CASCADE_LLM_PROVIDER = _cascade["llm"]["provider"]
        self.CASCADE_LLM_MODEL = _cascade["llm"]["openai_gpt"]["model"]
        self.GEMINI_MODEL = _cascade["llm"]["gemini"]["model"]

        # TTS configuration
        self.CASCADE_TTS_PROVIDER = _cascade["tts"]["provider"]
        self.CASCADE_TTS_TRIM_SILENCE = _cascade["tts"]["trim_silence"]
        self.CASCADE_TTS_VOICE = _cascade["tts"]["openai_tts"]["voice"]

        # Provider-specific TTS settings
        _tts_kokoro = _cascade["tts"]["kokoro"]
        _tts_elevenlabs = _cascade["tts"]["elevenlabs"]
        self.KOKORO_VOICE = _tts_kokoro["voice"]
        self.ELEVENLABS_VOICE_ID = _tts_elevenlabs["voice_id"]
        self.ELEVENLABS_MODEL = _tts_elevenlabs["model"]

        # Log configuration
        logger.debug(
            f"Cascade: ASR={self.CASCADE_ASR_PROVIDER}, LLM={self.CASCADE_LLM_PROVIDER} "
            f"({self.CASCADE_LLM_MODEL}), TTS={self.CASCADE_TTS_PROVIDER} "
            f"(trim_silence={self.CASCADE_TTS_TRIM_SILENCE})"
        )
        if self.CASCADE_ASR_PROVIDER == "parakeet":
            logger.debug(f"Parakeet: model={self.PARAKEET_MODEL}, precision={self.PARAKEET_PRECISION}")
        elif self.CASCADE_ASR_PROVIDER == "parakeet_streaming":
            logger.debug(
                f"Parakeet Streaming: model={self.PARAKEET_MODEL}, "
                f"precision={self.PARAKEET_PRECISION}, context={self.PARAKEET_STREAMING_CONTEXT}, "
                f"depth={self.PARAKEET_STREAMING_DEPTH}"
            )
        elif self.CASCADE_ASR_PROVIDER == "deepgram_streaming":
            logger.debug(f"Deepgram: model={self.DEEPGRAM_MODEL}")
        if self.CASCADE_LLM_PROVIDER == "gemini":
            logger.debug(f"Gemini: model={self.GEMINI_MODEL}")
        if self.CASCADE_TTS_PROVIDER == "openai_tts":
            logger.debug(f"OpenAI TTS: voice={self.CASCADE_TTS_VOICE}")
        elif self.CASCADE_TTS_PROVIDER == "kokoro":
            logger.debug(f"Kokoro: voice={self.KOKORO_VOICE}")
        elif self.CASCADE_TTS_PROVIDER == "elevenlabs":
            logger.debug(f"ElevenLabs: voice_id={self.ELEVENLABS_VOICE_ID}, model={self.ELEVENLABS_MODEL}")


# Singleton instance - loaded on import
config = CascadeConfig()
