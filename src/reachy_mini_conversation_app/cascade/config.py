"""Cascade-specific configuration, loaded only when cascade mode is active."""

import os
import logging
from typing import Any, Dict
from pathlib import Path

import yaml  # type: ignore[import-untyped]


logger = logging.getLogger(__name__)

# Suppress verbose debug logging from numba (used by librosa for audio resampling)
logging.getLogger("numba").setLevel(logging.WARNING)

# Metadata keys (not passed to provider constructor)
ASR_METADATA_KEYS = {"module", "class", "streaming", "location", "requires", "hardware", "description", "status"}
LLM_METADATA_KEYS = {"module", "class", "location", "requires", "description"}
TTS_METADATA_KEYS = {"module", "class", "location", "requires", "hardware", "description"}


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
            raise RuntimeError(f"cascade.yaml is missing required keys: {', '.join(missing_keys)}")

        logger.info("Cascade configuration loaded from cascade.yaml")
        return config

    except yaml.YAMLError as e:
        raise RuntimeError(f"Invalid YAML syntax in cascade.yaml:\n{e}\nPlease check the file for syntax errors.")


class CascadeConfig:
    """Configuration class for cascade pipeline mode."""

    def __init__(self) -> None:
        """Initialize cascade configuration by loading cascade.yaml."""
        # API keys from environment
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

        self._cascade = _load_cascade_config()

        # ASR config
        self.asr_provider = self._cascade["asr"]["provider"]
        self.asr_providers = self._cascade["asr"]["providers"]

        # LLM config
        self.llm_provider = self._cascade["llm"]["provider"]
        self.llm_providers = self._cascade["llm"]["providers"]

        # TTS config
        self.tts_provider = self._cascade["tts"]["provider"]
        self.tts_providers = self._cascade["tts"]["providers"]
        self.tts_trim_silence = self._cascade["tts"]["trim_silence"]

        # Transcript analysis config (optional section)
        ta = self._cascade.get("transcript_analysis", {})
        self.gliner_model: str = ta.get("gliner_model", "urchade/gliner_small-v2.1")

        self._log_config()

    def get_asr_provider_info(self, name: str | None = None) -> Dict[str, Any]:
        """Get full provider info from cascade.yaml."""
        provider_name = name or self.asr_provider
        if provider_name not in self.asr_providers:
            available = ", ".join(self.asr_providers.keys())
            raise ValueError(f"Unknown ASR provider: {provider_name}. Available: {available}")
        return self.asr_providers[provider_name]

    def get_asr_settings(self, name: str | None = None) -> Dict[str, Any]:
        """Get provider settings (excludes metadata like module, class, streaming, etc.)."""
        info = self.get_asr_provider_info(name)
        return {k: v for k, v in info.items() if k not in ASR_METADATA_KEYS}

    def is_asr_streaming(self, name: str | None = None) -> bool:
        """Check if provider supports streaming."""
        return self.get_asr_provider_info(name)["streaming"]

    def get_llm_provider_info(self, name: str | None = None) -> Dict[str, Any]:
        """Get full LLM provider info from cascade.yaml."""
        provider_name = name or self.llm_provider
        if provider_name not in self.llm_providers:
            available = ", ".join(self.llm_providers.keys())
            raise ValueError(f"Unknown LLM provider: {provider_name}. Available: {available}")
        return self.llm_providers[provider_name]

    def get_llm_settings(self, name: str | None = None) -> Dict[str, Any]:
        """Get LLM provider settings (excludes metadata)."""
        info = self.get_llm_provider_info(name)
        return {k: v for k, v in info.items() if k not in LLM_METADATA_KEYS}

    def get_tts_provider_info(self, name: str | None = None) -> Dict[str, Any]:
        """Get full TTS provider info from cascade.yaml."""
        provider_name = name or self.tts_provider
        if provider_name not in self.tts_providers:
            available = ", ".join(self.tts_providers.keys())
            raise ValueError(f"Unknown TTS provider: {provider_name}. Available: {available}")
        return self.tts_providers[provider_name]

    def get_tts_settings(self, name: str | None = None) -> Dict[str, Any]:
        """Get TTS provider settings (excludes metadata)."""
        info = self.get_tts_provider_info(name)
        return {k: v for k, v in info.items() if k not in TTS_METADATA_KEYS}

    def _log_config(self) -> None:
        """Log the loaded configuration."""
        logger.info(f"Cascade: ASR={self.asr_provider}, LLM={self.llm_provider}, TTS={self.tts_provider}")
        logger.debug(f"ASR provider info: {self.get_asr_provider_info()}")
        logger.debug(f"LLM provider info: {self.get_llm_provider_info()}")
        logger.debug(f"TTS provider info: {self.get_tts_provider_info()}")


# Singleton instance - loaded on import
config = CascadeConfig()
