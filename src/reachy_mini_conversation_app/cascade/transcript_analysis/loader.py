"""Load transcript reaction configurations from profile."""

from __future__ import annotations
import logging
import importlib
from typing import Any, Dict, Callable, Optional
from pathlib import Path

import yaml

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)

PROFILES_DIRECTORY = Path(__file__).parent.parent.parent / "profiles"


def get_profile_reactions() -> Optional[Dict[str, Any]]:
    """Load reactions from current profile's reactions.yaml.

    Returns:
        Dict with reactions config, or None if no profile or no reactions.yaml

    """
    profile = config.REACHY_MINI_CUSTOM_PROFILE
    if not profile:
        logger.debug("No profile set, transcript reactions disabled")
        return None

    reactions_file = PROFILES_DIRECTORY / profile / "reactions.yaml"
    if not reactions_file.exists():
        logger.debug(f"No reactions.yaml in profile '{profile}'")
        return None

    try:
        with open(reactions_file) as f:
            yaml_config = yaml.safe_load(f)

        if not yaml_config:
            return None

        reactions = {}

        # Resolve keyword callbacks
        if "keywords" in yaml_config:
            reactions["keywords"] = {}
            for keyword, callback_name in yaml_config["keywords"].items():
                callback = _import_callback(profile, callback_name)
                if callback:
                    reactions["keywords"][keyword] = callback

        # Resolve entity callbacks
        if "entities" in yaml_config:
            reactions["entities"] = {}
            for entity, callback_name in yaml_config["entities"].items():
                callback = _import_callback(profile, callback_name)
                if callback:
                    reactions["entities"][entity] = callback

        # Pass through gliner_model
        if "gliner_model" in yaml_config:
            reactions["gliner_model"] = yaml_config["gliner_model"]

        logger.info(f"Loaded reactions from profile '{profile}'")
        logger.info(
            f"  Keywords: {len(reactions.get('keywords', {}))}, Entities: {len(reactions.get('entities', {}))}"
        )

        return reactions if (reactions.get("keywords") or reactions.get("entities")) else None

    except Exception as e:
        logger.warning(f"Failed to load reactions from profile '{profile}': {e}")
        return None


def _import_callback(profile: str, callback_name: str) -> Optional[Callable]:
    """Import a callback function from profile module."""
    try:
        module_path = f"reachy_mini_conversation_app.profiles.{profile}.{callback_name}"
        module = importlib.import_module(module_path)
        return getattr(module, callback_name, None)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to import callback '{callback_name}' from profile '{profile}': {e}")
        return None
