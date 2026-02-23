import os
import sys
import logging
from typing import Optional
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


# Locked profile: set to a profile name (e.g., "astronomer") to lock the app
# to that profile and disable all profile switching. Leave as None for normal behavior.
LOCKED_PROFILE: str | None = None
DEFAULT_PROFILES_DIRECTORY = Path(__file__).parent / "profiles"

logger = logging.getLogger(__name__)
API_KEY_SOURCE_ENV = "REACHY_MINI_API_KEY_SOURCE"


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment flag.

    Accepted truthy values: 1, true, yes, on
    Accepted falsy values: 0, false, no, off
    """
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    logger.warning("Invalid boolean value for %s=%r, using default=%s", name, raw, default)
    return default


def _collect_profile_names(profiles_root: Path) -> set[str]:
    """Return profile folder names from a profiles root directory."""
    if not profiles_root.exists() or not profiles_root.is_dir():
        return set()
    return {p.name for p in profiles_root.iterdir() if p.is_dir()}


def _collect_tool_module_names(tools_root: Path) -> set[str]:
    """Return tool module names from a tools directory."""
    if not tools_root.exists() or not tools_root.is_dir():
        return set()
    ignored = {"__init__", "core_tools"}
    return {
        p.stem
        for p in tools_root.glob("*.py")
        if p.is_file() and p.stem not in ignored
    }


def _raise_on_name_collisions(
    *,
    label: str,
    external_root: Path,
    internal_root: Path,
    external_names: set[str],
    internal_names: set[str],
) -> None:
    """Raise with a clear message when external/internal names collide."""
    collisions = sorted(external_names & internal_names)
    if not collisions:
        return

    raise RuntimeError(
        f"Config.__init__(): Ambiguous {label} names found in both external and built-in libraries: {collisions}. "
        f"External {label} root: {external_root}. Built-in {label} root: {internal_root}. "
        f"Please rename the conflicting external {label}(s) to continue."
    )


# Validate LOCKED_PROFILE at startup
if LOCKED_PROFILE is not None:
    _profiles_dir = DEFAULT_PROFILES_DIRECTORY
    _profile_path = _profiles_dir / LOCKED_PROFILE
    _instructions_file = _profile_path / "instructions.txt"
    if not _profile_path.is_dir():
        print(f"Error: LOCKED_PROFILE '{LOCKED_PROFILE}' does not exist in {_profiles_dir}", file=sys.stderr)
        sys.exit(1)
    if not _instructions_file.is_file():
        print(f"Error: LOCKED_PROFILE '{LOCKED_PROFILE}' has no instructions.txt", file=sys.stderr)
        sys.exit(1)

_skip_dotenv = _env_flag("REACHY_MINI_SKIP_DOTENV", default=False)

if _skip_dotenv:
    logger.info("Skipping .env loading because REACHY_MINI_SKIP_DOTENV is set")
else:
    # Locate .env file (search upward from current working directory)
    dotenv_path = find_dotenv(usecwd=True)

    if dotenv_path:
        # Load .env and override environment variables
        load_dotenv(dotenv_path=dotenv_path, override=True)
        logger.info(f"Configuration loaded from {dotenv_path}")
    else:
        logger.warning("No .env file found, using environment variables")


class Config:
    """Configuration class for the conversation app."""

    # Required
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # The key is bootstrapped in console/gradio startup if needed

    # Optional
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-realtime")
    HF_HOME = os.getenv("HF_HOME", "./cache")
    LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, falls back to hf auth login if not set

    logger.debug(f"Model: {MODEL_NAME}, HF_HOME: {HF_HOME}, Vision Model: {LOCAL_VISION_MODEL}")

    _profiles_directory_env = os.getenv("REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY")
    PROFILES_DIRECTORY = (
        Path(_profiles_directory_env) if _profiles_directory_env else Path(__file__).parent / "profiles"
    )
    _tools_directory_env = os.getenv("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY")
    TOOLS_DIRECTORY = Path(_tools_directory_env) if _tools_directory_env else None
    AUTOLOAD_EXTERNAL_TOOLS = _env_flag("AUTOLOAD_EXTERNAL_TOOLS", default=False)
    REACHY_MINI_CUSTOM_PROFILE = LOCKED_PROFILE or os.getenv("REACHY_MINI_CUSTOM_PROFILE")

    logger.debug(f"Custom Profile: {REACHY_MINI_CUSTOM_PROFILE}")

    def __init__(self) -> None:
        """Initialize the configuration."""
        if self.REACHY_MINI_CUSTOM_PROFILE and self.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            selected_profile_path = self.PROFILES_DIRECTORY / self.REACHY_MINI_CUSTOM_PROFILE
            if not selected_profile_path.is_dir():
                available_profiles = sorted(_collect_profile_names(self.PROFILES_DIRECTORY))
                raise RuntimeError(
                    "Config.__init__(): Selected profile "
                    f"'{self.REACHY_MINI_CUSTOM_PROFILE}' was not found in external profiles root "
                    f"{self.PROFILES_DIRECTORY}. "
                    f"Available external profiles: {available_profiles}. "
                    "Either set 'REACHY_MINI_CUSTOM_PROFILE' to one of the available external profiles "
                    "or unset 'REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY' to use built-in profiles."
                )

        if self.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            external_profiles = _collect_profile_names(self.PROFILES_DIRECTORY)
            internal_profiles = _collect_profile_names(DEFAULT_PROFILES_DIRECTORY)
            _raise_on_name_collisions(
                label="profile",
                external_root=self.PROFILES_DIRECTORY,
                internal_root=DEFAULT_PROFILES_DIRECTORY,
                external_names=external_profiles,
                internal_names=internal_profiles,
            )

        if self.TOOLS_DIRECTORY is not None:
            builtin_tools_root = Path(__file__).parent / "tools"
            external_tools = _collect_tool_module_names(self.TOOLS_DIRECTORY)
            internal_tools = _collect_tool_module_names(builtin_tools_root)
            _raise_on_name_collisions(
                label="tool",
                external_root=self.TOOLS_DIRECTORY,
                internal_root=builtin_tools_root,
                external_names=external_tools,
                internal_names=internal_tools,
            )

        if self.PROFILES_DIRECTORY != DEFAULT_PROFILES_DIRECTORY:
            logger.warning(
                "Environment variable 'REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY' is set. "
                "Profiles (instructions.txt, ...) will be loaded from %s.",
                self.PROFILES_DIRECTORY,
            )
        else:
            logger.info(
                "'REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY' is not set. "
                "Using built-in profiles from %s.",
                DEFAULT_PROFILES_DIRECTORY,
            )

        if self.TOOLS_DIRECTORY is not None:
            logger.warning(
                "Environment variable 'REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY' is set. "
                "External tools will be loaded from %s.",
                self.TOOLS_DIRECTORY,
            )
        else:
            logger.info(
                "'REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY' is not set. "
                "Using built-in shared tools only."
            )


config = Config()


def set_custom_profile(profile: str | None) -> None:
    """Update the selected custom profile at runtime and expose it via env.

    This ensures modules that read `config` and code that inspects the
    environment see a consistent value.
    """
    if LOCKED_PROFILE is not None:
        return
    try:
        config.REACHY_MINI_CUSTOM_PROFILE = profile
    except Exception:
        pass
    try:
        if profile:
            os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile
        else:
            os.environ.pop("REACHY_MINI_CUSTOM_PROFILE", None)
    except Exception:
        pass


def read_env_lines(env_path: Path) -> list[str]:
    """Load env file contents or a template as a list of lines."""
    inst = env_path.parent
    try:
        if env_path.exists():
            try:
                return env_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                return []
        template_text = None
        for candidate in (
            inst / ".env.example",
            Path.cwd() / ".env.example",
            Path(__file__).parent / ".env.example",
        ):
            try:
                if candidate.exists():
                    template_text = candidate.read_text(encoding="utf-8")
                    break
            except Exception:
                continue
        return template_text.splitlines() if template_text else []
    except Exception:
        return []


def load_instance_env(instance_path: Optional[str], *, load_profile: bool = True) -> None:
    """Load `<instance_path>/.env` into the process and sync in-memory config."""
    if not instance_path:
        return
    env_path = Path(instance_path) / ".env"
    if not env_path.exists():
        return
    try:
        load_dotenv(dotenv_path=str(env_path), override=True)
    except Exception:
        return

    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if key:
        try:
            config.OPENAI_API_KEY = key
        except Exception:
            pass

    if not load_profile or LOCKED_PROFILE is not None:
        return

    profile = os.getenv("REACHY_MINI_CUSTOM_PROFILE")
    if profile is not None:
        set_custom_profile(profile.strip() or None)


def persist_api_key(
    key: str,
    instance_path: Optional[str],
    source: str = "settings_ui",
) -> None:
    """Persist an OpenAI API key to process env, config, and instance .env."""
    k = (key or "").strip()
    if not k:
        return
    source_value = (source or "").strip()

    try:
        os.environ["OPENAI_API_KEY"] = k
    except Exception:
        pass
    if source_value:
        try:
            os.environ[API_KEY_SOURCE_ENV] = source_value
        except Exception:
            pass
    try:
        config.OPENAI_API_KEY = k
    except Exception:
        pass

    if not instance_path:
        return

    try:
        env_path = Path(instance_path) / ".env"
        lines = read_env_lines(env_path)
        replaced = False
        for i, ln in enumerate(lines):
            if ln.strip().startswith("OPENAI_API_KEY="):
                lines[i] = f"OPENAI_API_KEY={k}"
                replaced = True
                break
        if not replaced:
            lines.append(f"OPENAI_API_KEY={k}")
        if source_value:
            source_replaced = False
            for i, ln in enumerate(lines):
                if ln.strip().startswith(f"{API_KEY_SOURCE_ENV}="):
                    lines[i] = f"{API_KEY_SOURCE_ENV}={source_value}"
                    source_replaced = True
                    break
            if not source_replaced:
                lines.append(f"{API_KEY_SOURCE_ENV}={source_value}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Persisted OPENAI_API_KEY to %s", env_path)
        try:
            load_dotenv(dotenv_path=str(env_path), override=True)
        except Exception:
            pass
    except Exception as e:
        logger.warning("Failed to persist OPENAI_API_KEY: %s", e)


def persist_personality(
    profile: Optional[str],
    instance_path: Optional[str],
    *,
    custom_logger: Optional[logging.Logger] = None,
) -> None:
    """Persist startup personality to config and instance .env."""
    if LOCKED_PROFILE is not None:
        return
    log = custom_logger or logger
    selection = (profile or "").strip() or None
    set_custom_profile(selection)

    if not instance_path:
        return

    try:
        env_path = Path(instance_path) / ".env"
        lines = read_env_lines(env_path)
        replaced = False
        for i, ln in enumerate(list(lines)):
            if ln.strip().startswith("REACHY_MINI_CUSTOM_PROFILE="):
                if selection:
                    lines[i] = f"REACHY_MINI_CUSTOM_PROFILE={selection}"
                else:
                    lines.pop(i)
                replaced = True
                break
        if selection and not replaced:
            lines.append(f"REACHY_MINI_CUSTOM_PROFILE={selection}")
        if selection is None and not env_path.exists():
            return
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.info("Persisted startup personality to %s", env_path)
        try:
            load_dotenv(dotenv_path=str(env_path), override=True)
        except Exception:
            pass
    except Exception as e:
        log.warning("Failed to persist REACHY_MINI_CUSTOM_PROFILE: %s", e)
