"""Synchronize external tool requirements into this project's dependencies."""

from __future__ import annotations
import os
import shutil
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass
from urllib.parse import urlparse

from huggingface_hub import HfApi, hf_hub_download


EXTERNAL_TOOLS_GROUP = "external-tools"
DEFAULT_EXTERNAL_TOOLS_DIRECTORY = Path("external_content") / "external_tools"
_TOOL_FILE_EXCLUSIONS = {
    "__init__.py",
    "app.py",
    "main.py",
    "setup.py",
}


@dataclass(frozen=True)
class ToolSpaceSyncResult:
    """Result of syncing tool-space requirements and tool module."""

    requirements_path: Path
    downloaded_tool_path: Path
    downloaded_tool_source: str
    dependency_group: str


def normalize_space_id(space: str) -> str:
    """Normalize a Hugging Face Space ID or URL to ``owner/repo``."""
    raw = (space or "").strip().rstrip("/")
    if not raw:
        raise ValueError("Space ID is empty.")

    if "huggingface.co" in raw:
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 3 and parts[0] == "spaces":
            return _validate_space_id(parts[1], parts[2])
        raise ValueError(
            "Invalid Hugging Face Space URL. Expected https://huggingface.co/spaces/<owner>/<repo>.",
        )

    if raw.startswith("spaces/"):
        parts = raw.split("/")
        if len(parts) == 3:
            return _validate_space_id(parts[1], parts[2])
        raise ValueError("Invalid Space ID. Expected spaces/<owner>/<repo>.")

    parts = raw.split("/")
    if len(parts) != 2:
        raise ValueError("Invalid Space ID. Expected <owner>/<repo>.")
    return _validate_space_id(parts[0], parts[1])


def _validate_space_id(owner: str, repo: str) -> str:
    owner_s = owner.strip()
    repo_s = repo.strip()
    if not owner_s or not repo_s:
        raise ValueError("Invalid Space ID. Owner and repo are required.")
    return f"{owner_s}/{repo_s}"


def _requirements_is_empty(requirements_path: Path) -> bool:
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return False
    return True


def _run_command(command: list[str], logger: logging.Logger) -> tuple[int, str, str]:
    logger.info("Running command: %s", " ".join(command))
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.stdout.strip():
        logger.info(completed.stdout.strip())
    if completed.stderr.strip():
        logger.warning(completed.stderr.strip())
    return completed.returncode, completed.stdout, completed.stderr


def _resolve_external_tools_directory(directory: str | None = None) -> Path:
    selected = directory or os.getenv("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY")
    resolved = (Path(selected).expanduser() if selected else DEFAULT_EXTERNAL_TOOLS_DIRECTORY).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _select_tool_python_file(space_id: str, token: str | None) -> str:
    """Pick exactly one tool Python file from a space snapshot listing.

    Selection heuristic:
    1) Keep only non-hidden ``*.py`` files, excluding app/entrypoint files.
    2) Prefer files named ``*_tool.py``.
    3) Require exactly one candidate.
    """
    api = HfApi(token=token)
    repo_files = api.list_repo_files(repo_id=space_id, repo_type="space", token=token)

    candidates: list[str] = []
    for repo_file in repo_files:
        path = Path(repo_file)
        filename = path.name
        if path.suffix != ".py":
            continue
        if filename.startswith("_") or filename in _TOOL_FILE_EXCLUSIONS:
            continue
        if any(part in {"tests", "__pycache__"} for part in path.parts):
            continue
        candidates.append(repo_file)

    if not candidates:
        raise ValueError(
            f"No candidate tool .py file found in space '{space_id}'.",
        )

    preferred = [entry for entry in candidates if Path(entry).name.endswith("_tool.py")]
    selected_pool = preferred if preferred else candidates

    if len(selected_pool) != 1:
        raise ValueError(
            "Ambiguous tool .py files in space "
            f"'{space_id}': {selected_pool}. Keep exactly one tool module or rename helper files.",
        )

    return selected_pool[0]


def _download_tool_module(
    *,
    space_id: str,
    token: str | None,
    target_directory: Path,
) -> tuple[Path, str]:
    repo_file = _select_tool_python_file(space_id, token)
    downloaded = Path(
        hf_hub_download(
            repo_id=space_id,
            repo_type="space",
            filename=repo_file,
            token=token,
        )
    )
    destination = target_directory / Path(repo_file).name
    shutil.copy2(downloaded, destination)
    return destination, repo_file


def sync_tool_space_dependencies(
    *,
    space: str,
    logger: logging.Logger,
    hf_token: str | None = None,
    dependency_group: str = EXTERNAL_TOOLS_GROUP,
    external_tools_directory: str | None = None,
) -> ToolSpaceSyncResult:
    """Add Space requirements to ``pyproject.toml`` using ``uv add``.

    The operation is rollback-safe: on any failure, both ``pyproject.toml``
    and ``uv.lock`` are restored to their prior content.
    """
    space_id = normalize_space_id(space)
    token = (hf_token or os.getenv("HF_TOKEN") or "").strip() or None

    requirements_path = Path(
        hf_hub_download(
            repo_id=space_id,
            repo_type="space",
            filename="requirements.txt",
            token=token,
        )
    )
    logger.info("Downloaded requirements from %s to %s", space_id, requirements_path)
    target_tools_directory = _resolve_external_tools_directory(external_tools_directory)

    if _requirements_is_empty(requirements_path):
        raise ValueError(f"{space_id} requirements.txt is empty.")

    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise RuntimeError("pyproject.toml not found in current working directory.")

    uv_lock_path = Path("uv.lock")
    pyproject_before = pyproject_path.read_bytes()
    uv_lock_exists_before = uv_lock_path.exists()
    uv_lock_before = uv_lock_path.read_bytes() if uv_lock_exists_before else b""

    command = [
        "uv",
        "add",
        "--group",
        dependency_group,
        "--requirements",
        str(requirements_path),
    ]
    copied_tool_path: Path | None = None
    try:
        code, _, stderr = _run_command(command, logger)
        if code != 0:
            raise RuntimeError(f"uv exited with code {code}. stderr: {stderr.strip()}")

        copied_tool_path, source_tool_file = _download_tool_module(
            space_id=space_id,
            token=token,
            target_directory=target_tools_directory,
        )
    except Exception as e:
        pyproject_path.write_bytes(pyproject_before)
        if uv_lock_exists_before:
            uv_lock_path.write_bytes(uv_lock_before)
        elif uv_lock_path.exists():
            uv_lock_path.unlink()
        if copied_tool_path is not None and copied_tool_path.exists():
            copied_tool_path.unlink()
        raise RuntimeError(
            "Dependency/tool synchronization failed and changes were rolled back. "
            f"Reason: {e}",
        ) from e

    logger.info(
        "Successfully synchronized Space dependencies into dependency group '%s'.",
        dependency_group,
    )
    logger.info(
        "Downloaded tool module %s to %s",
        source_tool_file,
        copied_tool_path,
    )
    return ToolSpaceSyncResult(
        requirements_path=requirements_path,
        downloaded_tool_path=copied_tool_path,
        downloaded_tool_source=source_tool_file,
        dependency_group=dependency_group,
    )
