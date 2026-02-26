"""Synchronize external tool requirements into this project's dependencies."""

from __future__ import annotations
import os
import re
import ast
import json
import shutil
import hashlib
import logging
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from urllib.parse import urlparse

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, EntryNotFoundError, RepositoryNotFoundError


EXTERNAL_TOOLS_GROUP = "external-tools"
DEFAULT_EXTERNAL_TOOLS_DIRECTORY = Path("external_content") / "external_tools"
_TOOL_FILE_EXCLUSIONS = {
    "__init__.py",
    "app.py",
    "main.py",
    "setup.py",
}
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UNSAFE_REQUIREMENTS_PREFIXES = (
    "-",
    "--",
)
_UNSAFE_REQUIREMENTS_SUBSTRINGS = (
    "://",
    "git+",
    "svn+",
    "hg+",
    "bzr+",
    "file:",
    " @ ",
)
_UNSAFE_REQUIREMENTS_PATH_PREFIXES = (
    "./",
    "../",
    "/",
    "~",
)


@dataclass(frozen=True)
class ToolSpaceSyncResult:
    """Result of syncing tool-space requirements and tool module."""

    requirements_path: Path | None
    downloaded_tool_path: Path
    downloaded_tool_source: str
    dependency_group: str
    resolved_revision: str
    synced_at_utc: str
    requirements_sha256: str | None
    metadata_path: Path


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


def _validate_requirements_for_safe_install(requirements_path: Path) -> None:
    """Validate requirements.txt blocks directives/URLs/paths before installation."""
    for line_no, raw_line in enumerate(requirements_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Drop inline comments for simple safety checks.
        requirement = stripped.split(" #", 1)[0].strip()
        lowered = requirement.lower()

        if lowered.startswith(_UNSAFE_REQUIREMENTS_PREFIXES):
            raise ValueError(
                f"Unsafe requirements.txt entry at line {line_no}: '{requirement}'. "
                "pip directives/options are not allowed for external tool dependency sync."
            )

        if lowered.startswith(_UNSAFE_REQUIREMENTS_PATH_PREFIXES) or re.match(r"^[a-z]:\\\\", lowered):
            raise ValueError(
                f"Unsafe requirements.txt entry at line {line_no}: '{requirement}'. "
                "Local file paths are not allowed for external tool dependency sync."
            )

        if any(token in lowered for token in _UNSAFE_REQUIREMENTS_SUBSTRINGS):
            raise ValueError(
                f"Unsafe requirements.txt entry at line {line_no}: '{requirement}'. "
                "Direct URL/VCS/file references are not allowed for external tool dependency sync."
            )


def _run_command(command: list[str], logger: logging.Logger) -> tuple[int, str, str]:
    logger.info("Running command: %s", " ".join(command))
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.stdout.strip():
        logger.info(completed.stdout.strip())
    if completed.stderr.strip():
        if completed.returncode == 0:
            logger.info(completed.stderr.strip())
        else:
            logger.warning(completed.stderr.strip())
    return completed.returncode, completed.stdout, completed.stderr


def _resolve_external_tools_directory(directory: str | None = None) -> Path:
    selected = directory or os.getenv("REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY")
    resolved = (Path(selected).expanduser() if selected else DEFAULT_EXTERNAL_TOOLS_DIRECTORY).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_space_revision(space_id: str, token: str | None) -> str:
    """Resolve Space to immutable commit SHA to avoid branch drift between downloads."""
    api = HfApi(token=token)
    info = api.space_info(repo_id=space_id, token=token)
    resolved_revision = (getattr(info, "sha", None) or "").strip()
    if not resolved_revision:
        raise RuntimeError(f"Unable to resolve revision for Hugging Face Space '{space_id}'.")
    return resolved_revision


def _select_tool_python_file(space_id: str, token: str | None, revision: str | None = None) -> str:
    """Pick exactly one tool Python file from a space snapshot listing.

    Selection heuristic:
    1) Keep only non-hidden ``*.py`` files, excluding app/entrypoint files.
    2) Prefer files named ``*_tool.py``.
    3) Require exactly one candidate.
    """
    api = HfApi(token=token)
    repo_files = api.list_repo_files(
        repo_id=space_id,
        repo_type="space",
        revision=revision,
        token=token,
    )

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
    revision: str | None = None,
) -> tuple[Path, str]:
    repo_file = _select_tool_python_file(space_id, token, revision)
    downloaded = Path(
        hf_hub_download(
            repo_id=space_id,
            repo_type="space",
            revision=revision,
            filename=repo_file,
            token=token,
        )
    )
    destination = target_directory / Path(repo_file).name
    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=target_directory,
        prefix=f".{destination.stem}.",
        suffix=".tmp.py",
        encoding="utf-8",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        shutil.copy2(downloaded, tmp_path)
        _validate_tool_module_format(tmp_path)
        os.replace(tmp_path, destination)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    if tmp_path.exists():
        tmp_path.unlink()
    return destination, repo_file


def _collect_class_level_attributes(node: ast.ClassDef) -> set[str]:
    """Collect class-level assigned attribute names."""
    attributes: set[str] = set()
    for member in node.body:
        if isinstance(member, ast.Assign):
            for target in member.targets:
                if isinstance(target, ast.Name):
                    attributes.add(target.id)
        elif isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
            attributes.add(member.target.id)
    return attributes


def _extract_class_level_string_value(node: ast.ClassDef, attribute_name: str) -> str | None:
    """Extract a class-level string literal assigned to ``attribute_name``."""
    for member in node.body:
        value_node: ast.expr | None = None
        if isinstance(member, ast.Assign):
            for target in member.targets:
                if isinstance(target, ast.Name) and target.id == attribute_name:
                    value_node = member.value
                    break
        elif isinstance(member, ast.AnnAssign):
            if isinstance(member.target, ast.Name) and member.target.id == attribute_name:
                value_node = member.value

        if value_node is not None:
            if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
                return value_node.value
            return None
    return None


def _is_tool_subclass(node: ast.ClassDef, aliases: set[str]) -> bool:
    """Return True when class bases include Tool or its known aliases."""
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in aliases:
            return True
        if isinstance(base, ast.Attribute) and base.attr == "Tool":
            return True
    return False


def _validate_tool_module_format(tool_path: Path) -> None:
    """Validate downloaded module has one concrete Tool-shaped class."""
    try:
        tree = ast.parse(tool_path.read_text(encoding="utf-8"), filename=str(tool_path))
    except SyntaxError as e:
        line = f" line {e.lineno}" if e.lineno is not None else ""
        raise ValueError(
            f"Invalid Python syntax in downloaded tool '{tool_path.name}' ({e.msg}{line})."
        ) from e

    tool_aliases = {"Tool"}
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                if alias.name == "Tool":
                    tool_aliases.add(alias.asname or alias.name)

    tool_subclasses: list[ast.ClassDef] = []
    concrete_tool_subclasses: list[ast.ClassDef] = []
    for statement in tree.body:
        if not isinstance(statement, ast.ClassDef):
            continue
        if not _is_tool_subclass(statement, tool_aliases):
            continue

        tool_subclasses.append(statement)
        has_async_call = any(
            isinstance(member, ast.AsyncFunctionDef) and member.name == "__call__"
            for member in statement.body
        )
        if has_async_call:
            concrete_tool_subclasses.append(statement)

    if not tool_subclasses:
        raise ValueError(
            "No Tool subclass found in downloaded module. "
            "Expected a class inheriting Tool with name, description, parameters_schema, and async __call__."
        )

    if len(concrete_tool_subclasses) != 1:
        if not concrete_tool_subclasses:
            raise ValueError(
                "Expected exactly one concrete Tool subclass with async __call__, but found none."
            )
        concrete_names = [node.name for node in concrete_tool_subclasses]
        raise ValueError(
            "Expected exactly one concrete Tool subclass with async __call__, "
            f"but found {len(concrete_tool_subclasses)}: {concrete_names}."
        )

    tool_class = concrete_tool_subclasses[0]
    attributes = _collect_class_level_attributes(tool_class)
    missing_fields = [name for name in ("name", "description", "parameters_schema") if name not in attributes]
    if missing_fields:
        raise ValueError(
            f"Invalid Tool implementation '{tool_class.name}': missing required class field(s): {missing_fields}."
        )

    tool_name = _extract_class_level_string_value(tool_class, "name")
    if tool_name is None or not tool_name.strip():
        raise ValueError(
            f"Invalid Tool implementation '{tool_class.name}': Tool.name must be a non-empty string literal."
        )
    if _TOOL_NAME_PATTERN.fullmatch(tool_name) is None:
        raise ValueError(
            f"Invalid Tool implementation '{tool_class.name}': "
            f"Tool.name '{tool_name}' must match pattern '[A-Za-z_][A-Za-z0-9_]*'."
        )


def _try_download_requirements(
    *,
    space_id: str,
    token: str | None,
    logger: logging.Logger,
    revision: str | None = None,
) -> Path | None:
    """Download requirements.txt if present; return None when absent or empty."""
    try:
        requirements_path = Path(
            hf_hub_download(
                repo_id=space_id,
                repo_type="space",
                revision=revision,
                filename="requirements.txt",
                token=token,
            )
        )
    except EntryNotFoundError:
        logger.info("No requirements.txt found in %s. Skipping dependency sync.", space_id)
        return None

    logger.info("Downloaded requirements from %s to %s", space_id, requirements_path)
    if _requirements_is_empty(requirements_path):
        logger.info("requirements.txt is empty in %s. Skipping dependency sync.", space_id)
        return None
    return requirements_path


def _is_hf_token_permission_issue(error: Exception) -> bool:
    """Return True when an HF Hub error likely means missing/insufficient token permissions."""
    if isinstance(error, (GatedRepoError, RepositoryNotFoundError)):
        return True
    if isinstance(error, HfHubHTTPError):
        status_code = getattr(error.response, "status_code", None)
        return status_code in {401, 403}

    message = str(error).lower()
    return (
        "401" in message
        or "403" in message
        or "gated" in message
        or "access to this repo is restricted" in message
    )


def _sha256_file(file_path: Path) -> str:
    """Return SHA-256 digest for file contents."""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _build_metadata_path(tool_path: Path) -> Path:
    """Build metadata sidecar path next to downloaded tool."""
    return tool_path.with_name(f"{tool_path.name}.metadata.json")


def _write_tool_metadata(
    *,
    metadata_path: Path,
    space_id: str,
    revision: str,
    synced_at_utc: str,
    requirements_sha256: str | None,
    source_tool_file: str,
) -> None:
    """Write tool metadata atomically."""
    payload = {
        "schema_version": 1,
        "space_id": space_id,
        "revision": revision,
        "synced_at_utc": synced_at_utc,
        "requirements_sha256": requirements_sha256,
        "source_tool_file": source_tool_file,
    }

    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=metadata_path.parent,
        prefix=f".{metadata_path.stem}.",
        suffix=".tmp.json",
        encoding="utf-8",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        json.dump(payload, tmp_file, indent=2, sort_keys=True)
        tmp_file.write("\n")

    try:
        os.replace(tmp_path, metadata_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


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

    try:
        resolved_revision = _resolve_space_revision(space_id, token)
        requirements_path = _try_download_requirements(
            space_id=space_id,
            token=token,
            logger=logger,
            revision=resolved_revision,
        )
    except Exception as e:
        if _is_hf_token_permission_issue(e):
            raise RuntimeError(
                f"Unable to access Hugging Face Space '{space_id}' (not found or access denied). "
                "Check owner/repo and, for private Spaces, ensure HF_TOKEN is set and has access."
            ) from e
        raise
    target_tools_directory = _resolve_external_tools_directory(external_tools_directory)

    pyproject_path: Path | None = None
    uv_lock_path: Path | None = None
    pyproject_before = b""
    uv_lock_exists_before = False
    uv_lock_before = b""
    dependency_sync_attempted = False
    dependencies_synced = False

    if requirements_path is not None:
        pyproject_path = Path("pyproject.toml")
        if not pyproject_path.exists():
            raise RuntimeError("pyproject.toml not found in current working directory.")

        uv_lock_path = Path("uv.lock")
        pyproject_before = pyproject_path.read_bytes()
        uv_lock_exists_before = uv_lock_path.exists()
        uv_lock_before = uv_lock_path.read_bytes() if uv_lock_exists_before else b""

    copied_tool_path: Path | None = None
    source_tool_file: str | None = None
    metadata_path: Path | None = None
    synced_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    requirements_sha256 = _sha256_file(requirements_path) if requirements_path is not None else None
    try:
        if requirements_path is not None:
            _validate_requirements_for_safe_install(requirements_path)
            dependency_sync_attempted = True
            command = [
                "uv",
                "add",
                "--group",
                dependency_group,
                "--requirements",
                str(requirements_path),
            ]
            code, _, stderr = _run_command(command, logger)
            if code != 0:
                raise RuntimeError(f"uv exited with code {code}. stderr: {stderr.strip()}")
            dependencies_synced = True

        copied_tool_path, source_tool_file = _download_tool_module(
            space_id=space_id,
            token=token,
            target_directory=target_tools_directory,
            revision=resolved_revision,
        )
        metadata_path = _build_metadata_path(copied_tool_path)
        _write_tool_metadata(
            metadata_path=metadata_path,
            space_id=space_id,
            revision=resolved_revision,
            synced_at_utc=synced_at_utc,
            requirements_sha256=requirements_sha256,
            source_tool_file=source_tool_file,
        )
    except Exception as e:
        permission_hint = ""
        if _is_hf_token_permission_issue(e):
            permission_hint = (
                f" If '{space_id}' is private, ensure HF_TOKEN is set and has access to this Space."
            )
        if dependency_sync_attempted and pyproject_path is not None and uv_lock_path is not None:
            pyproject_path.write_bytes(pyproject_before)
            if uv_lock_exists_before:
                uv_lock_path.write_bytes(uv_lock_before)
            elif uv_lock_path.exists():
                uv_lock_path.unlink()
        if copied_tool_path is not None and copied_tool_path.exists():
            copied_tool_path.unlink()
        if metadata_path is not None and metadata_path.exists():
            metadata_path.unlink()
        if dependency_sync_attempted:
            raise RuntimeError(
                "Dependency/tool synchronization failed and changes were rolled back. "
                f"Reason: {e}{permission_hint}",
            ) from e
        raise RuntimeError(f"Tool synchronization failed. Reason: {e}{permission_hint}") from e

    if dependencies_synced:
        logger.info(
            "Successfully synchronized external tool's dependencies into dependency group '%s'.",
            dependency_group,
        )
    else:
        logger.info("Dependency sync skipped (no requirements.txt or empty requirements.txt).")
    logger.info(
        "Downloaded tool module %s to %s",
        source_tool_file,
        copied_tool_path,
    )
    logger.info("Tool metadata saved to %s", metadata_path)
    if copied_tool_path is None or source_tool_file is None:
        raise RuntimeError("Tool synchronization failed. No tool file was copied.")
    if metadata_path is None:
        raise RuntimeError("Tool synchronization failed. Metadata was not written.")
    return ToolSpaceSyncResult(
        requirements_path=requirements_path,
        downloaded_tool_path=copied_tool_path,
        downloaded_tool_source=source_tool_file,
        dependency_group=dependency_group,
        resolved_revision=resolved_revision,
        synced_at_utc=synced_at_utc,
        requirements_sha256=requirements_sha256,
        metadata_path=metadata_path,
    )
