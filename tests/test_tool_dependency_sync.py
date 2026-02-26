from __future__ import annotations
import json
import hashlib
from typing import Any
from pathlib import Path

import httpx
import pytest
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

import reachy_mini_conversation_app.tool_dependency_sync as sync_mod


TEST_RESOLVED_REVISION = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def stub_resolved_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid network calls for revision lookup across sync tests."""
    monkeypatch.setattr(sync_mod, "_resolve_space_revision", lambda *_: TEST_RESOLVED_REVISION)


VALID_TOOL_MODULE = """
from reachy_mini_conversation_app.tools.core_tools import Tool


class SearchTool(Tool):
    name = "search_tool"
    description = "Search things"
    parameters_schema = {"type": "object", "properties": {}}

    async def __call__(self, deps, **kwargs):
        return {"ok": True}
"""

MULTI_TOOL_MODULE = """
from reachy_mini_conversation_app.tools.core_tools import Tool


class SearchToolA(Tool):
    name = "search_tool_a"
    description = "Search things A"
    parameters_schema = {"type": "object", "properties": {}}

    async def __call__(self, deps, **kwargs):
        return {"ok": "a"}


class SearchToolB(Tool):
    name = "search_tool_b"
    description = "Search things B"
    parameters_schema = {"type": "object", "properties": {}}

    async def __call__(self, deps, **kwargs):
        return {"ok": "b"}
"""

MISSING_FIELDS_TOOL_MODULE = """
from reachy_mini_conversation_app.tools.core_tools import Tool


class SearchTool(Tool):
    name = "search_tool"
    parameters_schema = {"type": "object", "properties": {}}

    async def __call__(self, deps, **kwargs):
        return {"ok": True}
"""

INVALID_NAME_TOOL_MODULE = """
from reachy_mini_conversation_app.tools.core_tools import Tool


class SearchTool(Tool):
    name = "search tool"
    description = "Search things"
    parameters_schema = {"type": "object", "properties": {}}

    async def __call__(self, deps, **kwargs):
        return {"ok": True}
"""

EMPTY_NAME_TOOL_MODULE = """
from reachy_mini_conversation_app.tools.core_tools import Tool


class SearchTool(Tool):
    name = ""
    description = "Search things"
    parameters_schema = {"type": "object", "properties": {}}

    async def __call__(self, deps, **kwargs):
        return {"ok": True}
"""

MISSING_RUNTIME_DEP_TOOL_MODULE = """
from reachy_mini_conversation_app.tools.core_tools import Tool
import definitely_missing_runtime_dependency_abc123


class SearchTool(Tool):
    name = "search_tool"
    description = "Search things"
    parameters_schema = {"type": "object", "properties": {}}

    async def __call__(self, deps, **kwargs):
        return {"ok": True}
"""


def test_normalize_space_id_variants() -> None:
    """Space IDs and URLs normalize to owner/repo form."""
    assert sync_mod.normalize_space_id("owner/repo") == "owner/repo"
    assert sync_mod.normalize_space_id("spaces/owner/repo") == "owner/repo"
    assert (
        sync_mod.normalize_space_id("https://huggingface.co/spaces/owner/repo")
        == "owner/repo"
    )


def test_sync_tool_space_dependencies_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Successful sync calls uv add and copies one tool module locally."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='x'\nversion='0.0.1'\ndependencies=[]\n",
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text("lock-before\n", encoding="utf-8")

    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("duckduckgo-search>=7.0.0\n", encoding="utf-8")
    source_tool = tmp_path / "search_tool.py"
    source_tool.write_text(VALID_TOOL_MODULE, encoding="utf-8")

    monkeypatch.setattr(sync_mod, "_select_tool_python_file", lambda *_: "search_tool.py")

    def fake_download(**kwargs: Any) -> str:
        filename = kwargs["filename"]
        assert kwargs["revision"] == TEST_RESOLVED_REVISION
        if filename == "requirements.txt":
            return str(requirements_path)
        if filename == "search_tool.py":
            return str(source_tool)
        raise AssertionError(f"Unexpected download filename: {filename}")

    monkeypatch.setattr(sync_mod, "hf_hub_download", fake_download)

    def fake_run_command(command: list[str], logger: Any) -> tuple[int, str, str]:
        del logger
        assert command[:4] == ["uv", "add", "--group", sync_mod.EXTERNAL_TOOLS_GROUP]
        # Simulate uv having updated files
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.1'\ndependencies=[]\n[dependency-groups]\nexternal-tools=[]\n",
            encoding="utf-8",
        )
        (tmp_path / "uv.lock").write_text("lock-after\n", encoding="utf-8")
        return 0, "ok", ""

    monkeypatch.setattr(sync_mod, "_run_command", fake_run_command)

    out = sync_mod.sync_tool_space_dependencies(
        space="owner/repo",
        logger=sync_mod.logging.getLogger("test"),
    )
    assert out.requirements_path == requirements_path
    assert out.downloaded_tool_source == "search_tool.py"
    assert out.downloaded_tool_path == tmp_path / "external_content" / "external_tools" / "search_tool.py"
    assert out.downloaded_tool_path.read_text(encoding="utf-8") == VALID_TOOL_MODULE
    assert out.resolved_revision == TEST_RESOLVED_REVISION
    assert out.requirements_sha256 == hashlib.sha256(requirements_path.read_bytes()).hexdigest()
    assert out.synced_at_utc.endswith("Z")
    assert out.metadata_path == out.downloaded_tool_path.with_name("search_tool.py.metadata.json")
    metadata = json.loads(out.metadata_path.read_text(encoding="utf-8"))
    assert metadata["space_id"] == "owner/repo"
    assert metadata["revision"] == TEST_RESOLVED_REVISION
    assert metadata["requirements_sha256"] == out.requirements_sha256
    assert metadata["source_tool_file"] == "search_tool.py"
    assert metadata["synced_at_utc"].endswith("Z")
    assert "external-tools" in (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert (tmp_path / "uv.lock").read_text(encoding="utf-8") == "lock-after\n"


def test_sync_tool_space_dependencies_rolls_back_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On uv failure, pyproject and uv.lock are restored."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='x'\nversion='0.0.1'\ndependencies=[]\n",
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text("lock-before\n", encoding="utf-8")

    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("badpkg==9999\n", encoding="utf-8")

    monkeypatch.setattr(sync_mod, "hf_hub_download", lambda **_: str(requirements_path))

    def fake_run_command(command: list[str], logger: Any) -> tuple[int, str, str]:
        del logger
        # Simulate partial writes before command failure
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.1'\ndependencies=['badpkg']\n",
            encoding="utf-8",
        )
        (tmp_path / "uv.lock").write_text("lock-mutated\n", encoding="utf-8")
        return 2, "", "resolver conflict"

    monkeypatch.setattr(sync_mod, "_run_command", fake_run_command)

    with pytest.raises(RuntimeError, match="rolled back"):
        sync_mod.sync_tool_space_dependencies(
            space="owner/repo",
            logger=sync_mod.logging.getLogger("test"),
        )

    assert (tmp_path / "pyproject.toml").read_text(encoding="utf-8") == (
        "[project]\nname='x'\nversion='0.0.1'\ndependencies=[]\n"
    )
    assert (tmp_path / "uv.lock").read_text(encoding="utf-8") == "lock-before\n"


@pytest.mark.parametrize(
    "unsafe_line",
    [
        "-r other-requirements.txt",
        "--index-url https://pypi.org/simple",
        "mypkg @ https://example.com/pkg.whl",
        "git+https://github.com/org/repo.git",
        "../local_package",
        "file:///tmp/local.whl",
    ],
)
def test_sync_tool_space_dependencies_rejects_unsafe_requirements_entries(
    unsafe_line: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsafe requirements entries should be rejected before running uv add."""
    monkeypatch.chdir(tmp_path)
    pyproject_initial = "[project]\nname='x'\nversion='0.0.1'\ndependencies=[]\n"
    (tmp_path / "pyproject.toml").write_text(pyproject_initial, encoding="utf-8")
    (tmp_path / "uv.lock").write_text("lock-before\n", encoding="utf-8")
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text(f"{unsafe_line}\n", encoding="utf-8")

    def fake_download(**kwargs: Any) -> str:
        if kwargs["filename"] == "requirements.txt":
            return str(requirements_path)
        raise AssertionError("Tool module download should not be reached for unsafe requirements")

    monkeypatch.setattr(sync_mod, "hf_hub_download", fake_download)

    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("uv add should not run for unsafe requirements")

    monkeypatch.setattr(sync_mod, "_run_command", fail_if_called)

    with pytest.raises(RuntimeError, match="Unsafe requirements.txt entry"):
        sync_mod.sync_tool_space_dependencies(
            space="owner/repo",
            logger=sync_mod.logging.getLogger("test"),
        )

    assert (tmp_path / "pyproject.toml").read_text(encoding="utf-8") == pyproject_initial
    assert (tmp_path / "uv.lock").read_text(encoding="utf-8") == "lock-before\n"


def test_sync_tool_space_dependencies_allows_empty_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty requirements.txt should be treated as no external dependencies."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='x'\nversion='0.0.1'\ndependencies=[]\n",
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text("lock-before\n", encoding="utf-8")
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("# no deps\n\n", encoding="utf-8")
    source_tool = tmp_path / "search_tool.py"
    source_tool.write_text(VALID_TOOL_MODULE, encoding="utf-8")

    monkeypatch.setattr(sync_mod, "_select_tool_python_file", lambda *_: "search_tool.py")

    def fake_download(**kwargs: Any) -> str:
        filename = kwargs["filename"]
        if filename == "requirements.txt":
            return str(requirements_path)
        if filename == "search_tool.py":
            return str(source_tool)
        raise AssertionError(f"Unexpected download filename: {filename}")

    monkeypatch.setattr(sync_mod, "hf_hub_download", fake_download)

    def fail_if_called(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("uv add should not run for empty requirements")

    monkeypatch.setattr(sync_mod, "_run_command", fail_if_called)

    result = sync_mod.sync_tool_space_dependencies(
        space="owner/repo",
        logger=sync_mod.logging.getLogger("test"),
    )
    assert result.requirements_path is None
    assert result.downloaded_tool_path.exists()
    assert result.requirements_sha256 is None
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["requirements_sha256"] is None
    assert (tmp_path / "uv.lock").read_text(encoding="utf-8") == "lock-before\n"


def test_sync_tool_space_dependencies_allows_missing_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing requirements.txt should not fail if tool module exists."""
    monkeypatch.chdir(tmp_path)
    source_tool = tmp_path / "search_tool.py"
    source_tool.write_text(VALID_TOOL_MODULE, encoding="utf-8")

    monkeypatch.setattr(sync_mod, "_select_tool_python_file", lambda *_: "search_tool.py")

    def fake_download(**kwargs: Any) -> str:
        if kwargs["filename"] == "requirements.txt":
            raise EntryNotFoundError("requirements missing")
        if kwargs["filename"] == "search_tool.py":
            return str(source_tool)
        raise AssertionError(f"Unexpected download filename: {kwargs['filename']}")

    monkeypatch.setattr(sync_mod, "hf_hub_download", fake_download)
    result = sync_mod.sync_tool_space_dependencies(
        space="owner/repo",
        logger=sync_mod.logging.getLogger("test"),
    )
    assert result.requirements_path is None
    assert result.downloaded_tool_path.exists()
    assert result.requirements_sha256 is None
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["requirements_sha256"] is None


def test_sync_tool_space_dependencies_rolls_back_on_tool_download_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If tool download fails after uv add, dependency files are rolled back."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='x'\nversion='0.0.1'\ndependencies=[]\n",
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text("lock-before\n", encoding="utf-8")
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("ddgs\n", encoding="utf-8")

    monkeypatch.setattr(sync_mod, "hf_hub_download", lambda **_: str(requirements_path))
    monkeypatch.setattr(sync_mod, "_select_tool_python_file", lambda *_: (_ for _ in ()).throw(ValueError("ambiguous")))

    def fake_run_command(command: list[str], logger: Any) -> tuple[int, str, str]:
        del command, logger
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.1'\ndependencies=['ddgs']\n",
            encoding="utf-8",
        )
        (tmp_path / "uv.lock").write_text("lock-mutated\n", encoding="utf-8")
        return 0, "", ""

    monkeypatch.setattr(sync_mod, "_run_command", fake_run_command)

    with pytest.raises(RuntimeError, match="rolled back"):
        sync_mod.sync_tool_space_dependencies(
            space="owner/repo",
            logger=sync_mod.logging.getLogger("test"),
        )

    assert (tmp_path / "pyproject.toml").read_text(encoding="utf-8") == (
        "[project]\nname='x'\nversion='0.0.1'\ndependencies=[]\n"
    )
    assert (tmp_path / "uv.lock").read_text(encoding="utf-8") == "lock-before\n"


def test_sync_tool_space_dependencies_fails_when_no_tool_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing tool python module should raise a clear error."""
    monkeypatch.chdir(tmp_path)

    def fake_download(**kwargs: Any) -> str:
        if kwargs["filename"] == "requirements.txt":
            raise EntryNotFoundError("requirements missing")
        raise AssertionError(f"Unexpected download filename: {kwargs['filename']}")

    monkeypatch.setattr(sync_mod, "hf_hub_download", fake_download)
    monkeypatch.setattr(
        sync_mod,
        "_select_tool_python_file",
        lambda *_: (_ for _ in ()).throw(ValueError("No candidate tool .py file found")),
    )

    with pytest.raises(RuntimeError, match="No candidate tool \\.py file found"):
        sync_mod.sync_tool_space_dependencies(
            space="owner/repo",
            logger=sync_mod.logging.getLogger("test"),
        )


def test_sync_tool_space_dependencies_fails_when_tool_has_invalid_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tool module must follow Tool class contract to pass sync validation."""
    monkeypatch.chdir(tmp_path)

    source_tool = tmp_path / "search_tool.py"
    source_tool.write_text("print('hello world')\n", encoding="utf-8")
    monkeypatch.setattr(sync_mod, "_select_tool_python_file", lambda *_: "search_tool.py")

    def fake_download(**kwargs: Any) -> str:
        if kwargs["filename"] == "requirements.txt":
            raise EntryNotFoundError("requirements missing")
        if kwargs["filename"] == "search_tool.py":
            return str(source_tool)
        raise AssertionError(f"Unexpected download filename: {kwargs['filename']}")

    monkeypatch.setattr(sync_mod, "hf_hub_download", fake_download)

    with pytest.raises(RuntimeError, match="No Tool subclass found in downloaded module"):
        sync_mod.sync_tool_space_dependencies(
            space="owner/repo",
            logger=sync_mod.logging.getLogger("test"),
        )

    assert not (tmp_path / "external_content" / "external_tools" / "search_tool.py").exists()


def test_sync_tool_space_dependencies_does_not_execute_module_runtime_imports_during_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sync should not execute downloaded module imports during validation."""
    monkeypatch.chdir(tmp_path)

    source_tool = tmp_path / "search_tool.py"
    source_tool.write_text(MISSING_RUNTIME_DEP_TOOL_MODULE, encoding="utf-8")
    monkeypatch.setattr(sync_mod, "_select_tool_python_file", lambda *_: "search_tool.py")

    def fake_download(**kwargs: Any) -> str:
        if kwargs["filename"] == "requirements.txt":
            raise EntryNotFoundError("requirements missing")
        if kwargs["filename"] == "search_tool.py":
            return str(source_tool)
        raise AssertionError(f"Unexpected download filename: {kwargs['filename']}")

    monkeypatch.setattr(sync_mod, "hf_hub_download", fake_download)

    result = sync_mod.sync_tool_space_dependencies(
        space="owner/repo",
        logger=sync_mod.logging.getLogger("test"),
    )
    assert result.downloaded_tool_path.exists()


def test_sync_tool_space_dependencies_fails_when_tool_class_missing_required_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tool class must define required class fields."""
    monkeypatch.chdir(tmp_path)

    source_tool = tmp_path / "search_tool.py"
    source_tool.write_text(MISSING_FIELDS_TOOL_MODULE, encoding="utf-8")
    monkeypatch.setattr(sync_mod, "_select_tool_python_file", lambda *_: "search_tool.py")

    def fake_download(**kwargs: Any) -> str:
        if kwargs["filename"] == "requirements.txt":
            raise EntryNotFoundError("requirements missing")
        if kwargs["filename"] == "search_tool.py":
            return str(source_tool)
        raise AssertionError(f"Unexpected download filename: {kwargs['filename']}")

    monkeypatch.setattr(sync_mod, "hf_hub_download", fake_download)

    with pytest.raises(RuntimeError, match="missing required class field\\(s\\)"):
        sync_mod.sync_tool_space_dependencies(
            space="owner/repo",
            logger=sync_mod.logging.getLogger("test"),
        )


def test_sync_tool_space_dependencies_fails_when_multiple_concrete_tools_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Downloaded module must contain exactly one concrete Tool subclass."""
    monkeypatch.chdir(tmp_path)

    source_tool = tmp_path / "search_tool.py"
    source_tool.write_text(MULTI_TOOL_MODULE, encoding="utf-8")
    monkeypatch.setattr(sync_mod, "_select_tool_python_file", lambda *_: "search_tool.py")

    def fake_download(**kwargs: Any) -> str:
        if kwargs["filename"] == "requirements.txt":
            raise EntryNotFoundError("requirements missing")
        if kwargs["filename"] == "search_tool.py":
            return str(source_tool)
        raise AssertionError(f"Unexpected download filename: {kwargs['filename']}")

    monkeypatch.setattr(sync_mod, "hf_hub_download", fake_download)

    with pytest.raises(RuntimeError, match="Expected exactly one concrete Tool subclass"):
        sync_mod.sync_tool_space_dependencies(
            space="owner/repo",
            logger=sync_mod.logging.getLogger("test"),
        )


@pytest.mark.parametrize("tool_module", [INVALID_NAME_TOOL_MODULE, EMPTY_NAME_TOOL_MODULE])
def test_sync_tool_space_dependencies_fails_when_tool_name_invalid_or_empty(
    tool_module: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tool.name must be non-empty and match the expected identifier pattern."""
    monkeypatch.chdir(tmp_path)

    source_tool = tmp_path / "search_tool.py"
    source_tool.write_text(tool_module, encoding="utf-8")
    monkeypatch.setattr(sync_mod, "_select_tool_python_file", lambda *_: "search_tool.py")

    def fake_download(**kwargs: Any) -> str:
        if kwargs["filename"] == "requirements.txt":
            raise EntryNotFoundError("requirements missing")
        if kwargs["filename"] == "search_tool.py":
            return str(source_tool)
        raise AssertionError(f"Unexpected download filename: {kwargs['filename']}")

    monkeypatch.setattr(sync_mod, "hf_hub_download", fake_download)

    with pytest.raises(RuntimeError, match="Tool.name"):
        sync_mod.sync_tool_space_dependencies(
            space="owner/repo",
            logger=sync_mod.logging.getLogger("test"),
        )


def test_sync_tool_space_dependencies_keeps_existing_tool_when_new_download_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing tool file should not be replaced by an invalid downloaded update."""
    monkeypatch.chdir(tmp_path)

    existing_tool = tmp_path / "external_content" / "external_tools" / "search_tool.py"
    existing_tool.parent.mkdir(parents=True, exist_ok=True)
    existing_tool_content = VALID_TOOL_MODULE + "\n# existing-tool-version\n"
    existing_tool.write_text(existing_tool_content, encoding="utf-8")

    invalid_downloaded_tool = tmp_path / "downloaded_invalid_search_tool.py"
    invalid_downloaded_tool.write_text("print('hello world')\n", encoding="utf-8")

    monkeypatch.setattr(sync_mod, "_select_tool_python_file", lambda *_: "search_tool.py")

    def fake_download(**kwargs: Any) -> str:
        if kwargs["filename"] == "requirements.txt":
            raise EntryNotFoundError("requirements missing")
        if kwargs["filename"] == "search_tool.py":
            return str(invalid_downloaded_tool)
        raise AssertionError(f"Unexpected download filename: {kwargs['filename']}")

    monkeypatch.setattr(sync_mod, "hf_hub_download", fake_download)

    with pytest.raises(RuntimeError, match="No Tool subclass found in downloaded module"):
        sync_mod.sync_tool_space_dependencies(
            space="owner/repo",
            logger=sync_mod.logging.getLogger("test"),
        )

    assert existing_tool.read_text(encoding="utf-8") == existing_tool_content


def test_sync_tool_space_dependencies_fails_when_space_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing/private space should raise a clear access/not-found error."""
    monkeypatch.chdir(tmp_path)

    request = httpx.Request("GET", "https://huggingface.co/spaces/owner/repo")
    response = httpx.Response(404, request=request)
    repo_not_found_error = RepositoryNotFoundError("Repository Not Found", response=response)

    def fail_requirements(*, space_id: str, token: str | None, logger: Any, revision: str | None = None) -> None:
        del space_id, token, logger, revision
        raise repo_not_found_error

    monkeypatch.setattr(sync_mod, "_try_download_requirements", fail_requirements)

    with pytest.raises(RuntimeError, match="not found or access denied"):
        sync_mod.sync_tool_space_dependencies(
            space="owner/repo",
            logger=sync_mod.logging.getLogger("test"),
        )
