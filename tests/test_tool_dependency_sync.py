from __future__ import annotations
import subprocess
from typing import Any
from pathlib import Path

import pytest

import reachy_mini_conversation_app.tool_dependency_sync as sync_mod


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
    source_tool.write_text("# external tool\n", encoding="utf-8")

    monkeypatch.setattr(sync_mod, "_select_tool_python_file", lambda *_: "search_tool.py")

    def fake_download(**kwargs: Any) -> str:
        filename = kwargs["filename"]
        if filename == "requirements.txt":
            return str(requirements_path)
        if filename == "search_tool.py":
            return str(source_tool)
        raise AssertionError(f"Unexpected download filename: {filename}")

    monkeypatch.setattr(sync_mod, "hf_hub_download", fake_download)

    def fake_run(command: list[str], text: bool, capture_output: bool, check: bool) -> Any:
        del text, capture_output, check
        assert command[:4] == ["uv", "add", "--group", sync_mod.EXTERNAL_TOOLS_GROUP]
        # Simulate uv having updated files
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.1'\ndependencies=[]\n[dependency-groups]\nexternal-tools=[]\n",
            encoding="utf-8",
        )
        (tmp_path / "uv.lock").write_text("lock-after\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(sync_mod.subprocess, "run", fake_run)

    out = sync_mod.sync_tool_space_dependencies(
        space="owner/repo",
        logger=sync_mod.logging.getLogger("test"),
    )
    assert out.requirements_path == requirements_path
    assert out.downloaded_tool_source == "search_tool.py"
    assert out.downloaded_tool_path == tmp_path / "external_content" / "external_tools" / "search_tool.py"
    assert out.downloaded_tool_path.read_text(encoding="utf-8") == "# external tool\n"
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

    def fake_run(command: list[str], text: bool, capture_output: bool, check: bool) -> Any:
        del text, capture_output, check
        # Simulate partial writes before command failure
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.1'\ndependencies=['badpkg']\n",
            encoding="utf-8",
        )
        (tmp_path / "uv.lock").write_text("lock-mutated\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=command, returncode=2, stdout="", stderr="resolver conflict")

    monkeypatch.setattr(sync_mod.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="rolled back"):
        sync_mod.sync_tool_space_dependencies(
            space="owner/repo",
            logger=sync_mod.logging.getLogger("test"),
        )

    assert (tmp_path / "pyproject.toml").read_text(encoding="utf-8") == (
        "[project]\nname='x'\nversion='0.0.1'\ndependencies=[]\n"
    )
    assert (tmp_path / "uv.lock").read_text(encoding="utf-8") == "lock-before\n"


def test_sync_tool_space_dependencies_rejects_empty_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty requirements.txt should fail fast."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\nversion='0.0.1'\n", encoding="utf-8")
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("# no deps\n\n", encoding="utf-8")
    monkeypatch.setattr(sync_mod, "hf_hub_download", lambda **_: str(requirements_path))

    with pytest.raises(ValueError, match="empty"):
        sync_mod.sync_tool_space_dependencies(
            space="owner/repo",
            logger=sync_mod.logging.getLogger("test"),
        )


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

    def fake_run(command: list[str], text: bool, capture_output: bool, check: bool) -> Any:
        del command, text, capture_output, check
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname='x'\nversion='0.0.1'\ndependencies=['ddgs']\n",
            encoding="utf-8",
        )
        (tmp_path / "uv.lock").write_text("lock-mutated\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=["uv"], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(sync_mod.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="rolled back"):
        sync_mod.sync_tool_space_dependencies(
            space="owner/repo",
            logger=sync_mod.logging.getLogger("test"),
        )

    assert (tmp_path / "pyproject.toml").read_text(encoding="utf-8") == (
        "[project]\nname='x'\nversion='0.0.1'\ndependencies=[]\n"
    )
    assert (tmp_path / "uv.lock").read_text(encoding="utf-8") == "lock-before\n"
