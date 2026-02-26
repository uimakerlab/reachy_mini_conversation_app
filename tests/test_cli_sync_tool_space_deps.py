import sys
from typing import Generator

import pytest

from reachy_mini_conversation_app.utils import parse_args


@pytest.fixture
def restore_argv() -> Generator[None, None, None]:
    """Restore ``sys.argv`` after each test."""
    original = list(sys.argv)
    try:
        yield
    finally:
        sys.argv = original


def test_parse_args_sync_tool_space_deps_flags(restore_argv: None) -> None:
    """CLI parser exposes dependency sync flags."""
    del restore_argv
    sys.argv = [
        "reachy-mini-conversation-app",
        "--sync-tool-space-deps",
        "owner/repo",
        "--sync-tool-space-hf-token",
        "hf_abc",
    ]

    args, unknown = parse_args()
    assert unknown == []
    assert args.sync_tool_space_deps == "owner/repo"
    assert args.sync_tool_space_hf_token == "hf_abc"
