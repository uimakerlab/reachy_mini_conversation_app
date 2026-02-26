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


def test_parse_args_download_hf_tool_flag(restore_argv: None) -> None:
    """CLI parser exposes external HF tool download flag."""
    del restore_argv
    sys.argv = [
        "reachy-mini-conversation-app",
        "--download-hf-tool",
        "owner/repo",
    ]

    args, unknown = parse_args()
    assert unknown == []
    assert args.download_hf_tool == "owner/repo"
    assert not hasattr(args, "sync_tool_space_hf_token")
