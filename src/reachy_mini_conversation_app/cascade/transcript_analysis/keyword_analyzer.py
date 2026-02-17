"""Keyword-based transcript analyzer."""

from __future__ import annotations
import logging

from .base import TranscriptAnalyzer


logger = logging.getLogger(__name__)


class KeywordAnalyzer(TranscriptAnalyzer):
    """Pure keyword matcher. Returns reaction_name → matched words.

    No callbacks, no deduplication — that's the manager's job.
    """

    def __init__(self, reaction_words: dict[str, list[str]]):
        """Initialize keyword analyzer.

        Args:
            reaction_words: Dict mapping reaction_name to list of trigger words

        """
        # Pre-lowercase for fast matching
        self.reaction_words: dict[str, list[str]] = {
            name: [w.lower() for w in words] for name, words in reaction_words.items()
        }
        total = sum(len(ws) for ws in self.reaction_words.values())
        logger.info(f"KeywordAnalyzer initialized: {total} words across {len(self.reaction_words)} reactions")

    async def analyze(self, text: str, is_final: bool) -> dict[str, list[str]]:
        """Return {reaction_name: [matched_words]} for words found in text."""
        text_lower = text.lower()
        matches: dict[str, list[str]] = {}

        for reaction_name, words in self.reaction_words.items():
            matched = [w for w in words if w in text_lower]
            if matched:
                matches[reaction_name] = matched

        return matches
