"""Transcript analysis for real-time reaction to user speech.

This module provides tools for analyzing transcripts (both partial and final)
and triggering demo-configured reactions based on keywords and named entities.

Example usage in a demo:

TODO : UPDATE THIS

"""

from .base import Reaction, TranscriptAnalyzer
from .loader import get_profile_reactions
from .manager import NoOpTranscriptManager, TranscriptAnalysisManager
from .keyword_analyzer import KeywordAnalyzer


# EntityAnalyzer is optional (requires gliner extra)
try:
    from .entity_analyzer import EntityAnalyzer

    __all__ = [
        "Reaction",
        "TranscriptAnalyzer",
        "KeywordAnalyzer",
        "EntityAnalyzer",
        "TranscriptAnalysisManager",
        "NoOpTranscriptManager",
        "get_profile_reactions",
    ]
except ImportError:
    __all__ = [
        "Reaction",
        "TranscriptAnalyzer",
        "KeywordAnalyzer",
        "TranscriptAnalysisManager",
        "NoOpTranscriptManager",
        "get_profile_reactions",
    ]
