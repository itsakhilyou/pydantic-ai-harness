"""Freshness/staleness signals for tracked files (private, not re-exported at top level)."""

from pydantic_ai_harness.experimental._warn import warn_experimental
from pydantic_ai_harness.experimental.staleness._capability import (
    DEFAULT_TRACK,
    PathExtractor,
    StalenessTracker,
    TrackValue,
)

warn_experimental('staleness')

__all__ = [
    'DEFAULT_TRACK',
    'PathExtractor',
    'StalenessTracker',
    'TrackValue',
]
