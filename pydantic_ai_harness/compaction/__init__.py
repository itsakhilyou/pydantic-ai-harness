"""Compaction capabilities: keep an agent's conversation history within the context window."""

from pydantic_ai_harness.compaction._capability import (
    ClearToolResults,
    CompactionStrategy,
    DeduplicateFileReads,
    LimitWarner,
    SlidingWindow,
    SummarizingCompaction,
    TieredCompaction,
    WarningKind,
    estimate_token_count,
)

__all__ = [
    'ClearToolResults',
    'CompactionStrategy',
    'DeduplicateFileReads',
    'LimitWarner',
    'SlidingWindow',
    'SummarizingCompaction',
    'TieredCompaction',
    'WarningKind',
    'estimate_token_count',
]
