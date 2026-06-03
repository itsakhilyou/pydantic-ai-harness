"""Pydantic AI capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .compaction import (
        ClearToolResults,
        DeduplicateFileReads,
        LimitWarner,
        SlidingWindow,
        SummarizingCompaction,
        TieredCompaction,
    )
    from .filesystem import FileSystem
    from .logfire import ManagedPrompt
    from .shell import Shell

__all__ = [
    'ClearToolResults',
    'CodeMode',
    'DeduplicateFileReads',
    'FileSystem',
    'LimitWarner',
    'ManagedPrompt',
    'Shell',
    'SlidingWindow',
    'SummarizingCompaction',
    'TieredCompaction',
]


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    if name == 'FileSystem':
        from .filesystem import FileSystem

        return FileSystem
    if name == 'ManagedPrompt':
        from .logfire import ManagedPrompt

        return ManagedPrompt
    if name == 'Shell':
        from .shell import Shell

        return Shell
    if name in {
        'ClearToolResults',
        'DeduplicateFileReads',
        'LimitWarner',
        'SlidingWindow',
        'SummarizingCompaction',
        'TieredCompaction',
    }:
        from . import compaction

        return getattr(compaction, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
