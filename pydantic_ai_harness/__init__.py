"""Pydantic AI capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .filesystem import FileSystem
    from .logfire import ManagedPrompt
    from .shell import Shell
    from .web import SummarizingFetch, WebResearch

__all__ = ['CodeMode', 'FileSystem', 'ManagedPrompt', 'Shell', 'SummarizingFetch', 'WebResearch']


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
    if name == 'SummarizingFetch':
        from .web import SummarizingFetch

        return SummarizingFetch
    if name == 'WebResearch':
        from .web import WebResearch

        return WebResearch
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
