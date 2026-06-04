"""Pydantic AI capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .filesystem import FileSystem
    from .logfire import ManagedPrompt
    from .planning import Planning
    from .shell import Shell
    from .subagents import SubAgents

__all__ = ['CodeMode', 'FileSystem', 'ManagedPrompt', 'Planning', 'Shell', 'SubAgents']


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
    if name == 'Planning':
        from .planning import Planning

        return Planning
    if name == 'Shell':
        from .shell import Shell

        return Shell
    if name == 'SubAgents':
        from .subagents import SubAgents

        return SubAgents
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
