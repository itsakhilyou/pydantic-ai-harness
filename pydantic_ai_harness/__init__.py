"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .sql_mode import SQLMode, SQLModeToolset

__all__ = ['CodeMode', 'SQLMode', 'SQLModeToolset']


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    if name in ('SQLMode', 'SQLModeToolset'):
        from . import sql_mode

        return getattr(sql_mode, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
