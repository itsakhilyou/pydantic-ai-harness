"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .sql_mode import SQLModeBuilder, SQLModeToolset

__all__ = ['CodeMode', 'SQLModeBuilder', 'SQLModeToolset']


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    if name in ('SQLModeBuilder', 'SQLModeToolset'):
        from . import sql_mode

        return getattr(sql_mode, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
