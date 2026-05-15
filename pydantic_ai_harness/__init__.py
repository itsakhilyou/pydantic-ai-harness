"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .code_mode_dynamic_catalog import CodeModeDynamicCatalog

__all__ = ['CodeMode', 'CodeModeDynamicCatalog']


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    if name == 'CodeModeDynamicCatalog':
        from .code_mode_dynamic_catalog import CodeModeDynamicCatalog

        return CodeModeDynamicCatalog
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
