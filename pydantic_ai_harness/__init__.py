"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .execution_env import ExecutionEnv

__all__ = ['CodeMode', 'ExecutionEnv']


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    if name == 'ExecutionEnv':
        from .execution_env import ExecutionEnv

        return ExecutionEnv
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
