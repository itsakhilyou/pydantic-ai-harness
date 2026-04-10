"""Agent harness for composable, reusable AI agent capabilities, for Pydantic AI."""

try:
    from .code_mode import CodeMode
except ImportError:  # pragma: no cover — pydantic-monty not installed
    pass
else:
    __all__ = ['CodeMode']
