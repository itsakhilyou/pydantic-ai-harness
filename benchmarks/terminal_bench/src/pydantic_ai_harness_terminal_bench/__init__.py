"""Terminal-Bench reference agent for Pydantic AI on Harbor.

A minimal Pydantic AI agent whose weight lives in harness capabilities, wrapped
in a Harbor `BaseAgent` adapter so it can be evaluated on Terminal-Bench 2.x.

See the README for how to run it. The Harbor adapter
(`PydanticAITerminalBenchAgent`) imports `harbor`, so it is exported lazily:
importing this package does not require Harbor unless you touch the adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai_harness_terminal_bench.agent import build_agent, build_compaction
from pydantic_ai_harness_terminal_bench.config import (
    build_usage_limits,
    convert_model_name,
)
from pydantic_ai_harness_terminal_bench.tools import (
    CommandExecutor,
    CommandResult,
    TerminalBenchDeps,
    bash,
    build_bash_toolset,
)

if TYPE_CHECKING:
    from pydantic_ai_harness_terminal_bench.harbor_agent import PydanticAITerminalBenchAgent

__all__ = [
    'CommandExecutor',
    'CommandResult',
    'PydanticAITerminalBenchAgent',
    'TerminalBenchDeps',
    'bash',
    'build_agent',
    'build_bash_toolset',
    'build_compaction',
    'build_usage_limits',
    'convert_model_name',
]


def __getattr__(name: str) -> object:
    if name == 'PydanticAITerminalBenchAgent':
        from pydantic_ai_harness_terminal_bench.harbor_agent import PydanticAITerminalBenchAgent

        return PydanticAITerminalBenchAgent
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
