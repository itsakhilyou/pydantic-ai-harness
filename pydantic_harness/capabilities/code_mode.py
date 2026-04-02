"""Code mode capability that routes all tool execution through a Monty sandbox."""

from dataclasses import dataclass

from pydantic_ai import AbstractToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_harness.toolsets import CodeExecutionToolset


@dataclass
class CodeMode(AbstractCapability[AgentDepsT]):
    """Capability that wraps an agent's tools behind a single `run_code` tool.

    When applied, the LLM writes Python code to call tools instead of invoking
    them directly. The code runs in a Monty sandbox with the original tools
    available as callable functions.
    """

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Wrap the agent's assembled toolset in a `CodeExecutionToolset`."""
        return CodeExecutionToolset(wrapped=toolset)
