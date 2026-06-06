"""Back many tools' framing with a single Logfire-managed variable holding a map of overrides."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from logfire.variables import Variable
from pydantic_ai import AbstractToolset, Tool
from pydantic_ai.tools import AgentDepsT, ToolFuncEither
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from pydantic_ai_harness.logfire._managed_tool import ManagedToolOverride, ManagedToolsToolset
from pydantic_ai_harness.logfire._managed_variable import ManagedVariableCapability

# `toolset__` namespaces the backing variable, mirroring `ManagedTool`'s `tool__` and
# `ManagedPrompt`'s `prompt__`. One `toolset__<name>` variable holds a map of per-tool overrides.
_TOOLSET_VARIABLE_PREFIX = 'toolset__'

_ToolOverrides = dict[str, ManagedToolOverride]


@dataclass
class ManagedToolset(ManagedVariableCapability[AgentDepsT, _ToolOverrides]):
    """Back many tools' framing with a single Logfire-managed variable holding a map of overrides.

    Where [`ManagedTool`][pydantic_ai_harness.logfire.ManagedTool] manages one tool per variable,
    `ManagedToolset` manages a whole group of tools from one
    [managed variable](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/) -- a
    `dict` mapping tool name to [`ManagedToolOverride`][pydantic_ai_harness.logfire.ManagedToolOverride].
    A group named `support` resolves the variable `toolset__support`. This is handy for reframing a
    whole MCP server or a large toolset from one place in the Logfire UI -- versioned, labelled, and
    rolled out -- without redeploying.

    Pass `tools` to have this capability both provide and manage those tools; omit it to manage tools
    registered elsewhere on the agent. Either way, the resolved map's keys select which tools are
    overridden by their original name (keys that match no tool are ignored).

    ```python
    import logfire
    from pydantic_ai import Agent

    from pydantic_ai_harness.logfire import ManagedToolset

    logfire.configure()

    def get_weather(city: str) -> str:
        return f'The weather in {city} is sunny.'

    def get_forecast(city: str) -> str:
        return f'The forecast for {city} is sunny.'

    agent = Agent(
        'openai:gpt-5',
        capabilities=[ManagedToolset('weather', tools=[get_weather, get_forecast], label='production')],
    )
    ```

    Each entry follows the same rules as `ManagedTool`: the advertised name, description, and
    parameter descriptions can be overridden, but the parameter schema's structure stays fixed in
    code. The map resolves **once per run** inside
    [`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run], with the
    [`ResolvedVariable`][logfire.variables.ResolvedVariable] kept open so the selected label and
    version propagate as baggage to every child span.

    Pass an existing [`logfire.variables.Variable`][logfire.variables.Variable] as `name` to use a
    variable you declared yourself.
    """

    name: str | Variable[_ToolOverrides]
    """The managed variable name (declared as `toolset__<name>`), or a pre-built `logfire.Variable`."""

    tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] | None = None
    """Tools to provide and manage. When `None`, only tools registered elsewhere are managed."""

    default: _ToolOverrides | None = None
    """Code-default override map. When `None`, an empty map (no changes) is used. Ignored when
    `name` is a `Variable`."""

    def __post_init__(self) -> None:
        self._resolved = self._new_resolved()
        if isinstance(self.name, Variable):
            self._warn_logfire_instance_ignored('name')
            self._variable = self.name
            return

        default: _ToolOverrides = self.default if self.default is not None else {}
        self._variable = self._build_managed_variable(
            self.name, prefix=_TOOLSET_VARIABLE_PREFIX, value_type=_ToolOverrides, default=default
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Provide the owned tools, when this capability was constructed with `tools`."""
        if not self.tools:
            return None
        return FunctionToolset(list(self.tools))

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Wrap the assembled toolset so the managed overrides are applied to the matching tools."""
        return ManagedToolsToolset(wrapped=toolset, get_overrides=self._current_overrides)

    def _current_overrides(self) -> Mapping[str, ManagedToolOverride]:
        resolved = self._resolved.get()
        if resolved is None:
            return {}
        return resolved.value
