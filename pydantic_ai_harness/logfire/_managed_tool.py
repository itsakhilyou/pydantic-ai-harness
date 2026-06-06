"""Back a tool's advertised name, description, and parameter descriptions with a Logfire-managed variable."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, TypeAlias

from logfire.variables import Variable
from pydantic import BaseModel
from pydantic_ai import AbstractToolset, RunContext, Tool, ToolDefinition, WrapperToolset
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset, FunctionToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

from pydantic_ai_harness.logfire._managed_variable import ManagedVariableCapability

# Unlike Logfire's Prompt management (which reserves `prompt__`), there is no first-party
# Logfire "tool management" feature, so `tool__` is a harness convention: it namespaces the
# backing managed variable and keeps it visually grouped with `ManagedPrompt`'s `prompt__` vars.
_TOOL_VARIABLE_PREFIX = 'tool__'


class ManagedToolOverride(BaseModel):
    """A patch over a tool's advertised name, description, and parameter descriptions.

    Resolved from a Logfire managed variable. Every field is optional and `None` means *keep the
    tool's own value* -- so you can tweak just a tool's `description` from the Logfire UI without
    restating its `name`. The override only changes what the model is **shown**; the parameter
    schema's structure (names, types, required fields), argument validation, and execution all stay
    exactly as defined in code.
    """

    name: str | None = None
    """Replacement tool name shown to the model. `None` keeps the original name."""

    description: str | None = None
    """Replacement tool description shown to the model. `None` keeps the original description."""

    parameter_descriptions: dict[str, str] | None = None
    """Replacement descriptions for individual parameters, keyed by parameter name.

    Only the `description` text of each named top-level parameter is changed; names, types, and
    which parameters are required stay exactly as defined in code, so the tool's argument validator
    is unaffected. Parameter names that aren't in the tool's schema are ignored."""


# Returns the overrides to apply for the active run, keyed by each tool's *original* name. The
# capabilities supply a closure over their per-run resolution so the wrapper stays agnostic to
# whether it's managing one tool (`ManagedTool`) or many (`ManagedToolset`).
OverridesProvider: TypeAlias = Callable[[], Mapping[str, ManagedToolOverride]]


def _with_parameter_descriptions(
    parameters_json_schema: dict[str, Any], parameter_descriptions: dict[str, str]
) -> dict[str, Any]:
    """Return the schema with the given top-level parameters' `description` replaced.

    Patches only the `description` of each named parameter under `properties`; structure (names,
    types, required) is untouched, so the tool's argument validator still applies. Parameters not
    present in the schema are ignored. Returns the same object when nothing changed.
    """
    if not isinstance(parameters_json_schema.get('properties'), dict):
        return parameters_json_schema
    # Read via indexing (yielding `Any`) rather than the `isinstance`-narrowed value, so the
    # handle stays typed as `dict[str, Any]` instead of `dict[Unknown, Unknown]`.
    properties: dict[str, Any] = parameters_json_schema['properties']

    new_properties: dict[str, Any] = {}
    changed = False
    for name, schema in properties.items():
        if name in parameter_descriptions and isinstance(schema, dict):
            param_schema: dict[str, Any] = properties[name]
            new_properties[name] = {**param_schema, 'description': parameter_descriptions[name]}
            changed = True
        else:
            new_properties[name] = schema
    if not changed:
        return parameters_json_schema
    return {**parameters_json_schema, 'properties': new_properties}


def _apply_override(tool_def: ToolDefinition, override: ManagedToolOverride) -> ToolDefinition:
    """Return `tool_def` with the override's set fields applied, or the same object when it's a no-op."""
    changes: dict[str, Any] = {}
    if override.name is not None and override.name != tool_def.name:
        changes['name'] = override.name
    if override.description is not None and override.description != tool_def.description:
        changes['description'] = override.description
    if override.parameter_descriptions:
        new_schema = _with_parameter_descriptions(tool_def.parameters_json_schema, override.parameter_descriptions)
        if new_schema is not tool_def.parameters_json_schema:
            changes['parameters_json_schema'] = new_schema
    if not changes:
        return tool_def
    return replace(tool_def, **changes)


@dataclass
class ManagedToolsToolset(WrapperToolset[AgentDepsT]):
    """Applies Logfire-resolved overrides to the matching tools of the wrapped toolset.

    `get_overrides` returns the overrides for the active run keyed by each tool's original name, so
    listing tools and routing calls both see the same resolution. A tool whose name an override
    changes is re-keyed under the new name and routed back to the original on call.
    """

    get_overrides: OverridesProvider = field(repr=False, compare=False)

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools = await super().get_tools(ctx)
        overrides = self.get_overrides()
        # No active resolution, or nothing to override (e.g. the empty default): leave tools as-is.
        if not overrides:
            return tools

        result: dict[str, ToolsetTool[AgentDepsT]] = {}
        source_of: dict[str, str] = {}
        for original_name, tool in tools.items():
            override = overrides.get(original_name)
            new_tool_def = tool.tool_def if override is None else _apply_override(tool.tool_def, override)
            new_name = new_tool_def.name
            if new_name in result:
                raise UserError(
                    f'A managed tool override produces a duplicate tool name {new_name!r} '
                    f'(from {source_of[new_name]!r} and {original_name!r}). Choose different override names.'
                )
            # `replace` preserves the wrapped tool's concrete class, which inner toolsets
            # (e.g. `CombinedToolset`) assert on when dispatching the call.
            result[new_name] = (
                tool if new_tool_def is tool.tool_def else replace(tool, toolset=self, tool_def=new_tool_def)
            )
            source_of[new_name] = original_name
        return result

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        # At most one override can advertise a given name (`get_tools` rejects duplicates), so the
        # first match is the original tool the model is really calling.
        for original_name, override in self.get_overrides().items():
            if override.name == name and original_name != name:
                ctx = replace(ctx, tool_name=original_name)
                tool = replace(tool, tool_def=replace(tool.tool_def, name=original_name))
                return await super().call_tool(original_name, tool_args, ctx, tool)
        return await super().call_tool(name, tool_args, ctx, tool)


@dataclass
class ManagedTool(ManagedVariableCapability[AgentDepsT, ManagedToolOverride]):
    """Back a single tool's advertised name, description, and parameter descriptions with a Logfire variable.

    The tool keeps its code-defined implementation **and parameter schema structure**; this
    capability only overrides what the model is **shown**, resolved from a [managed variable](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
    so you can iterate on a tool's framing from the Logfire UI -- versioned, labelled, and rolled
    out -- without redeploying. A tool named `get_weather` resolves the variable `tool__get_weather`.

    Pass a tool **name** to manage a tool registered elsewhere on the agent, or pass a `Tool` to have
    this capability both provide and manage it:

    ```python
    import logfire
    from pydantic_ai import Agent, Tool

    from pydantic_ai_harness.logfire import ManagedTool

    logfire.configure()

    def get_weather(city: str) -> str:
        return f'The weather in {city} is sunny.'

    agent = Agent('openai:gpt-5', capabilities=[ManagedTool(Tool(get_weather), label='production')])
    ```

    The backing variable resolves to a [`ManagedToolOverride`][pydantic_ai_harness.logfire.ManagedToolOverride]
    -- a patch whose unset fields leave the tool's own definition untouched. The code default is an
    empty override, so the agent keeps using the tool's code-defined framing until a remote value is
    published. The advertised name, description, and individual parameter descriptions can be
    overridden, but the parameter schema's structure (names, types, required fields) is deliberately
    fixed so a remote value can never drift from the validator the tool runs against.

    Resolution happens **once per run** inside [`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run],
    keeping the [`ResolvedVariable`][logfire.variables.ResolvedVariable] open as a context manager for
    the whole run so the selected label and version are attached as baggage to every child span.

    Pass an existing [`logfire.variables.Variable`][logfire.variables.Variable] as `variable` to use a
    variable you declared yourself; otherwise pass a name (defaults to the tool's name).
    """

    tool: str | Tool[AgentDepsT]
    """The tool to manage: its **name** (to manage a tool registered elsewhere on the agent), or a
    `Tool` (which this capability then both provides and manages). Wrap a bare function with
    [`Tool`][pydantic_ai.tools.Tool] (e.g. `ManagedTool(Tool(my_func))`)."""

    variable: str | Variable[ManagedToolOverride] | None = None
    """Managed variable name (declared as `tool__<name>`), or a pre-built `logfire.Variable`.
    When `None`, defaults to the tool's name."""

    default: ManagedToolOverride | None = None
    """Code-default override. When `None`, an empty override (no changes) is used. Ignored when
    `variable` is a `Variable`."""

    _tool_name: str = field(init=False, repr=False, compare=False)
    """The original name of the managed tool (its own name, or the name of the owned tool)."""

    _owned: Tool[AgentDepsT] | None = field(init=False, repr=False, compare=False)
    """The tool this capability provides, when constructed from a `Tool` rather than a name."""

    def __post_init__(self) -> None:
        self._resolved = self._new_resolved()
        if isinstance(self.tool, str):
            self._tool_name = self.tool
            self._owned = None
        else:
            self._tool_name = self.tool.name
            self._owned = self.tool

        if isinstance(self.variable, Variable):
            self._warn_logfire_instance_ignored('variable')
            self._variable = self.variable
            return

        name = self.variable if self.variable is not None else self._tool_name
        default = self.default if self.default is not None else ManagedToolOverride()
        self._variable = self._build_managed_variable(
            name, prefix=_TOOL_VARIABLE_PREFIX, value_type=ManagedToolOverride, default=default
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Provide the owned tool, when this capability was built from a `Tool`."""
        if self._owned is None:
            return None
        return FunctionToolset([self._owned])

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Wrap the assembled toolset so the managed override is applied to the target tool."""
        return ManagedToolsToolset(wrapped=toolset, get_overrides=self._current_overrides)

    def _current_overrides(self) -> Mapping[str, ManagedToolOverride]:
        resolved = self._resolved.get()
        if resolved is None:
            return {}
        return {self._tool_name: resolved.value}
