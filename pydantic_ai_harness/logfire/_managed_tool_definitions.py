"""Override the LLM-facing definitions of an agent's tools with a Logfire-managed variable."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, TypeAlias

from logfire.variables import Variable
from pydantic import BaseModel, Field
from pydantic_ai import AbstractToolset, RunContext, ToolDefinition, WrapperToolset
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.abstract import ToolsetTool

from pydantic_ai_harness.logfire._managed_variable import ManagedVariableCapability

# Unlike Logfire's Prompt management (which reserves `prompt__`), there is no first-party Logfire
# "tool management" feature reserving this prefix, so `tool_definitions__` is a harness convention:
# it namespaces the backing managed variable and keeps it visually grouped with `ManagedPrompt`'s
# `prompt__` vars. One variable holds the whole list of overrides for an agent's tools.
_TOOL_DEFINITIONS_VARIABLE_PREFIX = 'tool_definitions__'


class ToolDefinitionOverride(BaseModel):
    """A patch over a single tool's LLM-facing definition (name, description, parameter docs).

    Resolved as one entry in a Logfire-managed list. Every field but `name` is optional and `None`
    means *keep the tool's own value* -- so you can tweak just a tool's `description` from the
    Logfire UI without restating anything else. The override only changes what the model is
    **shown**; the parameter schema's structure (names, types, required fields), argument
    validation, and execution all stay exactly as defined in code.
    """

    # Reject empty string so a UI-side edit of `{"name": ""}` can't produce an entry that targets no
    # tool. `min_length` is enforced by Pydantic; the SDK falls back to the code default on failure.
    name: str = Field(min_length=1)
    """The original (code-side) name of the tool this override targets. This is the lookup key, not
    a rename -- use `new_name` to change what the model is shown."""

    new_name: str | None = Field(default=None, min_length=1)
    """Replacement tool name shown to the model. `None` keeps the original name. A rename still
    routes the model's call back to the original code implementation."""

    description: str | None = None
    """Replacement tool description shown to the model. `None` keeps the original description."""

    parameter_descriptions: dict[str, str] | None = None
    """Replacement descriptions for individual parameters, keyed by parameter name.

    Only the `description` text of each named top-level parameter is changed; names, types, and
    which parameters are required stay exactly as defined in code, so the tool's argument validator
    is unaffected. Parameter names that aren't in the tool's schema are ignored."""


# Returns the overrides to apply for the active run, keyed by each tool's *original* name. The
# capability supplies a closure over its per-run resolution so the wrapper stays agnostic to how the
# overrides were resolved.
OverridesProvider: TypeAlias = Callable[[], Mapping[str, ToolDefinitionOverride]]


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


def _apply_override(tool_def: ToolDefinition, override: ToolDefinitionOverride) -> ToolDefinition:
    """Return `tool_def` with the override's set fields applied, or the same object when it's a no-op.

    `override.name` is the lookup key (matched before this is called); `override.new_name` drives the
    rename shown to the model.
    """
    changes: dict[str, Any] = {}
    if override.new_name is not None and override.new_name != tool_def.name:
        changes['name'] = override.new_name
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
class _ToolDefinitionOverridesToolset(WrapperToolset[AgentDepsT]):
    """Applies Logfire-resolved tool-definition overrides to the matching tools of the wrapped toolset.

    `get_overrides` returns the overrides for the active run keyed by each tool's original name, so
    listing tools and routing calls both see the same resolution. A tool whose name an override
    changes is re-keyed under the new name and routed back to the original on call.
    """

    get_overrides: OverridesProvider = field(repr=False, compare=False)

    def _effective_tools(
        self, tools: dict[str, ToolsetTool[AgentDepsT]], *, warn: bool
    ) -> dict[str, tuple[str, ToolsetTool[AgentDepsT]]]:
        """Map each advertised name to `(original_name, patched tool)` under the active overrides.

        Deterministic on `(tools, overrides)` so listing tools and routing calls agree on which
        renames took effect -- a rename that collides with a name another tool already advertises is
        dropped (keeping any description/parameter patches), because a bad managed value must never
        break a run. `warn` controls whether the drop emits a warning (once, from `get_tools`).
        """
        overrides = self.get_overrides()
        result: dict[str, tuple[str, ToolsetTool[AgentDepsT]]] = {}
        for original_name, tool in tools.items():
            override = overrides.get(original_name)
            new_tool_def = tool.tool_def if override is None else _apply_override(tool.tool_def, override)
            new_name = new_tool_def.name
            if new_name != original_name and (new_name in tools or new_name in result):
                if warn:
                    warnings.warn(
                        f'Managed tool definition override renames {original_name!r} to {new_name!r}, '
                        f'which is already advertised by another tool; keeping the original '
                        f'name {original_name!r}.',
                        stacklevel=2,
                    )
                new_tool_def = replace(new_tool_def, name=original_name)
                new_name = original_name
            # `replace` preserves the wrapped tool's concrete class, which inner toolsets
            # (e.g. `CombinedToolset`) assert on when dispatching the call.
            result[new_name] = (
                original_name,
                tool if new_tool_def is tool.tool_def else replace(tool, toolset=self, tool_def=new_tool_def),
            )
        return result

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools = await super().get_tools(ctx)
        # No active resolution, or nothing to override (e.g. the empty default): leave tools as-is.
        if not self.get_overrides():
            return tools
        return {name: tool for name, (_, tool) in self._effective_tools(tools, warn=True).items()}

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        overrides = self.get_overrides()
        if not overrides or not any(o.new_name for o in overrides.values()):
            return await super().call_tool(name, tool_args, ctx, tool)
        # A rename is in play somewhere: recompute the effective mapping (the same deterministic
        # logic `get_tools` used, including dropped collisions) to find which original tool the
        # advertised name routes to. This re-lists the wrapped tools once per call, which is cheap
        # for function toolsets and cached by remote ones; correctness over micro-optimization.
        effective = self._effective_tools(await super().get_tools(ctx), warn=False)
        entry = effective.get(name)
        if entry is not None and entry[0] != name:
            original_name = entry[0]
            ctx = replace(ctx, tool_name=original_name)
            tool = replace(tool, tool_def=replace(tool.tool_def, name=original_name))
            return await super().call_tool(original_name, tool_args, ctx, tool)
        return await super().call_tool(name, tool_args, ctx, tool)


@dataclass
class ManagedToolDefinitions(ManagedVariableCapability[AgentDepsT, 'list[ToolDefinitionOverride]']):
    """Override the LLM-facing definitions of an agent's tools with a Logfire-managed variable.

    Drop this capability onto any agent and it can override the **definition** each of the agent's
    tools advertises to the model -- name, description, and parameter descriptions -- while every
    tool keeps its code-defined implementation and parameter schema structure. A tool is the
    executable unit (it stays in code); the tool *definition* is the LLM-facing spec, and that is the
    managed surface. The overrides are resolved from one [managed variable](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)
    so you can iterate on your tools' framing from the Logfire UI -- versioned, labelled, and rolled
    out -- without redeploying. A name of `checkout_assistant` resolves the variable
    `tool_definitions__checkout_assistant`.

    ```python
    import logfire
    from pydantic_ai import Agent

    from pydantic_ai_harness.logfire import ManagedToolDefinitions

    logfire.configure()

    def get_weather(city: str) -> str:
        return f'The weather in {city} is sunny.'

    def get_forecast(city: str) -> str:
        return f'The forecast for {city} is sunny all week.'

    agent = Agent(
        'openai:gpt-5',
        tools=[get_weather, get_forecast],
        capabilities=[ManagedToolDefinitions('checkout_assistant', label='production')],
    )
    result = agent.run_sync('What should I wear in London?')
    ```

    The backing variable resolves to a `list` of [`ToolDefinitionOverride`][pydantic_ai_harness.logfire.ToolDefinitionOverride]
    entries, each keyed to a tool by its original (code-side) `name`. Every override field but `name`
    is optional and unset fields keep the tool's own definition (patch semantics), so an override can
    tweak just a `description` while leaving everything else as written in code. The parameter
    schema's structure (parameter names, types, required fields) is deliberately fixed -- only the
    `description` strings inside it can be patched -- so a remote value can never drift from the
    validator the tool actually runs against.

    **Fallback:** with no override list published, or when the remote value can't be validated, the
    tools are advertised exactly as defined in code -- the logfire SDK returns the code default (an
    empty list) on validation errors, so a bad remote value never crashes a run. An override whose
    `name` matches no tool on the agent is inert: it simply never matches when tools are listed. That
    is the drift case (the tool may have been removed or renamed in code); rather than warn on every
    run, the Logfire UI's before/after view is where the drift becomes visible.

    **Renames round-trip:** setting `new_name` changes the name the model is shown, and a call to the
    renamed tool is routed back to the original code implementation -- `ctx.tool_name` inside the
    tool is the original name. A rename that collides with a name another tool already advertises is
    dropped with a warning (any description or parameter patches still apply), so a bad remote value
    degrades instead of breaking the run.

    Resolution happens **once per run** inside [`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run],
    keeping the [`ResolvedVariable`][logfire.variables.ResolvedVariable] open as a context manager for
    the whole run so the selected label and version are attached as baggage to every child span.

    Declaring the same name more than once is fine -- each `ManagedToolDefinitions` constructs its own
    backing variable, so sharing one override list across several agents just works. Pass an existing
    [`logfire.variables.Variable`][logfire.variables.Variable] as `name` instead of a name when you
    want to use a variable you declared yourself.
    """

    name: str | Variable[list[ToolDefinitionOverride]]
    """The managed name (declared as the variable `tool_definitions__<name>`), or a pre-built
    `logfire.Variable`."""

    default: list[ToolDefinitionOverride] | None = None
    """Code-default override list. When omitted, an empty list (no overrides) is used -- a sensible
    default meaning "no overrides yet", which is also the auto-create story. Ignored when `name` is a
    `Variable`."""

    def __post_init__(self) -> None:
        self._resolved = self._new_resolved()
        if not isinstance(self.name, str):
            self._warn_logfire_instance_ignored('name')
            self._variable = self.name
            return

        self._variable = self._build_managed_variable(
            self.name,
            prefix=_TOOL_DEFINITIONS_VARIABLE_PREFIX,
            value_type=list[ToolDefinitionOverride],
            default=self.default if self.default is not None else [],
        )

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Wrap the assembled run toolset so the resolved overrides are applied to every tool."""
        return _ToolDefinitionOverridesToolset(wrapped=toolset, get_overrides=self._current_overrides)

    def _current_overrides(self) -> Mapping[str, ToolDefinitionOverride]:
        resolved = self._resolved.get()
        if resolved is None:
            return {}
        # Fold the list into a lookup keyed by the original tool name. A server-side merge bug could
        # emit two entries for one tool; the later entry wins (matching how the UI form is edited),
        # but warn rather than silently drop -- a duplicate is a signal something upstream is wrong.
        overrides: dict[str, ToolDefinitionOverride] = {}
        for override in resolved.value:
            if override.name in overrides:
                warnings.warn(
                    f'Multiple managed tool definition overrides target {override.name!r}; the last one wins.',
                    stacklevel=2,
                )
            overrides[override.name] = override
        return overrides
