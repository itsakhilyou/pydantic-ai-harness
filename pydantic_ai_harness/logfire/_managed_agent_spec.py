"""Back a whole agent's shape with a Logfire-managed [`AgentSpec`][pydantic_ai.agent.spec.AgentSpec]."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from logfire.variables import Variable
from pydantic_ai import Agent, TemplateStr, Tool, models
from pydantic_ai._template import validate_from_spec_args
from pydantic_ai.agent.spec import (
    AgentSpec,
    CapabilitySpecContext,
    capability_spec_context,
    get_capability_registry,
    load_capability_from_nested_spec,
)
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentCapability,
    CapabilityOrdering,
    CombinedCapability,
    Instrumentation,
)
from pydantic_ai.models import Model, ModelRequestContext, infer_model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext, ToolFuncEither
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.logfire._managed_variable import ManagedVariableCapability

if TYPE_CHECKING:
    from logfire.variables import ResolvedVariable
    from pydantic_ai._instructions import AgentInstructions
    from pydantic_ai.capabilities.abstract import WrapRunHandler
    from pydantic_ai.run import AgentRunResult

# Logfire's "Agent Specs" surface exposes the managed spec for an agent with slug `<slug>` as a
# variable named `agentspec__<slug>`, hyphens replaced by underscores. `agentspec__` is reserved
# for these system-managed agent-spec variables. One variable holds the whole agent shape.
_AGENT_SPEC_VARIABLE_PREFIX = 'agentspec__'


@dataclass
class ManagedAgentSpec(ManagedVariableCapability[AgentDepsT, AgentSpec]):
    """Back a whole agent's shape with a Logfire-managed [`AgentSpec`][pydantic_ai.agent.spec.AgentSpec].

    Where [`ManagedPrompt`][pydantic_ai_harness.logfire.ManagedPrompt],
    [`ManagedToolDefinitions`][pydantic_ai_harness.logfire.ManagedToolDefinitions], and
    [`ManagedSettings`][pydantic_ai_harness.logfire.ManagedSettings] each manage one surface of the
    agent, `ManagedAgentSpec` manages the whole shape at once, atomically: instructions, model,
    model settings, and a list of [capabilities](https://ai.pydantic.dev/capabilities/) all come
    from a single Logfire-managed variable and roll out together as one versioned unit. Reach for
    it when you want to steer the agent's overall configuration from the Logfire UI without
    redeploying; reach for the per-surface capabilities when you only need to manage one thing.

    A name of `checkout_assistant` resolves the variable `agentspec__checkout_assistant`, matching
    the naming Logfire's "Agent Specs" surface uses.

    ```python
    import logfire
    from pydantic_ai import Agent

    from pydantic_ai_harness.logfire import ManagedAgentSpec

    logfire.configure()

    agent = Agent(
        'openai:gpt-5',
        capabilities=[ManagedAgentSpec('checkout_assistant', label='production')],
    )
    result = agent.run_sync('Refund my last order.')
    ```

    The spec is resolved **once per run**, inside [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run]
    (earlier than the per-surface capabilities' `wrap_run`, because the resolved spec decides what the
    run is assembled from), and the [`ResolvedVariable`][logfire.variables.ResolvedVariable] is then
    kept open as a context manager for the whole run -- so the selected label and version ride as
    baggage on every child span of the agent run.

    **Additive, not a replacement:** the managed spec's contributions layer **on top of** the
    code-defined agent -- its `instructions` add to the agent's own, and its `model_settings` merge
    over the agent's constructor settings but under per-run `model_settings=` (run arguments win).
    Local tools, toolsets, and code-defined capabilities stay in code; the spec never removes them.

    **Spec capabilities materialize per run:** each entry in the spec's `capabilities` list is
    instantiated from the capability registry (extend it with `custom_capability_types` for your own
    capability classes) and its instructions, model settings, toolset, native tools, and hooks all
    flow through the run normally, exactly as if you had listed the capability in code. An unknown
    capability name, or one whose construction fails, is skipped with a warning rather than crashing
    the run -- a bad managed value must never break a run.

    **Model, two layers:** when the spec sets `model`, it overrides the model per request via
    [`before_model_request`][pydantic_ai.capabilities.AbstractCapability.before_model_request], for
    agents that already have a code-side model. Two known limits, both pending future pydantic-ai
    (run-spec) work: a fully model-less agent still requires a code-side model today, and a managed
    model currently overrides even a per-run `model=` passed at the call site -- the hook can't
    distinguish a run argument from the agent default, so the run-arguments-win precedence that
    settings enjoy doesn't yet hold for the model itself. (A forward-compatible `get_model` hook is
    already wired up for the day pydantic-ai grows the framework-level surface for it.)

    **Fallback semantics:** if the remote value is missing, invalid, or unreachable, the logfire SDK
    falls back to the code default (an empty [`AgentSpec`][pydantic_ai.agent.spec.AgentSpec] when
    none is given) and records the reason on the resolve span, so the run degrades to exactly the
    agent the developer wrote -- never a crashed run.

    Pass an existing [`logfire.variables.Variable`][logfire.variables.Variable] as `name` instead of
    an Agent Specs name when you want to use a variable you defined yourself.
    """

    name: str | Variable[AgentSpec]
    """The Agent Specs name (declared as the variable `agentspec__<name>`), or a pre-built `logfire.Variable`."""

    default: AgentSpec | None = None
    """Code-default spec. When omitted, an empty `AgentSpec()` is used -- nothing is managed until a
    value is configured in Logfire. Ignored when `name` is a `Variable`."""

    custom_capability_types: Sequence[type[AbstractCapability[Any]]] = ()
    """Custom capability classes to add to the registry used to materialize the spec's `capabilities`.
    Extend this to reference your own capability classes by name from the managed spec; built-in
    capability names (e.g. `Thinking`) are always available."""

    def __post_init__(self) -> None:
        self._resolved = self._new_resolved()
        # Inferred `Model` instances keyed by model string, so a repeated override isn't re-inferred.
        # Kept on the (run-shared) capability rather than the per-run resolution so the cache spans runs.
        self._model_cache: dict[str, Model] = {}
        if not isinstance(self.name, str):
            self._warn_logfire_instance_ignored('name')
            self._variable = self.name
            return

        self._variable = self._build_managed_variable(
            self.name,
            prefix=_AGENT_SPEC_VARIABLE_PREFIX,
            value_type=AgentSpec,
            default=self.default or AgentSpec(),
        )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Resolve the managed spec and assemble the per-run capability from it.

        Resolution happens here (not in `wrap_run`) because the spec decides what the run is built
        from: its `capabilities` are materialized now so their `get_*()` surfaces and hooks are in
        place before the framework extracts them. The returned `CombinedCapability` carries a private
        `_ResolvedAgentSpec` (contributing the spec's instructions/model/settings and holding the
        baggage open) alongside the materialized child capabilities, so all of them flow through the
        run as siblings of the agent's own capabilities.
        """
        resolved = self._resolve(ctx)
        spec = resolved.value
        children = await self._materialize_capabilities(spec, ctx)
        # Pass the run-shared plumbing in explicitly (rather than a back-reference to `self`) so the
        # per-run capability only touches public fields -- the private members are read here, in the
        # owning class. `_resolved` is the base's per-run context variable behind the `resolved`
        # property; `_model_cache` and `_maybe_auto_create_for` come from the base.
        resolved_spec = _ResolvedAgentSpec[AgentDepsT](
            spec=spec,
            resolution=resolved,
            resolution_holder=self._resolved,
            model_cache=self._model_cache,
            trigger_auto_create=self._maybe_auto_create_for,
        )
        return CombinedCapability([resolved_spec, *children])

    async def _materialize_capabilities(
        self, spec: AgentSpec, ctx: RunContext[AgentDepsT]
    ) -> list[AbstractCapability[AgentDepsT]]:
        """Instantiate the spec's `capabilities`, skipping (with a warning) any that can't be built.

        Reuses the framework's spec-loading machinery: the capability registry (extended with
        `custom_capability_types`) and `load_capability_from_nested_spec`, run under a
        `CapabilitySpecContext` so nested capability specs (e.g. `PrefixTools`) resolve against the
        same registry. Each capability is `for_run`-resolved before it is returned, mirroring how the
        framework prepares capabilities, so a bad entry is caught here rather than mid-run.
        """
        registry = get_capability_registry(tuple(self.custom_capability_types))
        deps_type = ctx.agent.deps_type if ctx.agent is not None else None
        template_context: dict[str, Any] = {
            'deps_type': deps_type if deps_type is not type(None) else None,
            'deps_schema': spec.deps_schema,
        }

        def instantiate(
            cap_cls: type[AbstractCapability[Any]], args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> AbstractCapability[Any]:
            args, kwargs = validate_from_spec_args(cap_cls, args, kwargs, template_context)
            return cap_cls.from_spec(*args, **kwargs)

        token = capability_spec_context.set(CapabilitySpecContext(registry=registry, instantiate=instantiate))
        try:
            capabilities: list[AbstractCapability[AgentDepsT]] = []
            for cap_spec in spec.capabilities:
                try:
                    capability = load_capability_from_nested_spec(cap_spec)
                    capability = await capability.for_run(ctx)
                except Exception as exc:
                    warnings.warn(
                        f'Skipping managed spec capability {cap_spec.name!r}: {exc}',
                        stacklevel=2,
                    )
                    continue
                capabilities.append(capability)
            return capabilities
        finally:
            capability_spec_context.reset(token)


@dataclass
class _ResolvedAgentSpec(AbstractCapability[AgentDepsT]):
    """The per-run capability carrying an already-resolved managed [`AgentSpec`][pydantic_ai.agent.spec.AgentSpec].

    Built by [`ManagedAgentSpec.for_run`][pydantic_ai_harness.logfire.ManagedAgentSpec.for_run] and
    combined with the spec's materialized child capabilities. It contributes the spec's own fields
    (instructions, model, model settings) and holds the resolution's baggage open for the run;
    resolution and child materialization already happened in `for_run`.
    """

    spec: AgentSpec = field(compare=False)
    """The resolved spec value driving this run's instructions, model, and settings."""

    resolution: ResolvedVariable[AgentSpec] = field(repr=False, compare=False)
    """The resolved variable for this run (resolved in `for_run`, baggage entered in `wrap_run`)."""

    resolution_holder: ContextVar[ResolvedVariable[AgentSpec] | None] = field(repr=False, compare=False)
    """The owner's per-run context variable, set for the run so `ManagedAgentSpec.resolved` reflects it."""

    model_cache: dict[str, Model] = field(repr=False, compare=False)
    """The owner's inferred-`Model` cache, keyed by model string, shared across runs."""

    trigger_auto_create: Callable[[ResolvedVariable[AgentSpec]], None] = field(repr=False, compare=False)
    """The owner's auto-create hook, invoked when the provider doesn't recognize the variable yet."""

    def get_ordering(self) -> CapabilityOrdering:
        """Run outermost so the resolution's baggage envelops the whole run, including the run span."""
        return CapabilityOrdering(position='outermost', wraps=[Instrumentation])

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Contribute the spec's instructions, used verbatim (no deps-templating in v1)."""
        raw = self.spec.instructions
        if raw is None:
            return None
        items = raw if isinstance(raw, list) else [raw]
        # `AgentSpec.instructions` may parse `{{...}}` strings into `TemplateStr`; coerce those back
        # to their raw source so the managed instructions are used verbatim. Plain strings pass through.
        instructions = [str(item) if isinstance(item, TemplateStr) else item for item in items]
        return instructions or None

    def get_model_settings(self) -> ModelSettings | None:
        """Contribute the spec's model settings, merged over the agent's under run arguments."""
        if not self.spec.model_settings:
            return None
        return cast(ModelSettings, self.spec.model_settings)

    def get_model(self) -> str | None:
        """Return the spec's model string.

        Forward-compatible: pydantic-ai does not call a capability-level `get_model` hook yet, so this
        is inert until it grows one. Today the model override happens via `before_model_request` below.
        """
        return self.spec.model

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Override the request's model when the resolved spec sets one."""
        model_string = self.spec.model
        if model_string is None:
            return request_context
        model = self.model_cache.get(model_string)
        if model is None:
            model = self.model_cache[model_string] = infer_model(model_string)
        return dataclasses.replace(request_context, model=model)

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Keep the (already-resolved) variable's baggage active for the run, and auto-create if new."""
        self.trigger_auto_create(self.resolution)
        with self.resolution:
            token = self.resolution_holder.set(self.resolution)
            try:
                return await handler()
            finally:
                self.resolution_holder.reset(token)


def ManagedAgent(
    name: str | Variable[AgentSpec],
    *,
    label: str | None = None,
    targeting_key: str | Callable[[RunContext[None]], str | None] | None = None,
    attributes: Mapping[str, Any] | Callable[[RunContext[None]], Mapping[str, Any] | None] | None = None,
    default: AgentSpec | None = None,
    custom_capability_types: Sequence[type[AbstractCapability[Any]]] = (),
    model: models.Model | models.KnownModelName | str | None = None,
    tools: Sequence[Tool[None] | ToolFuncEither[None, ...]] = (),
    toolsets: Sequence[AgentToolset[None]] | None = None,
    capabilities: Sequence[AgentCapability[None]] = (),
    **agent_kwargs: Any,
) -> Agent[None, str]:
    """Build an [`Agent`][pydantic_ai.Agent] whose whole shape is backed by a Logfire-managed spec.

    Sugar over `Agent(..., capabilities=[ManagedAgentSpec(name, ...), *capabilities])`: it constructs
    a real agent **once**, with a [`ManagedAgentSpec`][pydantic_ai_harness.logfire.ManagedAgentSpec]
    that resolves the managed [`AgentSpec`][pydantic_ai.agent.spec.AgentSpec] fresh on every run. It is
    not a builder -- the returned agent is a normal `Agent`; the managed values just flow in per run
    via the capability. Local `tools`, `toolsets`, and extra `capabilities` you pass here stay in
    code and compose with whatever the managed spec adds.

    ```python
    import logfire

    from pydantic_ai_harness.logfire import ManagedAgent

    logfire.configure()

    agent = ManagedAgent('checkout_assistant', model='openai:gpt-5', label='production')
    result = agent.run_sync('Refund my last order.')
    ```

    Until pydantic-ai ships a framework-level `get_model` hook, pass a fallback `model` so the agent
    can run before any spec is published; the managed spec's `model`, when set, then overrides it per
    request (see [`ManagedAgentSpec`][pydantic_ai_harness.logfire.ManagedAgentSpec] for the model-override limits).

    Args:
        name: The Agent Specs name (declared as `agentspec__<name>`), or a pre-built `logfire.Variable`.
        label: Explicit Logfire targeting label to resolve (e.g. `'production'`).
        targeting_key: Stable key seeding Logfire's deterministic rollout assignment, or a callable
            deriving it from the [`RunContext`][pydantic_ai.tools.RunContext].
        attributes: Attributes for condition-based targeting rules, or a callable deriving them.
        default: Code-default spec used until a value is published (an empty `AgentSpec()` when omitted).
        custom_capability_types: Custom capability classes to make available to the managed spec.
        model: Fallback model for the agent (overridden per request by the spec's `model` when set).
        tools: Local tools to register on the agent, kept in code.
        toolsets: Local toolsets to register on the agent, kept in code.
        capabilities: Extra capabilities to compose alongside the managed spec.
        **agent_kwargs: Forwarded to the [`Agent`][pydantic_ai.Agent] constructor (e.g. `name`, `retries`).

    Returns:
        A configured [`Agent`][pydantic_ai.Agent] with the managed spec capability in place.
    """
    managed = ManagedAgentSpec[None](
        name,
        label=label,
        targeting_key=targeting_key,
        attributes=attributes,
        default=default,
        custom_capability_types=custom_capability_types,
    )
    return Agent(
        model,
        tools=tools,
        toolsets=toolsets,
        capabilities=[managed, *capabilities],
        **agent_kwargs,
    )
