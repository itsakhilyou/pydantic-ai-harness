"""Back an agent's configuration with a Logfire-managed agent spec."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any, cast

import logfire
from logfire.variables import Variable
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model

if TYPE_CHECKING:
    from logfire import Logfire
    from pydantic_ai.mcp import MCPToolset


# Logfire stores agent specs as variables named `agentspec__<slug>`. The
# OFREP-side compiler returns the assembled spec shape (instructions, model,
# model_settings, tools, mcp_servers, skills, capabilities), so this module
# never has to follow refs itself -- it just consumes the compiled payload.
_AGENT_SPEC_VARIABLE_PREFIX = 'agentspec__'

# Empty default for the logfire Variable. When the variable hasn't been
# published yet, the SDK returns this dict and the builder produces a
# minimum-viable Agent (using the code-default model below).
_EMPTY_SPEC: dict[str, Any] = {}

# Last-resort model when the spec doesn't pin one. Matches the snippet shown
# in the Logfire UI's "Use this spec" card so the two stay in sync.
_DEFAULT_MODEL = 'openai:gpt-5'


def _empty_tools() -> dict[str, Callable[..., Any]]:
    return {}


def _empty_capabilities() -> dict[str, type[AbstractCapability[Any]]]:
    return {}


@dataclass
class ManagedAgentSpec:
    """Build a `pydantic_ai.Agent` from a Logfire-managed agent spec.

    Pass the spec name (a `<slug>` from the Logfire UI's Agent Specs surface)
    plus your app's tool implementations and any harness capability classes
    you want the spec to be able to enable. Each `build()` call resolves the
    spec fresh and returns a ready-to-run Agent -- changes published from
    the Logfire UI take effect on the next call, no redeploy.

    ```python
    from pydantic_ai_harness import CodeMode
    from pydantic_ai_harness.logfire import ManagedAgentSpec


    def fetch_weather(city: str) -> str:
        return f'sunny and 72°F in {city}'


    spec = ManagedAgentSpec(
        'checkout_assistant',
        tools={'tool__weather': fetch_weather},
        capability_classes={'code_mode': CodeMode},
    )
    agent = await spec.build()
    result = await agent.run('What is the weather in Lisbon?')
    ```

    The Logfire UI owns instructions, model, model_settings, tool/MCP/skill
    metadata, and harness-capability config. Tool *implementations* and
    capability *classes* stay in your code -- tools because the function
    body is application logic, capabilities because the harness package is
    optional.

    Pass an existing [`logfire.variables.Variable`][logfire.variables.Variable]
    as `name` when you want to use a variable you defined yourself (for
    example a typed `Variable[YourSpecModel]`).
    """

    name: str | Variable[dict[str, Any]]
    """The agent-spec name (declared as the variable `agentspec__<name>`),
    or a pre-built `logfire.Variable[dict]`."""

    _: KW_ONLY

    tools: Mapping[str, Callable[..., Any]] = field(default_factory=_empty_tools)
    """Map from a managed tool's variable name (e.g. `tool__weather`) to the
    Python callable that implements it. Tools the spec lists but you haven't
    mapped are silently skipped -- the model just won't see them."""

    capability_classes: Mapping[str, type[AbstractCapability[Any]]] = field(default_factory=_empty_capabilities)
    """Map from a capability `type` key in the spec (e.g. `code_mode`) to the
    class to instantiate. The class is called with `**config` from the spec.
    Unknown keys in the spec are silently dropped so the harness package can
    stay optional -- a spec referencing CodeMode doesn't break callers who
    haven't installed it."""

    targeting_key: str | None = None
    """`targetingKey` sent to Logfire's rollout/condition evaluation. When
    `None`, Logfire falls back to its own targeting context and then the
    active trace id."""

    attributes: Mapping[str, Any] | None = None
    """Attributes for condition-based targeting rules on the agent spec."""

    label: str | None = None
    """Explicit label on the managed spec to resolve (e.g. `'production'`).
    When `None`, the targeting rules pick the label."""

    default_model: str | Model = _DEFAULT_MODEL
    """Model to use when the spec doesn't pin one. Lets a Logfire-side
    rollback to an empty spec keep your app running. Accepts either a
    provider-prefixed model id or a pydantic-ai `Model` instance (e.g.
    `TestModel()` in tests)."""

    logfire_instance: Logfire | None = None
    """Logfire instance to resolve the variable on. When `None`, the global
    default instance is used. Ignored when `name` is a `Variable`."""

    _variable: Variable[dict[str, Any]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            if self.logfire_instance is not None:
                warnings.warn(
                    '`logfire_instance` is ignored when `name` is a `Variable`; '
                    'the variable already carries its own Logfire instance.',
                    stacklevel=2,
                )
            self._variable = self.name
            return

        name = self.name
        if name.startswith(_AGENT_SPEC_VARIABLE_PREFIX):
            warnings.warn(
                f'The {_AGENT_SPEC_VARIABLE_PREFIX!r} prefix is added automatically; '
                f'pass the bare spec name rather than {name!r}.',
                stacklevel=2,
            )
            name = name[len(_AGENT_SPEC_VARIABLE_PREFIX) :]

        variable_name = f'{_AGENT_SPEC_VARIABLE_PREFIX}{name.replace("-", "_")}'
        if not variable_name.isidentifier():
            raise ValueError(
                f'Agent-spec name {self.name!r} produces an invalid variable name {variable_name!r}; '
                'names may only contain letters, digits, hyphens, and underscores.'
            )

        # Construct the Variable directly rather than via `logfire.var` so the
        # same spec can be built repeatedly without tripping the per-instance
        # duplicate-registration check.
        instance = self.logfire_instance if self.logfire_instance is not None else logfire.DEFAULT_LOGFIRE_INSTANCE
        self._variable = Variable(variable_name, type=dict, default=_EMPTY_SPEC, logfire_instance=instance)

    async def build(self) -> Agent[None, str]:
        """Resolve the spec and return a ready-to-run `Agent`.

        Hits Logfire fresh on every call. Cache the returned `Agent` between
        requests if you want to amortise the resolve cost.
        """
        resolved = self._variable.get(targeting_key=self.targeting_key, attributes=self.attributes, label=self.label)
        with resolved:
            # `resolved.value` is the dict the variable resolves to; cast it
            # to a typed `Mapping[str, Any]` so the wirers don't propagate
            # Unknowns through pyright.
            spec: Mapping[str, Any] = cast('Mapping[str, Any]', resolved.value)

            kwargs: dict[str, Any] = {
                'model': spec.get('model', self.default_model),
                'instructions': _instructions_with_skills(spec),
                'tools': _wire_tools(spec, self.tools),
            }
            model_settings = spec.get('model_settings')
            if isinstance(model_settings, dict) and model_settings:
                kwargs['model_settings'] = model_settings
            toolsets = _wire_mcp_servers(spec)
            if toolsets:
                kwargs['toolsets'] = toolsets
            capabilities = _wire_capabilities(spec, self.capability_classes)
            if capabilities:
                kwargs['capabilities'] = capabilities

            return Agent(**kwargs)


def _iter_dicts(raw: object) -> Iterable[dict[str, Any]]:
    """Yield each `dict[str, Any]` entry from `raw`, skipping anything else.

    `raw` comes from a resolved `Variable[dict]`, so it's `object`. The
    helper centralises the list-and-dict narrowing the four wirers all
    repeat, and keeps pyright happy with one cast instead of four.
    """
    if not isinstance(raw, list):
        return
    for entry in cast('list[Any]', raw):
        if isinstance(entry, dict):
            yield cast('dict[str, Any]', entry)


def _wire_tools(spec: Mapping[str, Any], impls: Mapping[str, Callable[..., Any]]) -> list[Callable[..., Any]]:
    """Resolve each spec tool to its registered Python callable."""
    out: list[Callable[..., Any]] = []
    for entry in _iter_dicts(spec.get('tools')):
        name = entry.get('name')
        if not isinstance(name, str):
            continue
        impl = impls.get(name)
        if impl is not None:
            out.append(impl)
    return out


def _wire_mcp_servers(spec: Mapping[str, Any]) -> list[MCPToolset]:
    """Build `MCPToolset` for each compiled-spec MCP entry.

    Lazy-imports `pydantic_ai.mcp` only when there's at least one valid
    entry to instantiate, so a spec without (usable) MCP servers doesn't
    require the `pydantic-ai-slim[mcp]` extra. Callers who include MCP
    servers in their spec must install the extra themselves; the import
    error fires at build time with a clear `pydantic-ai` suggestion.
    """
    configs: list[tuple[str, dict[str, Any] | None]] = []
    for entry in _iter_dicts(spec.get('mcp_servers')):
        url = entry.get('url')
        if not isinstance(url, str) or not url:
            continue
        headers = entry.get('headers')
        configs.append((url, cast('dict[str, Any]', headers) if isinstance(headers, dict) else None))
    if not configs:
        return []

    # Tested via a happy-path test that mocks `pydantic_ai.mcp.MCPToolset` --
    # the harness's default install doesn't include the `[mcp]` extra so we
    # can't actually instantiate one here.
    from pydantic_ai.mcp import MCPToolset

    return [MCPToolset(url, headers=headers) for url, headers in configs]


def _wire_capabilities(
    spec: Mapping[str, Any],
    classes: Mapping[str, type[AbstractCapability[Any]]],
) -> list[AbstractCapability[Any]]:
    """Instantiate each compiled-spec capability the caller has registered.

    Capability entries the caller hasn't registered are silently dropped --
    `CodeMode`/`FileSystem`/`Shell` are optional installs, and a spec
    referencing one shouldn't fail to build just because the caller hasn't
    enabled it in their app.
    """
    out: list[AbstractCapability[Any]] = []
    for entry in _iter_dicts(spec.get('capabilities')):
        type_key = entry.get('type')
        if not isinstance(type_key, str):
            continue
        cls = classes.get(type_key)
        if cls is None:
            continue
        config = entry.get('config')
        if config is None:
            out.append(cls())
        elif isinstance(config, dict):
            try:
                out.append(cls(**config))
            except TypeError as exc:
                raise UserError(f'ManagedAgentSpec: capability {type_key!r} rejected config {config!r}: {exc}') from exc
    return out


def _instructions_with_skills(spec: Mapping[str, Any]) -> str | None:
    """Append the skill catalog to the spec's instructions.

    Skills are progressively-loaded: the model picks one by description, then
    the harness loads its full instructions on demand. The catalog has to
    land in the system prompt so the model can ASK for a skill in the first
    place. (Full-skill on-demand loading is future work -- today only the
    descriptions reach the model.)
    """
    base = spec.get('instructions')
    base_text = base if isinstance(base, str) else None

    blurbs: list[str] = []
    for entry in _iter_dicts(spec.get('skills')):
        name = entry.get('name')
        description = entry.get('description')
        if isinstance(name, str) and isinstance(description, str) and description:
            blurbs.append(f'- {name}: {description}')
    if not blurbs:
        return base_text

    catalog = 'Available skills:\n' + '\n'.join(blurbs)
    return f'{base_text}\n\n{catalog}' if base_text else catalog
