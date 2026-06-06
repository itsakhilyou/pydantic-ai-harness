"""Shared base for capabilities backed by a Logfire managed variable.

`ManagedPrompt`, `ManagedTool`, and `ManagedToolset` all resolve a Logfire
[managed variable](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/) once per
run and keep its baggage active for the whole run. This base owns that shared plumbing -- the
targeting inputs, the per-run resolution context variable, `get_ordering`, and `wrap_run` -- so each
capability only declares its own variable and exposes the resolved value through its own surface.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic

import logfire
from logfire.variables import Variable
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, Instrumentation
from pydantic_ai.tools import AgentDepsT, RunContext
from typing_extensions import TypeVar

if TYPE_CHECKING:
    from logfire import Logfire
    from logfire.variables import ResolvedVariable
    from pydantic_ai.capabilities.abstract import WrapRunHandler
    from pydantic_ai.run import AgentRunResult

# `AgentDepsT` carries a PEP 696 default, so `ValueT` needs one too to follow it in the type
# parameter list. Subclasses always bind it explicitly, so the default is never actually used.
ValueT = TypeVar('ValueT', default=object)


@dataclass
class ManagedVariableCapability(AbstractCapability[AgentDepsT], Generic[AgentDepsT, ValueT]):
    """Base for capabilities that resolve a Logfire managed variable once per run.

    Subclasses set `self._variable` (and `self._resolved`) in their `__post_init__` -- typically via
    `_build_managed_variable` for a declared name, or directly from a pre-built
    [`logfire.variables.Variable`][logfire.variables.Variable] -- then expose the active run's
    resolution through their own surface (instructions, a toolset wrapper, ...).
    """

    label: str | None = field(default=None, kw_only=True)
    """Explicit targeting label on the Logfire managed variable to resolve (e.g. `'production'`).
    When `None`, the targeting rules on the managed variable select the label."""

    targeting_key: str | Callable[[RunContext[AgentDepsT]], str | None] | None = field(default=None, kw_only=True)
    """Stable key that seeds Logfire's deterministic rollout assignment -- the same key always
    lands in the same percentage bucket. Accepts a static value or a callable that derives it from
    the [`RunContext`][pydantic_ai.tools.RunContext]. When `None`, Logfire falls back to its own
    targeting context and then the active trace id."""

    attributes: Mapping[str, Any] | Callable[[RunContext[AgentDepsT]], Mapping[str, Any] | None] | None = field(
        default=None, kw_only=True
    )
    """Attributes for condition-based targeting rules, or a callable that derives them
    from the [`RunContext`][pydantic_ai.tools.RunContext]."""

    logfire_instance: Logfire | None = field(default=None, kw_only=True)
    """Logfire instance to resolve the variable on. When `None`, the global default instance is
    used. Ignored when the capability is given a pre-built `Variable`."""

    _variable: Variable[ValueT] = field(init=False, repr=False, compare=False)
    """The managed variable backing this capability (declared by the subclass)."""

    _resolved: ContextVar[ResolvedVariable[ValueT] | None] = field(init=False, repr=False, compare=False)
    """Per-run resolution, isolated across concurrent runs via the context variable."""

    def _new_resolved(self) -> ContextVar[ResolvedVariable[ValueT] | None]:
        """A fresh per-run resolution context variable; `None` means nothing is resolved yet."""
        return ContextVar('managed_variable_resolved', default=None)

    def _warn_logfire_instance_ignored(self, field_name: str) -> None:
        if self.logfire_instance is not None:
            warnings.warn(
                f'`logfire_instance` is ignored when `{field_name}` is a `Variable`; '
                'the variable already carries its own Logfire instance.',
                stacklevel=3,
            )

    def _build_managed_variable(
        self, name: str, *, prefix: str, value_type: type[ValueT], default: ValueT
    ) -> Variable[ValueT]:
        """Declare the backing variable as `<prefix><name>`, normalizing and validating the name."""
        # Strip the prefix if the user accidentally passed it so we can still apply
        # hyphen-to-underscore normalization, then re-add the prefix below.
        if name.startswith(prefix):
            warnings.warn(
                f'The {prefix!r} prefix is added automatically; pass the bare name rather than {name!r}.',
                stacklevel=3,
            )
            name = name[len(prefix) :]

        variable_name = f'{prefix}{name.replace("-", "_")}'
        if not variable_name.isidentifier():
            raise ValueError(
                f'Name {name!r} produces an invalid variable name {variable_name!r}; '
                'names may only contain letters, digits, hyphens, and underscores.'
            )

        # Construct the variable directly (rather than via `logfire.var`) so redeclaring the
        # same name is idempotent: `logfire.var` registers in a per-instance registry and raises
        # on a duplicate name, which would break sharing one variable across agents.
        instance = self.logfire_instance if self.logfire_instance is not None else logfire.DEFAULT_LOGFIRE_INSTANCE
        return Variable(variable_name, type=value_type, default=default, logfire_instance=instance)

    @property
    def resolved(self) -> ResolvedVariable[ValueT] | None:
        """The resolution for the active run, or `None` outside a run.

        Exposes the full [`ResolvedVariable`][logfire.variables.ResolvedVariable] (`value`, `label`,
        `version`, `reason`, ...) so callers can inspect which version is in play.
        """
        return self._resolved.get()

    def get_ordering(self) -> CapabilityOrdering:
        """Run outermost so the resolution's baggage envelops the whole run, including the run span."""
        return CapabilityOrdering(position='outermost', wraps=[Instrumentation])

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Resolve the variable once and keep its baggage active for the duration of the run."""
        if callable(self.targeting_key):
            targeting_key = self.targeting_key(ctx)
        else:
            targeting_key = self.targeting_key

        if callable(self.attributes):
            attributes = self.attributes(ctx)
        else:
            attributes = self.attributes

        resolved = self._variable.get(targeting_key=targeting_key, attributes=attributes, label=self.label)
        with resolved:
            token = self._resolved.set(resolved)
            try:
                return await handler()
            finally:
                self._resolved.reset(token)
