"""Shared base for capabilities backed by a Logfire managed variable.

`ManagedPrompt` and `ManagedToolDefinitions` both resolve a Logfire
[managed variable](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/) once per
run and keep its baggage active for the whole run. This base owns that shared plumbing -- the
targeting inputs, the per-run resolution context variable, `get_ordering`, and `wrap_run` -- so each
capability only declares its own variable and exposes the resolved value through its own surface.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic

import logfire
from logfire.variables import Variable, VariableAlreadyExistsError
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

# Names of variables we have already attempted to auto-create in this process, guarded by a lock.
# The contract is one attempt per process per name: we mark a name when spawning the creation thread
# (not on success), so a failed create -- e.g. a read-only token -- does not retry on every run.
_auto_create_attempted: set[str] = set()
_auto_create_lock = threading.Lock()


def _reset_auto_create_guard() -> None:  # pyright: ignore[reportUnusedFunction]
    """Clear the once-per-process auto-create guard. Intended for tests only."""
    with _auto_create_lock:
        _auto_create_attempted.clear()


def _spawn_create(variable: Variable[Any]) -> None:
    """Run the (blocking, sync-HTTP) creation off the run's thread so it never blocks or fails it.

    Isolated as a module-level function so tests can monkeypatch it to run `_create_variable`
    inline for determinism.
    """
    threading.Thread(target=_create_variable, args=(variable,), daemon=True).start()


def _create_variable(variable: Variable[Any]) -> None:
    """Create the variable in Logfire from its code default, JSON schema, and description.

    Best-effort: an already-existing variable (a race with another process or the UI) is fine, and
    any other failure is surfaced once as a warning rather than crashing the background thread.
    """
    provider = variable.logfire_instance.config.get_variable_provider()
    # Duck-type the write path: a provider without it (or without persistence) can't be created into.
    create = getattr(provider, 'create_variable', None)
    if not callable(create):  # pragma: no cover
        return
    try:
        create(variable.to_config())
    except VariableAlreadyExistsError:
        # The variable already exists server-side (another process or the UI created it first).
        pass
    except Exception as exc:
        warnings.warn(f'Failed to auto-create Logfire managed variable {variable.name!r}: {exc}')


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

    auto_create: bool = field(default=True, kw_only=True)
    """Whether to create the variable in Logfire the first time it is used but doesn't exist there yet.

    When the variable is unknown to the configured Logfire provider, it is created in the background
    with the code default as its value (plus the payload's JSON schema and description), so the
    Logfire UI becomes the editing surface without a manual create-in-UI step. Until someone
    configures a label there, resolution keeps falling back to the code default. Creation happens
    off the run's thread and never blocks or fails the run; it is attempted at most once per process
    per variable. Set to `False` to opt out."""

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
                # 1=warn, 2=_warn_logfire_instance_ignored, 3=__post_init__,
                # 4=dataclass-generated __init__, 5=user's `ManagedTool(...)` call.
                stacklevel=4,
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
                # Same chain as `_warn_logfire_instance_ignored`: helper → __post_init__ → dataclass __init__ → user.
                stacklevel=4,
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

    def _maybe_auto_create(self) -> None:
        """Kick off background creation of the backing variable, at most once per process per name."""
        name = self._variable.name
        with _auto_create_lock:
            if name in _auto_create_attempted:
                return
            # Mark before spawning: one attempt per process, so a failed create doesn't retry.
            _auto_create_attempted.add(name)
        _spawn_create(self._variable)

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
        # `'unrecognized_variable'` means a provider is configured but doesn't know this name yet --
        # the case auto-create is for. Reasons like `'no_provider'`/`'missing_config'` mean there's
        # no provider (or config) to create into, so they must not trigger. `ResolvedVariable` only
        # exposes the reason privately today (a known SDK gap we're flagging upstream).
        if self.auto_create and resolved._reason == 'unrecognized_variable':  # pyright: ignore[reportPrivateUsage]
            self._maybe_auto_create()
        with resolved:
            token = self._resolved.set(resolved)
            try:
                return await handler()
            finally:
                self._resolved.reset(token)
