"""Capability that exposes the execution environment to the agent."""

import os
from dataclasses import dataclass, field
from typing import Literal, assert_never

from pydantic_ai import FunctionToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from ..environments.abstract import AbstractEnvironment
from ..environments.docker import DockerEnvironment
from ..environments.local import LocalEnvironment
from ._toolset import build_toolset


@dataclass
class ExecutionEnv(AbstractCapability[AgentDepsT]):
    """Capability that exposes the execution environment to the agent.

    Defaults to running against your current working directory via `LocalEnvironment`,
    so `ExecutionEnv()` "just works" for the common case. Pass `'docker'` for the
    skeleton Docker backend (Slice 5; not yet usable), or an `AbstractEnvironment`
    instance to configure the backend (e.g. `LocalEnvironment(root=...)`,
    `DockerEnvironment(image=...)`, or a custom backend).

    Bounds are applied at this presentation layer, not in the backend: `read_file`
    fetches the whole file then windows/truncates it, and `ls` fetches the whole
    listing then caps it. On a remote backend this ships bytes/entries over the wire
    only to discard the tail -- a real cost we accept for now rather than push limits
    into the backend contract, which would grow the surface area every backend must
    implement correctly. Keeping every tool consistent here is the deliberate trade-off;
    revisit it for all of them together if a remote backend's cost says otherwise.

    The tools themselves live in `_toolset.py`; this class owns the generic `AgentDepsT`
    and the environment, and delegates tool construction to `build_toolset`.
    """

    environment: AbstractEnvironment | Literal['local', 'docker'] = 'local'
    """User-facing backend selector: a string shorthand for the sensible default of a
    backend flavor (autocomplete-discoverable), or an `AbstractEnvironment` instance for
    a configured backend. `__post_init__` normalizes this into `_environment`."""

    _environment: AbstractEnvironment = field(init=False)
    """Normalized backend used by `get_toolset`. Single, internal type; resolved once in
    `__post_init__`. Note: `__post_init__` only *selects* the backend -- async startup
    (Docker container, remote connect) belongs in a run-lifecycle hook, not here."""

    def __post_init__(self) -> None:
        """Normalize `environment` into `_environment` -- a single internal type."""
        if isinstance(self.environment, AbstractEnvironment):
            self._environment = self.environment
            return
        # `self.environment` is a Literal at this point; exhaustive match with assert_never
        # guarantees a new arm added to the Literal forces a corresponding branch here.
        if self.environment == 'local':
            self._environment = LocalEnvironment(root=os.getcwd())
        elif self.environment == 'docker':
            self._environment = DockerEnvironment()
        else:
            assert_never(self.environment)

    def get_toolset(self) -> FunctionToolset[AgentDepsT]:
        """Build the toolset bound to the resolved environment."""
        return build_toolset(self._environment, FunctionToolset[AgentDepsT]())
