"""Capability that exposes the execution environment to the agent."""

import os
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import FunctionToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import WrapRunHandler
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext

from ..environments.abstract import AbstractEnvironment
from ..environments.local import LocalEnvironment
from ._toolset import build_toolset


def _default_environment() -> AbstractEnvironment:
    """Return a `LocalEnvironment` rooted at the current working directory.

    Resolved at instance creation (via `default_factory`), not at class definition,
    so `os.getcwd()` reflects where the user actually constructed the capability.
    Lifted out of the class body to keep the field declaration single-line.
    """
    return LocalEnvironment(root=os.getcwd())


@dataclass
class ExecutionEnv(AbstractCapability[AgentDepsT]):
    """Capability that exposes the execution environment to the agent.

    Defaults to running against your current working directory via `LocalEnvironment`,
    so `ExecutionEnv()` "just works" for the common case. Pass an `AbstractEnvironment`
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

    environment: AbstractEnvironment = field(default_factory=_default_environment)
    """Backend to delegate to. Defaults to a `LocalEnvironment` rooted at the current
    working directory; pass a configured instance to use a different root or backend
    (e.g. `DockerEnvironment(image=...)`).

    No string shorthand: any choice complex enough to be worth naming (Docker image,
    custom root) needs configuration the string can't carry, and a single-arm
    `Literal['local']` would be cosmetic. The default factory carries the only sensible
    no-arg behavior; everything else is an instance.
    """

    def get_toolset(self) -> FunctionToolset[AgentDepsT]:
        """Build the toolset bound to the environment."""
        return build_toolset(self.environment, FunctionToolset[AgentDepsT]())

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Manage the environment's lifecycle around an agent run.

        Captures ownership at entry: if the environment was already started by an outer scope
        (e.g. the user wrapped the env in `async with` to share it across many runs / agents),
        this run does not start or stop it. Otherwise, this run owns the lifecycle and pairs
        a `start()` with a `stop()` even if the run raises or is cancelled.
        """
        # Capture once, honor at exit: if an outer scope flips `_started` between entry and
        # finally we must not double-start or double-stop. Same pattern as a re-entrant lock.
        owns = not self.environment._started  # pyright: ignore[reportPrivateUsage]
        if owns:
            await self.environment.start()
        try:
            return await handler()
        finally:
            if owns:
                await self.environment.stop()
