"""Entry points for serving a Pydantic AI agent over ACP."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Literal

import acp
from acp import schema
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.models import KnownModelName
from pydantic_ai.output import OutputDataT
from pydantic_ai.tools import AgentDepsT

from ._adapter import DEFAULT_VERSION, PydanticAIACPAgent
from ._permission import PermissionPolicy
from ._present import ToolCallPresenter
from ._session import SessionConfigFunc
from ._store import SessionStore


async def run_acp_stdio(
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    deps: AgentDepsT = None,
    name: str | None = None,
    version: str = DEFAULT_VERSION,
    session_config: SessionConfigFunc[AgentDepsT] | None = None,
    permission_policy: PermissionPolicy | None = None,
    prompt_capabilities: schema.PromptCapabilities | None = None,
    tool_presenter: ToolCallPresenter | None = None,
    session_store: SessionStore | None = None,
    models: Sequence[KnownModelName | str] | Literal['all'] | None = None,
) -> None:
    """Serve `agent` as an ACP agent over stdin/stdout until the client disconnects.

    This is the entry point an ACP client (such as an editor or terminal UI) launches as a
    subprocess. It blocks for the lifetime of the connection.

    Args:
        agent: The Pydantic AI agent to expose over ACP.
        deps: Dependencies passed to every agent run.
        name: Name advertised to the client. Defaults to the agent's name, then `'pydantic-ai-agent'`.
        version: Version advertised to the client.
        session_config: Per-session factory deriving deps/toolsets from the client's workspace setup.
            See [`PydanticAIACPAgent`][pydantic_ai_harness.acp.PydanticAIACPAgent].
        permission_policy: Scopes how "always allow"/"always reject" decisions are remembered.
            See [`PydanticAIACPAgent`][pydantic_ai_harness.acp.PydanticAIACPAgent].
        prompt_capabilities: Prompt content types the agent advertises support for.
            See [`PydanticAIACPAgent`][pydantic_ai_harness.acp.PydanticAIACPAgent].
        tool_presenter: Maps tool calls to rich ACP presentation (kind, file locations, diffs).
            See [`PydanticAIACPAgent`][pydantic_ai_harness.acp.PydanticAIACPAgent].
        session_store: Enables `session/load` by persisting each session. See
            [`PydanticAIACPAgent`][pydantic_ai_harness.acp.PydanticAIACPAgent].
        models: Models the client may switch between with `session/set_model`. See
            [`PydanticAIACPAgent`][pydantic_ai_harness.acp.PydanticAIACPAgent].
    """
    adapter = PydanticAIACPAgent(
        agent,
        deps=deps,
        name=name,
        version=version,
        session_config=session_config,
        permission_policy=permission_policy,
        prompt_capabilities=prompt_capabilities,
        tool_presenter=tool_presenter,
        session_store=session_store,
        models=models,
    )
    # `session/set_model` and `session/close` are still UNSTABLE in the ACP SDK, and the SDK's
    # router rejects unstable methods with `method_not_found` unless this flag is set -- so without
    # it, the model picker and session-close affordances we advertise at `initialize` are dead over
    # the wire even though their handlers exist. Keep enabled until those methods stabilize.
    await acp.run_agent(adapter, use_unstable_protocol=True)


def run_acp_stdio_sync(
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    deps: AgentDepsT = None,
    name: str | None = None,
    version: str = DEFAULT_VERSION,
    session_config: SessionConfigFunc[AgentDepsT] | None = None,
    permission_policy: PermissionPolicy | None = None,
    prompt_capabilities: schema.PromptCapabilities | None = None,
    tool_presenter: ToolCallPresenter | None = None,
    session_store: SessionStore | None = None,
    models: Sequence[KnownModelName | str] | Literal['all'] | None = None,
) -> None:
    """Synchronous wrapper around [`run_acp_stdio`][pydantic_ai_harness.acp.run_acp_stdio].

    Convenient as the `main()` of an ACP agent script, which clients launch as a subprocess.
    """
    asyncio.run(
        run_acp_stdio(
            agent,
            deps=deps,
            name=name,
            version=version,
            session_config=session_config,
            permission_policy=permission_policy,
            prompt_capabilities=prompt_capabilities,
            tool_presenter=tool_presenter,
            session_store=session_store,
            models=models,
        )
    )
