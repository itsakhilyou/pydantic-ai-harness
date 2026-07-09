"""Back a managed MCP server connection with a Logfire-managed variable."""

from __future__ import annotations

from dataclasses import dataclass

from logfire.variables import Variable
from pydantic import BaseModel
from pydantic_ai.capabilities import MCP, AbstractCapability, CombinedCapability, PrefixTools
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness.logfire._managed_variable import ManagedVariableCapability

# There is no first-party Logfire "MCP management" feature reserving this prefix, so `mcp__` is a
# harness convention: it namespaces the backing managed variable and keeps it visually grouped with
# the other managed-capability variables. One variable holds the whole connection config for one
# managed MCP server.
_MCP_VARIABLE_PREFIX = 'mcp__'


class ManagedMCPValue(BaseModel):
    """The value backing a [`ManagedMCP`][pydantic_ai_harness.logfire.ManagedMCP] capability.

    Manages exactly the **connection, filtering, and framing** of an MCP server -- never executable
    code. A managed value points the agent at an MCP server (`url`), authenticates the connection
    (`authorization`/`headers`), narrows which of the server's tools the model may use (`tools`),
    namespaces them (`tool_prefix`), and frames the server for the model (`description`). The tools
    themselves run on the MCP server, so nothing runnable is ever downloaded from Logfire -- only the
    knobs that decide *which already-trusted server* the agent talks to and *how*.

    An empty value (`url` unset -- the default when nothing is configured in Logfire yet) contributes
    no MCP server at all, so the agent runs exactly as coded until a connection is published.
    """

    url: str | None = None
    """The URL of the MCP server to connect to. `None` (the default) contributes no server -- the
    agent runs with no managed MCP connection until a URL is published."""

    authorization: str | None = None
    """`Authorization` header value for MCP server requests (e.g. a bearer token). Merged into the
    request headers alongside `headers`."""

    headers: dict[str, str] | None = None
    """HTTP headers for MCP server requests."""

    tools: list[str] | None = None
    """Filter the server's tools to only these names. `None` keeps every tool the server exposes.
    Applied to the server's own tool names, before any `tool_prefix` is added."""

    tool_prefix: str | None = None
    """Prefix added to every tool name the server exposes, to namespace them against the agent's
    other tools (e.g. `'gh'` turns `'search'` into `'gh_search'`). `None` keeps the bare names."""

    description: str | None = None
    """Human-readable description of the MCP server. Framing only; does not change which tools run."""


@dataclass
class ManagedMCP(ManagedVariableCapability[AgentDepsT, ManagedMCPValue]):
    """Back a managed MCP server connection with a Logfire-managed variable.

    Drop this capability onto any agent and you can point it at an [MCP](https://ai.pydantic.dev/mcp/client/)
    server -- and steer which of that server's tools it may use, how they are namespaced, and how the
    server is framed -- from the Logfire UI, versioned, labelled, and rolled out, without redeploying.
    A name of `github` resolves the variable `mcp__github`.

    ```python
    import logfire
    from pydantic_ai import Agent

    from pydantic_ai_harness.logfire import ManagedMCP

    logfire.configure()

    agent = Agent(
        'openai:gpt-5',
        capabilities=[ManagedMCP('github', label='production')],
    )
    result = agent.run_sync('Summarize my open pull requests.')
    ```

    **Connection, not code:** the managed value
    ([`ManagedMCPValue`][pydantic_ai_harness.logfire.ManagedMCPValue]) carries only the connection
    (`url`, `authorization`, `headers`), a tool filter (`tools`), a `tool_prefix`, and a
    `description`. Nothing executable is ever downloaded from Logfire -- the server runs the tools;
    the managed value just decides *which already-trusted server* the agent connects to and *how*.
    The server is connected locally (via [`MCP`][pydantic_ai.capabilities.MCP]), so credentials,
    hooks, and tracing stay under your control.

    The value is resolved **once per run**, inside
    [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run] (earlier than the per-surface
    capabilities' `wrap_run`, because the resolved connection decides what toolset the run is
    assembled from), and the [`ResolvedVariable`][logfire.variables.ResolvedVariable] is kept open as
    a context manager for the whole run -- so the selected label and version ride as baggage on every
    child span of the agent run. The materialized [`MCP`][pydantic_ai.capabilities.MCP] capability's
    toolset and hooks then flow through the run exactly as if you had listed it in code.

    **Fallback semantics:** with no connection published (or when the remote value can't be
    validated), the logfire SDK falls back to the code default -- an empty
    [`ManagedMCPValue`][pydantic_ai_harness.logfire.ManagedMCPValue] with no `url`, which contributes
    no MCP server -- so the run degrades to exactly the agent the developer wrote, never a crashed
    run. Connecting an MCP server locally requires the `mcp` extra (`pip install
    "pydantic-ai-slim[mcp]"`); the error surfaces only once a `url` is actually published.

    Pass an existing [`logfire.variables.Variable`][logfire.variables.Variable] as `name` instead of
    an MCP name when you want to use a variable you defined yourself.
    """

    name: str | Variable[ManagedMCPValue] | None = None
    """The managed MCP name (declared as the variable `mcp__<name>`), or a pre-built
    `logfire.Variable`. When omitted, the variable is derived from the agent's own `name` at run time
    (`mcp__<agent name>`); the agent must then have a `name`."""

    default: ManagedMCPValue | None = None
    """Code-default connection. When omitted, an empty `ManagedMCPValue()` (no `url`) is used --
    nothing is managed until a connection is configured in Logfire. Ignored when `name` is a
    `Variable`."""

    def __post_init__(self) -> None:
        self._setup_variable(
            self.name,
            prefix=_MCP_VARIABLE_PREFIX,
            value_type=ManagedMCPValue,
            default=self.default or ManagedMCPValue(),
        )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Resolve the managed connection and assemble the per-run capability from it.

        Resolution happens here (not in `wrap_run`) because the connection decides what toolset the
        run is built from: the [`MCP`][pydantic_ai.capabilities.MCP] capability is materialized now so
        its toolset and hooks are in place before the framework extracts them. The returned
        `CombinedCapability` carries the base's baggage holder alongside the materialized MCP
        capability, so both flow through the run as siblings of the agent's own capabilities. When no
        `url` is published, only the baggage holder is returned -- the run gains no MCP server.
        """
        resolved = self._resolve(ctx)
        children = await self._materialize_mcp(resolved.value, ctx)
        return CombinedCapability([self._resolved_holder(resolved), *children])

    async def _materialize_mcp(
        self, value: ManagedMCPValue, ctx: RunContext[AgentDepsT]
    ) -> list[AbstractCapability[AgentDepsT]]:
        """Build the [`MCP`][pydantic_ai.capabilities.MCP] capability for the resolved connection.

        Returns an empty list when no `url` is published (nothing to connect to). A `tool_prefix`
        wraps the MCP capability in [`PrefixTools`][pydantic_ai.capabilities.PrefixTools] so the
        server's tools are namespaced. The materialized capability is `for_run`-resolved before being
        returned, mirroring how the framework prepares capabilities.
        """
        if value.url is None:
            return []

        mcp: AbstractCapability[AgentDepsT] = MCP[AgentDepsT](
            url=value.url,
            authorization_token=value.authorization,
            headers=value.headers,
            allowed_tools=value.tools,
            description=value.description,
        )
        if value.tool_prefix:
            mcp = PrefixTools(wrapped=mcp, prefix=value.tool_prefix)
        return [await mcp.for_run(ctx)]
