# GitHub

Give an agent the official [GitHub MCP server](https://github.com/github/github-mcp-server),
run as a Docker subprocess over stdio, with fine-grained control over which tools
the model can see.

## The problem

GitHub ships a capable MCP server, but it exposes a large surface — dozens of
tools across repositories, issues, pull requests, actions, code security, and
more. Handing all of that to a model is wasteful (every tool description costs
tokens) and risky (a coding agent rarely needs to delete repositories). Wiring
the server up also means assembling a `docker run` command, forwarding a token
into the container, and filtering tools — boilerplate every project reinvents.

## The solution

`GitHub` spawns the `ghcr.io/github/github-mcp-server` container and exposes its
tools to the agent, with two layers of limiting. Both keep tools out of the
model's view (and its token budget); they differ in granularity:

- **server-side** — the server is told to only enable whole toolset groups, so it
  never advertises the rest. This is the GitHub MCP server's own scoping
  mechanism (`GITHUB_TOOLSETS`, `GITHUB_READ_ONLY`, `GITHUB_DYNAMIC_TOOLSETS`).
- **client-side** — the advertised tools are filtered by name before reaching the
  model, for per-tool control on top of the enabled groups.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import GitHub

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        GitHub(toolsets=['repos', 'issues', 'pull_requests'], read_only=True),
    ],
)

result = agent.run_sync('Summarize the open issues labelled "bug" in pydantic/pydantic-ai.')
print(result.output)
```

Install the extra and make sure Docker is available:

```bash
pip install "pydantic-ai-harness[github]"
```

## Authentication

The token is resolved in this order:

1. the `token=` argument
2. the `GITHUB_PERSONAL_ACCESS_TOKEN` environment variable
3. the `GITHUB_TOKEN` environment variable

If none is found, constructing the toolset raises a `UserError`. The token is
passed into the server process and forwarded into the container with `-e`; it is
never written into the `docker run` arguments.

## Limiting tools

### Server-side

| Field | Effect |
|---|---|
| `toolsets` | Toolset groups to enable, e.g. `['repos', 'issues', 'pull_requests']`. `None` uses the server default; `['all']` enables everything. Maps to `GITHUB_TOOLSETS`. |
| `read_only` | Disable every state-mutating tool (`GITHUB_READ_ONLY`). |
| `dynamic_toolsets` | Start with only discovery tools and let the model enable toolsets on demand (`GITHUB_DYNAMIC_TOOLSETS`), instead of listing them all up front. |

Group names (`repos`, `issues`, `pull_requests`, `actions`, `code_security`,
`notifications`, …) are defined by the
[GitHub MCP server](https://github.com/github/github-mcp-server#tool-configuration),
not by this capability — `GitHub` forwards them untouched, so it stays decoupled
from the server version. These options map to the server's own environment
variables; the capability sets them in the container's environment.

### Client-side (per-tool control)

| Field | Effect |
|---|---|
| `allowed_tools` | If set, only tools whose name is listed are exposed. |
| `denied_tools` | Tools whose name is listed are hidden. Combines with `allowed_tools` (a tool must be allowed *and* not denied). |
| `tool_filter` | A `(ctx, tool_def) -> bool` predicate (sync or async) for anything the lists can't express, applied last. |

Names match the server's own tool names (e.g. `get_issue`, `create_pull_request`)
and are matched **before** any `tool_prefix` is applied, so the lists stay stable
regardless of prefix.

```python
GitHub(
    toolsets=['issues', 'pull_requests'],
    denied_tools=['delete_pull_request_review'],
    tool_prefix='gh',  # tools surface to the model as gh_get_issue, gh_create_issue, ...
)
```

## How it composes

`get_toolset` returns the MCP server wrapped as
`PrefixedToolset(FilteredToolset(MCPServerStdio))` (each layer only added when
needed), so it behaves like any other Pydantic AI toolset: it works alongside
other capabilities, inside `ToolSearch`, and within
[`CodeMode`](../code_mode/README.md). The container starts when the agent run
begins and is torn down when it ends.

## Configuration

```python
GitHub(
    token=None,              # PAT; falls back to GITHUB_PERSONAL_ACCESS_TOKEN / GITHUB_TOKEN
    toolsets=None,           # server-side toolset groups (None = server default)
    read_only=False,         # disable mutating tools
    dynamic_toolsets=False,  # discover toolsets on demand
    allowed_tools=None,      # client-side allowlist (unprefixed names)
    denied_tools=None,       # client-side denylist (unprefixed names)
    tool_filter=None,        # (ctx, tool_def) -> bool predicate
    tool_prefix=None,        # prefix added to every tool name
    host=None,               # GitHub Enterprise host (GITHUB_HOST)
    docker_image='ghcr.io/github/github-mcp-server',
    docker_command='docker', # set to 'podman' for Podman
    docker_args=(),          # extra docker run args (e.g. -v mounts)
    env=None,                # extra env set in the process and forwarded into the container
    init_timeout=30.0,       # seconds to wait for startup (a cold run pulls the image)
    read_timeout=300.0,      # seconds before an idle connection is considered stale
    include_instructions=True,
    id='github',             # stable id for durable execution backends
)
```

## Agent spec (YAML/JSON)

`GitHub` works with Pydantic AI's
[agent spec](https://ai.pydantic.dev/agent-spec/):

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - GitHub:
      toolsets: ['repos', 'issues', 'pull_requests']
      read_only: true
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import GitHub

agent = Agent.from_file('agent.yaml', custom_capability_types=[GitHub])
```

## Safety

`read_only` and the deny list are real reductions of the tool surface, but the
GitHub token still grants whatever its scopes allow. Scope the token to the
repositories and permissions the agent actually needs, and prefer `read_only`
plus a narrow `toolsets` list as the baseline. The server runs in a container,
which provides process isolation but shares your network and the mounted token.

## Further reading

- [GitHub MCP server](https://github.com/github/github-mcp-server)
- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Toolsets](https://ai.pydantic.dev/toolsets/)
