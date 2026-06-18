# Pydantic AI Harness

[![CI](https://github.com/pydantic/pydantic-ai-harness/actions/workflows/main.yml/badge.svg?event=push)](https://github.com/pydantic/pydantic-ai-harness/actions/workflows/main.yml?query=branch%3Amain)
[![PyPI](https://img.shields.io/pypi/v/pydantic-ai-harness.svg)](https://pypi.python.org/pypi/pydantic-ai-harness)
[![versions](https://img.shields.io/pypi/pyversions/pydantic-ai-harness.svg)](https://github.com/pydantic/pydantic-ai-harness)
[![license](https://img.shields.io/github/license/pydantic/pydantic-ai-harness.svg)](https://github.com/pydantic/pydantic-ai-harness/blob/main/LICENSE)

**The batteries for your [Pydantic AI](https://ai.pydantic.dev/) agent.**

---

Pydantic AI's [capabilities](https://ai.pydantic.dev/capabilities/) and [hooks](https://ai.pydantic.dev/hooks/) API is how you give an agent its harness -- bundles of tools, lifecycle hooks, instructions, and model settings that extend what the agent can do without any framework changes.

**Pydantic AI Harness** is the official capability library for Pydantic AI, maintained by the [Pydantic AI](https://github.com/pydantic/pydantic-ai) team. Pydantic AI core ships capabilities that require model or framework support, and capabilities fundamental to every agent -- [web search](https://ai.pydantic.dev/capabilities/#provider-adaptive-tools), [tool search](https://ai.pydantic.dev/deferred-tools/), [thinking](https://ai.pydantic.dev/capabilities/#thinking). Everything else lives here: standalone building blocks you pick and choose to turn your agent into a coding agent, a research assistant, or anything else. This is also where new capabilities start -- as they stabilize and prove themselves broadly essential, they can graduate into core.

[What's available today](#whats-available-today) lists what you can install and use right now. The [roadmap](#roadmap) tracks what's coming -- [tell us what to prioritize](#help-us-prioritize).

**Contents:** [Installation](#installation) · [Quick start](#quick-start) · [What's available today](#whats-available-today) · [Stability tiers](#stability-tiers) · [Roadmap](#roadmap) · [An ecosystem agent](#an-ecosystem-agent) · [Help us prioritize](#help-us-prioritize) · [Build your own](#build-your-own) · [Contributing](#contributing) · [Version policy](#version-policy) · [Pydantic AI references](#pydantic-ai-references) · [License](#license)

## Installation

```bash
uv add pydantic-ai-harness
```

Extras for specific capabilities:

```bash
uv add "pydantic-ai-harness[code-mode]"   # CodeMode (adds the Monty sandbox)
uv add "pydantic-ai-harness[logfire]"     # ManagedPrompt (Logfire-managed prompts)
```

The `codemode` extra (no hyphen) is also supported as an alias.

Requires Python 3.10+ and `pydantic-ai-slim>=1.105.0`.

## Quick start

```bash
uv add "pydantic-ai-slim[anthropic,mcp,duckduckgo,logfire]" "pydantic-ai-harness[code-mode]"
```

Set the API key for whichever model you use (`ANTHROPIC_API_KEY` for the example below); see the [Pydantic AI models guide](https://ai.pydantic.dev/models/) for other providers.

```bash
export ANTHROPIC_API_KEY=sk-...
```

```python
import logfire
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP, WebSearch
from pydantic_ai_harness import CodeMode

# See https://ai.pydantic.dev/logfire/ for setup details.
logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    'anthropic:claude-opus-4-7',
    capabilities=[
        # Wraps every tool into a single run_code tool, sandboxed by Monty
        # (https://github.com/pydantic/monty -- pulled in by the [code-mode] extra).
        # The model writes Python that calls multiple tools with loops, conditionals,
        # asyncio.gather, and local filtering -- one model round-trip for N tool calls.
        CodeMode(),
        # Connect to any MCP server -- here, the open-source Hacker News server
        # (https://github.com/cyanheads/hn-mcp-server). native=False forces the
        # local MCP toolset so CodeMode can wrap the tools; without it,
        # providers that natively support MCP server connectors execute the tools
        # server-side and bypass the sandbox.
        MCP('https://hn.caseyjhand.com/mcp', native=False),
        # Provider-adaptive web search; native=False routes through the local
        # DuckDuckGo fallback (the [duckduckgo] extra above) so CodeMode can batch
        # web searches alongside the HN calls in a single run_code.
        WebSearch(native=False),
    ],
)

result = agent.run_sync(
    "Across the top, best, and 'show HN' Hacker News feeds, find the most-discussed "
    "story with at least 100 points. Pull its comment thread, its submitter's profile, "
    "and any web coverage. Summarize what you find in one paragraph."
)
print(result.output)
"""
The most-discussed HN story across top/best/show clearing 100 points is "Vibe coding
and agentic engineering are getting closer than I'd like" by Simon Willison (748 points,
853 comments, on the Best feed), submitted by long-time HNer e12e. The piece argues
that the two modes Willison once kept mentally separate -- throwaway "vibe coding" and
disciplined "agentic engineering" -- are blurring, since agents like Claude Code now
reliably handle non-trivial tasks like "build a JSON API endpoint that runs a SQL query"
with tests and docs on the first pass. The HN thread is unusually substantive, with
commenters debating whether LLMs created or merely *exposed* sloppy engineering
practices and warning of a "normalization of deviance" as engineers stop reviewing diffs.
"""
```

[![Logfire trace from the Quick start run](docs/images/quick-start-trace.png)](https://logfire-us.pydantic.dev/public-trace/84bcf123-2106-49da-9f6f-5c26395339bb?spanId=7650806a0785b946)

**[See this run as a public Logfire trace →](https://logfire-us.pydantic.dev/public-trace/84bcf123-2106-49da-9f6f-5c26395339bb?spanId=7650806a0785b946)** Each `run_code` span fans out into the tool calls the model issued from inside the sandbox -- it's the easiest way to understand what code mode actually did.

### Capabilities compose

Each capability is independent; you stack the ones you need on one `Agent`. Two stable first-party capabilities, no extras or network, make a minimal coding agent that can read/edit files and run commands inside one directory:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import FileSystem, Shell

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        # Sandboxed file tools scoped to ./workspace.
        FileSystem(root_dir='./workspace'),
        # Run commands in the same directory; allowlist keeps it tight.
        Shell(cwd='./workspace', allowed_commands=['python', 'pytest', 'ls', 'cat']),
    ],
)

result = agent.run_sync('Run the tests, and if any fail, read the failing file and fix it.')
print(result.output)
```

## What's available today

These ship in the package right now. Each links to its own README with a walkthrough, a runnable example, and the full options.

### Stable

Imported from the top level (`from pydantic_ai_harness import ...`). Covered by the [version policy](#version-policy).

| Capability | What it does |
|---|---|
| [`CodeMode`](pydantic_ai_harness/code_mode/) | Wraps your tools into one sandboxed `run_code`. The model writes Python that calls many tools with loops and `asyncio.gather` -- one model round-trip for N tool calls. |
| [`FileSystem`](pydantic_ai_harness/filesystem/) | Sandboxed file tools (read/write/edit/search) scoped to one directory, with path-traversal prevention and allow/deny/protected glob patterns. |
| [`Shell`](pydantic_ai_harness/shell/) | Run commands with allow/deny lists, timeouts, output truncation, and auto-cleaned background processes. |
| [`ManagedPrompt`](pydantic_ai_harness/logfire/) | Back an agent's instructions with a [Logfire-managed prompt](https://logfire.pydantic.dev/docs/reference/advanced/prompt-management/), so you iterate, A/B test, and roll back from the UI without redeploying. |

### Experimental

Imported from `pydantic_ai_harness.experimental.*`. Working and documented, but the API may change -- see [Stability tiers](#stability-tiers).

| Capability | What it does |
|---|---|
| [`Planning`](pydantic_ai_harness/experimental/planning/) | A model-owned, self-updating task plan surfaced as an ephemeral reminder, so it never invalidates the prompt cache. |
| [`SubAgents`](pydantic_ai_harness/experimental/subagents/) | Delegate self-contained tasks to named child agents via one `delegate_task` tool, with per-delegate usage/timeout/call budgets. |
| [Compaction strategies](pydantic_ai_harness/experimental/compaction/) | Keep history within the context window: `SlidingWindow`, `ClearToolResults`, `DeduplicateFileReads`, `ClampOversizedMessages`, `SummarizingCompaction`, `LimitWarner`, and the tiered `TieredCompaction` default. |
| [`OverflowingToolOutput`](pydantic_ai_harness/experimental/overflow/) | Move an oversized tool return *out* of the window at production time -- truncate, spill to a queryable file, or summarize -- so it isn't re-sent every turn. |
| [`StepPersistence`](pydantic_ai_harness/experimental/step_persistence/) | An append-only step-event log plus provider-valid continuable snapshots, for resuming or forking a run. |
| [Media stores](pydantic_ai_harness/experimental/media/) | Content-addressed byte stores (`DiskMediaStore`, `SqliteMediaStore`, `S3MediaStore`) for moving large binary parts out of message history. |

## Stability tiers

The harness is one package with two import surfaces:

- **Stable** -- top-level (`from pydantic_ai_harness import CodeMode`). Part of the public API and subject to the [version policy](#version-policy) below.
- **Experimental** -- under `pydantic_ai_harness.experimental` (`from pydantic_ai_harness.experimental.planning import Planning`). May change or be removed in any release, without a deprecation period. A capability graduates to stable once its API settles.

Importing any experimental capability emits a `HarnessExperimentalWarning`. Silence the whole category with one filter:

```python
import warnings
from pydantic_ai_harness.experimental import HarnessExperimentalWarning

warnings.filterwarnings('ignore', category=HarnessExperimentalWarning)
```

## Roadmap

We studied leading coding agents, agent frameworks, and Claw-style assistants to map every capability area that matters for production agents. The table below is what we have **not** shipped yet; for what exists today, see [What's available today](#whats-available-today). Each row is tracked as an [issue](https://github.com/pydantic/pydantic-ai-harness/issues) in this repo.

**Vote on whatever is linked in the Status column** -- PRs if we're actively building it, issues if it's planned -- to help us decide what to work on next.

| Category | Capability | Description | Status | Community&nbsp;alternatives |
|---|---|---|---|---|
| **Tools &&nbsp;execution** | **Repo context injection** | Auto-load CLAUDE.md/AGENTS.md and repo structure | :construction: [PR&nbsp;#175](https://github.com/pydantic/pydantic-ai-harness/pull/175) | [pydantic-deep](https://github.com/vstorm-co/pydantic-deepagents) (vstorm&#8209;co) |
| | **Verification loop** | Run tests after edits, auto-fix failures | :construction: [PR&nbsp;#169](https://github.com/pydantic/pydantic-ai-harness/pull/169) | |
| **Context management** | **System reminders** | Inject periodic reminders to counteract instruction drift | :construction: [PR&nbsp;#181](https://github.com/pydantic/pydantic-ai-harness/pull/181) | |
| **Memory &&nbsp;persistence** | **Memory** | Persistent key-value memory across sessions | :construction: [PR&nbsp;#179](https://github.com/pydantic/pydantic-ai-harness/pull/179) | [pydantic-deep](https://github.com/vstorm-co/pydantic-deepagents) (vstorm&#8209;co) |
| | **Checkpointing** | Full graph-state save, rewind, and fork (beyond [`StepPersistence`](pydantic_ai_harness/experimental/step_persistence/)'s message-level snapshots) | :memo: [#196](https://github.com/pydantic/pydantic-ai-harness/issues/196) | [pydantic-deep](https://github.com/vstorm-co/pydantic-deepagents) (vstorm&#8209;co) |
| **Agent orchestration** | **Skills** | Progressive tool loading -- search, activate, deactivate | :construction: [PR&nbsp;#183](https://github.com/pydantic/pydantic-ai-harness/pull/183) | [pydantic-ai-skills](https://github.com/DougTrajano/pydantic-ai-skills) (DougTrajano), [pydantic-deep](https://github.com/vstorm-co/pydantic-deepagents) (vstorm&#8209;co) |
| | **Task tracking** | Track tasks, subtasks, and dependencies | :memo: [#65](https://github.com/pydantic/pydantic-ai-harness/issues/65) | [pydantic-ai-todo](https://github.com/vstorm-co/pydantic-ai-todo) (vstorm&#8209;co) |
| | **Teams** | Multi-agent teams with shared state and message bus | :memo: [#195](https://github.com/pydantic/pydantic-ai-harness/issues/195) | [pydantic-deep](https://github.com/vstorm-co/pydantic-deepagents) (vstorm&#8209;co) |
| **Safety &&nbsp;guardrails** | **Input guardrails** | Validate user input before the agent run starts | :construction: [PR&nbsp;#182](https://github.com/pydantic/pydantic-ai-harness/pull/182) | [pydantic-ai-shields](https://github.com/vstorm-co/pydantic-ai-shields) (vstorm&#8209;co) |
| | **Output guardrails** | Validate model output after the run completes | :construction: [PR&nbsp;#182](https://github.com/pydantic/pydantic-ai-harness/pull/182) | [pydantic-ai-shields](https://github.com/vstorm-co/pydantic-ai-shields) (vstorm&#8209;co) |
| | **Cost/token budgets** | Enforce token and cost limits per run | :construction: [PR&nbsp;#182](https://github.com/pydantic/pydantic-ai-harness/pull/182) | [pydantic-ai-shields](https://github.com/vstorm-co/pydantic-ai-shields) (vstorm&#8209;co) |
| | **Tool access control** | Block tools or require approval before execution | :construction: [PR&nbsp;#182](https://github.com/pydantic/pydantic-ai-harness/pull/182) | [pydantic-ai-shields](https://github.com/vstorm-co/pydantic-ai-shields) (vstorm&#8209;co) |
| | **Async guardrails** | Run validation concurrently with model requests | :construction: [PR&nbsp;#182](https://github.com/pydantic/pydantic-ai-harness/pull/182) | [pydantic-ai-shields](https://github.com/vstorm-co/pydantic-ai-shields) (vstorm&#8209;co) |
| | **Secret masking** | Detect and redact secrets in agent I/O | :construction: [PR&nbsp;#172](https://github.com/pydantic/pydantic-ai-harness/pull/172) | [pydantic-ai-shields](https://github.com/vstorm-co/pydantic-ai-shields) (vstorm&#8209;co) |
| | **Approval workflows** | Require human approval for sensitive operations | :construction: [PR&nbsp;#173](https://github.com/pydantic/pydantic-ai-harness/pull/173) | [Pydantic&nbsp;AI](https://ai.pydantic.dev/deferred-tools/#human-in-the-loop-tool-approval) (built&#8209;in) |
| | **Tool budget** | Limit total tool calls or cost per run | :construction: [PR&nbsp;#168](https://github.com/pydantic/pydantic-ai-harness/pull/168) | |
| **Reliability** | **Stuck loop detection** | Detect and break out of repetitive agent loops | :construction: [PR&nbsp;#186](https://github.com/pydantic/pydantic-ai-harness/pull/186) | |
| | **Tool error recovery** | Retry failed tool calls with backoff and budget | :construction: [PR&nbsp;#171](https://github.com/pydantic/pydantic-ai-harness/pull/171) | |
| | **Tool orphan repair** | Fix orphaned tool calls in conversation history | :construction: [PR&nbsp;#184](https://github.com/pydantic/pydantic-ai-harness/pull/184) | |
| **Reasoning** | **Adaptive reasoning** | Adjust thinking effort based on task complexity | :construction: [PR&nbsp;#174](https://github.com/pydantic/pydantic-ai-harness/pull/174) | |
| | **Current time** | Inject current date/time into system prompt | :construction: [PR&nbsp;#170](https://github.com/pydantic/pydantic-ai-harness/pull/170) | |

> Packages by [vstorm-co](https://github.com/vstorm-co) are endorsed by the Pydantic AI team. We're working with them to upstream some of their implementations into this repo.

## An ecosystem agent

The Quick start above is deliberately small. Here's the other end of the spectrum -- an agent wired up with capabilities drawn from across the Pydantic AI ecosystem: this repo, core `pydantic-ai`, and the community packages we vouch for in the [roadmap](#roadmap) above.

```python
import logfire
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP, Thinking, ToolSearch, WebSearch
from pydantic_ai_harness import CodeMode

# Community packages, alphabetical:
from pydantic_ai_backends import ConsoleCapability
from pydantic_ai_shields import CostTracking, InputGuard, SecretRedaction, ToolGuard
from pydantic_ai_skills import SkillsCapability
from pydantic_ai_summarization import ContextManagerCapability
from pydantic_ai_todo import TodoCapability
from pydantic_deep import MemoryCapability, StuckLoopDetection
from subagents_pydantic_ai import SubAgentCapability, SubAgentConfig

# See https://ai.pydantic.dev/logfire/ for setup details.
logfire.configure()
logfire.instrument_pydantic_ai()

agent = Agent(
    'anthropic:claude-opus-4-7',
    capabilities=[
        # --- Tool execution & discovery ---
        # Wraps every tool into a single run_code, sandboxed by Monty.
        CodeMode(),

        # Progressive tool discovery for large tool sets; discovered tools fold into run_code.
        ToolSearch(),

        # --- Reasoning ---
        # Provider-adaptive thinking; uses native extended thinking on supporting models.
        Thinking(effort='xhigh'),

        # --- Context management ---
        # Sliding window + LLM compaction. By @vstorm-co:
        # https://github.com/vstorm-co/summarization-pydantic-ai
        # Pydantic AI also ships `AnthropicCompaction` and `OpenAICompaction` for
        # provider-native compaction.
        ContextManagerCapability(max_tokens=180_000),

        # --- Tools ---
        # Connect to any MCP server -- here, the open-source Hacker News server
        # (https://github.com/cyanheads/hn-mcp-server).
        MCP('https://hn.caseyjhand.com/mcp'),

        # Provider-adaptive web search; falls back to a local DuckDuckGo implementation.
        WebSearch(),

        # Filesystem + shell. By @vstorm-co: https://github.com/vstorm-co/pydantic-ai-backend
        ConsoleCapability(),

        # --- Memory & persistence ---
        # Persistent ./MEMORY.md per agent name. By @vstorm-co:
        # https://github.com/vstorm-co/pydantic-deepagents
        MemoryCapability(agent_name='harness-example'),

        # --- Orchestration ---
        # Agent skills (Anthropic's spec) by @DougTrajano:
        # https://github.com/DougTrajano/pydantic-ai-skills
        # @vstorm-co's pydantic-deep also offers skills loading; the two have different
        # spec footprints (Doug's is closer to programmatic skills).
        SkillsCapability(directories=['./skills']),

        # Spawn sub-agents with their own toolsets and instructions. By @vstorm-co:
        # https://github.com/vstorm-co/subagents-pydantic-ai
        SubAgentCapability(subagents=[
            SubAgentConfig(
                name='researcher',
                description='Deep research on a topic',
                instructions='You are a thorough research assistant.',
            ),
        ]),

        # Track tasks and subtasks; in-memory by default, AsyncPostgresStorage available.
        # By @vstorm-co: https://github.com/vstorm-co/pydantic-ai-todo
        TodoCapability(enable_subtasks=True),

        # --- Safety & reliability ---
        # The next four are by @vstorm-co: https://github.com/vstorm-co/pydantic-ai-shields
        # Per-run cost cap with a callback hook.
        CostTracking(budget_usd=5.0),

        # Reject prompts that look like prompt-injection attempts.
        InputGuard(guard=lambda p: 'ignore previous instructions' not in p.lower()),

        # Block or require approval per tool name.
        ToolGuard(blocked=['rm'], require_approval=['write_file']),

        # Detect API keys/tokens in tool I/O and redact before they reach the model.
        SecretRedaction(),

        # Bail out if the agent gets stuck calling the same tools in a loop.
        # By @vstorm-co: https://github.com/vstorm-co/pydantic-deepagents
        StuckLoopDetection(),
    ],
)
```

This snippet is illustrative, not literally copy-pasteable: a few capabilities have setup requirements (a `./skills` directory, a Postgres database for `TodoCapability`'s persistent storage), and the community packages move independently of this one. [What's available today](#whats-available-today) tracks what has shipped first-party, and the [roadmap](#roadmap) tracks the rest. As the harness ships first-party versions, the imports above will collapse onto fewer packages -- but the example will keep working, since the API surface is the same.

## Help us prioritize

**Vote on whatever is linked in the Status column above.** If there's a PR, vote on the PR -- it means we're actively building it. If there's only an issue, vote on the issue.

Want something that's not on the list? [Open a capability request](https://github.com/pydantic/pydantic-ai-harness/issues/new?template=capability-request.yml).

## Build your own

[Capabilities](https://ai.pydantic.dev/capabilities/#building-custom-capabilities) are the primary extension point for Pydantic AI. Any of the existing capabilities in this repo can serve as a reference for building your own -- [`CodeMode`](pydantic_ai_harness/code_mode/) is the exemplar.

**Contributing a capability to this repo?** The [`agent_docs/`](agent_docs/index.md) guides cover capability authoring, the harness/core boundary, testing, and the review checklist.

**Publishing as a standalone package?** Use the `pydantic-ai-<name>` naming convention. See [Publishing capability packages](https://ai.pydantic.dev/extensibility/#publishing-capability-packages).

## Contributing

We welcome capability contributions. Here's how:

1. **Start with an issue.** [Open a capability request](https://github.com/pydantic/pydantic-ai-harness/issues/new?template=capability-request.yml) describing the behavior you want. This lets us discuss the approach and priority before code is written -- we can close an approach without closing the problem.
2. **Then open a PR.** Once the issue exists, you're welcome to open a PR with an implementation. Link the issue in your PR. We review based on community interest -- upvotes on both the issue and PR count.
3. **Don't chase green CI.** Get the approach working, then let us know. We'll take it from there -- we may push to your branch, rewrite, or open a follow-up PR. You'll be credited as the original author. (See the [Pydantic AI contributing guide](https://github.com/pydantic/pydantic-ai/blob/main/CONTRIBUTING.md).)

> **Note**: PRs that modify `pyproject.toml` or `uv.lock` from non-team members are auto-closed by CI to prevent supply chain risk. If you need a new dependency, [open an issue](https://github.com/pydantic/pydantic-ai-harness/issues/new).

### Development

```bash
make install   # install dependencies
make format    # ruff format
make lint      # ruff check
make typecheck # pyright strict
make test      # pytest
make testcov   # pytest with 100% branch coverage
```

## Version policy

Pydantic AI Harness uses **0.x versioning** to signal that APIs are still stabilizing. During 0.x:

- **Minor releases** (0.1 → 0.2) may include breaking changes -- renamed parameters, changed defaults, restructured APIs. As the library grows, especially as capabilities gain provider-native support (starting as a local implementation, then auto-switching to the provider's built-in API when available), we may need to reshape APIs we couldn't fully anticipate in the initial design.
- **Patch releases** (0.1.0 → 0.1.1) will not intentionally break existing behavior.
- **All breaking changes** are documented in release notes with migration guidance.
- Where practical, we'll keep the previous behavior available under a deprecated name or configuration option before removing it.

This is why Pydantic AI Harness is a separate package from [Pydantic AI](https://github.com/pydantic/pydantic-ai), which has a [stricter version policy](https://ai.pydantic.dev/version-policy/). As the core capabilities stabilize, we'll move toward 1.0 with stability guarantees to match.

## Pydantic AI references

- [Capabilities](https://ai.pydantic.dev/capabilities/) -- what capabilities are, built-in capabilities, building your own
- [Hooks](https://ai.pydantic.dev/hooks/) -- lifecycle hooks reference, ordering, error handling
- [Extensibility](https://ai.pydantic.dev/extensibility/) -- publishing packages, third-party ecosystem
- [Toolsets](https://ai.pydantic.dev/toolsets/) -- building tools for capabilities
- [API reference](https://ai.pydantic.dev/api/capabilities/) -- full API docs

## License

MIT -- see [LICENSE](LICENSE).
