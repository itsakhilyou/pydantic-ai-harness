# Pydantic Harness

Composable, reusable capabilities for [Pydantic AI](https://ai.pydantic.dev/) agents.

## What is it?

Pydantic Harness provides a library of **capabilities** -- self-contained bundles of system prompts, tools, and lifecycle hooks -- that you can attach to any Pydantic AI agent to give it new powers without writing boilerplate.

Each capability is an [`AbstractCapability`](https://ai.pydantic.dev/capabilities/) subclass that plugs into the agent loop via Pydantic AI's capabilities API.

## Installation

```bash
pip install pydantic-harness
```

Requires Python 3.10+ and `pydantic-ai-slim>=1.78.0`.

## Quick start

```python
from pydantic_ai import Agent
from pydantic_harness import Memory, Skills, Compaction

agent = Agent(
    'openai:gpt-4o',
    capabilities=[Memory(), Skills(), Compaction()],
)

result = agent.run_sync('Remember that my favourite colour is blue.')
```

## Available capabilities

| Capability | Description |
|---|---|
| AdaptiveReasoning | Dynamically adjust reasoning effort based on task complexity |
| Approval | Require human approval before executing sensitive operations |
| Compaction | Compress conversation history to stay within context limits |
| FileSystem | Read, write, and navigate the local filesystem |
| Guardrails | Validate inputs/outputs and enforce cost and tool constraints |
| KnowsCurrentTime | Inject the current date and time into the system prompt |
| Memory | Persistent key-value memory across agent sessions |
| Planning | Break complex tasks into plans before execution |
| RepoContextInjection | Inject repository structure and context into the system prompt |
| SecretMasking | Detect and redact secrets in agent inputs and outputs |
| SessionPersistence | Save and restore full conversation sessions |
| Shell | Execute shell commands with safety controls |
| Skills | Progressive tool loading via search and activate |
| SlidingWindow | Keep conversation history within a sliding token window |
| StuckLoopDetection | Detect and break out of repetitive agent loops |
| SubAgent | Delegate subtasks to specialised child agents |
| SystemReminders | Inject periodic reminders into the conversation |
| ToolErrorRecovery | Automatically retry or recover from tool execution errors |
| ToolOrphanRepair | Repair orphaned tool calls in conversation history |
| ToolOutputManagement | Control and format tool output for the model |

## Code Mode

The `CodeMode` capability replaces direct tool calls with a single `run_code` tool. Instead of calling tools one at a time, the model writes Python code that calls them as async functions inside a sandboxed [Monty](https://github.com/pydantic/monty) runtime. This lets the model chain multiple tool calls, use control flow (loops, conditionals), and post-process results -- all in a single round-trip.

See also: [Tool use is also code generation](https://www.anthropic.com/engineering/tool-use-is-also-code-generation) (Anthropic) and [How we built our AI agent's tool use pipeline](https://blog.cloudflare.com/how-we-built-our-ai-agents-tool-use-pipeline/) (Cloudflare).

### Basic usage

```python
from pydantic_ai import Agent
from pydantic_harness import CodeMode

agent = Agent('anthropic:claude-sonnet-4-20250514', capabilities=[CodeMode()])

@agent.tool_plain
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@agent.tool_plain
def greet(name: str) -> str:
    """Greet someone."""
    return f'Hello, {name}!'

result = agent.run_sync('Add 2 and 3, then greet the result')
```

The model sees a single `run_code` tool whose description includes the signatures of all available tools as async Python functions. It writes code like:

```python
total = await add(a=2, b=3)
msg = await greet(name=str(total))
msg  # last expression is returned automatically
```

### Selective tool sandboxing

By default, `CodeMode(tools='all')` sandboxes every tool. You can selectively choose which tools to sandbox:

```python
# By name
CodeMode(tools=['search', 'fetch'])

# By predicate
CodeMode(tools=lambda ctx, td: td.name != 'dangerous_tool')

# By metadata -- use with SetToolMetadata capability or .with_metadata() on toolsets
CodeMode(tools={'code_mode': True})
```

When using the metadata dict selector, mark tools for sandboxing with the `SetToolMetadata` capability or the `.with_metadata()` method on toolset instances:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata
from pydantic_harness import CodeMode

agent = Agent(
    'anthropic:claude-sonnet-4-20250514',
    capabilities=[
        SetToolMetadata(tools=['search', 'fetch'], metadata={'code_mode': True}),
        CodeMode(tools={'code_mode': True}),
    ],
)
```

Tools that match the selector are wrapped inside `run_code`; non-matching tools remain available as regular tool calls.

### Return values

The last expression in the code snippet is automatically captured as the return value -- the model does **not** need to `print()` it. `print()` output is only useful for supplementary logging.

- **No print output**: the last expression's value is returned directly.
- **With print output**: returns `{"output": "<printed text>", "result": <last expression>}`.
- **Multimodal content** (e.g. binary images from tools): returned natively so the model can process them.

### Nested tool call metadata

The `run_code` tool return includes metadata with all nested tool calls and their results, keyed by tool call ID:

```python
result = await agent.run('...')

# Access the run_code ToolReturnPart from messages
for msg in result.all_messages():
    for part in msg.parts:
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
            tool_calls = part.metadata['tool_calls']      # dict[str, ToolCallPart]
            tool_returns = part.metadata['tool_returns']   # dict[str, ToolReturnPart]
```

This is useful for observability, audit logging, or building UIs that show what happened inside each `run_code` invocation. When the agent is instrumented with Logfire/OTel, nested tool calls produce their own spans.

## Documentation

- [Pydantic AI docs](https://ai.pydantic.dev/)
- [Capabilities API](https://ai.pydantic.dev/capabilities/)

## Development

```bash
make install   # install dependencies
make lint      # ruff format check + lint
make typecheck # pyright strict
make test      # pytest
make testcov   # pytest with coverage
```

## License

MIT
