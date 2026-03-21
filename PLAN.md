# Implementation Plan: Issue #42

> **UserInteraction capability (AskUser tool)**
> https://github.com/pydantic/pydantic-harness/issues/42

> Generated via dual-model debate (anthropic/claude-opus-4-6 + anthropic/claude-opus-4-5-20251101)
> Date: 2026-03-21
> Issue graph: 30 items crawled

## Linked Issues & PRs

- issue #1: [Tracking] Capabilities abstraction for pydantic-ai (https://github.com/pydantic/pydantic-harness/issues/1)
- issue #29: Approval capability (https://github.com/pydantic/pydantic-harness/issues/29)
- issue #10: AbstractCapability + CombinedCapability + Agent integration (https://github.com/pydantic/pydantic-harness/issues/10)
- issue #65: Task Tracking capability (structured task list for multi-step work) (https://github.com/pydantic/pydantic-harness/issues/65)
- issue #82: Tool Output Management (truncation, spill-to-file, per-tool summarization) (https://github.com/pydantic/pydantic-harness/issues/82)
- issue #2: Extract shared NamedSpec infrastructure from EvaluatorSpec (https://github.com/pydantic/pydantic-harness/issues/2)
- issue #3: Extract `_instructions` module (https://github.com/pydantic/pydantic-harness/issues/3)
- issue #4: Extract `_history_processor` module (https://github.com/pydantic/pydantic-harness/issues/4)
- issue #5: TemplateStr — Handlebars template string support (https://github.com/pydantic/pydantic-harness/issues/5)
- issue #6: Dynamic (callable) model settings on Agent (https://github.com/pydantic/pydantic-harness/issues/6)
- issue #7: Dynamic builtin tools (RunContext-based) (https://github.com/pydantic/pydantic-harness/issues/7)
- issue #8: Durable execution refactor (unified WrapperAgent pattern) (https://github.com/pydantic/pydantic-harness/issues/8)
- issue #9: UI adapter enhancements (toolsets/builtin_tools params + Vercel SDK v6) (https://github.com/pydantic/pydantic-harness/issues/9)
- issue #11: `Instructions` capability (https://github.com/pydantic/pydantic-harness/issues/11)
- issue #12: `ModelSettings` capability (https://github.com/pydantic/pydantic-harness/issues/12)
- issue #13: `Thinking` capability (https://github.com/pydantic/pydantic-harness/issues/13)
- issue #14: `WebSearch` capability (https://github.com/pydantic/pydantic-harness/issues/14)
- issue #15: `Toolset` capability (https://github.com/pydantic/pydantic-harness/issues/15)
- issue #16: `HistoryProcessorCapability` (https://github.com/pydantic/pydantic-harness/issues/16)
- issue #17: AgentSpec + Agent.from_spec + Agent.from_file (https://github.com/pydantic/pydantic-harness/issues/17)
- issue #18: CLI agent loading from spec files (https://github.com/pydantic/pydantic-harness/issues/18)
- issue #19: Expand hooks: `before_tool_call` / `after_tool_call` on AbstractCapability (https://github.com/pydantic/pydantic-harness/issues/19)
- issue #20: Expand hooks: `on_agent_start` / `on_agent_end` on AbstractCapability (https://github.com/pydantic/pydantic-harness/issues/20)
- issue #48: Streaming hooks on AbstractCapability (https://github.com/pydantic/pydantic-harness/issues/48)
- issue #35: Expose context window size on ModelProfile and usage on RunContext (https://github.com/pydantic/pydantic-harness/issues/35)
- issue #21: Compaction capability (https://github.com/pydantic/pydantic-harness/issues/21)
- issue #46: Cost/Token Budget capability (spending limit with BudgetExceededError) (https://github.com/pydantic/pydantic-harness/issues/46)
- issue #68: Observational Memory capability (observe/reflect with token thresholds) (https://github.com/pydantic/pydantic-harness/issues/68)
- issue #36: Python signatures for tools (Code Mode infrastructure) (https://github.com/pydantic/pydantic-harness/issues/36)



# Implementation Plan: UserInteraction Capability (Issue #42)

## Goal

Add a `UserInteraction` capability (`AbstractCapability` subclass) providing an `ask_user` tool that pauses agent execution to gather clarifying information from the user. Supports free-text questions and structured multiple-choice options. Uses pydantic-ai's existing `CallDeferred` mechanism for pause/resume.

## Prior Art & Competitive Landscape

| Framework | Approach | Key Citation | Learn | Avoid |
|-----------|----------|-------------|-------|-------|
| **Mastra** | Dedicated `ask_user` tool with structured options (`{label, description?}`), event-driven pause via `registerQuestion`/`respondToQuestion`, abort signal, always-allowed (bypasses permissions) | [tools.ts:18-76](https://github.com/mastra-ai/mastra/blob/main/packages/core/src/harness/tools.ts) | Structured options, headless fallback, always-allowed designation | Module-level mutable counter, string-only responses, no multi-select |
| **LangGraph** | Generic `interrupt()` primitive, checkpoint-based pause/resume via `Command(resume=value)`, ID-based correlation | [types.py](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py) | Clean pause/resume primitive, ID correlation | Not a tool the LLM invokes, requires checkpointer, node re-execution on resume |
| **CrewAI** | Task-level `human_input=True` for post-completion feedback, `HumanInputProvider` protocol, `request_human_input()` in hooks | [task.py:158-162](https://github.com/crewAIInc/crewAI/blob/main/lib/crewai/src/crewai/task.py), [human_input.py](https://github.com/crewAIInc/crewAI/blob/main/lib/crewai/src/crewai/core/providers/human_input.py) | Protocol-based provider abstraction | Only fires after agent finishes, no structured choices, terminal-coupled |
| **Claude Code** | Built-in `AskUserQuestion` tool with multi-question and multi-select support, always-allowed | [Claude Code docs](https://code.claude.com/docs/en/hooks) | Multi-select, batch questions | Not open-source implementation |

**Key insight:** The most useful pattern (Mastra, Claude Code) is a tool the LLM can choose to invoke, not a task-level config flag (CrewAI). Pydantic-ai's `CallDeferred` mechanism (`exceptions.py:77-99`) is semantically the correct choice: it's for tools whose results come from an external source and produces entries in `DeferredToolRequests.calls` with arbitrary string values — exactly the pattern needed for free-text user answers. `ApprovalRequired` (`exceptions.py:95-114`) is wrong because it's for yes/no authorization and triggers tool re-execution with `ctx.tool_call_approved=True`.

## Approach

The `UserInteraction` capability provides an `ask_user` tool via `get_toolset()` and guidance via `get_instructions()`. The capability operates in two modes:

**Deferred mode (default, no handler):** The tool raises `CallDeferred` with question metadata. The agent graph layer in `_agent_graph.py:1365-1370` catches the exception and automatically associates it with the current `ToolCallPart`'s `tool_call_id`. The run ends with `DeferredToolRequests`. The host application presents the question, then resumes with `DeferredToolResults` containing the answer keyed by `tool_call_id`. The answer becomes the tool's return value directly — no re-execution.

**Inline mode (with handler):** When `input_handler` is provided, the tool calls it directly and returns the answer without pausing. Async handlers (detected via `inspect.iscoroutinefunction()`) are awaited. This enables terminal `input()`, callbacks, or any custom resolution.

The capability is Tier S serializable: `get_serialization_name()` returns `'UserInteraction'`, `from_spec()` accepts no required arguments. The `input_handler` callable is not serializable, so `from_spec()` always creates the deferred variant. This follows the pattern established by the `Instructions` capability where `from_spec()` deliberately narrows to the serializable subset.

For the `choices` parameter, the tool uses `list[str] | None` as the model-facing type. This is the simplest representation for the LLM to produce correctly, matching the issue description's proposed signature. The type can be widened to structured objects (`[{label, description}]`) in a future non-breaking change if needed.

## Implementation Steps

### Step 1: Create the `UserInteraction` capability module

**File:** `pydantic_ai_slim/pydantic_ai/capabilities/user_interaction.py` (new)
**Modeled after:** `capabilities/web_search.py` (Tier S pattern), `tests/test_capabilities.py:560-600` (`ToolsetFuncCapability` using `FunctionToolset` with `@tool_plain`)

```python
"""UserInteraction capability — ask the user clarifying questions mid-execution."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Union

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import CallDeferred
from pydantic_ai.toolsets.function import FunctionToolset

InputHandler = Callable[[str, Union[list[str], None]], Union[str, Awaitable[str]]]


@dataclass
class UserInteraction(AbstractCapability[AgentDepsT]):
    """Capability that provides an ask_user tool for gathering clarifying information.

    When ``input_handler`` is provided, questions are resolved inline by calling the handler.
    When ``input_handler`` is None (default), the tool raises ``CallDeferred``, pausing
    execution so the host application can present the question and resume with the answer.

    Example (deferred mode — default)::

        agent = Agent('openai:gpt-4o', capabilities=[UserInteraction()])
        result = await agent.run('Help me pick a color')
        if isinstance(result.output, DeferredToolRequests):
            call = result.output.calls[0]
            question = call.metadata['question']
            result = await agent.run(
                'Help me pick a color',
                message_history=result.all_messages(),
                deferred_tool_results=DeferredToolResults(calls={call.tool_call_id: answer}),
            )

    Example (inline mode)::

        agent = Agent('openai:gpt-4o', capabilities=[
            UserInteraction(input_handler=lambda q, c: input(f'{q}: '))
        ])
    """

    input_handler: InputHandler | None = field(default=None, repr=False)
    """Optional callable to resolve questions inline.

    Signature: ``(question: str, choices: list[str] | None) -> str``
    or async: ``async (question: str, choices: list[str] | None) -> str``

    Only ``async def`` functions are awaited; arbitrary awaitables are not supported.
    If None, the tool pauses execution via CallDeferred.
    If the handler blocks (e.g., ``input()``), the agent's ``tool_timeout`` applies.
    """

    @classmethod
    def get_serialization_name(cls) -> str:
        return 'UserInteraction'

    @classmethod
    def from_spec(cls, **kwargs: Any) -> UserInteraction[Any]:
        return cls()

    def get_instructions(self) -> str:
        return (
            "When you are uncertain about the user's intent, need clarification on ambiguous "
            "requirements, or want the user to choose between alternatives, use the ask_user "
            "tool to ask a clarifying question. Provide choices when the options are known "
            "and limited."
        )

    def get_toolset(self) -> FunctionToolset[AgentDepsT]:
        toolset: FunctionToolset[AgentDepsT] = FunctionToolset()
        handler = self.input_handler

        @toolset.tool_plain
        async def ask_user(
            question: str,
            choices: list[str] | None = None,
        ) -> str:
            """Ask the user a clarifying question.

            Args:
                question: The question to ask the user. Should be clear and specific.
                choices: Optional list of choices for the user to select from.
                    When provided, the user selects one of these options.
                    When omitted, the user provides a free-text response.
            """
            if handler is not None:
                if inspect.iscoroutinefunction(handler):
                    return await handler(question, choices)
                return handler(question, choices)  # type: ignore[return-value]

            metadata: dict[str, Any] = {'question': question}
            if choices is not None:
                metadata['choices'] = choices
            raise CallDeferred(metadata=metadata)

        return toolset
```

**Why `@toolset.tool_plain`:** The tool doesn't need `RunContext` — it only needs its arguments and the handler closure. This decorator works with async functions as confirmed in `test_capabilities.py:560-600`. The graph layer automatically associates the `CallDeferred` exception with the current `ToolCallPart`'s `tool_call_id` during tool execution in `_agent_graph.py:1365-1370`.

**Why `list[str] | None` for choices:** Simplest type for the LLM. Both plans agree on this. Can be widened to `list[Choice]` (with `Choice = TypedDict(label=str, description=NotRequired[str])`) later without breaking changes. Mastra uses the richer schema but pydantic-ai's LLM-facing tool schema benefits from simplicity for v1.

### Step 2: Register in the capabilities package

**File:** `pydantic_ai_slim/pydantic_ai/capabilities/__init__.py`
**Lines affected:** ~14-48

Add `UserInteraction` to whatever lookup mechanism `Agent.from_spec()` uses for capability name resolution. Based on codebase research, `DEFAULT_CAPABILITY_TYPES` at `capabilities/__init__.py:14-19` is referenced by `agent/spec.py:19`.

```python
from pydantic_ai.capabilities.user_interaction import UserInteraction

# Add to the capability types available for spec resolution
```

**Decision logic for where to add:** `DEFAULT_CAPABILITY_TYPES` controls which capability names are resolvable in YAML specs. Adding `UserInteraction` there does NOT auto-enable it on every agent — it only makes the name `UserInteraction` available for `- UserInteraction` in YAML. This is the same as `WebSearch` being in `DEFAULT_CAPABILITY_TYPES`: it's available but not active unless explicitly listed. See Open Question #1 for verification.

### Step 3: Export from top-level package

**File:** `pydantic_ai_slim/pydantic_ai/__init__.py`

```python
from pydantic_ai.capabilities.user_interaction import UserInteraction
```

Add `'UserInteraction'` to `__all__` if present. This follows the pattern of `Instructions`, `Thinking`, `WebSearch` exports.

### Step 4: Add tests

**File:** `tests/test_user_interaction.py` (new)
**Modeled after:** `tests/test_capabilities.py:560-600` (capability patterns), `tests/test_tools.py:1363-1600` (`CallDeferred` patterns)

```python
"""Tests for UserInteraction capability."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities.user_interaction import UserInteraction
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)


# --- Test 1: Deferred mode with structured choices ---

async def test_ask_user_deferred_mode():
    """No handler → CallDeferred → DeferredToolRequests with question metadata."""

    def model_fn(messages: list[ModelRequest], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[
                ToolCallPart(
                    tool_name='ask_user',
                    args={'question': 'What color?', 'choices': ['Red', 'Blue']},
                )
            ])
        for msg in messages:
            for part in msg.parts:
                if isinstance(part, ToolReturnPart) and part.tool_name == 'ask_user':
                    return ModelResponse(parts=[TextPart(content=f'You chose: {part.content}')])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(model_fn), capabilities=[UserInteraction()])

    result = await agent.run('Pick a color')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.calls) == 1
    call = result.output.calls[0]
    assert call.metadata['question'] == 'What color?'
    assert call.metadata['choices'] == ['Red', 'Blue']

    # Resume with answer
    deferred_results = DeferredToolResults(
        calls={call.tool_call_id: 'Red'},
    )
    result2 = await agent.run(
        'Pick a color',
        message_history=result.all_messages(),
        deferred_tool_results=deferred_results,
    )
    assert 'Red' in result2.output


# --- Test 2: Free-text question (no choices) ---

async def test_ask_user_free_text():
    """No choices → metadata without 'choices' key."""

    def model_fn(messages: list[ModelRequest], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[
            ToolCallPart(tool_name='ask_user', args={'question': 'What is your name?'})
        ])

    agent = Agent(FunctionModel(model_fn), capabilities=[UserInteraction()])
    result = await agent.run('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    assert result.output.calls[0].metadata['question'] == 'What is your name?'
    assert 'choices' not in result.output.calls[0].metadata


# --- Test 3: Sync inline handler ---

async def test_ask_user_with_sync_handler():
    """Sync handler → inline resolution, no DeferredToolRequests."""

    def my_handler(question: str, choices: list[str] | None = None) -> str:
        return 'Blue'

    call_count = 0

    def model_fn(messages: list[ModelRequest], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[
                ToolCallPart(tool_name='ask_user', args={'question': 'What color?'})
            ])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(model_fn), capabilities=[UserInteraction(input_handler=my_handler)])
    result = await agent.run('Pick a color')
    assert isinstance(result.output, str)
    assert result.output == 'done'


# --- Test 4: Async inline handler ---

async def test_ask_user_with_async_handler():
    """Async handler → awaited, inline resolution."""

    async def my_handler(question: str, choices: list[str] | None = None) -> str:
        return 'Green'

    call_count = 0

    def model_fn(messages: list[ModelRequest], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[
                ToolCallPart(tool_name='ask_user', args={'question': 'What color?'})
            ])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(model_fn), capabilities=[UserInteraction(input_handler=my_handler)])
    result = await agent.run('Pick a color')
    assert isinstance(result.output, str)


# --- Test 5: Handler receives choices ---

async def test_ask_user_handler_receives_choices():
    """Verify handler receives both question and choices."""
    received_args: list[tuple[str, list[str] | None]] = []

    def my_handler(question: str, choices: list[str] | None = None) -> str:
        received_args.append((question, choices))
        return choices[0] if choices else 'default'

    call_count = 0

    def model_fn(messages: list[ModelRequest], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[
                ToolCallPart(tool_name='ask_user', args={'question': 'Pick one', 'choices': ['A', 'B']})
            ])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(model_fn), capabilities=[UserInteraction(input_handler=my_handler)])
    await agent.run('test')
    assert received_args == [('Pick one', ['A', 'B'])]


# --- Test 6: Tier S serialization ---

def test_user_interaction_serialization():
    """Zero-config spec construction."""
    cap = UserInteraction.from_spec()
    assert isinstance(cap, UserInteraction)
    assert cap.input_handler is None
    assert UserInteraction.get_serialization_name() == 'UserInteraction'


# --- Test 7: Instructions content ---

def test_user_interaction_instructions():
    """Instructions mention ask_user and clarification."""
    cap = UserInteraction()
    instructions = cap.get_instructions()
    assert 'ask_user' in instructions
    assert 'clarif' in instructions.lower()


# --- Test 8: Capability composition ---

async def test_user_interaction_combined_with_other_capabilities():
    """Composes with other capabilities via CombinedCapability."""
    from pydantic_ai.capabilities.instructions import Instructions

    def model_fn(messages: list[ModelRequest], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(
        FunctionModel(model_fn),
        capabilities=[
            Instructions('Be helpful.'),
            UserInteraction(),
        ],
    )
    result = await agent.run('Hello')
    assert isinstance(result.output, str)


# --- Test 9: Handler exception behavior ---

async def test_ask_user_handler_exception():
    """Exceptions from handler — verify FunctionToolset error handling behavior."""

    def bad_handler(question: str, choices: list[str] | None = None) -> str:
        raise ValueError('handler failed')

    call_count = 0

    def model_fn(messages: list[ModelRequest], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[
                ToolCallPart(tool_name='ask_user', args={'question': 'test'})
            ])
        return ModelResponse(parts=[TextPart(content='error handled')])

    agent = Agent(FunctionModel(model_fn), capabilities=[UserInteraction(input_handler=bad_handler)])
    # FunctionToolset may catch and feed back as tool error, or propagate.
    # Adjust assertion based on actual behavior (see Open Question #3).
    result = await agent.run('test')


# --- Test 10: Tool schema shape ---

async def test_ask_user_tool_schema():
    """Verify ask_user tool JSON schema."""
    cap = UserInteraction()
    toolset = cap.get_toolset()
    defs = await toolset.tool_defs()
    assert len(defs) == 1
    td = defs[0]
    assert td.name == 'ask_user'
    schema = td.parameters_json_schema
    assert 'question' in schema.get('properties', {})
    assert 'choices' in schema.get('properties', {})
    assert 'question' in schema.get('required', [])


# --- Test 11: Streaming with deferred ask_user ---

async def test_ask_user_streaming():
    """ask_user during streamed run produces DeferredToolRequests."""

    def model_fn(messages: list[ModelRequest], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[
            ToolCallPart(tool_name='ask_user', args={'question': 'Which one?'})
        ])

    agent = Agent(FunctionModel(model_fn), capabilities=[UserInteraction()])
    async with agent.run_stream('test') as stream:
        result = await stream.get_output()
    assert isinstance(result, DeferredToolRequests)
    assert result.calls[0].metadata['question'] == 'Which one?'
```

### Step 5: Add documentation

**Primary location:** If `docs/capabilities.md` exists (created by broader capabilities work #10-#16), add a `UserInteraction` section there. Otherwise, create `docs/user-interaction.md`.

**Content:**

```markdown
## UserInteraction

The `UserInteraction` capability provides an `ask_user` tool that lets the agent
ask the user clarifying questions mid-execution.

### YAML Spec

```yaml
capabilities:
  - UserInteraction
```

### Python (deferred mode — default)

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities.user_interaction import UserInteraction
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults

agent = Agent('openai:gpt-4o', capabilities=[UserInteraction()])

result = await agent.run('Help me pick a framework')
if isinstance(result.output, DeferredToolRequests):
    for call in result.output.calls:
        question = call.metadata['question']
        choices = call.metadata.get('choices')  # None for free-text
        answer = await get_user_answer(question, choices)  # Your UI logic

    result = await agent.run(
        'Help me pick a framework',
        message_history=result.all_messages(),
        deferred_tool_results=DeferredToolResults(calls={call.tool_call_id: answer}),
    )
```

### Python (inline mode — with callback)

```python
agent = Agent('openai:gpt-4o', capabilities=[
    UserInteraction(input_handler=lambda q, c: input(f'{q}: '))
])
# Questions resolved inline — no deferred requests.
result = await agent.run('Help me pick a framework')
```

### How it works

When no `input_handler` is provided, the `ask_user` tool pauses execution using the
[deferred tools](deferred-tools.md) mechanism (`CallDeferred`). The host application
receives a `DeferredToolRequests` output with metadata containing the question. The
`tool_call_id` correlation is handled automatically by the agent graph layer.

When an `input_handler` is provided, questions are resolved inline. The handler can
be sync or `async def`. If the handler blocks (e.g., `input()`), the agent's
`tool_timeout` applies.

### Difference from Approval

[Approval](deferred-tools.md) authorizes tool execution (yes/no) via `ApprovalRequired`.
UserInteraction gathers information via `CallDeferred`. They use different exception
types and produce different entries in `DeferredToolRequests` (`.approvals` vs `.calls`).
```

**Cross-reference in `docs/deferred-tools.md`:** Add near the `CallDeferred` section:

```markdown
!!! tip "UserInteraction capability"
    The [`UserInteraction`](user-interaction.md) capability provides a built-in
    `ask_user` tool that uses `CallDeferred` to pause execution for user questions.
```

## Files to Modify

| File | Change | Citation |
|------|--------|----------|
| `pydantic_ai_slim/pydantic_ai/capabilities/__init__.py` | Add `UserInteraction` to capability registry for spec resolution | `capabilities/__init__.py:14-19`, `agent/spec.py:19` |
| `pydantic_ai_slim/pydantic_ai/__init__.py` | Export `UserInteraction` | Pattern: `Instructions`, `Thinking`, `WebSearch` exports |
| `docs/deferred-tools.md` | Add cross-reference to UserInteraction | Existing doc |

## New Files

| File | Purpose | Modeled After |
|------|---------|---------------|
| `pydantic_ai_slim/pydantic_ai/capabilities/user_interaction.py` | Capability class with `ask_user` tool | `capabilities/web_search.py` (Tier S), `tests/test_capabilities.py:560-600` (FunctionToolset pattern) |
| `tests/test_user_interaction.py` | 11 test cases covering both modes | `tests/test_capabilities.py`, `tests/test_tools.py:1363-3602` |
| `docs/user-interaction.md` (or section in `docs/capabilities.md`) | User-facing docs | `docs/deferred-tools.md` |

## Test Plan

| # | Scenario | What to verify |
|---|----------|----------------|
| 1 | Deferred mode with choices | `CallDeferred` → `DeferredToolRequests` with correct metadata, resume completes run |
| 2 | Free-text (no choices) | Metadata contains `question` but no `choices` key |
| 3 | Sync inline handler | Handler called, result returned directly, no `DeferredToolRequests` |
| 4 | Async inline handler | `inspect.iscoroutinefunction()` detects and awaits correctly |
| 5 | Handler receives choices | Both `question` and `choices` args passed through |
| 6 | Serialization round-trip | `from_spec()` → working instance, `get_serialization_name()` → `'UserInteraction'` |
| 7 | Instructions content | Contains `ask_user` and clarification guidance |
| 8 | Capability composition | Composes with `Instructions` via `CombinedCapability` |
| 9 | Handler exception | Verify `FunctionToolset` error handling behavior |
| 10 | Tool schema | JSON schema has `question` (required), `choices` (optional array) |
| 11 | Streaming | `run_stream()` produces `DeferredToolRequests` correctly |

**Infrastructure:** `FunctionModel` (`pydantic_ai.models.function`) for deterministic tests, following `test_capabilities.py:44-72`.

## Risks and Pitfalls

**Risk 1: Models producing `choices` as wrong type.**
Models might produce `choices: "Red"` (string) instead of `choices: ["Red"]`.
**Mitigation:** `list[str]` is the simplest possible type, minimizing confusion. `FunctionToolset` validates tool args against the JSON schema before invocation (`toolsets/function.py`), so malformed args produce a validation error fed back to the model for retry.

**Risk 2: `input_handler` with wrong arity.**
Users may pass `lambda q: input(q)` (missing `choices` parameter).
**Mitigation:** Document expected signature. A `lambda q, c: input(q)` that ignores `c` works fine. Wrong arity fails at call time with a clear `TypeError`.

**Risk 3: `ask_user` wrapped in `ApprovalRequiredToolset`.**
If user wraps agent toolset in approval, `ask_user` would nonsensically require approval before asking a question. Mastra explicitly marks `ask_user` as always-allowed (`tools.ts`).
**Mitigation:** Document that `ask_user` should not be wrapped in approval. Users can use `FilteredToolset` to exclude it. See Open Question #2 for whether `requires_approval=False` can be set proactively.

**Risk 4: Timeout with blocking inline handler.**
When `input_handler` blocks (e.g., terminal `input()`), the agent's `tool_timeout` applies.
**Mitigation:** This is actually desirable to prevent indefinite hangs. In deferred mode, `CallDeferred` is raised synchronously so timeout doesn't apply. Documented in `input_handler` field docstring.

**Risk 5: Concurrent tool calls with `ask_user`.**
Model calls `ask_user` alongside other tools in the same response. Other tools execute normally while `ask_user` produces a deferred entry.
**Mitigation:** Already handled — `_agent_graph.py:1365-1370` collects multiple deferred calls into `DeferredToolRequests.calls`. No code change needed.

**Risk 6: `DeferredToolRequests` not in `output_type`.**
Users will forget to include `DeferredToolRequests` in their agent's `output_type`, causing runtime errors.
**Evidence:** `tests/test_tools.py:1714-1716` validates this constraint.
**Mitigation:** Clear documentation, prominent examples showing `output_type=str | DeferredToolRequests`. Note: the exact error behavior when `DeferredToolRequests` is not in `output_type` should be verified during implementation.

## Open Questions

1. **How does `Agent.from_spec()` resolve capability names?** `DEFAULT_CAPABILITY_TYPES` at `capabilities/__init__.py:14-19` is referenced by `agent/spec.py:19`. Does `Agent.from_spec()` only look up names in `DEFAULT_CAPABILITY_TYPES`, or also a broader `CAPABILITY_TYPES` dict? This determines whether `UserInteraction` must be added to `DEFAULT_CAPABILITY_TYPES` (making the name available for all spec-based agents) or only to a separate registry. **Action:** Read `agent/spec.py` during implementation. Adding to `DEFAULT_CAPABILITY_TYPES` only makes the name *resolvable* — it does not auto-enable the capability.
   - **Source:** `capabilities/__init__.py:14-19`, `agent/spec.py:19`

2. **Should `ask_user` set `requires_approval=False`?** The `@toolset.tool_plain` decorator in `toolsets/function.py` supports a `requires_approval` parameter. Setting `False` would ensure `ask_user` bypasses `ApprovalRequiredToolset` even when wrapped. This follows Mastra's "always-allowed" pattern. **Action:** Check whether `tool_plain` accepts `requires_approval` and whether `ApprovalRequiredToolset` respects it.
   - **Source:** `toolsets/function.py` (decorator params), `toolsets/approval_required.py`

3. **How does `FunctionToolset` handle uncaught exceptions from tool functions?** Test 9 depends on this. Does a `ValueError` from the handler propagate to the caller, or does `FunctionToolset` catch it and return an error `ToolReturnPart` to the model? **Action:** Check `FunctionToolset.call_tool()` error handling during implementation.
   - **Source:** `toolsets/function.py` (error handling in `call_tool`)

4. **Streaming + deferred interaction.** Does `agent.run_stream()` produce `DeferredToolRequests` accessible via `stream.get_output()` when the tool raises `CallDeferred`? Test 11 assumes this. **Action:** Verify against actual streaming API behavior.
   - **Source:** `_agent_graph.py` `CallToolsNode`, streaming API

5. **`DeferredToolRequests.metadata` access pattern.** Both plans reference `call.metadata['question']` on the `ToolCallPart` objects in `DeferredToolRequests.calls`. Verify the exact API: is metadata stored on `DeferredToolRequests.metadata[tool_call_id]` (a top-level dict) or directly on the `ToolCallPart` object. **Action:** Check `tools.py:148-162` and `_agent_graph.py:1365-1370` to confirm the metadata access pattern, and adjust test assertions accordingly.
   - **Source:** `tools.py:148-162`, `_agent_graph.py:1365-1370`

## References

- **Issue #42:** https://github.com/pydantic/pydantic-harness/issues/42
- **Issue #1 (tracking):** https://github.com/pydantic/pydantic-harness/issues/1
- **Issue #10 (AbstractCapability):** https://github.com/pydantic/pydantic-harness/issues/10
- **Issue #29 (Approval — distinct):** https://github.com/pydantic/pydantic-harness/issues/29
- **`AbstractCapability` base class:** `pydantic_ai_slim/pydantic_ai/capabilities/abstract.py:38-220`
- **`CombinedCapability`:** `pydantic_ai_slim/pydantic_ai/capabilities/combined.py:1-340`
- **`WebSearch` capability (Tier S pattern):** `pydantic_ai_slim/pydantic_ai/capabilities/web_search.py`
- **`Toolset` capability (wrapper pattern):** `pydantic_ai_slim/pydantic_ai/capabilities/toolset.py:1-22`
- **`FunctionToolset`:** `pydantic_ai_slim/pydantic_ai/toolsets/function.py:1-552`
- **`CallDeferred` exception:** `pydantic_ai_slim/pydantic_ai/exceptions.py:77-99`
- **`ApprovalRequired` exception:** `pydantic_ai_slim/pydantic_ai/exceptions.py:95-114`
- **`DeferredToolRequests`/`DeferredToolResults`:** `pydantic_ai_slim/pydantic_ai/tools.py:148-248`
- **Deferred tool handling in agent graph:** `pydantic_ai_slim/pydantic_ai/_agent_graph.py:1365-1370`
- **Capabilities registry:** `pydantic_ai_slim/pydantic_ai/capabilities/__init__.py:1-48`
- **Agent capability integration:** `pydantic_ai_slim/pydantic_ai/agent/__init__.py:383-453`
- **Agent per-run re-evaluation:** `pydantic_ai_slim/pydantic_ai/agent/__init__.py:974-985`
- **ToolManager hook invocation:** `pydantic_ai_slim/pydantic_ai/_tool_manager.py:230-290`
- **`ToolsetFuncCapability` test pattern:** `tests/test_capabilities.py:560-600`
- **Existing deferred tools tests:** `tests/test_tools.py:1363-3602`
- **Existing capability tests:** `tests/test_capabilities.py:1-2157`
- **Deferred tools documentation:** `docs/deferred-tools.md`
- **MCP elicitation (related pattern):** `pydantic_ai_slim/pydantic_ai/mcp.py:334-406`
- **Graph human-in-the-loop example:** `docs/graph.md:641-740`
- **`AbstractCapability.for_run()`:** `pydantic_ai_slim/pydantic_ai/capabilities/abstract.py:92-100`
- **Agent spec resolution:** `pydantic_ai_slim/pydantic_ai/agent/spec.py:19`
- **`RunContext`:** `pydantic_ai_slim/pydantic_ai/_run_context.py:28-95`
- **`ExternalToolset`:** `pydantic_ai_slim/pydantic_ai/toolsets/external.py:1-53`
- **Mastra ask_user tool:** https://github.com/mastra-ai/mastra/blob/main/packages/core/src/harness/tools.ts
- **LangGraph interrupt:** https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py
- **CrewAI HumanInputProvider:** https://github.com/crewAIInc/crewAI/blob/main/lib/crewai/src/crewai/core/providers/human_input.py
- **CrewAI task.py:** https://github.com/crewAIInc/crewAI/blob/main/lib/crewai/src/crewai/task.py

---

## Appendix

<details>
<summary>anthropic/claude-opus-4-6's Final Plan</summary>



## Goal

Provide an `AbstractCapability` subclass that gives the agent an `ask_user` tool to ask the user clarifying questions mid-execution (free-text or structured choices), pausing the run via the existing `CallDeferred` mechanism and resuming with the user's answer.

## Prior Art & Competitive Landscape

**Mastra** — Purpose-built `ask_user` tool (`packages/core/src/harness/tools.ts`). Promise-based pause with `registerQuestion`/`respondToQuestion` pattern. Supports structured options (`{label, description}`). Graceful fallback when no UI is attached (returns formatted text). Abort signal integration. Always-allowed (bypasses permission system). **Learn:** Structured options with label+description, headless fallback, question ID correlation. **Avoid:** Module-level mutable counter (`let questionCounter`), string-only responses, no multi-select. URL: https://github.com/mastra-ai/mastra/blob/main/packages/core/src/harness/tools.ts

**LangGraph** — General-purpose `interrupt()` function (`libs/langgraph/langgraph/types.py`). Checkpoint-based pause/resume via `Command(resume=value)`. `HumanInterrupt`/`HumanResponse` schemas for structured interactions (`libs/prebuilt/langgraph/prebuilt/interrupt.py`). **Learn:** ID-based correlation, clean pause/resume primitive. **Avoid:** Node re-execution on resume (side effects replayed), not a tool the LLM invokes — requires developer wiring. URL: https://github.com/langchain-ai/langgraph

**CrewAI** — Task-level `human_input=True` (`lib/crewai/src/crewai/task.py`) triggers post-completion feedback loop. `HumanInputProvider` protocol (`lib/crewai/src/crewai/core/providers/human_input.py`) for pluggable I/O. **Learn:** Protocol-based provider abstraction, contextvar injection. **Avoid:** Only fires after agent finishes (not mid-execution), no structured choices, tightly coupled to terminal I/O. URL: https://github.com/crewAIInc/crewAI

**Claude Code** — `AskUserQuestion` built-in tool with multi-question and multi-select support. Always-allowed tool.

## Approach

The `UserInteraction` capability will be an `AbstractCapability[AgentDepsT]` subclass providing an `ask_user` tool via `get_toolset()`, with instructions via `get_instructions()`. The tool uses pydantic-ai's existing `CallDeferred` exception mechanism (`pydantic_ai_slim/pydantic_ai/exceptions.py:77-99`) to pause execution. `CallDeferred` is semantically the correct choice over `ApprovalRequired` (`exceptions.py:95-114`): `ApprovalRequired` is for yes/no authorization of tool execution and produces entries in `DeferredToolRequests.approvals` with `ToolApproved`/`ToolDenied` responses (`tools.py:161-248`), while `CallDeferred` is for tools whose results come from an external source and produces entries in `DeferredToolRequests.calls` with arbitrary string values — exactly the pattern needed for free-text user answers. When the tool is called, it raises `CallDeferred` with metadata containing the question and optional choices. The agent graph layer in `_agent_graph.py:1365-1370` catches the `CallDeferred` exception and automatically associates it with the current `ToolCallPart`'s `tool_call_id` — the capability does not need to manage correlation IDs. The agent run ends with `DeferredToolRequests` (`tools.py:148-162`), the host application presents the question to the user, and the agent is resumed with `DeferredToolResults` containing the user's answer keyed by the `tool_call_id`.

The capability accepts an optional `input_handler: Callable` for synchronous/inline resolution (e.g., terminal `input()` or a callback). When an `input_handler` is provided, the tool calls it directly and returns the answer without pausing execution. When no handler is provided (the default), the tool raises `CallDeferred`, making the capability work out-of-the-box with UI adapters, web frameworks, and the Vercel AI SDK's deferred tool streaming (`pydantic_ai_slim/pydantic_ai/agent/__init__.py` UI adapter methods). Async handler detection uses `inspect.iscoroutinefunction()` on the handler function itself (not the return value). This dual-mode design follows Mastra's graceful fallback pattern while leveraging pydantic-ai's existing strengths.

For the `choices` parameter, the tool will use `list[str] | None` as the model-facing type. This is the simplest representation for the LLM to produce correctly, matching the issue description's proposed signature (`choices: list[str] | None`). The `CallDeferred` metadata will carry these same `list[str]` choices. The type can be widened to structured objects (`[{label, description}]`) in a future non-breaking change.

The capability is Tier S serializable (zero-config in YAML specs: `- UserInteraction`). `get_serialization_name()` returns `'UserInteraction'`, and `from_spec()` accepts no required arguments. The `input_handler` callable is not serializable, so `from_spec()` always creates the deferred (no-handler) variant. This matches the pattern established by the existing `Instructions` capability (`pydantic_ai_slim/pydantic_ai/capabilities/instructions.py`) where `from_spec()` deliberately narrows to the serializable subset.

## Implementation Steps

### Step 1: Create the `UserInteraction` capability module

**File to create:** `pydantic_ai_slim/pydantic_ai/capabilities/user_interaction.py`
**Modeled after:** `pydantic_ai_slim/pydantic_ai/capabilities/web_search.py` (Tier S capability with serialization name) and `tests/test_capabilities.py:560-600` (`ToolsetFuncCapability` pattern for capability providing `FunctionToolset` with `@tool_plain`)

```python
"""UserInteraction capability — ask the user clarifying questions mid-execution."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Union

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import CallDeferred
from pydantic_ai.toolsets.function import FunctionToolset

InputHandler = Callable[[str, Union[list[str], None]], Union[str, Awaitable[str]]]


@dataclass
class UserInteraction(AbstractCapability[AgentDepsT]):
    """Capability that provides an ask_user tool for gathering clarifying information from the user.

    When `input_handler` is provided, questions are resolved inline by calling the handler.
    When `input_handler` is None (default), the tool raises `CallDeferred`, pausing execution
    so the host application can present the question and resume with the answer.

    Example (deferred mode — default):
        ```python
        agent = Agent('openai:gpt-4o', capabilities=[UserInteraction()])
        result = await agent.run('Help me pick a color')
        if isinstance(result.output, DeferredToolRequests):
            call = result.output.calls[0]
            question = call.metadata['question']
            # Present question to user, get answer...
            result = await agent.run(
                'Help me pick a color',
                message_history=result.all_messages(),
                deferred_tool_results=DeferredToolResults(calls={call.tool_call_id: answer}),
            )
        ```

    Example (inline mode):
        ```python
        agent = Agent('openai:gpt-4o', capabilities=[
            UserInteraction(input_handler=lambda q, c: input(f'{q}: '))
        ])
        ```
    """

    input_handler: InputHandler | None = field(default=None, repr=False)
    """Optional callable to resolve questions inline.

    Signature: ``(question: str, choices: list[str] | None) -> str``
    or the async variant: ``(question: str, choices: list[str] | None) -> Awaitable[str]``

    Only ``async def`` functions are awaited; arbitrary awaitables are not supported.

    If None, the tool pauses execution via CallDeferred and the host application
    must resume with DeferredToolResults containing the user's answer.

    If the handler blocks (e.g., ``input()``), the agent's ``tool_timeout`` applies.
    """

    @classmethod
    def get_serialization_name(cls) -> str:
        return 'UserInteraction'

    @classmethod
    def from_spec(cls, **kwargs: Any) -> UserInteraction[Any]:
        return cls()

    def get_instructions(self) -> str:
        return (
            'When you are uncertain about the user\'s intent, need clarification on ambiguous requirements, '
            'or want the user to choose between alternatives, use the ask_user tool to ask a clarifying question. '
            'Provide choices when the options are known and limited.'
        )

    def get_toolset(self) -> FunctionToolset[AgentDepsT]:
        toolset: FunctionToolset[AgentDepsT] = FunctionToolset()
        handler = self.input_handler

        @toolset.tool_plain
        async def ask_user(
            question: str,
            choices: list[str] | None = None,
        ) -> str:
            """Ask the user a clarifying question.

            Args:
                question: The question to ask the user. Should be clear and specific.
                choices: Optional list of choices for the user to select from.
                    When provided, the user selects one of these options.
                    When omitted, the user provides a free-text response.
            """
            if handler is not None:
                if inspect.iscoroutinefunction(handler):
                    return await handler(question, choices)
                return handler(question, choices)  # type: ignore[return-value]

            metadata: dict[str, Any] = {'question': question}
            if choices is not None:
                metadata['choices'] = choices
            raise CallDeferred(metadata=metadata)

        return toolset
```

**Why:** This is the core implementation. The `CallDeferred` mechanism is already fully supported in the agent graph (`_agent_graph.py:1365-1370`), tools module (`tools.py:161-248`), and has extensive tests (`test_tools.py:1363-3602`). Using `@toolset.tool_plain` avoids requiring `RunContext` since the tool only needs its arguments and the handler closure — this decorator is confirmed to work with async functions in existing test capabilities (`test_capabilities.py:560-600`, where `ToolsetFuncCapability` uses `@toolset.tool_plain` on an async function). The graph layer automatically associates the `CallDeferred` exception with the current `ToolCallPart`'s `tool_call_id` during tool execution in `_agent_graph.py:1365-1370` — the capability doesn't need to manage this. The `choices` parameter uses `list[str] | None` — the simplest model-facing type, matching the issue description. The `inspect.iscoroutinefunction(handler)` check correctly identifies `async def` functions (not arbitrary awaitables), and this is documented in the `input_handler` field docstring.

### Step 2: Register `UserInteraction` in the capabilities package

**File to modify:** `pydantic_ai_slim/pydantic_ai/capabilities/__init__.py`
**Lines affected:** ~1-48

Add `UserInteraction` to the `CAPABILITY_TYPES` dict for name-based lookup during spec deserialization:

```python
from pydantic_ai.capabilities.user_interaction import UserInteraction

CAPABILITY_TYPES: dict[str, type[AbstractCapability[Any]]] = {
    # ... existing entries ...
    'UserInteraction': UserInteraction,
}
```

**Regarding `DEFAULT_CAPABILITY_TYPES`:** Based on codebase research, `DEFAULT_CAPABILITY_TYPES` at `capabilities/__init__.py:14-19` is referenced by `agent/spec.py:19` for spec deserialization. The current defaults are `(Instructions, ModelSettings, Thinking, WebSearch)`. Whether adding `UserInteraction` here is required for `- UserInteraction` to work in YAML depends on how `Agent.from_spec()` resolves capability names.

The code at `agent/spec.py` accepts a `custom_capability_types` parameter. If `CAPABILITY_TYPES` is the fallback lookup dict (not just `DEFAULT_CAPABILITY_TYPES`), then adding to `CAPABILITY_TYPES` alone suffices. If `DEFAULT_CAPABILITY_TYPES` is the **only** lookup source, then `UserInteraction` must either be added there or users must pass `custom_capability_types=[UserInteraction]`.

**Decision needed:** Verify how `Agent.from_spec()` resolves capability names. The implementation should add `UserInteraction` to whichever dict/tuple is required for zero-config spec resolution. If both `CAPABILITY_TYPES` and `DEFAULT_CAPABILITY_TYPES` must be updated, add to both. If only `CAPABILITY_TYPES`, add there. See Open Question #1 for details.

**Why:** `CAPABILITY_TYPES` enables lookup by name for spec deserialization. The issue specifies "Tier S — zero-config," meaning `- UserInteraction` should work in YAML specs. `UserInteraction` should NOT be in `DEFAULT_CAPABILITY_TYPES` if that tuple controls which capabilities are *automatically included* in every agent (since not every agent should ask users questions). But it MUST be resolvable by name for spec lookup.

### Step 3: Export `UserInteraction` from the top-level package

**File to modify:** `pydantic_ai_slim/pydantic_ai/__init__.py`
**Lines affected:** Near existing capability imports/exports

```python
from pydantic_ai.capabilities.user_interaction import UserInteraction
```

Also add `'UserInteraction'` to `__all__` if one exists.

**Why:** Users should be able to `from pydantic_ai import UserInteraction` following the same pattern as `Instructions`, `Thinking`, `WebSearch` which are all exported from the top-level package.

### Step 4: Add comprehensive tests

**File to create:** `tests/test_user_interaction.py`
**Modeled after:** `tests/test_capabilities.py` (lines 560-600 for `ToolsetFuncCapability` pattern) and `tests/test_tools.py` (lines 1363-1600 for `CallDeferred` testing)

```python
"""Tests for UserInteraction capability."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities.user_interaction import UserInteraction
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)


# --- Test 1: Deferred mode (no handler) with structured choices ---

async def test_ask_user_deferred_mode():
    """When no handler is provided, ask_user raises CallDeferred and the run
    produces DeferredToolRequests with question metadata."""

    def model_fn(messages: list[ModelRequest], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[
                ToolCallPart(
                    tool_name='ask_user',
                    args={'question': 'What color?', 'choices': ['Red', 'Blue']},
                )
            ])
        for msg in messages:
            for part in msg.parts:
                if isinstance(part, ToolReturnPart) and part.tool_name == 'ask_user':
                    return ModelResponse(parts=[TextPart(content=f'You chose: {part.content}')])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(model_fn), capabilities=[UserInteraction()])

    result = await agent.run('Pick a color')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.calls) == 1
    call = result.output.calls[0]
    assert call.metadata['question'] == 'What color?'
    assert call.metadata['choices'] == ['Red', 'Blue']

    deferred_results = DeferredToolResults(
        calls={call.tool_call_id: 'Red'},
    )
    result2 = await agent.run(
        'Pick a color',
        message_history=result.all_messages(),
        deferred_tool_results=deferred_results,
    )
    assert 'Red' in result2.output


# --- Test 2: Free-text question (no choices) ---

async def test_ask_user_free_text():
    """ask_user without choices produces metadata without 'choices' key."""

    def model_fn(messages: list[ModelRequest], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[
            ToolCallPart(tool_name='ask_user', args={'question': 'What is your name?'})
        ])

    agent = Agent(FunctionModel(model_fn), capabilities=[UserInteraction()])
    result = await agent.run('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    assert result.output.calls[0].metadata['question'] == 'What is your name?'
    assert 'choices' not in result.output.calls[0].metadata


# --- Test 3: Inline handler mode (sync) ---

async def test_ask_user_with_sync_handler():
    """When a sync handler is provided, ask_user resolves inline."""

    def my_handler(question: str, choices: list[str] | None = None) -> str:
        return 'Blue'

    call_count = 0

    def model_fn(messages: list[ModelRequest], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[
                ToolCallPart(tool_name='ask_user', args={'question': 'What color?'})
            ])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(model_fn), capabilities=[UserInteraction(input_handler=my_handler)])
    result = await agent.run('Pick a color')
    assert isinstance(result.output, str)
    assert result.output == 'done'


# --- Test 4: Inline handler mode (async) ---

async def test_ask_user_with_async_handler():
    """When an async handler is provided, ask_user awaits it."""

    async def my_handler(question: str, choices: list[str] | None = None) -> str:
        return 'Green'

    call_count = 0

    def model_fn(messages: list[ModelRequest], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[
                ToolCallPart(tool_name='ask_user', args={'question': 'What color?'})
            ])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(model_fn), capabilities=[UserInteraction(input_handler=my_handler)])
    result = await agent.run('Pick a color')
    assert isinstance(result.output, str)


# --- Test 5: Inline handler receives choices ---

async def test_ask_user_handler_receives_choices():
    """Verify the handler receives the choices argument from the model."""
    received_args: list[tuple[str, list[str] | None]] = []

    def my_handler(question: str, choices: list[str] | None = None) -> str:
        received_args.append((question, choices))
        return choices[0] if choices else 'default'

    call_count = 0

    def model_fn(messages: list[ModelRequest], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[
                ToolCallPart(tool_name='ask_user', args={'question': 'Pick one', 'choices': ['A', 'B']})
            ])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(model_fn), capabilities=[UserInteraction(input_handler=my_handler)])
    await agent.run('test')
    assert received_args == [('Pick one', ['A', 'B'])]


# --- Test 6: Spec serialization (Tier S) ---

def test_user_interaction_serialization():
    """UserInteraction should be constructible from spec with zero args."""
    cap = UserInteraction.from_spec()
    assert isinstance(cap, UserInteraction)
    assert cap.input_handler is None
    assert UserInteraction.get_serialization_name() == 'UserInteraction'


# --- Test 7: Instructions are provided ---

def test_user_interaction_instructions():
    """UserInteraction should provide system prompt instructions."""
    cap = UserInteraction()
    instructions = cap.get_instructions()
    assert 'ask_user' in instructions
    assert 'clarif' in instructions.lower()


# --- Test 8: Combined with other capabilities ---

async def test_user_interaction_combined_with_other_capabilities():
    """UserInteraction should compose with other capabilities."""
    from pydantic_ai.capabilities.instructions import Instructions

    def model_fn(messages: list[ModelRequest], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(
        FunctionModel(model_fn),
        capabilities=[
            Instructions('Be helpful.'),
            UserInteraction(),
        ],
    )
    result = await agent.run('Hello')
    assert isinstance(result.output, str)


# --- Test 9: Handler exception behavior ---

async def test_ask_user_handler_exception():
    """Exceptions from input_handler — verify behavior matches FunctionToolset handling."""

    def bad_handler(question: str, choices: list[str] | None = None) -> str:
        raise ValueError('handler failed')

    call_count = 0

    def model_fn(messages: list[ModelRequest], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[
                ToolCallPart(tool_name='ask_user', args={'question': 'test'})
            ])
        # After error, model should see the error message and produce final output
        return ModelResponse(parts=[TextPart(content='error handled')])

    agent = Agent(FunctionModel(model_fn), capabilities=[UserInteraction(input_handler=bad_handler)])
    # FunctionToolset may catch the exception and return it as a tool error to the model,
    # or it may propagate. This test verifies whichever behavior actually occurs.
    # If FunctionToolset catches and feeds back to model:
    result = await agent.run('test')
    # If it propagates instead, change to: pytest.raises(ValueError, match='handler failed')


# --- Test 10: Tool schema shape ---

async def test_ask_user_tool_schema():
    """Verify the ask_user tool has the expected JSON schema."""
    cap = UserInteraction()
    toolset = cap.get_toolset()
    defs = await toolset.tool_defs()
    assert len(defs) == 1
    td = defs[0]
    assert td.name == 'ask_user'
    schema = td.parameters_json_schema
    assert 'question' in schema.get('properties', {})
    assert 'choices' in schema.get('properties', {})
    assert 'question' in schema.get('required', [])


# --- Test 11: Streaming with deferred ask_user ---

async def test_ask_user_streaming():
    """Verify ask_user works correctly during streamed runs."""

    def model_fn(messages: list[ModelRequest], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[
            ToolCallPart(tool_name='ask_user', args={'question': 'Which one?'})
        ])

    agent = Agent(FunctionModel(model_fn), capabilities=[UserInteraction()])
    async with agent.run_stream('test') as stream:
        result = await stream.get_output()
    assert isinstance(result, DeferredToolRequests)
    assert result.calls[0].metadata['question'] == 'Which one?'
```

**Why:** Tests cover both modes (deferred and inline), sync and async handlers, free-text and structured choices, handler receives correct arguments, serialization round-trip, instructions content, capability composition, handler exception behavior, tool JSON schema shape, and streaming behavior. The `FunctionModel` pattern is used extensively in existing tests (`test_capabilities.py`, `test_tools.py`) for deterministic model behavior.

**Note on Test 9 (handler exceptions):** The prior plan assumed `ValueError` would propagate directly via `pytest.raises`. However, `FunctionToolset` may catch tool function exceptions and convert them to error `ToolReturnPart` messages fed back to the model (this is standard behavior for tool validation errors — see `toolsets/function.py` error handling). The test is written to accommodate whichever behavior actually occurs. During implementation, verify how `FunctionToolset` handles uncaught exceptions from `tool_plain` functions and adjust the assertion accordingly.

**Note on Test 10 (tool schema):** This test creates a standalone `UserInteraction()` and tests its `get_toolset()` directly — it does not create a separate agent. This verifies the tool definition shape without conflating with agent integration behavior.

### Step 5: Add documentation

**Where documentation lives:** Based on codebase research, there is no dedicated `docs/capabilities.md` file. Capabilities are documented in docstrings (`abstract.py:42-62`) and referenced from other docs. The most relevant existing docs page is `docs/deferred-tools.md` which documents the `CallDeferred`/`DeferredToolRequests` mechanism.

**Option A (preferred):** If a `docs/capabilities.md` exists or is created as part of the broader capabilities work (#10, #11-#16), add a `UserInteraction` section there.

**Option B (fallback):** Add documentation as a standalone `docs/user-interaction.md` page.

**In either location, add:**

```markdown
## UserInteraction

The `UserInteraction` capability provides an `ask_user` tool that lets the agent
ask the user clarifying questions mid-execution.

### YAML Spec

```yaml
capabilities:
  - UserInteraction
```

### Python (deferred mode — default)

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities.user_interaction import UserInteraction
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults

agent = Agent('openai:gpt-4o', capabilities=[UserInteraction()])

result = await agent.run('Help me pick a framework')
if isinstance(result.output, DeferredToolRequests):
    for call in result.output.calls:
        question = call.metadata['question']
        choices = call.metadata.get('choices')  # None for free-text
        answer = await get_user_answer(question, choices)  # Your UI logic

    result = await agent.run(
        'Help me pick a framework',
        message_history=result.all_messages(),
        deferred_tool_results=DeferredToolResults(calls={call.tool_call_id: answer}),
    )
```

### Python (inline mode — with callback)

```python
agent = Agent('openai:gpt-4o', capabilities=[
    UserInteraction(input_handler=lambda q, c: input(f'{q}: '))
])
# Questions are resolved inline — no deferred requests.
result = await agent.run('Help me pick a framework')
```

### How it works

When no `input_handler` is provided, the `ask_user` tool pauses execution using the
[deferred tools](deferred-tools.md) mechanism (`CallDeferred`). The host application
receives a `DeferredToolRequests` output with metadata containing the question, and
resumes the run with the user's answer via `DeferredToolResults`.

The `tool_call_id` correlation is handled automatically by the agent graph layer —
when `CallDeferred` is raised during tool execution, the graph associates it with
the current `ToolCallPart`'s ID. Your code only needs to map `call.tool_call_id` to
the user's answer in `DeferredToolResults.calls`.

When an `input_handler` is provided, questions are resolved inline by calling the
handler directly. The handler can be sync or async (`async def`).

### Difference from Approval

[Approval](deferred-tools.md#approval) authorizes tool execution (yes/no).
UserInteraction gathers information to continue the task (free-text or choices).
They use different exception types (`ApprovalRequired` vs `CallDeferred`) and
produce different entries in `DeferredToolRequests` (`.approvals` vs `.calls`).
```

**In `docs/deferred-tools.md`, add a cross-reference** near the `CallDeferred` section:

```markdown
!!! tip "UserInteraction capability"
    The [`UserInteraction`](capabilities.md#userinteraction) capability provides a
    built-in `ask_user` tool that uses `CallDeferred` to pause execution for user
    questions. See [UserInteraction](capabilities.md#userinteraction) for details.
```

**Why:** Users need to understand both modes, the deferred workflow, the automatic `tool_call_id` correlation, and how this differs from the approval mechanism.

## Files to Modify

| File | Change | Lines Affected | Citation |
|------|--------|----------------|----------|
| `pydantic_ai_slim/pydantic_ai/capabilities/__init__.py` | Add `UserInteraction` to `CAPABILITY_TYPES` dict; possibly to `DEFAULT_CAPABILITY_TYPES` depending on Open Question #1 | ~14-48 | Issue #42 Tier S requirement; `capabilities/__init__.py:14-19` |
| `pydantic_ai_slim/pydantic_ai/__init__.py` | Export `UserInteraction` | Near other capability imports | Pattern: `Instructions`, `Thinking`, `WebSearch` exports |
| `docs/deferred-tools.md` | Add cross-reference to UserInteraction | Near `CallDeferred` section | Existing doc at `docs/deferred-tools.md` |

## New Files

| File | Purpose | Modeled After |
|------|---------|---------------|
| `pydantic_ai_slim/pydantic_ai/capabilities/user_interaction.py` | `UserInteraction` capability class with `ask_user` tool | `pydantic_ai_slim/pydantic_ai/capabilities/web_search.py` (Tier S serializable capability), `tests/test_capabilities.py:560-600` (`ToolsetFuncCapability` pattern for `FunctionToolset` with `@tool_plain`) |
| `tests/test_user_interaction.py` | Tests for both deferred and inline modes, streaming, edge cases | `tests/test_capabilities.py` (capability testing patterns), `tests/test_tools.py:1363-3602` (`CallDeferred` testing patterns) |
| `docs/user-interaction.md` (or section in `docs/capabilities.md` if it exists) | User-facing documentation | `docs/deferred-tools.md` (structure and examples) |

## Test Plan

- **Deferred mode (no handler):** Verify `ask_user` raises `CallDeferred`, run produces `DeferredToolRequests` with correct metadata (`question`, `choices`), and resume with `DeferredToolResults` completes the run. `tool_call_id` correlation is automatic. Follow `test_tools.py:1363-1600` patterns. (Test 1)
- **Free-text vs. structured choices:** Verify metadata contains `choices` only when provided; omits key for free-text questions. (Test 2)
- **Inline handler (sync):** Verify sync handler is called and result returned directly (no `DeferredToolRequests`). (Test 3)
- **Inline handler (async):** Verify async handler is awaited correctly via `inspect.iscoroutinefunction()`. (Test 4)
- **Handler receives choices:** Verify the handler callable receives both `question` and `choices` arguments as passed by the model. (Test 5)
- **Serialization round-trip:** Verify `from_spec()` produces a working `UserInteraction` with no handler (Tier S), `get_serialization_name()` returns `'UserInteraction'`. (Test 6)
- **Instructions content:** Verify `get_instructions()` mentions `ask_user` and clarification guidance. (Test 7)
- **Capability composition:** Verify `UserInteraction` composes correctly with `Instructions` via `CombinedCapability`. Follow `test_capabilities.py` composition patterns. (Test 8)
- **Handler exception behavior:** Verify behavior when `input_handler` raises an exception — check whether `FunctionToolset` catches and feeds back as tool error, or propagates. (Test 9)
- **Tool schema:** Verify the `ask_user` tool's JSON schema has `question` (required string) and `choices` (optional array of strings). Test only the standalone capability's `get_toolset()` output. (Test 10)
- **Streaming:** Verify `ask_user` during a streamed run produces `DeferredToolRequests` correctly. (Test 11)

**Test infrastructure:** Use `FunctionModel` (`pydantic_ai.models.function`) for deterministic tests, following the pattern in `test_capabilities.py:44-72`.

## Documentation Changes

- **`pydantic_ai_slim/pydantic_ai/capabilities/user_interaction.py`:** Full module docstring explaining both modes. Docstrings on `UserInteraction` class, `input_handler` field (including note that only `async def` functions are awaited, and that `tool_timeout` applies to blocking handlers), and `ask_user` tool function.
- **User-facing documentation page:** New section or page covering YAML spec usage, Python usage with both modes, the deferred workflow with code examples, `tool_call_id` correlation explanation, and explicit comparison with Approval to clarify the distinction.
- **`docs/deferred-tools.md`:** Add a "See also" reference to `UserInteraction` as a built-in capability that uses the `CallDeferred` mechanism.

## Risks and Pitfalls

**Risk 1: Models producing `choices` as wrong type.**
The `choices` parameter is `list[str] | None`. Models might produce `choices: "Red"` (string instead of list) or `choices: [{"label": "Red"}]`.
- **Where:** `user_interaction.py`, the `ask_user` tool parameter definition.
- **Evidence:** Mastra uses a more complex Zod schema with explicit `.describe()` annotations for structured options (`packages/core/src/harness/tools.ts`).
- **Mitigation:** `list[str]` is the simplest possible type, minimizing model confusion. Pydantic-ai's `FunctionToolset` validates tool arguments against the JSON schema before invoking the tool (`toolsets/function.py`), so malformed args will produce a validation error that gets fed back to the model for retry. The `question` and `choices` fields have clear docstrings that appear in the JSON schema description.

**Risk 2: `input_handler` type not enforced at runtime.**
Users may pass lambdas with wrong signatures (e.g., `lambda q: input(q)` missing the `choices` parameter).
- **Where:** `user_interaction.py`, `input_handler` field and `ask_user` tool body.
- **Mitigation:** Document the expected signature clearly in the field docstring. A `lambda q, c: input(q)` that ignores `c` works fine. A `lambda q: input(q)` with wrong arity will fail at call time with a clear `TypeError`. Runtime signature validation is over-engineering for v1.

**Risk 3: `ask_user` tool wrapped in `ApprovalRequiredToolset`.**
If a user wraps the agent's toolset in `ApprovalRequiredToolset`, the `ask_user` tool would require approval before asking a question, which is nonsensical.
- **Where:** `toolsets/approval_required.py` and how `CombinedToolset` interacts with capability-provided toolsets.
- **Evidence:** Mastra explicitly marks `ask_user` as always-allowed, bypassing the permission system (`packages/core/src/harness/tools.ts`).
- **Mitigation:** Document that `ask_user` should not be wrapped in approval toolsets. Users can use `FilteredToolset` to exclude `ask_user` from approval requirements. A future enhancement could add `requires_approval=False` metadata to the tool definition, but this is not needed for v1. See Open Question #2 for whether the `@toolset.tool_plain` decorator's `requires_approval` parameter could be used proactively.

**Risk 4: Timeout behavior with inline handler.**
When `input_handler` is provided and blocks (e.g., terminal `input()`), the `FunctionToolset`'s tool timeout may fire.
- **Where:** `toolsets/function.py:83` (timeout parameter), `user_interaction.py` (tool definition).
- **Mitigation:** In deferred mode, `CallDeferred` is raised synchronously so timeout doesn't apply. In inline mode with a blocking handler, the agent's `tool_timeout` applies — this is actually desirable behavior to prevent indefinite hangs. Documented in the `input_handler` field docstring.

**Risk 5: Concurrent tool calls with `ask_user`.**
If the model calls `ask_user` alongside other tool calls in the same response, the other tool calls execute normally while `ask_user` produces a deferred entry. The existing `CallDeferred` handling in `_agent_graph.py:1365-1370` already supports this — multiple deferred calls are collected into `DeferredToolRequests.calls`.
- **Where:** `_agent_graph.py:1365-1370`
- **Mitigation:** No code change needed — already handled. Document that multiple `ask_user` calls in one model response produce multiple entries in `DeferredToolRequests.calls`.

**Risk 6: `for_run()` state isolation.**
The `input_handler` closure is captured at capability construction time. If a user needs different handlers per run, they'd need to create separate `UserInteraction` instances.
- **Where:** `abstract.py:92-100` (`for_run()` method), `user_interaction.py`
- **Evidence:** The `AbstractCapability` docs at `abstract.py:92-100` state "For per-run state isolation, use `for_run()` to return new instance."
- **Mitigation:** The default `for_run()` returns `self`, which is correct for `UserInteraction` — the same handler should be used across all runs of the same agent. Document that per-run handler customization requires creating separate `UserInteraction` instances. No `for_run()` override needed.

## Open Questions

1. **How does `Agent.from_spec()` resolve capability names?** The issue specifies "Tier S — zero-config," meaning `- UserInteraction` must work in YAML. `DEFAULT_CAPABILITY_TYPES` at `capabilities/__init__.py:14-19` is referenced by `agent/spec.py:19`. Does `Agent.from_spec()` look up capability names only in `DEFAULT_CAPABILITY_TYPES`, or does it also fall back to the broader `CAPABILITY_TYPES` dict? This determines whether `UserInteraction` must be added to `DEFAULT_CAPABILITY_TYPES` (which would make it "default" for all spec-based agents) or only to `CAPABILITY_TYPES` (available but opt-in). **Action:** During implementation, read `agent/spec.py` to determine the exact lookup path. If `CAPABILITY_TYPES` is used as fallback, add only there. If `DEFAULT_CAPABILITY_TYPES` is the sole source, add there too but note that this means spec-based agents will always have access to the `UserInteraction` name.
   - **Source:** `capabilities/__init__.py:14-19`, `agent/spec.py:19`

2. **Should `ask_user` proactively set `requires_approval=False`?** The `@toolset.tool_plain` decorator at `toolsets/function.py` may support a `requires_approval` parameter. Setting it to `False` would ensure that even if wrapped in `ApprovalRequiredToolset`, `ask_user` bypasses approval. This follows Mastra's "always-allowed" pattern. **Action:** During implementation, check whether `tool_plain` accepts `requires_approval` and whether `ApprovalRequiredToolset` respects it. If so, set `requires_approval=False` on the `ask_user` tool.
   - **Source:** `toolsets/function.py` (tool decorator), `toolsets/approval_required.py`

3. **How does `FunctionToolset` handle uncaught exceptions from tool functions?** Test 9 depends on this: does a `ValueError` from the handler propagate to the caller, or does `FunctionToolset` catch it and return an error `ToolReturnPart` to the model? **Action:** During implementation, check `FunctionToolset.call_tool()` error handling and write the test assertion accordingly.
   - **Source:** `toolsets/function.py` (error handling in `call_tool`)

4. **Streaming + deferred interaction.** When the agent calls `ask_user` during a streamed run (`agent.run_stream()`), does the stream end with `DeferredToolRequests` accessible via `stream.get_output()`? Test 11 assumes this but it should be verified against actual streaming API behavior with deferred tools.
   - **Source:** `_agent_graph.py` handles `CallDeferred` in `CallToolsNode`, but streaming + deferred interaction is not explicitly documented.

## References

- **Issue #42:** https://github.com/pydantic/pydantic-harness/issues/42
- **Issue #1 (tracking):** https://github.com/pydantic/pydantic-harness/issues/1
- **Issue #10 (AbstractCapability):** https://github.com/pydantic/pydantic-harness/issues/10
- **Issue #29 (Approval — different from UserInteraction):** https://github.com/pydantic/pydantic-harness/issues/29
- **`AbstractCapability` base class:** `pydantic_ai_slim/pydantic_ai/capabilities/abstract.py:1-247`
- **`CombinedCapability`:** `pydantic_ai_slim/pydantic_ai/capabilities/combined.py:1-340`
- **`WebSearch` capability (Tier S pattern):** `pydantic_ai_slim/pydantic_ai/capabilities/web_search.py`
- **`Toolset` capability (non-serializable pattern):** `pydantic_ai_slim/pydantic_ai/capabilities/toolset.py:1-22`
- **`FunctionToolset`:** `pydantic_ai_slim/pydantic_ai/toolsets/function.py:1-552`
- **`CallDeferred` exception:** `pydantic_ai_slim/pydantic_ai/exceptions.py:77-99`
- **`ApprovalRequired` exception:** `pydantic_ai_slim/pydantic_ai/exceptions.py:95-114`
- **`DeferredToolRequests`/`DeferredToolResults`:** `pydantic_ai_slim/pydantic_ai/tools.py:148-248`
- **Deferred tool handling in agent graph:** `pydantic_ai_slim/pydantic_ai/_agent_graph.py:1365-1370`
- **Capabilities registry:** `pydantic_ai_slim/pydantic_ai/capabilities/__init__.py:1-48`
- **Agent capability integration:** `pydantic_ai_slim/pydantic_ai/agent/__init__.py:383-453`
- **Agent per-run capability re-evaluation:** `pydantic_ai_slim/pydantic_ai/agent/__init__.py:974-985`
- **ToolManager hook invocation:** `pydantic_ai_slim/pydantic_ai/_tool_manager.py:230-290`
- **`ToolsetFuncCapability` test pattern:** `tests/test_capabilities.py:560-600`
- **Existing deferred tools tests:** `tests/test_tools.py:1363-3602`
- **Existing capability tests:** `tests/test_capabilities.py:1-2157`
- **Deferred tools documentation:** `docs/deferred-tools.md`
- **MCP elicitation (related pattern):** `pydantic_ai_slim/pydantic_ai/mcp.py:334-406`
- **Graph human-in-the-loop example:** `docs/graph.md:641-740`
- **`AbstractCapability.for_run()`:** `pydantic_ai_slim/pydantic_ai/capabilities/abstract.py:92-100`
- **Agent spec resolution:** `pydantic_ai_slim/pydantic_ai/agent/spec.py:19`
- **`DynamicToolset` / `ToolsetFunc`:** `pydantic_ai_slim/pydantic_ai/toolsets/_dynamic.py:16`
- **Mastra ask_user tool:** https://github.com/mastra-ai/mastra/blob/main/packages/core/src/harness/tools.ts
- **LangGraph interrupt:** https://github.com/langchain-ai/langgraph
- **CrewAI HumanInputProvider:** https://github.com/crewAIInc/crewAI

</details>

<details>
<summary>anthropic/claude-opus-4-5-20251101's Final Plan</summary>

## Goal

Provide a `UserInteraction` capability with an `ask_user` tool that pauses agent execution to gather clarifying information from the user, supporting both free-text questions and structured multiple-choice options.

## Prior Art & Competitive Landscape

| Framework | Approach | Citation | Learn/Avoid |
|-----------|----------|----------|-------------|
| **Mastra** | Dedicated `ask_user` tool with structured options (label + description), event-driven (`ask_question` event), promise-based resolution, abort signal support | [tools.ts:18-76](https://github.com/mastra-ai/mastra/blob/main/packages/core/src/harness/tools.ts) | **Learn:** Structured options schema, event-driven architecture, always-allowed tool. **Avoid:** No timeout mechanism |
| **LangGraph** | Generic `interrupt()` primitive that pauses anywhere, `Command(resume=...)` to continue, checkpointer-required | [types.py interrupt function](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py) | **Learn:** Generic primitive enables flexibility. **Avoid:** Too generic—no tool abstraction, no instructions, requires checkpointer |
| **CrewAI** | `human_input=True` on tasks for post-completion review, `request_human_input()` in hooks | [task.py:158-162, human_input.py](https://github.com/crewAIInc/crewAI/blob/main/lib/crewai/src/crewai/task.py) | **Learn:** Protocol-based provider abstraction. **Avoid:** Task-level only—not mid-execution questioning |

**Key insight from Mastra**: The `ask_user` tool emits an event (`ask_question`) and awaits resolution via `respondToQuestion()`. This event-driven pattern enables UI integration without blocking.

**Key insight from pydantic-ai**: The deferred tool infrastructure in `pydantic_ai_slim/pydantic_ai/tools.py:148-240` provides `CallDeferred` exception support. When a tool raises `CallDeferred`, the agent pauses and returns `DeferredToolRequests`. The `metadata` from `CallDeferred` is stored in `DeferredToolRequests.metadata` keyed by `tool_call_id` (handled in `_agent_graph.py:1365-1370`). When resuming with `DeferredToolResults.calls[tool_call_id]`, that value is injected as the tool's return value—the tool function is **not** re-executed.

## Approach

The `UserInteraction` capability will provide an `ask_user` tool that raises `CallDeferred` when invoked, pausing agent execution and returning `DeferredToolRequests` to the caller. The caller provides the user's response via `DeferredToolResults`, and that response becomes the tool's return value directly—no re-execution occurs.

This approach leverages pydantic-ai's existing deferred tool infrastructure (`pydantic_ai_slim/pydantic_ai/tools.py:148-240`). The `CallDeferred` exception supports `metadata` for passing structured data (the question and choices) to the UI layer. The code in `_agent_graph.py:1365-1370` catches `CallDeferred` and populates `DeferredToolRequests.metadata[tool_call_id]` with the exception's metadata. When the agent resumes with `deferred_tool_results`, the value in `DeferredToolResults.calls[tool_call_id]` is returned as the tool result.

The issue specifies "Tier S" serializability (zero-config), so `UserInteraction()` will work with no arguments and `from_spec()` will accept no arguments.

**Why `CallDeferred` over `ApprovalRequired`?** `ApprovalRequired` is semantically for authorization (yes/no) and triggers tool re-execution with `ctx.tool_call_approved=True`. `CallDeferred` is for external execution returning arbitrary results—the response is information, not a boolean approval. The issue explicitly distinguishes UserInteraction from Approval (#29).

## Implementation Steps

### Step 1: Create Choice TypedDict with proper required/optional handling

Create a typed schema for structured choices matching Mastra's pattern, with `label` required and `description` optional.

**File:** `pydantic_ai_slim/pydantic_ai/capabilities/user_interaction.py` (new file)

```python
from __future__ import annotations

from typing import NotRequired, TypedDict

class Choice(TypedDict):
    """A choice option for ask_user tool.
    
    Attributes:
        label: Short display text for this option (1-5 words). Required.
        description: Explanation of what this option means. Optional.
    """
    label: str
    description: NotRequired[str]
```

**Why:** Mastra uses `{label: string, description?: string}` for options ([tools.ts:27-31](https://github.com/mastra-ai/mastra/blob/main/packages/core/src/harness/tools.ts#L27-L31)). Using `NotRequired` from typing ensures `label` is required while `description` remains optional. `TypedDict` with `NotRequired` generates correct JSON schema via pydantic v2's schema generation.

### Step 2: Implement the `ask_user` tool using decorator pattern

Create the toolset using `FunctionToolset`'s decorator pattern. Use `@toolset.tool_plain` since the tool doesn't need `RunContext`.

**File:** `pydantic_ai_slim/pydantic_ai/capabilities/user_interaction.py` (continued)

```python
from typing import Any

from pydantic_ai.exceptions import CallDeferred
from pydantic_ai.toolsets.function import FunctionToolset

def _create_ask_user_toolset() -> FunctionToolset[Any]:
    """Create the ask_user toolset."""
    toolset: FunctionToolset[Any] = FunctionToolset()

    @toolset.tool_plain
    async def ask_user(
        question: str,
        choices: list[Choice] | None = None,
    ) -> str:
        """Ask the user a clarifying question.

        Use this when you need clarification, want to validate assumptions,
        or need the user to make a decision between options.

        Args:
            question: The question to ask. Should be clear and specific.
            choices: Optional list of choices. If provided, user selects one.
                     If omitted, user provides free-text response.

        Returns:
            The user's response as a string.
        """
        # Build metadata to surface to the UI layer
        metadata: dict[str, Any] = {'question': question}
        if choices is not None:
            metadata['choices'] = choices

        # Raise CallDeferred - the result from DeferredToolResults.calls[tool_call_id]
        # will be returned as this tool's result (no re-execution occurs)
        raise CallDeferred(metadata=metadata)

    return toolset
```

**Why:** 
- Uses `@toolset.tool_plain` decorator since `RunContext` is not needed—the tool only raises `CallDeferred` (`pydantic_ai_slim/pydantic_ai/toolsets/function.py:179-240`)
- The tool raises `CallDeferred`—when resumed, the value from `DeferredToolResults.calls[tool_call_id]` is returned as the tool result
- Defined inside a factory function to avoid module-level generic type issues and enable clean testing

### Step 3: Implement the `UserInteraction` capability class

Create the capability that provides the tool and instructions.

**File:** `pydantic_ai_slim/pydantic_ai/capabilities/user_interaction.py` (continued)

```python
from dataclasses import dataclass
from typing import Any

from pydantic_ai import _instructions
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.toolsets.abstract import AbstractToolset

@dataclass
class UserInteraction(AbstractCapability[Any]):
    """Capability for asking users clarifying questions during agent execution.

    Provides an `ask_user` tool that pauses execution to gather user input.
    Supports both free-text questions and structured multiple-choice options.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.capabilities import UserInteraction
        from pydantic_ai.tools import DeferredToolRequests

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[UserInteraction()],
            output_type=str | DeferredToolRequests,
        )

        result = await agent.run("Book me a flight")
        if isinstance(result.output, DeferredToolRequests):
            # Handle the question - see deferred-tools docs
            ...
        ```
    """

    def get_instructions(self) -> _instructions.Instructions[Any] | None:
        return (
            "When uncertain about the user's intent or need clarification, "
            "use the ask_user tool to ask a clarifying question before proceeding."
        )

    def get_toolset(self) -> AbstractToolset[Any] | None:
        return _create_ask_user_toolset()

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'UserInteraction'

    @classmethod
    def from_spec(cls) -> 'UserInteraction':
        """Create UserInteraction from spec (zero-config)."""
        return cls()
```

**Why:** 
- Follows the `Toolset` capability pattern from `capabilities/toolset.py:1-19`
- Uses `AbstractCapability[Any]` matching the pattern in existing capabilities
- Instructions match issue specification: "When uncertain about the user's intent, use the ask_user tool to clarify"
- `get_serialization_name()` returns `'UserInteraction'` for Tier S YAML support
- `from_spec()` accepts no arguments for zero-config. If users pass unexpected arguments in YAML, the registry will raise an error (better than silent ignoring).

### Step 4: Export from capabilities package

**File:** `pydantic_ai_slim/pydantic_ai/capabilities/__init__.py`

Add to existing exports (around line 14-19 where `DEFAULT_CAPABILITY_TYPES` is defined):

```python
from pydantic_ai.capabilities.user_interaction import UserInteraction, Choice

# Add to DEFAULT_CAPABILITY_TYPES tuple
DEFAULT_CAPABILITY_TYPES = (Instructions, ModelSettings, Thinking, WebSearch, UserInteraction)

__all__ = [
    # ... existing exports ...
    'UserInteraction',
    'Choice',
]
```

**Why:** 
- Makes `UserInteraction` importable from `pydantic_ai.capabilities`
- Adding to `DEFAULT_CAPABILITY_TYPES` registers it for spec parsing, per the pattern at `capabilities/__init__.py:14-19`
- Export `Choice` for users who want to construct choices programmatically

### Step 5: Add tests

**File:** `tests/test_user_interaction.py` (new file)

```python
from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.capabilities import UserInteraction, Choice
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelResponse, ToolCallPart


def make_ask_user_model(question: str, choices: list[dict[str, str]] | None = None):
    """Create a FunctionModel that calls ask_user with the given args."""
    call_count = 0

    def model_function(messages: list, info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # First call: have the model call ask_user
            args: dict[str, Any] = {'question': question}
            if choices is not None:
                args['choices'] = choices
            return ModelResponse(
                parts=[ToolCallPart(tool_name='ask_user', args=args, tool_call_id='call_1')]
            )
        else:
            # After resumption: return final text
            return ModelResponse.from_text('Task completed with user input.')

    return FunctionModel(model_function)


async def test_ask_user_defers_with_question():
    """Test that ask_user tool pauses execution with question metadata."""
    agent = Agent(
        make_ask_user_model('What city?'),
        capabilities=[UserInteraction()],
        output_type=str | DeferredToolRequests,
    )

    result = await agent.run('Book a flight')

    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.calls) == 1

    call = result.output.calls[0]
    assert call.tool_name == 'ask_user'

    # Verify metadata is stored under the tool_call_id
    metadata = result.output.metadata.get(call.tool_call_id, {})
    assert metadata == snapshot({'question': 'What city?'})


async def test_ask_user_with_choices():
    """Test ask_user with structured choices."""
    choices = [
        {'label': 'Morning', 'description': 'Before noon'},
        {'label': 'Afternoon', 'description': 'After noon'},
    ]

    agent = Agent(
        make_ask_user_model('When?', choices),
        capabilities=[UserInteraction()],
        output_type=str | DeferredToolRequests,
    )

    result = await agent.run('Schedule meeting')

    assert isinstance(result.output, DeferredToolRequests)
    metadata = result.output.metadata[result.output.calls[0].tool_call_id]
    assert metadata == snapshot({
        'question': 'When?',
        'choices': [
            {'label': 'Morning', 'description': 'Before noon'},
            {'label': 'Afternoon', 'description': 'After noon'},
        ],
    })


async def test_ask_user_resume_with_answer():
    """Test resuming after user provides answer."""
    agent = Agent(
        make_ask_user_model('Which city?'),
        capabilities=[UserInteraction()],
        output_type=str | DeferredToolRequests,
    )

    # First call: agent asks question
    result = await agent.run('Book flight')

    assert isinstance(result.output, DeferredToolRequests)
    call_id = result.output.calls[0].tool_call_id

    # Second call: resume with answer
    # The answer string is directly injected as the tool result
    deferred_results = DeferredToolResults(
        calls={call_id: 'New York'},
    )

    result2 = await agent.run(
        None,  # No new user message - verified against Agent.run() signature
        message_history=result.all_messages(),
        deferred_tool_results=deferred_results,
    )

    assert result2.output == 'Task completed with user input.'


def test_user_interaction_instructions():
    """Test that capability provides instructions."""
    cap = UserInteraction()
    instructions = cap.get_instructions()
    assert instructions is not None
    assert 'ask_user' in instructions
    assert 'uncertain' in instructions.lower()


def test_user_interaction_serialization():
    """Test Tier S serialization support."""
    assert UserInteraction.get_serialization_name() == 'UserInteraction'

    cap = UserInteraction.from_spec()
    assert isinstance(cap, UserInteraction)


def test_user_interaction_toolset():
    """Test that capability provides a toolset with ask_user tool."""
    cap = UserInteraction()
    toolset = cap.get_toolset()
    assert toolset is not None


async def test_ask_user_tool_schema_has_choices_structure():
    """Test that the ask_user tool's JSON schema correctly represents Choice TypedDict."""
    from pydantic_ai.toolsets.function import FunctionToolset
    
    toolset = _create_ask_user_toolset()
    # Get tool definitions to inspect schema
    # This verifies TypedDict with NotRequired generates correct JSON schema
    tools = await toolset.get_tool_defs(None)  # type: ignore
    ask_user_tool = next(t for t in tools if t.name == 'ask_user')
    
    # Verify choices parameter schema
    choices_schema = ask_user_tool.parameters_json_schema['properties']['choices']
    assert choices_schema['type'] == 'array'
    # Items should have 'label' as required, 'description' as optional
    items_schema = choices_schema['items']
    assert 'label' in items_schema['properties']
    assert 'description' in items_schema['properties']
    assert 'label' in items_schema.get('required', [])
```

**Why:** 
- Uses `FunctionModel` for reliable deferred tool testing (pattern from `tests/test_tools.py`)
- Tests use plain string for `DeferredToolResults.calls` value—verified against existing deferred tool tests
- Uses `snapshot()` for metadata assertions per existing test patterns
- Tests `agent.run(None, ...)` for resumption—verified that `Agent.run()` accepts `None` for user_prompt
- Includes schema test to verify `TypedDict` with `NotRequired` generates correct JSON schema for LLM consumption
- Covers: basic deferral, choices metadata, resumption flow, instructions, serialization, toolset creation, schema generation

### Step 6: Add documentation

**File:** `docs/user-interaction.md` (new file)

```markdown
# UserInteraction Capability

The `UserInteraction` capability provides an `ask_user` tool that allows agents to pause execution and ask the user clarifying questions.

## Basic Usage

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import UserInteraction
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults

agent = Agent(
    'openai:gpt-4o',
    capabilities=[UserInteraction()],
    output_type=str | DeferredToolRequests,  # Required for deferred tools
)

# Agent may pause to ask questions
result = await agent.run("Book me a flight to see my family")

if isinstance(result.output, DeferredToolRequests):
    # Agent asked a question
    for call in result.output.calls:
        metadata = result.output.metadata.get(call.tool_call_id, {})
        question = metadata.get('question')
        choices = metadata.get('choices')  # May be None for free-text

        # Display question to user, get their answer
        answer = await get_user_response(question, choices)

        # Resume with answer - the answer string becomes the tool result
        deferred_results = DeferredToolResults(
            calls={call.tool_call_id: answer}
        )
        result = await agent.run(
            None,
            message_history=result.all_messages(),
            deferred_tool_results=deferred_results,
        )
```

## Structured Choices

The `ask_user` tool supports optional structured choices:

```python
# Agent might call ask_user with:
# question="What time works best?"
# choices=[
#     {"label": "Morning", "description": "9 AM - 12 PM"},
#     {"label": "Afternoon", "description": "1 PM - 5 PM"},
# ]
```

When choices are provided, your UI should present them for selection rather than free-text input. The `description` field is optional.

## YAML Agent Spec

```yaml
model: openai:gpt-4o
capabilities:
  - UserInteraction
output_type: str | DeferredToolRequests
```

## Relationship to Approval

UserInteraction is for **gathering information** — "Which city?", "What date?"

[Approval](deferred-tools.md) is for **authorizing actions** — "Can I delete this file?"

Use UserInteraction when the agent needs more context to proceed.
Use Approval when the agent knows what to do but needs permission.

## How It Works

Under the hood, `ask_user` raises `CallDeferred` with the question metadata. This:

1. Pauses the agent and returns `DeferredToolRequests`
2. The metadata contains `question` and optionally `choices`
3. When you resume with `DeferredToolResults`, the answer string you provide becomes the tool's return value
4. The agent continues with the user's response in context

See [Deferred Tools](deferred-tools.md) for more details on the underlying mechanism.
```

**Why:** 
- Placed at `docs/user-interaction.md` since there's no existing `docs/capabilities/` directory
- Documents that answer is a plain string (not a dict), matching the actual API
- Cross-references existing deferred-tools documentation
- Shows both Python and YAML usage

## Files to Modify

| File | Change | Lines Affected | Citation |
|------|--------|----------------|----------|
| `pydantic_ai_slim/pydantic_ai/capabilities/__init__.py` | Export UserInteraction, Choice; add to DEFAULT_CAPABILITY_TYPES | ~5 lines | Pattern from lines 14-19 |

## New Files

| File | Purpose | Modeled After |
|------|---------|---------------|
| `pydantic_ai_slim/pydantic_ai/capabilities/user_interaction.py` | UserInteraction capability implementation | `capabilities/toolset.py` (wrapper pattern), `toolsets/function.py` (decorator pattern) |
| `tests/test_user_interaction.py` | Unit tests | `tests/test_tools.py` (deferred tool tests with FunctionModel) |
| `docs/user-interaction.md` | Documentation | `docs/deferred-tools.md` |

## Test Plan

**Existing test patterns to follow:**
- `tests/test_tools.py`: Uses `FunctionModel` for controlling tool call behavior
- `tests/test_capabilities.py:559-606`: Capability providing toolset pattern

**Scenarios to cover:**
1. Basic question without choices → verify `CallDeferred` raised, metadata contains question
2. Question with structured choices → verify choices in metadata
3. Resumption with answer → verify answer string becomes tool result, agent continues
4. Multiple questions in one run → verify multiple calls in DeferredToolRequests (if model calls tool twice)
5. Instructions present → verify `get_instructions()` returns expected text
6. Serialization → verify `get_serialization_name()` and `from_spec()` work correctly
7. Toolset creation → verify `get_toolset()` returns a toolset with ask_user tool
8. **JSON schema generation** → verify `Choice` TypedDict generates correct schema with required `label` and optional `description`

**Test infrastructure:**
- Use `FunctionModel` to simulate model calling `ask_user` with specific arguments
- Use `snapshot()` for metadata assertions
- Assert on `result.all_messages()` for resumption flow

## Documentation Changes

- **New file:** `docs/user-interaction.md` — full capability documentation
- **Update:** `docs/index.md` (if exists) — add link to UserInteraction
- **Update:** `docs/deferred-tools.md` — add cross-reference to UserInteraction as a use case
- **Docstrings:** Full docstrings on `UserInteraction` class, `Choice` TypedDict, and `ask_user` function

## Risks and Pitfalls

### Risk 1: `DeferredToolRequests` must be in output_type union
**What:** Users will forget to include `DeferredToolRequests` in their agent's `output_type`, causing runtime errors.
**Where:** `pydantic_ai_slim/pydantic_ai/tools.py:148-150`, `tests/test_tools.py:1714-1716` validates this constraint.
**Evidence:** Test explicitly checks that `DeferredToolRequests` alone fails validation.
**Mitigation:** Clear documentation, helpful error messages. The example code prominently shows `output_type=str | DeferredToolRequests`.

### Risk 2: Confusion between CallDeferred and ApprovalRequired
**What:** Developers may use `ApprovalRequired` for user questions or vice versa.
**Where:** `pydantic_ai_slim/pydantic_ai/exceptions.py:73-107` defines both.
**Evidence:** Issue #42 explicitly states "Different from Approval (#29)".
**Mitigation:** Documentation clearly explains the distinction. CallDeferred is for external results (including user answers); ApprovalRequired is for yes/no authorization that triggers re-execution.

### Risk 3: Answer format assumptions
**What:** Users might pass dicts like `{'answer': 'New York'}` instead of plain strings.
**Where:** `DeferredToolResults.calls` value type is `Any` per codebase research.
**Evidence:** The `ExternalToolset` pattern shows that `calls` values are injected directly as tool results.
**Mitigation:** Document clearly that the answer should be a plain string. Add a note in the docs showing the correct format.

### Risk 4: No timeout mechanism
**What:** If user never responds, agent remains paused indefinitely.
**Where:** This is inherent in the deferred tool design.
**Evidence:** Mastra's implementation also lacks timeout (noted in competitive analysis).
**Mitigation:** This is an application-level concern—the caller can implement timeouts. Document that long-running pauses should be handled by the orchestration layer.

### Risk 5: Choice validation at application layer
**What:** The capability doesn't validate that user's answer matches one of the provided choices.
**Where:** The `ask_user` tool accepts any string response.
**Evidence:** Mastra's implementation also doesn't validate at tool level—UI handles it.
**Mitigation:** Document that choice validation is the application's responsibility. The agent will receive whatever answer is provided.

### Risk 6: TypedDict JSON schema generation
**What:** `TypedDict` with `NotRequired` might not generate correct JSON schema for LLM consumption.
**Where:** `pydantic_ai_slim/pydantic_ai/toolsets/function.py` generates schemas from type hints.
**Evidence:** Pydantic v2 supports `TypedDict` schema generation, but edge cases may exist.
**Mitigation:** Include explicit test (`test_ask_user_tool_schema_has_choices_structure`) that verifies the generated schema has `label` as required and `description` as optional.

## Open Questions

1. **Multiple questions per turn:** Can the model call `ask_user` multiple times before yielding? If so, `DeferredToolRequests.calls` will contain multiple entries. The current design supports this, but the UX implications (batch vs sequential questions) are left to the application.
   - **Source of ambiguity:** `DeferredToolRequests.calls` is a list, suggesting batching is supported.

2. **Integration with AG-UI:** Should this capability integrate with the AG-UI protocol documented in `docs/ui/ag-ui.md`? The `ask_question` event from Mastra aligns with AG-UI's event-driven model.
   - **Source of ambiguity:** AG-UI is for frontend protocols; unclear if `ask_user` should emit AG-UI compatible events.

3. **Choice-only responses:** Should there be a way to enforce that only choice labels are valid responses (not arbitrary strings)? Currently this is left to the application.
   - **Source of ambiguity:** Mastra handles this at UI layer, not tool layer.

## References

- **Issue #42:** https://github.com/pydantic/pydantic-harness/issues/42
- **Issue #10 (AbstractCapability):** https://github.com/pydantic/pydantic-harness/issues/10
- **Issue #29 (Approval - distinct from UserInteraction):** https://github.com/pydantic/pydantic-harness/issues/29
- **Issue #1 (Tracking):** https://github.com/pydantic/pydantic-harness/issues/1
- **CallDeferred/ApprovalRequired:** `pydantic_ai_slim/pydantic_ai/exceptions.py:73-107`
- **DeferredToolRequests/Results:** `pydantic_ai_slim/pydantic_ai/tools.py:148-240`
- **Deferred handling in agent graph:** `pydantic_ai_slim/pydantic_ai/_agent_graph.py:1365-1370`
- **AbstractCapability:** `pydantic_ai_slim/pydantic_ai/capabilities/abstract.py:38-220`
- **Toolset capability pattern:** `pydantic_ai_slim/pydantic_ai/capabilities/toolset.py:1-19`
- **FunctionToolset decorators:** `pydantic_ai_slim/pydantic_ai/toolsets/function.py:83-150, 179-240`
- **DEFAULT_CAPABILITY_TYPES:** `pydantic_ai_slim/pydantic_ai/capabilities/__init__.py:14-19`
- **Agent.run() signature:** `pydantic_ai_slim/pydantic_ai/agent/__init__.py` (accepts `None` for user_prompt)
- **Deferred tools tests:** `tests/test_tools.py:1330-1720`
- **Capability tests:** `tests/test_capabilities.py:559-606`
- **Mastra ask_user implementation:** https://github.com/mastra-ai/mastra/blob/main/packages/core/src/harness/tools.ts
- **LangGraph interrupt:** https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/types.py
- **CrewAI human input:** https://github.com/crewAIInc/crewAI/blob/main/lib/crewai/src/crewai/task.py

</details>

<details>
<summary>anthropic/claude-opus-4-6's Research</summary>

## Codebase Research

Now I have a comprehensive understanding. Let me compile the findings.

---

## Relevant Code — file path, lines, what's there, what imports it

### AbstractCapability base class
- **File:** `pydantic_ai_slim/pydantic_ai/capabilities/abstract.py` (lines 1-247)
- **What:** `AbstractCapability[AgentDepsT]` is a dataclass with 18 hook methods covering run, model request, tool validate, tool execute, event stream, and tool preparation lifecycles. Also includes `for_run()` for per-run state isolation, `get_instructions()`, `get_model_settings()`, `get_toolset()`, `get_builtin_tools()`, and serialization methods (`get_serialization_name()`, `from_spec()`).
- **Imported by:** `_agent_graph.py:24`, `_tool_manager.py:25`, `agent/__init__.py:46`, `capabilities/__init__.py`, all capability subclasses, `agent/spec.py:19`, `tests/test_capabilities.py:21`

### CombinedCapability
- **File:** `pydantic_ai_slim/pydantic_ai/capabilities/combined.py` (lines 1-340)
- **What:** Composite pattern. Instructions concatenated, settings merged, toolsets combined via `CombinedToolset`, hooks chained (forward for `before_*`, reversed for `after_*`, reversed-wrapped for `wrap_*`).
- **Imported by:** `capabilities/__init__.py`, `agent/__init__.py:46`

### Toolset capability (existing pattern for capability providing tools)
- **File:** `pydantic_ai_slim/pydantic_ai/capabilities/toolset.py` (lines 1-22)
- **What:** Simple wrapper that takes an `AbstractToolset[AgentDepsT]` and returns it via `get_toolset()`. Opts out of serialization (`get_serialization_name() -> None`).

### FunctionToolset (tool registration)
- **File:** `pydantic_ai_slim/pydantic_ai/toolsets/function.py` (lines 1-552)
- **What:** `FunctionToolset` lets Python functions be used as tools. Key decorators: `@toolset.tool` (takes `RunContext`) and `@toolset.tool_plain` (no `RunContext`). Supports `requires_approval`, `sequential`, `metadata`, and `timeout` options.

### ToolsetFunc pattern (dynamic tool creation from RunContext)
- **File:** `pydantic_ai_slim/pydantic_ai/toolsets/_dynamic.py` (line 16: `ToolsetFunc`)
- **What:** `ToolsetFunc` is a type alias for `Callable[[RunContext[AgentDepsT]], AbstractToolset[AgentDepsT] | Awaitable[...]]`. The `DynamicToolset` wrapper evaluates the factory at `for_run()` or `for_run_step()` time.
- **Used in:** `AbstractCapability.get_toolset()` return type includes `ToolsetFunc` (abstract.py line 103)

### Capabilities `__init__.py` (registry)
- **File:** `pydantic_ai_slim/pydantic_ai/capabilities/__init__.py` (lines 1-48)
- **What:** Exports `DEFAULT_CAPABILITY_TYPES = (Instructions, ModelSettings, Thinking, WebSearch)` for spec serialization. `CAPABILITY_TYPES` dict maps names to all built-in capability classes.

### Agent integration with capabilities
- **File:** `pydantic_ai_slim/pydantic_ai/agent/__init__.py` (lines 383-453)
- **What:** `Agent.__init__` wraps capabilities into `CombinedCapability` at `self._root_capability` (line 388). Extracts instructions (line 409), builtin tools (line 422), model settings (line 424), and toolsets (line 450) at construction time.
- **Per-run re-extraction:** Lines 974-985: `for_run()` called on root capability; if different instance returned, all `get_*()` methods re-evaluated.
- **ToolManager creation:** Line 1022-1023: `ToolManager` receives `root_capability=run_capability` for hook invocation.

### ToolManager hook invocation
- **File:** `pydantic_ai_slim/pydantic_ai/_tool_manager.py` (lines 230-290)
- **What:** `_run_validate_hooks()` chains `before_tool_validate` → `wrap_tool_validate` → `after_tool_validate`. `_run_execute_hooks()` chains `before_tool_execute` → `wrap_tool_execute` → `after_tool_execute`. Both delegate to `root_capability` (the `CombinedCapability` or per-run variant).

### Deferred tool infrastructure (existing pause/resume mechanism)
- **File:** `pydantic_ai_slim/pydantic_ai/exceptions.py` (lines 77-114)
- **What:** `CallDeferred(Exception)` and `ApprovalRequired(Exception)` — raised from tool functions to pause execution. Both carry optional `metadata: dict[str, Any] | None`.
- **File:** `pydantic_ai_slim/pydantic_ai/tools.py` (lines 161-248)
- **What:** `DeferredToolRequests` (output type with `calls` and `approvals` lists), `ToolApproved`, `ToolDenied`, `DeferredToolResults` (input for resumption with `calls`, `approvals`, `metadata` dicts).
- **File:** `pydantic_ai_slim/pydantic_ai/_agent_graph.py` (lines 1365-1370)
- **What:** Tool execution catches `CallDeferred` and `ApprovalRequired`, populates deferred call metadata.

### MCP Elicitation (existing user-input pattern)
- **File:** `pydantic_ai_slim/pydantic_ai/mcp.py` (lines 334-335, 390-406)
- **What:** `elicitation_callback: ElicitationFnT | None` on MCPServer. Callback receives `ElicitRequestParams` and returns `ElicitResult` with actions: accept/decline/cancel.
- **Docs:** `docs/mcp/client.md` (lines 609-746) documents the full elicitation flow.

### ExternalToolset (tools whose results are produced externally)
- **File:** `pydantic_ai_slim/pydantic_ai/toolsets/external.py` (lines 1-53)
- **What:** `ExternalToolset` holds tool definitions with `kind='external'`. `call_tool()` raises `NotImplementedError`. Tools produce `DeferredToolRequests` with the `calls` list.

## Type System — key types/protocols/interfaces with actual code

### `AbstractCapability.get_toolset()` return type
```python
# abstract.py:103
def get_toolset(self) -> AbstractToolset[AgentDepsT] | ToolsetFunc[AgentDepsT] | None:
```
Returning `ToolsetFunc` (a callable) wraps it in `DynamicToolset` inside `CombinedCapability.get_toolset()` (combined.py:79-80).

### `RunContext[AgentDepsT]` — available during tool execution
- **File:** `pydantic_ai_slim/pydantic_ai/_run_context.py` (lines 28-95)
- **Key fields:** `deps`, `model`, `usage`, `prompt`, `messages`, `run_step`, `tool_call_id`, `tool_name`, `retry`, `tool_call_approved`, `tool_call_metadata`, `model_settings`

### `Instructions[AgentDepsT]` type alias
- **File:** `pydantic_ai_slim/pydantic_ai/_instructions.py` (lines 1-19)
- **What:** `TemplateStr | str | SystemPromptFunc | Sequence[...] | None`

### `ToolDefinition` — what's sent to the model
- **File:** `pydantic_ai_slim/pydantic_ai/tools.py` (line 509ff)
- **Key fields:** `name`, `description`, `parameters_json_schema`, `kind` (Literal: 'function', 'output', 'unapproved', 'external')

### `AgentDepsT` — generic type for dependencies
- **File:** `pydantic_ai_slim/pydantic_ai/_run_context.py` (line 23)
- **What:** `TypeVar('AgentDepsT', default=None, contravariant=True)`

## Call Chains — trace execution paths

### Capability → Tool available to model
1. `Agent.__init__` creates `CombinedCapability(capabilities)` → `self._root_capability` (agent/__init__.py:388)
2. `self._root_capability.get_toolset()` called at construction (agent/__init__.py:450), stored in `self._cap_toolsets`
3. At run time, `for_run()` re-evaluates if capability changed (agent/__init__.py:975-985)
4. `self._get_toolset()` (agent/__init__.py:2014ff) combines all toolset sources into `CombinedToolset`
5. `toolset.for_run(initial_ctx)` called (agent/__init__.py:1021)
6. `ToolManager` created with the toolset (agent/__init__.py:1022-1023)
7. `ToolManager.for_run_step()` gets tool definitions (tool_manager.py:102-120)
8. `ModelRequestNode` sends tool definitions to the model

### Tool call execution with capability hooks
1. Model responds with `ToolCallPart`
2. `CallToolsNode` processes response (_agent_graph.py:736ff)
3. `ToolManager.validate_tool_call()` → `_run_validate_hooks()` → capability hooks chain (_tool_manager.py:229-262)
4. `ToolManager.execute_tool_call()` → `_run_execute_hooks()` → capability hooks chain (_tool_manager.py:264-290)
5. If tool raises `CallDeferred` or `ApprovalRequired`, execution pauses and `DeferredToolRequests` is produced (_agent_graph.py:1365-1370)

### Deferred tool resume
1. Next `agent.run()` call with `deferred_tool_results=DeferredToolResults(...)` and `message_history=messages`
2. `CallToolsNode` checks `tool_call_results` for deferred results (_agent_graph.py:740)
3. If `ToolApproved`, tool is re-executed with approval; if `ToolDenied`, denial message returned to model

## Test Coverage — test files, what they cover, gaps

### `tests/test_capabilities.py` (2157+ lines)
- **Covers:** spec construction, `from_spec`, `for_run`, combined capabilities, hook ordering, tool validate/execute hooks, wrap hooks, streaming hooks, event stream wrapping, `prepare_tools`, `wrap_run_step`, concurrent run isolation, `ToolsetFunc` capabilities
- **Key test patterns:** `LoggingCapability` for hook order verification, `FunctionModel` for deterministic testing
- **Gaps:** No test for a capability that uses `CallDeferred` or `ApprovalRequired` from within a capability-provided tool. No test for a capability that provides a tool requiring user input.

### `tests/test_tools.py` — deferred tools testing
- **Lines 1363-3602:** Extensive testing of `ApprovalRequired`, `CallDeferred`, `ToolApproved`, `ToolDenied`, `DeferredToolRequests`, `DeferredToolResults`, override args, metadata
- **Covers:** The full pause/resume lifecycle for approval and external tools

### `tests/test_template.py`
- Line 16: Imports `AbstractCapability` for template string capability tests (152ff)

## Documentation — docs files, docstrings, discrepancies

### Deferred tools documentation
- **File:** `docs/deferred-tools.md` — Full documentation on `ApprovalRequired`, `CallDeferred`, `DeferredToolRequests`, `DeferredToolResults` with complete examples
- Shows the pattern: tool raises exception → run ends with `DeferredToolRequests` → user gathers results → resumes with `DeferredToolResults`

### Toolsets documentation
- **File:** `docs/toolsets.md` (lines 328-332) — `ApprovalRequiredToolset` documentation
- Links to deferred-tools.md for human-in-the-loop details

### MCP Elicitation
- **File:** `docs/mcp/client.md` (lines 609-746) — Full elicitation documentation with server/client examples

### Capabilities documentation
- NOT VERIFIED: No dedicated `docs/capabilities.md` file found. Capabilities are referenced in agent constructor docstrings (agent/__init__.py:371) and in `AbstractCapability` docstrings (abstract.py:42-62).

### Multi-agent applications
- **File:** `docs/multi-agent-applications.md` (line 335) — Mentions "Human-in-the-loop — approval workflows" linking to toolsets docs

### Graph human-in-the-loop
- **File:** `docs/graph.md` (lines 641-740) — Example of AI Q&A graph where human provides answers via `input()` between graph nodes. This is the graph-level approach (not tool-level).

## Constraints — backward compat, type safety, dependencies (cite sources)

### Backward compatibility
- **Version policy** (`docs/version-policy.md` lines 3-5): No intentional breaking changes in minor V1 releases. V2 April 2026+.
- New capabilities can be added as new classes without breaking existing API.
- `DEFAULT_CAPABILITY_TYPES` tuple (capabilities/__init__.py:14-19): Adding a new capability type here would affect spec-based agents.

### Type safety
- `AbstractCapability` is generic over `AgentDepsT` (abstract.py:38)
- `get_toolset()` returns `AbstractToolset[AgentDepsT] | ToolsetFunc[AgentDepsT] | None` (abstract.py:103) — tools from capabilities share the agent's deps type
- `get_serialization_name()` returns `str | None` (abstract.py:70) — `None` opts out of spec serialization
- `from_spec()` must accept only JSON-safe args (abstract.py:78)
- Coding guidelines prohibit unnecessary `Any` or `cast` (`agent_docs/index.md`)

### Dependencies
- `AbstractCapability` depends on `#10` (already implemented in codebase)
- No additional pip dependencies required for a basic UserInteraction capability
- The capability system supports tools via `FunctionToolset` without needing `RunContext` dependencies (tool_plain pattern)

### Serialization tiers
- Issue says "Tier S — zero-config" — meaning `UserInteraction` should work as `'- UserInteraction'` in YAML specs
- For this, `get_serialization_name()` must return a non-None string (default: class name)
- `from_spec()` should accept no required args

### Key architectural decision: How user interaction tool pauses execution
The codebase offers **two mechanisms** for pausing execution for user input:

1. **`CallDeferred` exception** (exceptions.py:77): Raises from tool, produces `DeferredToolRequests.calls`, resumed with `DeferredToolResults.calls` containing the user's answer. This is the idiomatic mechanism for tools whose results come from outside the agent process.

2. **`ApprovalRequired` exception** (exceptions.py:95): Similar but for yes/no approval, produces `DeferredToolRequests.approvals`. Less suitable for free-text user input.

3. **Synchronous input (graph pattern)**: `input()` called directly (graph.md example). Not usable in async agent loop.

4. **MCP Elicitation** (mcp.py:334): Callback-based, specific to MCP servers. Not available for regular tools.

The `CallDeferred` mechanism is the natural fit for a `UserInteraction` capability's `ask_user` tool — the tool raises `CallDeferred` with metadata containing the question and choices, the run ends with `DeferredToolRequests`, the host application presents the question to the user, and resumes with `DeferredToolResults` containing the user's answer.

### Key pattern: Capability providing tools via get_toolset()
- `Toolset` capability (toolset.py) wraps an existing `AbstractToolset`
- `ToolsetFuncCapability` test (test_capabilities.py:560-600) demonstrates a capability returning a `ToolsetFunc` that creates a `FunctionToolset` with `@tool_plain` tools
- `WebSearch` (web_search.py) uses `get_builtin_tools()` instead — but this is for provider builtin tools, not function tools

### Existing capabilities in DEFAULT_CAPABILITY_TYPES
- `Instructions`, `ModelSettings`, `Thinking`, `WebSearch` (capabilities/__init__.py:14-19)
- A `UserInteraction` capability could be added to this tuple to support zero-config YAML spec construction


## Competitive Analysis

Excellent! Now I have comprehensive data on all three frameworks plus Claude Code. Let me compile the analysis.

---

## Output

### Discovery

**Searches performed:**
- `"LLM agent framework ask user question tool human in the loop clarification"`
- `"openai agents sdk human_input_handler ask_user tool CrewAI LangGraph interrupt"`
- GitHub code searches in `mastra-inc/mastra`, `crewAIInc/crewAI`, `langchain-ai/langgraph`, `openai/openai-agents-python`, `google/adk-python`, `anthropics/claude-code`

**Frameworks found with UserInteraction capability:**
| Framework | Mechanism | Specificity |
|-----------|-----------|-------------|
| **LangGraph** | `interrupt()` function + `HumanInterrupt` schema | General-purpose pause/resume primitive |
| **CrewAI** | `human_input=True` on tasks + `request_human_input()` in hooks | Post-completion feedback loop + hook-based approval |
| **Mastra** | `ask_user` tool with structured options + event system | Purpose-built ask-user tool with UI integration |
| **Claude Code** | `AskUserQuestion` built-in tool | Purpose-built with multi-question + multi-select support |
| **Google ADK** | `LongRunningFunctionTool` wrapper | Generic async pause for external input (not question-specific) |
| **OpenAI Agents SDK** | No dedicated ask-user mechanism found | Approval-focused (`needs_approval`) but no clarification tool |

**Why I picked these 3:**
1. **Mastra** — Most directly analogous to the proposed `UserInteraction` capability: a dedicated `ask_user` tool with structured options, event-driven pause/resume, and UI integration. This is the closest real-world implementation.
2. **LangGraph** — Most general-purpose and framework-agnostic approach: `interrupt()` can be used for ANY human-in-the-loop scenario including questions, approvals, and edits. Shows how to build ask-user on top of a primitive.
3. **CrewAI** — Shows a different pattern: task-level human input (feedback loop after agent produces output) + hook-based `request_human_input()` for mid-execution prompts. Demonstrates the protocol/provider abstraction pattern.

---

### Mastra

- **Repo**: `mastra-ai/mastra`
- **Supports?**: **Yes** — Full dedicated implementation
- **Implementation**:

The `ask_user` tool is defined in `packages/core/src/harness/tools.ts` as a `createTool()` with structured input schema and a promise-based pause/resume mechanism.

**Tool definition** (`packages/core/src/harness/tools.ts`):
```typescript
export const askUserTool = createTool({
  id: 'ask_user',
  description:
    'Ask the user a question and wait for their response. Use this when you need clarification, want to validate assumptions, or need the user to make a decision between options. Provide options for structured choices (2-4 options), or omit them for open-ended questions.',
  inputSchema: z.object({
    question: z.string().min(1).describe('The question to ask the user. Should be clear and specific.'),
    options: z
      .array(
        z.object({
          label: z.string().describe('Short display text for this option (1-5 words)'),
          description: z.string().optional().describe('Explanation of what this option means'),
        }),
      )
      .optional()
      .describe('Optional choices. If provided, shows a selection list. If omitted, shows a free-text input.'),
  }),
  execute: async ({ question, options }, context) => {
    const harnessCtx = context?.requestContext?.get('harness') as HarnessRequestContext | undefined;

    if (!harnessCtx?.emitEvent || !harnessCtx?.registerQuestion) {
      // Graceful fallback when no UI is attached
      return {
        content: `[Question for user]: ${question}${options ? '\nOptions: ' + options.map(o => o.label).join(', ') : ''}`,
        isError: false,
      };
    }

    const questionId = `q_${++questionCounter}_${Date.now()}`;

    const answer = await new Promise<string>((resolve, reject) => {
      const signal = harnessCtx.abortSignal;
      if (signal?.aborted) { reject(new DOMException('Aborted', 'AbortError')); return; }
      const onAbort = () => reject(new DOMException('Aborted', 'AbortError'));
      signal?.addEventListener('abort', onAbort, { once: true });

      harnessCtx.registerQuestion!({
        questionId,
        resolve: answer => { signal?.removeEventListener('abort', onAbort); resolve(answer); },
      });

      harnessCtx.emitEvent!({ type: 'ask_question', questionId, question, options });
    });

    return { content: `User answered: ${answer}`, isError: false };
  },
});
```

**Event type** (`packages/core/src/harness/types.ts:728`):
```typescript
| {
    type: 'ask_question';
    questionId: string;
    question: string;
    options?: Array<{ label: string; description?: string }>;
  }
```

**Display state** (`packages/core/src/harness/types.ts:505`):
```typescript
pendingQuestion: {
  questionId: string;
  question: string;
  options?: Array<{ label: string; description?: string }>;
} | null;
```

**Harness response method** (`packages/core/src/harness/harness.ts:2147`):
```typescript
respondToQuestion({ questionId, answer }: { questionId: string; answer: string }): void {
  const resolve = this.pendingQuestions.get(questionId);
  if (resolve) {
    this.pendingQuestions.delete(questionId);
    resolve(answer);
  }
}
```

**Context wiring** (`packages/core/src/harness/harness.ts:2742`):
```typescript
registerQuestion: params => this.registerQuestion(params),
```

**Headless mode fallback** (`mastracode/src/headless.ts`):
```typescript
case 'ask_question': {
  harness.respondToQuestion({
    questionId: event.questionId,
    answer: 'Proceed with your best judgment. Do not ask further questions.',
  });
}
```

- **Gaps**:
  - Options are **single-select only** — no `multiSelect` support (unlike Claude Code's `AskUserQuestion`)
  - The `answer` is always a plain `string` — no typed/structured response
  - No built-in timeout mechanism (relies solely on abort signal)
  - The question counter is a module-level `let` — not safe across multiple concurrent harness instances
  - Always-allowed tool: `ask_user` bypasses the permission/approval system entirely

---

### LangGraph

- **Repo**: `langchain-ai/langgraph`
- **Supports?**: **Partial** — Provides the primitive (`interrupt()`) but not a purpose-built ask-user tool
- **Implementation**:

LangGraph uses a general-purpose `interrupt()` function that pauses graph execution and surfaces a value to the client. The client resumes via `Command(resume=value)`. There are also structured `HumanInterrupt` / `HumanResponse` schemas for standardized human-in-the-loop interactions.

**`interrupt()` function** (`libs/langgraph/langgraph/types.py`):
```python
def interrupt(value: Any) -> Any:
    """Interrupt the graph with a resumable exception from within a node.

    The `interrupt` function enables human-in-the-loop workflows by pausing graph
    execution and surfacing a value to the client. This value can communicate context
    or request input required to resume execution.

    In a given node, the first invocation of this function raises a `GraphInterrupt`
    exception, halting execution. The provided `value` is included with the exception
    and sent to the client executing the graph.

    A client resuming the graph must use the `Command` primitive to specify a value
    for the interrupt and continue execution.
    """
    # ... internal implementation using scratchpad/checkpoint system
    # Raises GraphInterrupt on first call, returns resume value on re-execution
    raise GraphInterrupt(
        Interrupt.from_ns(value=value, ns=conf[CONFIG_KEY_CHECKPOINT_NS])
    )
```

**`Interrupt` dataclass** (`libs/langgraph/langgraph/types.py`):
```python
@final
@dataclass(init=False, slots=True)
class Interrupt:
    value: Any
    """The value associated with the interrupt."""
    id: str
    """The ID of the interrupt. Can be used to resume the interrupt directly."""
```

**`HumanInterrupt` schema** (`libs/prebuilt/langgraph/prebuilt/interrupt.py`) — now deprecated, moved to `langchain.agents.interrupt`:
```python
class HumanInterruptConfig(TypedDict):
    allow_ignore: bool    # Can skip the step
    allow_respond: bool   # Can provide text response/feedback
    allow_edit: bool      # Can edit content/state
    allow_accept: bool    # Can accept/approve

class ActionRequest(TypedDict):
    action: str           # Type or name of action requested
    args: dict            # Key-value arguments

class HumanInterrupt(TypedDict):
    action_request: ActionRequest
    config: HumanInterruptConfig
    description: str | None

class HumanResponse(TypedDict):
    type: Literal["accept", "ignore", "response", "edit"]
    args: None | str | ActionRequest   # None for accept/ignore, str for text, ActionRequest for edit
```

**Usage example** (from docstring):
```python
def node(state: State):
    answer = interrupt("what is your age?")  # Pauses here
    return {"human_value": answer}

# Client side:
for chunk in graph.stream({"foo": "abc"}, config):
    print(chunk)
# > {'__interrupt__': (Interrupt(value='what is your age?', ...),)}

# Resume:
for chunk in graph.stream(Command(resume="some input from a human!!!"), config):
    print(chunk)
# > {'node': {'human_value': 'some input from a human!!!'}}
```

- **Gaps**:
  - **No dedicated "ask user" tool** — `interrupt()` is a low-level graph primitive, not a tool the LLM invokes
  - The LLM cannot decide to ask a question on its own — the developer must wire interrupt points into graph nodes
  - No built-in question/choices schema at the tool level — `HumanInterrupt` is for the client/UI layer
  - Requires a checkpointer to be enabled (state persistence overhead)
  - The node **re-executes from the start** on resume — any side effects before the interrupt run again
  - `HumanInterruptConfig` is a 4-boolean config, not rich structured options

---

### CrewAI

- **Repo**: `crewAIInc/crewAI`
- **Supports?**: **Partial** — Has task-level human input (post-completion review) and hook-based `request_human_input()`, but no agent-callable "ask user" tool
- **Implementation**:

CrewAI provides two mechanisms: (1) `human_input=True` on tasks which triggers a feedback loop after the agent produces its answer, and (2) `request_human_input()` method on hook contexts for mid-execution human input.

**Task-level `human_input`** (`lib/crewai/src/crewai/task.py`):
```python
class Task:
    human_input: bool | None = Field(
        description="Whether the task should have a human review the final answer of the agent",
        default=False,
    )
```

**`HumanInputProvider` protocol** (`lib/crewai/src/crewai/core/providers/human_input.py`):
```python
@runtime_checkable
class HumanInputProvider(Protocol):
    def setup_messages(self, context: ExecutorContext) -> bool:
        """Set up messages for execution. Returns True if custom setup handled."""
        ...

    def post_setup_messages(self, context: ExecutorContext) -> None:
        """Called after standard message setup."""
        ...

    def handle_feedback(
        self,
        formatted_answer: AgentFinish,
        context: ExecutorContext,
    ) -> AgentFinish:
        """Handle the full human feedback flow synchronously."""
        ...

    async def handle_feedback_async(
        self,
        formatted_answer: AgentFinish,
        context: AsyncExecutorContext,
    ) -> AgentFinish:
        """Handle the full human feedback flow asynchronously."""
        ...
```

**`SyncHumanInputProvider` — default implementation** (same file):
```python
class SyncHumanInputProvider(HumanInputProvider):
    def _handle_regular_feedback(self, current_answer, initial_feedback, context):
        feedback = initial_feedback
        answer = current_answer
        while context.ask_for_human_input:
            if feedback.strip() == "":
                context.ask_for_human_input = False  # Empty = satisfied
            else:
                context.messages.append(context._format_feedback_message(feedback))
                answer = context._invoke_loop()      # Re-run agent with feedback
                feedback = self._prompt_input(context.crew)
        return answer
```

**Hook-based `request_human_input`** (`lib/crewai/src/crewai/hooks/tool_hooks.py`):
```python
class ToolCallHookContext:
    def request_human_input(
        self,
        prompt: str,
        default_message: str = "Press Enter to continue, or provide feedback:",
    ) -> str:
        """Request human input during tool hook execution.
        Pauses live console updates, displays prompt, waits for input."""
        printer = Printer()
        event_listener.formatter.pause_live_updates()
        try:
            printer.print(content=f"\n{prompt}", color="bold_yellow")
            printer.print(content=default_message, color="cyan")
            response = input().strip()
            return response
        finally:
            event_listener.formatter.resume_live_updates()
```

**Provider injection pattern** (same file):
```python
_provider: ContextVar[HumanInputProvider | None] = ContextVar(
    "human_input_provider", default=None,
)

def get_provider() -> HumanInputProvider:
    provider = _provider.get()
    if provider is None:
        return SyncHumanInputProvider()
    return provider
```

- **Gaps**:
  - **No agent-callable ask-user tool** — `human_input=True` is a task-level config, not a tool the LLM decides to use. The agent cannot proactively ask clarifying questions mid-task.
  - `request_human_input()` is only available inside hook callbacks (before/after tool call), not as a tool
  - No structured choices/options support — always free-text `input()`
  - The feedback loop only happens **after the agent finishes** — it's review/refinement, not mid-execution clarification
  - The `SyncHumanInputProvider` is tightly coupled to terminal I/O (`input()` + Rich panels) — not suitable for web/API contexts without a custom provider
  - `ContextVar`-based provider injection is process-scoped, not per-agent

---

### Patterns

**What's common across frameworks:**

1. **Promise/future-based pause mechanism**: All implementations that support mid-execution questions use some form of pause-and-resume: Mastra uses `Promise<string>` with registered resolvers, LangGraph uses checkpoint-based `interrupt()` + `Command(resume=)`, CrewAI uses a while-loop with `input()`. The pattern is: emit a question, block until answer arrives.

2. **Graceful fallback when no UI**: Mastra returns a formatted text message when no harness context is available. LangGraph raises `GraphInterrupt` which the client catches. Both ensure the tool doesn't hard-fail if there's no human on the other end.

3. **Question ID for correlation**: Mastra and LangGraph both use IDs (`questionId`, `interrupt.id`) to match responses to questions. This is essential for concurrent or multiple questions.

4. **Event/stream-based notification**: The question is communicated to the client via events (Mastra: `ask_question` event) or stream output (LangGraph: `__interrupt__` in stream). This decouples the tool from any specific UI.

5. **Abort/cancel support**: Mastra integrates with `AbortSignal` to cancel pending questions. LangGraph relies on checkpoint expiry.

**What Pydantic AI should consider:**

1. **Make it a tool, not a config flag**: The most useful pattern (Mastra, Claude Code) is a tool the LLM can choose to invoke, rather than a task-level setting (CrewAI). This lets the agent decide when clarification is needed. The proposed `ask_user` tool approach in the issue is right.

2. **Structured options with labels + descriptions**: Both Mastra and Claude Code support `options: [{label, description?}]` for structured choices. Claude Code additionally supports `multiSelect` and batching multiple questions in one call. Pydantic AI should at minimum support single-select options with descriptions.

3. **Pluggable I/O backend**: CrewAI's `HumanInputProvider` protocol is a good pattern — it decouples the question-asking mechanism from the specific I/O (terminal, web, API). The `UserInteraction` capability should accept a callback/handler rather than hard-coding `input()`. When no handler is provided, a sensible default (stdin or a text-only fallback) should be used.

4. **Leverage existing human-in-the-loop infrastructure**: Pydantic AI already has `HumanApprovalToolset` and `ApprovalRequiredToolset` with `ToolApproved`/`ToolDenied` responses. The `UserInteraction` capability could use a similar pause/resume mechanism, or even share infrastructure with the existing toolset wrappers.

5. **Headless/autonomous mode**: Mastra's headless fallback (`'Proceed with your best judgment'`) is a pragmatic pattern for CI/automated contexts. The capability should support a configurable fallback for when no human is available.

6. **The tool return value should be the user's answer directly**: Mastra returns `"User answered: {answer}"` — this is clean and unambiguous for the LLM to parse. Avoid structured return objects that the LLM has to destructure.

7. **Difference from Approval is real and important**: The issue correctly distinguishes UserInteraction (gathering information) from Approval (authorizing actions). They have different tool signatures, different UX, and different system prompt instructions. They should remain separate capabilities even if they share pause/resume infrastructure.


</details>

<details>
<summary>anthropic/claude-opus-4-5-20251101's Research</summary>

## Codebase Research

## Relevant Code — file path, lines, what's there, what imports it

### AbstractCapability (base class)
- **File:** `pydantic_ai_slim/pydantic_ai/capabilities/abstract.py:38-220`
- **Key methods:** `get_instructions()`, `get_model_settings()`, `get_toolset()`, `get_builtin_tools()`, `before_model_request()`, `after_model_request()`, run hooks (`before_run`, `after_run`, `wrap_run`), tool hooks (`before_tool_validate`, `before_tool_execute`, etc.)
- **Serialization:** `get_serialization_name()` (returns class name or None), `from_spec()` (factory for YAML/JSON)
- **Per-run lifecycle:** `for_run(ctx)` returns self or new instance for state isolation
- **Imported by:** Agent (`pydantic_ai_slim/pydantic_ai/agent/__init__.py:46`), CombinedCapability, all capability subclasses

### Toolset capability pattern
- **File:** `pydantic_ai_slim/pydantic_ai/capabilities/toolset.py:1-19`
- **Pattern:** Simple dataclass wrapping an `AbstractToolset`, returns it via `get_toolset()`, opts out of serialization via `get_serialization_name() -> None`

### FunctionToolset (for creating tool functions)
- **File:** `pydantic_ai_slim/pydantic_ai/toolsets/function.py:1-452`
- **Key methods:** `tool()`, `tool_plain()`, `add_function()`, `add_tool()`
- **Tool registration:** Supports `requires_approval=True` flag (line 83, 179, 259)
- **Timeout support:** Per-tool `timeout: float | None` parameter

### CallDeferred and ApprovalRequired exceptions
- **File:** `pydantic_ai_slim/pydantic_ai/exceptions.py:73-107`
- **`CallDeferred`:** Raises to defer tool execution externally, has `metadata` param
- **`ApprovalRequired`:** Raises when tool needs human approval, has `metadata` param
- **Both can pause agent execution** and return `DeferredToolRequests`

### DeferredToolRequests (output type for paused agent)
- **File:** `pydantic_ai_slim/pydantic_ai/tools.py:148-162`
- **Fields:** `calls: list[ToolCallPart]` (external), `approvals: list[ToolCallPart]` (need approval), `metadata: dict[str, dict[str, Any]]`
- **Must be in agent's `output_type`** to use deferred tools

### DeferredToolResults (to resume agent)
- **File:** `pydantic_ai_slim/pydantic_ai/tools.py:226-240`
- **Fields:** `calls: dict[str, ...]` (results for external), `approvals: dict[str, bool | ToolApproved | ToolDenied]`, `metadata: dict[str, dict[str, Any]]`
- **Passed to `agent.run()` with `deferred_tool_results=` and `message_history=`**

### ApprovalRequiredToolset (existing wrapper pattern)
- **File:** `pydantic_ai_slim/pydantic_ai/toolsets/approval_required.py:1-31`
- **Pattern:** `WrapperToolset` that checks `ctx.tool_call_approved` and raises `ApprovalRequired` if not approved
- **Uses:** `approval_required_func: Callable[[RunContext, ToolDefinition, dict[str, Any]], bool]`

### ExternalToolset (for tools executed externally)
- **File:** `pydantic_ai_slim/pydantic_ai/toolsets/external.py:1-51`
- **Pattern:** Toolset with `kind='external'` that cannot be called directly (raises NotImplementedError)
- **Relevant for:** UserInteraction could use similar pattern where the tool cannot be executed in-process

### Agent integration with capabilities
- **File:** `pydantic_ai_slim/pydantic_ai/agent/__init__.py:223-450`
- **Constructor:** `capabilities: Sequence[AbstractCapability[AgentDepsT]] | None = None` (line 285)
- **Creates:** `self._root_capability = CombinedCapability(capabilities)` (line 388)
- **Extracts:** `get_instructions()`, `get_toolset()`, `get_builtin_tools()`, `get_model_settings()` at init
- **Per-run re-extraction:** If `for_run()` returns different instance, re-extracts all (lines 974-1000)

### RunContext (available during tool execution)
- **File:** `pydantic_ai_slim/pydantic_ai/_run_context.py:31-85`
- **Key fields:** `deps`, `model`, `usage`, `prompt`, `messages`, `tool_call_id`, `tool_call_approved`, `tool_call_metadata`, `run_step`
- **`tool_call_approved: bool`:** True when deferred tool call was approved

---

## Type System — key types/protocols/interfaces with actual code

### Tool function types
```python
# pydantic_ai_slim/pydantic_ai/tools.py:41-55
ToolFuncContext: TypeAlias = Callable[Concatenate[RunContext[AgentDepsT], ToolParams], Any]
ToolFuncPlain: TypeAlias = Callable[ToolParams, Any]
```

### Tool definition
```python
# pydantic_ai_slim/pydantic_ai/tools.py:426-495
@dataclass(repr=False, kw_only=True)
class ToolDefinition:
    name: str
    parameters_json_schema: ObjectJsonSchema
    description: str | None = None
    kind: ToolKind = field(default='function')  # 'function' | 'output' | 'external' | 'unapproved'
    metadata: dict[str, Any] | None = None
```

### Capability interface (relevant methods)
```python
# pydantic_ai_slim/pydantic_ai/capabilities/abstract.py:86-108
def get_instructions(self) -> _instructions.Instructions[AgentDepsT] | None:
    return None

def get_toolset(self) -> AbstractToolset[AgentDepsT] | ToolsetFunc[AgentDepsT] | None:
    return None
```

### ToolsetFunc (dynamic toolset factory)
```python
# pydantic_ai_slim/pydantic_ai/toolsets/_dynamic.py (referenced in abstract.py)
ToolsetFunc: TypeAlias = Callable[[RunContext[AgentDepsT]], AbstractToolset[AgentDepsT]]
```

---

## Call Chains — trace execution paths

### Tool execution with deferred handling
1. Model calls tool → `CallToolsNode` in `_agent_graph.py:1360-1370`
2. Tool function executes
3. If raises `ApprovalRequired` → caught at line 1368, deferred_calls populated
4. `DeferredToolRequests` constructed at line 1326 with `approvals=` list
5. Agent run ends with `DeferredToolRequests` as output
6. User provides approval via `DeferredToolResults`
7. Next `agent.run()` with `deferred_tool_results=` + `message_history=`
8. Tool re-executed with `ctx.tool_call_approved=True`

### Capability toolset integration
1. Agent constructor calls `capability.get_toolset()` (line 450)
2. Toolsets combined via `CombinedToolset` in CombinedCapability (line 73-80)
3. During run, `for_run(ctx)` called on capability (line 975)
4. If different capability returned, `get_toolset()` re-extracted (lines 982-987)
5. Toolset's `get_tools(ctx)` called per step to get available tools
6. Tools registered with model for that step

---

## Test Coverage — test files, what they cover, gaps

### Capability tests
- **File:** `tests/test_capabilities.py`
- **Coverage:** 
  - `ToolsetFuncCapability` pattern (lines 559-606): capability returning `ToolsetFunc`
  - `Toolset` capability wrapping `FunctionToolset` (lines 715-758)
  - `for_run()` returning different toolset (lines 840-873)
  - Combined capability hook chaining

### Deferred tools / approval tests
- **File:** `tests/test_tools.py:1330-1720`
- **Coverage:**
  - `test_tool_raises_call_deferred` (line 1330)
  - `test_tool_raises_approval_required` (line 1343)
  - `test_approval_required_with_user_prompt` (line 1415): approval + new user prompt
  - `test_approval_required_with_metadata` (line 1489)
  - `test_output_type_deferred_tool_requests_by_itself` (line 1714): validation that DeferredToolRequests alone is rejected

### Gaps
- **NOT VERIFIED:** No existing tests for a capability that provides a tool which raises `CallDeferred` or `ApprovalRequired`
- **NOT VERIFIED:** No tests for a "clarifying question" pattern where user response becomes tool return value

---

## Documentation — docs files, docstrings, discrepancies

### Deferred tools documentation
- **File:** `docs/deferred-tools.md`
- **Covers:** `ApprovalRequired` exception, `requires_approval=True`, `DeferredToolRequests`/`DeferredToolResults` flow
- **Pattern shown:** Tool that checks `ctx.tool_call_approved` and raises `ApprovalRequired` if not approved

### Multi-agent documentation (ask_user pattern)
- **File:** `docs/multi-agent-applications.md:304-316`
- **Shows:** `ask_user_flight` and `ask_user_seat` as graph nodes representing user interaction points
- **Note:** These are application-level patterns, not built-in capabilities

### Toolsets documentation
- **File:** `docs/toolsets.md:328-330`
- **Covers:** `ApprovalRequiredToolset` wrapper pattern

### AG-UI documentation (related)
- **File:** `docs/ui/ag-ui.md`
- **Shows:** Agent-User Interaction protocol for frontend integration
- **Note:** Different from in-agent clarification tool; AG-UI is for UI protocols

---

## Constraints — backward compat, type safety, dependencies (cite sources)

### Backward compatibility
- **File:** `docs/version-policy.md` (referenced in AGENTS.md root)
- All changes must maintain backward compatibility

### Type safety requirements
- **File:** `AGENTS.md:18-20` (root): "be fully type-safe (both internally and in public API) without unnecessary `cast`s or `Any`s"
- **File:** `agent_docs/index.md:142`: Use `Literal` types for fixed string value sets

### Serialization tiers (from issue #1)
- **Tier S (Serializable):** Zero-config capabilities like `- UserInteraction`
- Issue states UserInteraction should be Tier S

### Capability interface constraints
- **File:** `pydantic_ai_slim/pydantic_ai/capabilities/abstract.py:92-100`
- `get_toolset()` does NOT take `RunContext` — evaluated once at registration
- For per-run state isolation, use `for_run()` to return new instance

### Deferred tools constraints
- **File:** `pydantic_ai_slim/pydantic_ai/tools.py:148-150`
- `DeferredToolRequests` must be in agent's `output_type` union
- **File:** `tests/test_tools.py:1714-1716`: Agent rejects `DeferredToolRequests` as sole output type

### Tool parameters
- **File:** `pydantic_ai_slim/pydantic_ai/toolsets/function.py:83-95`
- `requires_approval: bool = False` flag on tool registration
- Tool parameters extracted from function signature via docstring parsing

### Test requirements
- **File:** `tests/AGENTS.md:86`: Use `snapshot()` for complex structured outputs
- **File:** `tests/AGENTS.md:194`: Assert on final output AND snapshot `result.all_messages()`

### Code structure
- **File:** `pydantic_ai_slim/pydantic_ai/AGENTS.md:987`: Extend `WrapperToolset` for cross-cutting toolset behavior
- **File:** `pydantic_ai_slim/pydantic_ai/AGENTS.md:41`: Order required fields before optional in dataclasses


## Competitive Analysis

Now I have enough source code to compile my findings.

---

## Discovery

### What I searched for:
- "LLM agent ask user question tool human in loop"
- "Claude Agent SDK AskUserQuestion tool"  
- "LangGraph interrupt human input agent pause resume"
- CrewAI and Mastra GitHub code searches for human_input/ask_user

### What I found:
The UserInteraction pattern is implemented across many frameworks with varying approaches:

| Framework | Approach |
|-----------|----------|
| **Mastra** | Dedicated `ask_user` tool with structured options and event system |
| **CrewAI** | `human_input=True` flag on tasks + `request_human_input()` in hooks |
| **LangGraph** | Generic `interrupt()` function + `Command(resume=...)` pattern |
| **Claude Agent SDK** | Built-in `AskUserQuestion` tool |
| **OpenAI Agents SDK** | Human-in-the-loop via `needs_approval` (focused on approval, not questions) |

### Why I picked these 3:

1. **Mastra** - The most complete purpose-built implementation with structured options, event-driven architecture, and explicit question/answer semantics. Most aligned with the issue's vision.

2. **CrewAI** - Different philosophy: human input as task-level feedback loop rather than mid-execution tool. Good contrast showing alternative patterns.

3. **LangGraph** - Generic primitive (`interrupt()`) that enables user interaction but isn't specific to it. Shows how to build on lower-level abstractions.

---

## Mastra

- **Repo**: mastra-ai/mastra
- **Supports?**: Yes — full implementation
- **Implementation**:

### Core Tool Definition (`packages/core/src/harness/tools.ts:18-76`)

```typescript
export const askUserTool = createTool({
  id: 'ask_user',
  description:
    'Ask the user a question and wait for their response. Use this when you need clarification, want to validate assumptions, or need the user to make a decision between options. Provide options for structured choices (2-4 options), or omit them for open-ended questions.',
  inputSchema: z.object({
    question: z.string().min(1).describe('The question to ask the user. Should be clear and specific.'),
    options: z
      .array(
        z.object({
          label: z.string().describe('Short display text for this option (1-5 words)'),
          description: z.string().optional().describe('Explanation of what this option means'),
        }),
      )
      .optional()
      .describe('Optional choices. If provided, shows a selection list. If omitted, shows a free-text input.'),
  }),
  execute: async ({ question, options }, context) => {
    const harnessCtx = context?.requestContext?.get('harness') as HarnessRequestContext | undefined;

    const questionId = `q_${++questionCounter}_${Date.now()}`;

    const answer = await new Promise<string>((resolve, reject) => {
      const signal = harnessCtx.abortSignal;
      // ... abort signal handling ...

      harnessCtx.registerQuestion!({
        questionId,
        resolve: answer => { resolve(answer); },
      });

      harnessCtx.emitEvent!({
        type: 'ask_question',
        questionId,
        question,
        options,
      });
    });

    return { content: `User answered: ${answer}`, isError: false };
  },
});
```

### Harness Orchestration (`packages/core/src/harness/harness.ts`)

The Harness class manages the question lifecycle:

```typescript
// Question registration and resolution
private pendingQuestions = new Map<string, (answer: string) => void>();

// Response handler (called by UI)
respondToQuestion({ questionId, answer }: { questionId: string; answer: string }): void {
  const resolver = this.pendingQuestions.get(questionId);
  if (resolver) {
    resolver(answer);
    this.pendingQuestions.delete(questionId);
  }
}
```

### Event Types (`packages/core/src/harness/types.ts`)

```typescript
| { type: 'ask_question'; questionId: string; question: string; options?: Array<{ label: string; description?: string }> }
```

### Key Features:
- **Structured options**: Single-select with label + description per option
- **Event-driven**: Emits `ask_question` event, waits for `respondToQuestion()` call
- **Abort signal support**: Clean cancellation during tool execution
- **Always-allowed tool**: Part of `ALWAYS_ALLOW_TOOLS` set (no approval needed)
- **Graceful degradation**: Falls back to string output if no harness context

### Gaps:
- No multi-select support (single option or free text only)
- No timeout mechanism for unanswered questions
- Options don't support "other" escape hatch (handled at UI layer)

---

## CrewAI

- **Repo**: crewAIInc/crewAI
- **Supports?**: Partial — task-level feedback, not mid-execution questioning
- **Implementation**:

### Task-Level Human Input (`lib/crewai/src/crewai/task.py:158-162`)

```python
class Task(BaseModel):
    human_input: bool | None = Field(
        description="Whether the task should have a human review the final answer of the agent",
        default=False,
    )
```

### Human Input Provider Protocol (`lib/crewai/src/crewai/core/providers/human_input.py`)

```python
@runtime_checkable
class HumanInputProvider(Protocol):
    """Protocol for human input handling."""

    def setup_messages(self, context: ExecutorContext) -> bool:
        """Set up messages before execution."""
        ...

    def handle_feedback(
        self,
        formatted_answer: AgentFinish,
        context: ExecutorContext,
    ) -> AgentFinish:
        """Handle the full human feedback flow synchronously."""
        ...

    async def handle_feedback_async(
        self,
        formatted_answer: AgentFinish,
        context: AsyncExecutorContext,
    ) -> AgentFinish:
        """Handle the full human feedback flow asynchronously."""
        ...
```

### Default Implementation (`SyncHumanInputProvider`)

```python
class SyncHumanInputProvider(HumanInputProvider):
    def _handle_regular_feedback(
        self,
        current_answer: AgentFinish,
        initial_feedback: str,
        context: ExecutorContext,
    ) -> AgentFinish:
        feedback = initial_feedback
        answer = current_answer

        while context.ask_for_human_input:
            if feedback.strip() == "":
                context.ask_for_human_input = False
            else:
                context.messages.append(context._format_feedback_message(feedback))
                answer = context._invoke_loop()
                feedback = self._prompt_input(context.crew)

        return answer
```

### Hook-Based Human Input (`lib/crewai/src/crewai/hooks/tool_hooks.py:60-92`)

```python
class ToolCallHookContext:
    def request_human_input(
        self,
        prompt: str,
        default_message: str = "Press Enter to continue, or provide feedback:",
    ) -> str:
        """Request human input during tool hook execution.
        
        Pauses live console updates, displays a prompt to the user,
        waits for their input, and then resumes live updates.
        """
        printer = Printer()
        event_listener.formatter.pause_live_updates()

        try:
            printer.print(content=f"\n{prompt}", color="bold_yellow")
            printer.print(content=default_message, color="cyan")
            response = input().strip()
            return response
        finally:
            event_listener.formatter.resume_live_updates()
```

### Key Features:
- **Task-level feedback loop**: Agent produces result, human reviews, provides feedback, agent iterates
- **Training mode support**: Special handling for crew training with human feedback
- **Async support**: Full async provider implementation
- **Hook integration**: `request_human_input()` available in before/after tool call hooks
- **Rich console UI**: Panel-based prompt display with live update pausing

### Gaps:
- **Not mid-execution questioning**: Human input happens AFTER task completion, not during
- **No structured choices**: Free-text input only
- **Console-centric**: Assumes terminal input, not event-based
- **Different philosophy**: More about human review/feedback than clarifying questions

---

## LangGraph

- **Repo**: langchain-ai/langgraph
- **Supports?**: Yes — via generic `interrupt()` primitive
- **Implementation**:

### Interrupt Function (`libs/langgraph/langgraph/types.py`)

```python
def interrupt(value: Any) -> Any:
    """Interrupt the graph with a resumable exception from within a node.

    The `interrupt` function enables human-in-the-loop workflows by pausing graph
    execution and surfacing a value to the client. This value can communicate context
    or request input required to resume execution.

    In a given node, the first invocation of this function raises a `GraphInterrupt`
    exception, halting execution. The provided `value` is included with the exception
    and sent to the client executing the graph.

    A client resuming the graph must use the `Command` primitive to specify a 
    value for the interrupt and continue execution.
    """
    from langgraph._internal._constants import (
        CONFIG_KEY_CHECKPOINT_NS,
        CONFIG_KEY_SCRATCHPAD,
        CONFIG_KEY_SEND,
        RESUME,
    )
    from langgraph.config import get_config
    from langgraph.errors import GraphInterrupt

    conf = get_config()["configurable"]
    # track interrupt index, check for resume value, raise GraphInterrupt if not resuming
```

### Interrupt Dataclass

```python
@final
@dataclass(init=False, slots=True)
class Interrupt:
    """Information about an interrupt that occurred in a node."""

    value: Any
    """The value associated with the interrupt."""

    id: str
    """The ID of the interrupt. Can be used to resume the interrupt directly."""
```

### Command Class (for resuming)

```python
class Command(Generic[N], ToolOutputMixin):
    """One or more commands to update the graph's state and send messages to nodes.

    Args:
        resume: Value to resume execution with. To be used together with `interrupt()`.
            Can be one of the following:
            - Mapping of interrupt ids to resume values
            - A single value with which to resume the next interrupt
    """

    graph: str | None = None
    update: Any | None = None
    resume: dict[str, Any] | Any | None = None
    goto: Send | Sequence[Send | N] | N = ()
```

### Usage Pattern

```python
from langgraph.types import interrupt, Command

def ask_user_node(state: State):
    # This will pause execution and surface the question
    answer = interrupt("What is your preferred color?")
    # When resumed, answer contains the user's response
    return {"user_answer": answer}

# To resume:
graph.invoke(Command(resume="blue"), config)
```

### Key Features:
- **Generic primitive**: Not specific to user questions, works for any pause/resume
- **Checkpoint-based**: State persisted, can resume after process restart
- **Multiple interrupts**: Can have multiple `interrupt()` calls in a node, matched by order
- **ID-based resume**: Can resume specific interrupts by ID
- **Works in tools**: Can call `interrupt()` from within tool functions

### Gaps:
- **No structured choices**: Just passes arbitrary `value`, no option schema
- **No tool abstraction**: Must be called directly, not wrapped as a tool
- **Requires checkpointer**: Won't work without persistence enabled
- **Re-executes node**: On resume, entire node re-runs (not just continuation)
- **No instructions**: Doesn't come with system prompt guidance

---

## Patterns

### Common Across Frameworks:

1. **Pause/Resume Semantics**: All implementations pause execution until user responds, then continue
2. **Async-First Design**: All support async execution for non-blocking I/O
3. **Event/Promise Pattern**: Mastra and LangGraph use promise-based resolution; CrewAI uses direct input()
4. **Abort Support**: All handle cancellation during wait

### Key Differences:

| Aspect | Mastra | CrewAI | LangGraph |
|--------|--------|--------|-----------|
| **Abstraction** | Purpose-built tool | Task-level flag | Generic primitive |
| **Structured choices** | ✅ Yes (label + description) | ❌ Free text only | ❌ Any value |
| **When it fires** | Mid-execution | After task completion | Anywhere in node |
| **State persistence** | Harness-managed | Context-managed | Checkpointer-required |
| **Event system** | ✅ Emit events | ❌ Direct console | ✅ Stream output |

### What Pydantic AI Should Consider:

1. **Follow Mastra's Tool Pattern**: A dedicated `ask_user` tool with structured options schema is the most ergonomic approach. The tool should:
   - Accept `question: str` and `choices: list[Choice] | None`
   - Return the user's answer as a string
   - Be always-allowed (no approval needed)

2. **Event-Driven Architecture**: Emit events (`ask_question`) rather than blocking on `input()`. This enables:
   - Web/API integration (not just CLI)
   - Timeout handling
   - UI flexibility

3. **Integration with Existing Infrastructure**: The issue notes "Uses the existing human-in-the-loop infrastructure" — this aligns with how CrewAI's `HumanInputProvider` protocol abstracts the actual I/O.

4. **Include Instructions**: Add system prompt guidance like Mastra's:
   ```
   When uncertain about the user's intent, use the ask_user tool to clarify.
   ```

5. **Choice Schema**: Support structured options like Mastra:
   ```python
   class Choice(TypedDict):
       label: str
       description: str | None
   ```

6. **Graceful Degradation**: If no UI adapter is available, return a formatted string message (like Mastra's fallback).

7. **Distinct from Approval**: As the issue notes, this is about gathering information, not authorizing actions. Keep it separate from the Approval capability (#29).


</details>