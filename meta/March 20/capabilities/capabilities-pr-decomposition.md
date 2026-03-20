# Capabilities PR Decomposition: Individual Issues

> Extracted from [PR #4640](https://github.com/pydantic/pydantic-ai/pull/4640) — "Add new capabilities abstraction + make agents serializable"
>
> Parent issue: [#4303](https://github.com/pydantic/pydantic-ai/issues/4303) — New "capabilities" abstraction
> Also closes: [#4251](https://github.com/pydantic/pydantic-ai/issues/4251) — Agent Definition from JSON/YAML

---

## What Is a Capability?

A **capability** is a composable, reusable unit of agent behavior implemented as a subclass of `AbstractCapability[AgentDepsT]`. Users plug capabilities into an agent via `Agent(..., capabilities=[...])`.

A capability can provide **any combination** of these 6 facets:

| Facet | Method | When Called | Example |
|-------|--------|------------|---------|
| **Instructions** | `get_instructions()` | Once at Agent init | `Instructions("You are helpful")` |
| **Model settings** | `get_model_settings()` | Once at init (static) or per-step (callable) | `ModelSettings(temperature=0.7)` |
| **Tools** | `get_toolset()` | Once at Agent init | `Toolset(my_toolset)` |
| **Builtin tools** | `get_builtin_tools()` | Once at Agent init | `WebSearch()` |
| **Pre-request hook** | `before_model_request(ctx, request_context)` | Per model call | `HistoryProcessorCapability(processor)` |
| **Post-response hook** | `after_model_request(ctx, response)` | Per model call | (no built-in example yet) |

Key design properties:
- **`get_toolset()` does NOT take `RunContext`** — evaluated once at registration, not per-run (confirmed by DouweM in [#4347](https://github.com/pydantic/pydantic-ai/issues/4347))
- **Multiple capabilities compose** via `CombinedCapability`: instructions concatenated, settings merged, toolsets combined, hooks chained (forward for `before`, reverse LIFO for `after`)
- **Hooks receive `RunContext`** — which includes `ctx.usage` (tool call counts, token counts), `ctx.deps`, `ctx.messages`, `ctx.model`, `ctx.retry`, etc.

### Serializability

Capabilities optionally support YAML/JSON serialization via the `AgentSpec` system:

- **`get_serialization_name()`** — Returns a string name for spec lookup (e.g., `"Thinking"`), or `None` to opt out
- **`from_spec(*args, **kwargs)`** — Factory accepting only JSON-safe arguments (strings, numbers, dicts, lists)
- **`DEFAULT_CAPABILITY_TYPES`** — Registry of built-in serializable capabilities

A capability's serializability tier:

| Tier | Meaning | Example |
|------|---------|---------|
| **S (Fully serializable)** | Entire config expressible in YAML. Has `from_spec()`. In registry. | `- Thinking`, `- Instructions: "Be helpful"` |
| **P (Partially serializable)** | Config subset is YAML-safe. `from_spec()` accepts the serializable subset; Python-only features (callables) require code. | `- ModelSettings: {temperature: 0.7}` (static only; callable not serializable) |
| **N (Not serializable)** | Returns `None` from `get_serialization_name()`. Requires Python to instantiate. | `Toolset(my_toolset)`, `HistoryProcessorCapability(my_func)` |

The `from_spec()` narrowing pattern is important: `Instructions.__init__` accepts `str | callable | TemplateStr | Sequence`, but `Instructions.from_spec()` only accepts `str | TemplateStr` — the serializable subset.

### What Is NOT a Capability

- **Infrastructure** (e.g., `NamedSpec` extraction, `_instructions` module) — plumbing that enables the system, not pluggable behavior
- **Agent-level features** (e.g., dynamic model settings on Agent, dynamic builtin tools) — changes to `Agent` constructor/runtime, not composable units
- **Hook extensions** (e.g., adding `before_tool_call` to `AbstractCapability`) — extends what capabilities CAN do, not a capability itself

### Current Capabilities in the PR

| Capability | Serialization | `from_spec` | Interface Methods Used |
|---|---|---|---|
| `Instructions` | **Tier S** — `- Instructions: "text"` | `from_spec(instructions='')` — str/TemplateStr only | `get_instructions()` |
| `ModelSettings` | **Tier P** — `- ModelSettings: {temperature: 0.7}` | `from_spec(temp=0.7)` — static dicts only | `get_model_settings()` |
| `Thinking` | **Tier S** — `- Thinking` | `from_spec()` — no args | `get_model_settings()` (inherits from ModelSettings) |
| `WebSearch` | **Tier S** — `- WebSearch` | default `cls()` — no args | `get_builtin_tools()` |
| `Toolset` | **Tier N** — not serializable | N/A | `get_toolset()` |
| `HistoryProcessorCapability` | **Tier N** — not serializable | N/A | `before_model_request()` |

---

## Implementation Status Summary

| # | Issue | Code Status | Test Status | Key Gaps |
|---|-------|-------------|-------------|----------|
| 1 | NamedSpec extraction | COMPLETE | COMPLETE (40 tests) | 2 TODOs for pydantic version upgrades |
| 2 | `_instructions` module | COMPLETE | Indirect only | No dedicated test file |
| 3 | `_history_processor` module | COMPLETE | COMPLETE (28 tests) | None |
| 4 | TemplateStr | COMPLETE | COMPLETE (34 tests) | Tests skip if `pydantic-handlebars` not installed |
| 5 | Dynamic model settings | COMPLETE | COMPLETE (in test_agent.py) | None |
| 6 | Dynamic builtin tools | COMPLETE | COMPLETE (in test_agent.py) | None |
| 7 | AbstractCapability + Agent integration | COMPLETE | PARTIAL | `after_model_request` hook untested; hook chaining untested |
| 8 | Instructions capability | COMPLETE | PARTIAL (~70%) | No direct `get_instructions()` test; no template strings test |
| 9 | ModelSettings capability | COMPLETE | GOOD (~80%) | Dynamic resolution in capability chain untested |
| 10 | Thinking capability | COMPLETE | PARTIAL (~60%) | No agent run integration test |
| 11 | WebSearch capability | COMPLETE | LOW (~30%) | No integration test with agent.run() |
| 12 | Toolset capability | COMPLETE | PARTIAL (~70%) | CombinedCapability merge of multiple toolsets untested |
| 13 | HistoryProcessorCapability | COMPLETE | **NONE (0%)** | Zero tests for the capability wrapper itself |
| 14 | Durable exec refactor | COMPLETE | COMPLETE | None |
| 15 | UI adapter enhancements | COMPLETE | N/A (tested via integration) | None |
| 16 | AgentSpec + from_spec | COMPLETE | GOOD | `Agent.from_file()` not directly tested; file round-trip with capabilities untested |
| 17 | CLI agent loading | COMPLETE | COMPLETE | None |

---

## Dependency Graph (read top-to-bottom)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FOUNDATION (can be done independently)           │
│                                                                     │
│  [1] NamedSpec    [2] _instructions   [3] _history_processor        │
│  extraction       module              module                        │
│                                                                     │
│  [4] TemplateStr  [5] Dynamic model   [6] Dynamic builtin           │
│  (handlebars)     settings on Agent   tools on Agent                │
│                                                                     │
│  [14] Durable     [15] UI adapter                                   │
│  exec refactor    enhancements                                      │
└─────────────────┬──────────────────┬────────────────────────────────┘
                  │                  │
┌─────────────────▼──────────────────▼────────────────────────────────┐
│                    CORE ABSTRACTION                                  │
│                                                                     │
│  [7] AbstractCapability + CombinedCapability + Agent integration    │
│      (depends on: [2], [3])                                         │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────────┐
│                    INDIVIDUAL CAPABILITIES (each depends on [7])     │
│                                                                     │
│  [8] Instructions   [9] ModelSettings   [10] Thinking               │
│  capability         capability          capability                  │
│                                                                     │
│  [11] WebSearch     [12] Toolset        [13] HistoryProcessor-      │
│  capability         capability          Capability                  │
└─────────────────┬───────────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────────┐
│                    SERIALIZATION (depends on [1], [4], [7-13])       │
│                                                                     │
│  [16] AgentSpec + Agent.from_spec + Agent.from_file                 │
│  [17] CLI agent loading from spec files                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Issue 1: Extract shared `NamedSpec` infrastructure from `EvaluatorSpec`

**Type:** Refactoring (enabler)
**Can be done independently:** Yes — pure refactoring, no new features, no behavior change
**Files:** `pydantic_ai_slim/pydantic_ai/_spec.py` (new), `pydantic_evals/pydantic_evals/evaluators/spec.py` (simplified), `pydantic_evals/pydantic_evals/dataset.py` (updated imports)

### Implementation Status: COMPLETE

All code is fully implemented with no stubs, incomplete methods, or `NotImplementedError` raises:
- `_spec.py` — 268 lines, 5 public functions/classes fully implemented
- `evaluators/spec.py` — reduced to 19-line type alias (`EvaluatorSpec = NamedSpec`)
- `dataset.py` — helper functions at lines 1318-1361 delegate to `_spec.py` utilities

**Known TODOs in code:**
- Line 23 of `_spec.py`: `# TODO: Replace with public Pydantic API once available` — uses `pydantic._internal._typing_extra.get_function_type_hints()` (private API)
- Line 246 of `_spec.py`: `# TODO: Replace with pydantic.with_config once pydantic 2.11 is the min supported version`

**Type safety notes:**
- 2 justified `cast()` usages (type narrowing after runtime checks)
- 4 `pyright: ignore` comments for dynamic TypedDict construction and Pydantic internals

**Test coverage:** COMPLETE — 40 tests across `test_spec.py` (27 tests) and `test_evaluator_spec.py` (6 tests + integration), covering all public APIs, serialization forms, edge cases (non-string keys, list args), and error cases.

### Goal

Extract the generic "named specification" parsing logic currently embedded in `pydantic_evals` evaluator spec code into a shared module (`pydantic_ai._spec`) that can be reused for both evaluators and capabilities (and any future spec-based systems).

### Description

The existing `EvaluatorSpec` class in `pydantic_evals/evaluators/spec.py` contains ~190 lines of generic serialization/deserialization logic for "named specs" — objects defined by name + arguments in three short forms:

- `'ClassName'` (no arguments)
- `{'ClassName': single_arg}` (single positional argument)
- `{'ClassName': {key: value, ...}}` (keyword arguments)

This logic is not evaluator-specific. It should be extracted into a shared `NamedSpec` class in `pydantic_ai._spec`, along with these reusable utilities:

- `build_registry()` — Creates a name-to-class mapping from custom and default types
- `load_from_registry()` — Instantiates objects from a spec using a registry
- `build_schema_types()` — Generates JSON schema types for all registered classes

After extraction, `EvaluatorSpec` becomes a simple type alias: `EvaluatorSpec = NamedSpec`.

### Considerations

- The evaluator test suite (`tests/evals/test_evaluator_spec.py`) must continue to pass unchanged
- `_SerializedNamedSpec` (internal RootModel for handling serialized forms) is part of the extraction
- The `build_schema_types()` function uses `inspect.signature()` and `pydantic._internal._typing_extra.get_function_type_hints()` — the latter is a private Pydantic API (flagged in PR review). A TODO should note this dependency.
- `build_schema_types` also references `pydantic.with_config` which is only available in pydantic 2.11+ — a TODO should track this

### Linked Issues

- Prerequisite for [#4251](https://github.com/pydantic/pydantic-ai/issues/4251) (Agent from JSON/YAML)
- Prerequisite for [#4303](https://github.com/pydantic/pydantic-ai/issues/4303) (capabilities abstraction — needs spec-based capability construction)

---

## Issue 2: Extract `_instructions` module (instruction type normalization)

**Type:** Refactoring (enabler)
**Can be done independently:** Yes — pure extraction of existing logic into a dedicated module
**Files:** `pydantic_ai_slim/pydantic_ai/_instructions.py` (new), `pydantic_ai_slim/pydantic_ai/agent/__init__.py` (updated imports)

### Implementation Status: COMPLETE

Fully implemented, 26 lines total:
- `Instructions[AgentDepsT]` type alias — complete union type
- `normalize_instructions()` — complete with all branches (None, str/callable, sequence)

**Code quality:** No TODOs, no `cast()`, no `type: ignore`, fully type-safe.

**Test coverage:** Indirect only — tested through `test_agent.py` and `test_capabilities.py` (instructions merging, from_spec instructions). No dedicated test file for `_instructions.py` itself, but all code paths are exercised by downstream tests.

### Goal

Create a dedicated `_instructions` module that centralizes the `Instructions` type alias and `normalize_instructions()` utility, so that both the `Agent` class and `Capability` classes can share the same instruction types and normalization logic.

### Description

Currently, instruction handling types and normalization are defined inline in the agent module. This should be extracted into `pydantic_ai._instructions` containing:

- `Instructions[AgentDepsT]` type alias — Union of `str`, `SystemPromptFunc[AgentDepsT]`, `TemplateStr[AgentDepsT]`, `Sequence[...]`, or `None`
- `normalize_instructions()` — Converts any `Instructions` form into a flat `list`

### Considerations

- The `Instructions` type alias now includes `TemplateStr[AgentDepsT]` in the union — this is a public API type widening that should be documented
- PR review flagged inconsistent usage: the `Agent` module re-exports `Instructions` from `_instructions` but uses the qualified form (`_instructions.Instructions[AgentDepsT]`) in method signatures. Pick one style consistently.

### Linked Issues

- Prerequisite for [#4303](https://github.com/pydantic/pydantic-ai/issues/4303) (Instructions capability needs shared instruction types)

---

## Issue 3: Extract `_history_processor` module (unified history processor execution)

**Type:** Refactoring (enabler)
**Can be done independently:** Yes — extracts and unifies existing execution logic
**Files:** `pydantic_ai_slim/pydantic_ai/_history_processor.py` (new)

### Implementation Status: COMPLETE

Fully implemented, 60 lines total:
- `HistoryProcessor[DepsT]` — union of 4 processor type variants
- `run_history_processor()` — complete with 4-way dispatch (sync/async x with/without context)
- `_takes_ctx()` — parameter type inspection, handles None case

**Code quality:** No TODOs. 4 `cast()` usages — all justified for narrowing union types after control-flow checks.

**Test coverage:** COMPLETE — 28 tests in `test_history_processor.py` (1691 lines) covering:
- All 4 processor variants (sync/async x context/no-context)
- Error cases (empty history, history ending in response)
- Callable class processors
- Message identity tracking with deepcopy
- Resume-without-prompt scenarios
- Tool-call loop interactions
- `_takes_ctx` returns false for untyped processors

### Goal

Create a dedicated `_history_processor` module that provides a unified async executor for history processors, supporting all four calling conventions (sync/async x with/without RunContext).

### Description

History processor functions can currently be sync or async, and may or may not accept a `RunContext` as their first parameter. The dispatch logic for these four combinations should be centralized in a single module containing:

- `HistoryProcessor[DepsT]` — Union type of all four processor variants
- `run_history_processor()` — Unified async executor with automatic dispatch
- `_takes_ctx()` — Inspects first parameter type to determine if processor accepts `RunContext`

### Considerations

- PR review noted that the original `HistoryProcessorCapability` was reaching into private `_function_schema` internals (`_takes_ctx`, `is_async_callable`) — this extraction resolves that by providing a clean public utility
- The four-way dispatch uses casting to disambiguate union variants
- Sync processors are run in an executor to avoid blocking the async event loop

### Linked Issues

- Prerequisite for [#4303](https://github.com/pydantic/pydantic-ai/issues/4303) (HistoryProcessorCapability needs this module)

---

## Issue 4: `TemplateStr` — Handlebars template string support for instructions

**Type:** New feature (enabler)
**Can be done independently:** Yes — new module with no dependencies on capabilities
**Files:** `pydantic_ai_slim/pydantic_ai/_template.py` (new), `pydantic_ai_slim/pydantic_ai/__init__.py` (export), `pydantic_ai_slim/pyproject.toml` (optional dependency)
**New dependency:** `pydantic-handlebars` (optional, `>=0.1.0`)

### Implementation Status: COMPLETE

Fully implemented, 187 lines total:
- `TemplateStr` class with `__init__`, `render()`, `__call__()`, `__get_pydantic_core_schema__()`, `__repr__()`, `__str__()`
- `validate_from_spec_args()` utility for template resolution during spec loading
- `_hint_contains_template_str()` and `_import_pydantic_handlebars()` helpers

**Code quality:**
- No TODOs
- 1 justified `cast()` on line 92 (Pydantic validator return type narrowing)
- 1 protective `assert` on line 72 (guarding against impossible None state)
- Lazy import with informative error message if `pydantic-handlebars` not installed

**Dependency declaration:**
- `pyproject.toml` line 134: `handlebars = ["pydantic-handlebars>=0.1.0"]`
- Root `pyproject.toml` line 64: `handlebars = ["pydantic-ai-slim[handlebars]=={{ version }}"]`
- `TemplateStr` exported from `pydantic_ai` top-level

**Test coverage:** COMPLETE — 34 tests in `test_template.py` (317 lines), organized into 7 test classes:
- `TestTemplateStr` (8 tests) — construction, rendering, error handling
- `TestTemplateStrPydanticValidation` (7 tests) — union discrimination, serialization
- `TestValidateFromSpecArgs` (5 tests) — positional/keyword arg template resolution
- `TestAgentFromSpecDeps` (6 tests) — full Agent.from_spec with deps_type/deps_schema
- `TestAgentSpecTemplateFields` (7 tests) — top-level spec field templates
- `TestTemplateStrRender` (3 tests) — direct render method
- Integration test: `test_agent_run_with_template_instructions()` — full agent run with template rendering

**Note:** Tests use `pytest.importorskip('pydantic_handlebars')` — all 34 tests skip if optional dependency not installed.

### Goal

Add `TemplateStr` — a Handlebars template string type that can be used in agent instructions to render dynamic content against dependency objects at runtime.

### Description

`TemplateStr[AgentDepsT]` is a generic type that wraps a Handlebars template string (containing `{{placeholders}}`). It integrates with Pydantic's validation system and can be used wherever instructions are accepted:

- As a direct instruction string: `Agent(..., instructions=TemplateStr("Hello {{name}}"))`
- In spec files: strings containing `{{` are automatically compiled as templates
- As a callable: `TemplateStr` instances are callable and accept `RunContext`, rendering against `ctx.deps`

Key features:
- Dual compilation: typed (against `deps_type`) and untyped (generic)
- Pydantic `__get_pydantic_core_schema__` integration for automatic validation
- `validate_from_spec_args()` utility for template resolution during spec loading
- Lazy import of `pydantic-handlebars` (optional dependency)
- Exported from `pydantic_ai` top-level as `TemplateStr`

### Considerations

- PR review flagged that the validation logic relies on `ValueError` from union discrimination (strings without `{{` raise `ValueError` to fall through to plain `str` branch). This is fragile if Pydantic's union strategy changes. Consider adding a comment explaining the intentional pattern, or using a discriminated union.
- PR review flagged the example in the docstring should show template rendering in action, not just construction
- Uses `pydantic._internal._typing_extra.get_function_type_hints()` (private API) — should have a TODO for when a public alternative is available

### Linked Issues

- Prerequisite for [#4251](https://github.com/pydantic/pydantic-ai/issues/4251) (template instructions in YAML/JSON specs)
- Related to [#921](https://github.com/pydantic/pydantic-ai/issues/921) (prompt management, versioning, and optimization)
- Related to [PR #3656](https://github.com/pydantic/pydantic-ai/pull/3656) (customizable prompt templates)

---

## Issue 5: Dynamic (callable) model settings on `Agent`

**Type:** New feature (enabler)
**Can be done independently:** Yes — enhancement to existing Agent API
**Files:** `pydantic_ai_slim/pydantic_ai/agent/__init__.py` (model settings resolution), `docs/agent.md` (documentation)

### Implementation Status: COMPLETE

Fully implemented in the `get_model_settings()` closure within `Agent.iter()` (lines 953-969 of `agent/__init__.py`). The 4-level resolution chain:

1. **Model defaults** (`model_used.settings`) — base
2. **Agent-level** (`self.model_settings`) — can be callable, receives `RunContext` with base settings in `ctx.model_settings`
3. **Capability-level** (`self._root_capability.get_model_settings()`) — can be callable, receives `RunContext` with agent+base merged
4. **Run-level** (from `agent.run(model_settings=...)`) — can be callable, receives `RunContext` with capability+agent+base merged

Each stage updates `run_context.model_settings` for the next stage to inspect.

**Code quality:** No TODOs. The `RunContext.model_settings` field was added to `_run_context.py` (line 82-83) to support this.

**Test coverage:** COMPLETE — tested in `test_agent.py` (lines 8121-8225) with tests for:
- Dynamic settings with callable
- Static settings (baseline)
- Settings merging across levels
- `RunContext.model_settings` reflecting resolved-so-far state

### Goal

Allow `model_settings` on `Agent` (and in `agent.run()`) to be a callable `Callable[[RunContext], ModelSettings]` that resolves dynamically per agent step, enabling context-dependent settings like temperature scaling based on retry count.

### Description

Currently `model_settings` only accepts a static `ModelSettings` dict. This enhancement adds support for callables that receive `RunContext` and return `ModelSettings`, enabling patterns like:

```python
agent = Agent(
    'openai:gpt-5',
    model_settings=lambda ctx: ModelSettings(temperature=min(0.5 + ctx.run_step * 0.1, 1.0)),
)
```

The resolution order is: model defaults -> agent-level settings -> capability-level -> run-level settings. Inside a callable, `ctx.model_settings` reflects the settings resolved so far.

### Considerations

- PR review raised concerns about the mutation-as-communication pattern where `run_context.model_settings` is mutated three times during resolution to make "current resolved settings so far" available to the next callable. Consider passing accumulated settings explicitly instead.
- PR review noted asymmetry: if `agent_model_settings` is callable, it sees only `model_settings = base` (model defaults), but `run_model_settings` callable sees merged agent+model settings. This should be documented.
- The doc example uses `test="skip"` which PR review says should be avoidable — consider making it a testable example
- When both `model_settings=callable` on Agent AND a `ModelSettings(callable)` capability are used, there's a risk of double-application. The interaction should be clearly documented.

### Linked Issues

- Prerequisite for [#4303](https://github.com/pydantic/pydantic-ai/issues/4303) (ModelSettings capability uses this mechanism)

---

## Issue 6: Dynamic builtin tools (RunContext-based tool configuration)

**Type:** New feature (enabler)
**Can be done independently:** Yes — enhancement to existing Agent API
**Files:** `pydantic_ai_slim/pydantic_ai/agent/__init__.py`, `pydantic_ai_slim/pydantic_ai/_agent_graph.py`, `docs/builtin-tools.md`

### Implementation Status: COMPLETE

Fully implemented:
- `builtin_tools` parameter accepts `AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]`
- Resolution happens at runtime in `_agent_graph.py` when building `ModelRequestParameters`
- Callables receiving `RunContext` and returning `AbstractBuiltinTool | None`

**Test coverage:** COMPLETE — tested in `test_agent.py` with:
- `test_agent_builtin_tools_runtime_vs_agent_level` — merging runtime and agent-level tools
- `test_dynamic_builtin_tool_configured` — dynamic tool resolved with RunContext deps
- `test_dynamic_builtin_tool_omitted` — callable returning None excludes tool
- `test_mixed_static_and_dynamic_builtin_tools` — static + dynamic combined
- `test_sync_dynamic_tool` — synchronous callable support
- `test_dynamic_tool_in_run_call` — dynamic tools passed to `agent.run()`

### Goal

Allow builtin tools to be specified as functions that take `RunContext` and return `AbstractBuiltinTool | None`, enabling conditional tool inclusion and runtime configuration based on agent dependencies.

### Description

Currently, builtin tools (like `WebSearchTool`, `CodeExecutionTool`) can only be configured statically. This enhancement allows passing a callable instead:

```python
def maybe_web_search(ctx: RunContext[MyDeps]) -> WebSearchTool | None:
    if ctx.deps.user_location:
        return WebSearchTool(user_location=ctx.deps.user_location)
    return None

agent = Agent('openai:gpt-5', builtin_tools=[maybe_web_search])
```

When the callable returns `None`, the tool is excluded from that run.

### Considerations

- The `builtin_tools` parameter type signature widens to accept `AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]`
- Resolution happens at runtime in the graph layer, not at Agent init time
- UI adapter methods (`run_stream_native`, `run_stream`, `dispatch_request`) also gain `builtin_tools` parameter support

### Linked Issues

- Related to [#3212](https://github.com/pydantic/pydantic-ai/issues/3212) (builtin tool fallback for models without support)

---

## Issue 7: `AbstractCapability` base class + `CombinedCapability` + Agent integration

**Type:** New feature (core)
**Can be done independently:** No — depends on [2] and [3]
**Files:** `pydantic_ai_slim/pydantic_ai/capabilities/abstract.py`, `pydantic_ai_slim/pydantic_ai/capabilities/combined.py`, `pydantic_ai_slim/pydantic_ai/capabilities/__init__.py`, `pydantic_ai_slim/pydantic_ai/agent/__init__.py`, `pydantic_ai_slim/pydantic_ai/_agent_graph.py`, `pydantic_ai_slim/pydantic_ai/_run_context.py`
**Existing issue:** [#4303](https://github.com/pydantic/pydantic-ai/issues/4303)

### Implementation Status: COMPLETE (code) / PARTIAL (tests)

**Code status:** All files fully implemented with no stubs:
- `abstract.py` — `AbstractCapability` with all 8 lifecycle methods having proper defaults, `BeforeModelRequestContext` dataclass
- `combined.py` — `CombinedCapability` with forward/reverse hook chaining, settings merge (static + dynamic), toolset combination
- `__init__.py` — Registry (`DEFAULT_CAPABILITY_TYPES`, `CAPABILITY_TYPES`), all exports
- Agent integration — `capabilities` parameter, `_root_capability = CombinedCapability(capabilities)`, hook calls in `ModelRequestNode`
- Graph layer — `GraphAgentDeps.root_capability` field, `before_model_request` called in `_prepare_request()`, `after_model_request` called in `_finish_handling()`

**Code quality:**
- `abstract.py`: No TODOs, no `cast()`, no `type: ignore`. All methods have docstrings.
- `combined.py`: 1 `type: ignore[return-value]` on line 45 (justified), 1 `pyright: ignore` on line 57. No TODOs.
- `__init__.py`: Clean, uses walrus operator for registry construction.

**Test coverage gaps (CRITICAL):**
- `after_model_request` hook is **never tested** — no test verifies the reverse-order hook execution or response modification
- `before_model_request` hook chaining with multiple capabilities is **not tested** — only single-capability pass-through tested
- `CombinedCapability.get_toolset()` merging of multiple toolsets is **not tested**
- `CombinedCapability.get_builtin_tools()` flattening from multiple capabilities is **not tested**
- Dynamic model settings merging via `CombinedCapability.get_model_settings()` resolver function is **not tested** (only static merge tested)
- `before_model_request` mutation of messages/settings/parameters is **not tested** (only pass-through verified)

### Goal

Introduce the core `AbstractCapability` base class — the fundamental extensibility point for the capabilities system — along with `CombinedCapability` for composing multiple capabilities, and integrate both into the `Agent` constructor and the agent graph execution layer.

### Description

**`AbstractCapability[AgentDepsT]`** (abstract dataclass, generic over deps type) — the base class that all capabilities must subclass. Defines the capability lifecycle:

- `get_instructions()` — Return static instructions for the system prompt
- `get_model_settings()` — Return static or dynamic (callable) model settings
- `get_toolset()` — Return a toolset (or `ToolsetFunc` for dynamic toolsets)
- `get_builtin_tools()` — Return a list of builtin tools
- `before_model_request(ctx, request_context)` — Hook called before each model request (can modify messages, settings, request parameters)
- `after_model_request(ctx, response)` — Hook called after each model response (can modify response)
- `get_serialization_name()` — Class method returning name for spec-based construction (return `None` to opt out)
- `from_spec(*args, **kwargs)` — Class method factory for spec-based instantiation

**`BeforeModelRequestContext`** — dataclass wrapping messages, model_settings, and model_request_parameters for the before_model_request hook. PR review recommended this (originally a bare tuple) to allow the hook surface area to grow without breaking existing implementations.

**`CombinedCapability[AgentDepsT]`** — Composite pattern implementation:
- Concatenates instructions from all capabilities
- Merges model settings (static merged bottom-up; dynamic collected and resolved in order)
- Combines toolsets into `CombinedToolset`
- Flattens builtin tools
- Chains `before_model_request` forward (0->n) and `after_model_request` in reverse (n->0, LIFO)

**Agent integration:**
- New `capabilities` parameter on `Agent` constructor accepting `Sequence[AbstractCapability]`
- Agent composes all capabilities into a `CombinedCapability` stored as `_root_capability`
- Instructions, tools, toolsets, and builtin tools are extracted from root capability during init
- `_root_capability` is passed to the graph layer via `GraphAgentDeps.root_capability`
- `ModelRequestNode` calls `before_model_request` and `after_model_request` hooks during execution
- Capability model settings are resolved in the settings precedence chain: model defaults -> agent-level -> capability-level -> run-level

### Considerations

- PR review strongly recommended adding comprehensive docstrings to all `AbstractCapability` methods — this is the public extension point that users and third-party packages will subclass
- PR review flagged `root_capability` should be `_root_capability` (private) on the Agent class to avoid exposing internal wiring
- PR review flagged that `GraphAgentState.model_settings` uses `Any` type annotation to work around Pydantic schema issues with `httpx.Timeout` — should use `dict[str, Any] | None` or exclude from schema
- PR review flagged duplicate validation in `history_processor.py` and `_agent_graph.py` for empty messages / must-end-with-ModelRequest — centralize in graph layer only
- `before_model_request` runs forward, `after_model_request` runs in reverse — this LIFO pattern should be documented

### Important context from related issues

**The current hooks (`before/after_model_request`) are intentionally a starting point, not the full surface area.** Per [#2885](https://github.com/pydantic/pydantic-ai/issues/2885) and [#1197](https://github.com/pydantic/pydantic-ai/issues/1197), the capabilities system will eventually need:
- `before_tool_call` / `after_tool_call` hooks — for tool arg validation, output sanitization, PII replacement (see Appendix C2)
- `on_agent_start` / `on_agent_end` hooks — for per-run initialization/cleanup, output guardrails (see Appendix C3)
- Parallel input guardrails — guardrails running concurrently with model request (see Appendix C3)

The `AbstractCapability` base class should be designed so these hooks can be added later without breaking existing capability implementations (this is why `BeforeModelRequestContext` was made a dataclass instead of a bare tuple).

**DouweM confirmed** in [#4347](https://github.com/pydantic/pydantic-ai/issues/4347) that `get_toolset()` intentionally does NOT take `RunContext` — it's evaluated once at registration, not per-run. This is a deliberate design choice.

### Linked Issues

- [#4303](https://github.com/pydantic/pydantic-ai/issues/4303) — The parent issue for the entire capabilities abstraction
- [#2885](https://github.com/pydantic/pydantic-ai/issues/2885) — Middlewares/hooks for processing model requests/responses (current hooks partially address this; `before/after_tool_call` still needed)
- [#1197](https://github.com/pydantic/pydantic-ai/issues/1197) — Guardrails (capabilities provide the foundation; parallel execution and output guardrails are future work)
- [#4347](https://github.com/pydantic/pydantic-ai/issues/4347) — Toolset state isolation (affects how `get_toolset()` interacts with capabilities)
- [#4359](https://github.com/pydantic/pydantic-ai/issues/4359) — Tool budget reminders (confirmed as capability candidate by maintainers)
- [#4262](https://github.com/pydantic/pydantic-ai/issues/4262) — Tool output validation (needs `after_tool_call` hook)

---

## Issue 8: `Instructions` capability

**Type:** New feature (capability)
**Serializability:** Tier S — `- Instructions: "You are helpful"` or `- Instructions: "Hello {{name}}"` (TemplateStr)
**Can be done independently:** No — depends on [7] (AbstractCapability) and [2] (_instructions module)
**Files:** `pydantic_ai_slim/pydantic_ai/capabilities/instructions.py`

### Implementation Status: COMPLETE (code) / PARTIAL (tests, ~70%)

**Code status:** Fully implemented, no issues:
- `@dataclass` with `instructions` field
- `get_instructions()` returns stored instructions
- `from_spec(instructions='')` accepts optional string/TemplateStr (narrower than full type for serializability)
- Serialization name: `"Instructions"` (default from base class)

**Code quality:** No TODOs, no `cast()`, no `type: ignore`. Proper docstrings.

**Test coverage gaps:**
- No direct unit test of `get_instructions()` return value
- No test of `Instructions` capability with `TemplateStr` (only plain strings tested)
- No test of `Instructions` with callable instructions (dynamic system prompts)
- Merging with Agent-level instructions tested via `test_agent_from_spec_capabilities_merged()` — OK
- `from_spec()` tested indirectly via `Agent.from_spec()` — OK

### Goal

Provide a capability that wraps instructions (static strings, template strings, or callables) as a composable, spec-constructible capability.

### Description

`Instructions[AgentDepsT]` capability class:
- Accepts `instructions: Instructions[AgentDepsT]` (string, callable, TemplateStr, or sequence)
- `get_instructions()` returns stored instructions
- `from_spec()` narrows accepted types to serialize-safe variants only (strings and `TemplateStr`, not callables)

### Considerations

- `from_spec()` deliberately restricts to serializable instruction types — callables can't be represented in YAML/JSON
- When used alongside Agent-level `instructions`, they are additive (both contribute to system prompt)

### Linked Issues

- Part of [#4303](https://github.com/pydantic/pydantic-ai/issues/4303)

---

## Issue 9: `ModelSettings` capability

**Type:** New feature (capability)
**Serializability:** Tier P — `- ModelSettings: {temperature: 0.7}` (static only; callables not serializable)
**Can be done independently:** No — depends on [7] (AbstractCapability) and [5] (dynamic model settings)
**Files:** `pydantic_ai_slim/pydantic_ai/capabilities/model_settings.py`

### Implementation Status: COMPLETE (code) / GOOD (tests, ~80%)

**Code status:** Fully implemented:
- `@dataclass` with `settings` field accepting `_ModelSettings | Callable[[RunContext], _ModelSettings]`
- `get_serialization_name()` explicitly returns `'ModelSettings'`
- `from_spec(*args, **kwargs)` handles both positional dict and keyword arguments
- `get_model_settings()` returns settings (static or callable)

**Code quality:**
- 2 `cast()` usages in `from_spec()` (lines 38-39) — justified for `*args/**kwargs` -> `_ModelSettings` conversion
- No TODOs

**Test coverage:**
- `test_model_settings_from_spec_positional` and `test_model_settings_from_spec_kwargs` — OK
- `test_callable_model_settings` — tests callable returning settings
- `test_model_settings_static_before_model_request` — tests static pass-through
- `test_combined_model_settings` — tests merging of multiple static settings
- `test_combined_no_model_settings` — tests None return

**Test coverage gaps:**
- Dynamic (callable) settings resolution through the full `CombinedCapability.get_model_settings()` resolver chain is **not tested**
- Interaction between Agent-level callable `model_settings` and a `ModelSettings(callable)` capability (double-application risk) is **not tested**

### Goal

Provide a capability that contributes model settings (static or dynamic/callable) to the agent's settings resolution chain.

### Description

`ModelSettings[AgentDepsT]` capability class:
- Accepts `settings: ModelSettings | Callable[[RunContext], ModelSettings]`
- `get_model_settings()` returns static settings or callable
- `from_spec()` supports both positional dict and keyword arguments: `from_spec({'temperature': 0.7})` or `from_spec(temperature=0.7)`
- Settings are merged on top of agent-level defaults; capability settings take precedence over agent defaults but can be overridden by run-level settings

### Considerations

- **Naming collision:** PR review flagged that `ModelSettings` (the capability class) shadows `pydantic_ai.settings.ModelSettings` (the widely-used TypedDict). Consider renaming to `ModelSettingsCapability` or `DynamicModelSettings`. The `get_serialization_name()` can still return `'ModelSettings'` for YAML/JSON specs.
- PR review flagged double-merge for static settings: `get_model_settings()` merges at init time, and `before_model_request` merges again per-request. Consider skipping merge in `before_model_request` when settings are not callable.
- PR review flagged potential double-application if user sets both `model_settings=callable` on Agent AND adds a `ModelSettings(callable)` capability. The interaction should be documented.

### Linked Issues

- Part of [#4303](https://github.com/pydantic/pydantic-ai/issues/4303)

---

## Issue 10: `Thinking` capability

**Type:** New feature (capability)
**Serializability:** Tier S — `- Thinking` (zero-config, no args)
**Can be done independently:** No — depends on [9] (ModelSettings capability, which it extends)
**Files:** `pydantic_ai_slim/pydantic_ai/capabilities/thinking.py`

### Implementation Status: COMPLETE (code, placeholder) / PARTIAL (tests, ~60%)

**Code status:** Fully implemented but explicitly a **temporary placeholder**:
- Extends `ModelSettings[AgentDepsT]` (not a `@dataclass`)
- `__init__()` hardcodes provider-specific settings for 4 providers: OpenAI (`reasoning_effort='high'`), Anthropic (`thinking={'type': 'adaptive'}`), Google (`thinking_config={'include_thoughts': True}`), Gemini (`thinking_config={'include_thoughts': True}`)
- `from_spec()` raises `TypeError` if any arguments provided
- Serialization name: `"Thinking"`

**Code quality:**
- 1 `cast(_ModelSettings, {...})` on lines 34-35 — necessary because provider-specific keys aren't in base TypedDict
- **TODO on lines 14-16:** "This is a placeholder that hardcodes provider-specific thinking settings. It will be replaced by unified thinking settings from #3894, which will also allow configurable parameters (e.g. reasoning_effort, budget_tokens)."

**Test coverage:**
- `test_thinking_applies_settings` — verifies settings dict contents
- `test_thinking_from_spec_rejects_args` — verifies TypeError on args

**Test coverage gaps:**
- No integration test running an agent with `Thinking` capability to verify the model actually receives thinking settings
- No test of `Thinking` combined with other capabilities
- No provider-specific behavior validation

### Goal

Provide a zero-config capability that enables model thinking/reasoning across all supported providers.

### Description

`Thinking[AgentDepsT]` extends `ModelSettings`:
- Takes no arguments — `Thinking()` with no configuration
- Hardcodes provider-specific settings: `openai_reasoning_effort='high'`, `anthropic_thinking={'type': 'adaptive'}`, `google_thinking_config={'include_thoughts': True}`, `gemini_thinking_config={'include_thoughts': True}`
- `from_spec()` raises `TypeError` if any arguments are provided

### Considerations

- This is explicitly a **temporary placeholder** — the code comments note it will be replaced by unified thinking settings from [#3894](https://github.com/pydantic/pydantic-ai/issues/3894) (PR already exists with `thinking: bool` and `thinking_effort` fields)
- PR review flagged the `cast(_ModelSettings, {...})` as defeating type-safety
- PR review flagged that this is fragile: if a new provider adds thinking support, this capability won't cover it until manually updated
- PR review flagged the `__init__` pattern (non-dataclass overriding dataclass parent) as fragile: if `ModelSettings` gains another required field, `Thinking` will silently fail to initialize it
- Once [#3894](https://github.com/pydantic/pydantic-ai/issues/3894) lands, `Thinking` should use the unified `thinking=True, thinking_effort='high'` settings instead

### Linked Issues

- [#3894](https://github.com/pydantic/pydantic-ai/issues/3894) — Unified thinking settings across all model providers (will replace this implementation)
- Part of [#4303](https://github.com/pydantic/pydantic-ai/issues/4303)

---

## Issue 11: `WebSearch` capability

**Type:** New feature (capability)
**Serializability:** Tier S — `- WebSearch` (zero-config, no args)
**Can be done independently:** No — depends on [7] (AbstractCapability)
**Files:** `pydantic_ai_slim/pydantic_ai/capabilities/web_search.py`

### Implementation Status: COMPLETE (code) / LOW (tests, ~30%)

**Code status:** Fully implemented, minimal (17 lines):
- No `@dataclass` needed (no state)
- Module-level singleton: `_BUILTIN_WEB_SEARCH_TOOL = WebSearchTool()`
- `get_builtin_tools()` returns `[_BUILTIN_WEB_SEARCH_TOOL]`
- Serialization name: `"WebSearch"` (default from base class)

**Code quality:**
- **TODO on line 14:** `# TODO: Add toolset-based fallback for models without builtin web search (#3212)`
- No `cast()`, no `type: ignore`

**Test coverage gaps (SIGNIFICANT):**
- No integration test running an agent with `WebSearch` capability to verify it actually provides the builtin tool
- Only mentioned in `test_agent_from_spec_basic()` as part of a spec and in JSON schema tests
- `get_builtin_tools()` return value is **never verified** in any test
- No test of `WebSearch` combined with other capabilities

### Goal

Provide a capability that enables web search functionality via the provider's builtin `WebSearchTool`.

### Description

`WebSearch[AgentDepsT]` capability:
- No custom fields or initialization
- `get_builtin_tools()` returns a singleton `WebSearchTool()` instance
- Module-level singleton avoids creating multiple tool instances

### Considerations

- Currently only uses the provider builtin `WebSearchTool` — there is a TODO referencing [#3212](https://github.com/pydantic/pydantic-ai/issues/3212) for adding a toolset-based fallback (e.g., DuckDuckGo, Tavily) for models that don't support the builtin web search
- PR review noted that commented-out fallback code should be removed and tracked via the TODO/issue reference instead

### Linked Issues

- [#3212](https://github.com/pydantic/pydantic-ai/issues/3212) — Let builtin tools fallback on custom tools for models that don't support them
- Part of [#4303](https://github.com/pydantic/pydantic-ai/issues/4303)

---

## Issue 12: `Toolset` capability

**Type:** New feature (capability)
**Serializability:** Tier N — not serializable (toolsets are arbitrary code)
**Can be done independently:** No — depends on [7] (AbstractCapability)
**Files:** `pydantic_ai_slim/pydantic_ai/capabilities/toolset.py`

### Implementation Status: COMPLETE (code) / PARTIAL (tests, ~70%)

**Code status:** Fully implemented, minimal (20 lines):
- `@dataclass` with `toolset: AbstractToolset[AgentDepsT]`
- `get_toolset()` returns the stored toolset
- `get_serialization_name()` returns `None` (opts out of spec construction)

**Code quality:** No TODOs, no `cast()`, no `type: ignore`.

**Test coverage:**
- `test_toolset_capability` — verifies `get_toolset()` returns the toolset
- `test_toolset_capability_in_agent` — verifies toolset tools are available to the agent
- `test_capability_returning_toolset_func` and `test_toolset_func_with_combined_capabilities` — ToolsetFunc variant

**Test coverage gaps:**
- No test of multiple `Toolset` capabilities being merged via `CombinedCapability.get_toolset()`
- No test of `DynamicToolset` wrapping when `ToolsetFunc` is used inside `CombinedCapability`

### Goal

Provide a simple wrapper capability that contributes an `AbstractToolset` to the agent.

### Description

`Toolset[AgentDepsT]` capability:
- Accepts `toolset: AbstractToolset[AgentDepsT]`
- `get_toolset()` returns the stored toolset
- `get_serialization_name()` returns `None` — opts out of spec-based construction (toolsets are not serializable)

### Considerations

- Minimal wrapper — toolsets contain arbitrary code and cannot be serialized
- Useful for composing toolsets alongside other capabilities in the `capabilities` list
- **Toolset state isolation ([#4347](https://github.com/pydantic/pydantic-ai/issues/4347)):** Static toolsets passed via `Toolset(my_toolset)` share mutable state across runs. DouweM confirmed `get_toolset()` intentionally does NOT take `RunContext` — evaluated once at registration. Users needing per-run isolation must use `DynamicToolset`/factory pattern or await the future `per_run` flag on `AbstractToolset`. See Appendix C1 for full context.
- **Sub-agent isolation:** A `for_sub_agent()` method on `AbstractToolset` is being considered for controlling toolset sharing with sub-agents (Layer 3 of the proposal in #4347)

### Linked Issues

- Part of [#4303](https://github.com/pydantic/pydantic-ai/issues/4303)
- [#4347](https://github.com/pydantic/pydantic-ai/issues/4347) — Toolset state leaks / factory pattern (directly impacts this capability)

---

## Issue 13: `HistoryProcessorCapability`

**Type:** New feature (capability)
**Serializability:** Tier N — not serializable (processors are callables)
**Can be done independently:** No — depends on [7] (AbstractCapability) and [3] (_history_processor module)
**Files:** `pydantic_ai_slim/pydantic_ai/capabilities/history_processor.py`

### Implementation Status: COMPLETE (code) / **NONE (tests, 0%)**

**Code status:** Fully implemented, 28 lines:
- `@dataclass` with `processor: _history_processor.HistoryProcessor[AgentDepsT]`
- `before_model_request()` calls `run_history_processor()` to process `request_context.messages` in-place
- `get_serialization_name()` returns `None` (opts out of spec construction)

**Code quality:** No TODOs, no `cast()`, no `type: ignore`.

**Test coverage: ZERO tests for this capability wrapper.**

The underlying `_history_processor` module is well-tested (28 tests), and history processors passed via `history_processors=` on Agent are tested (they're auto-wrapped into `HistoryProcessorCapability`). However, there are **no tests** for:
- Creating `HistoryProcessorCapability` directly and passing via `capabilities=`
- The `before_model_request` hook implementation on this capability
- The `get_serialization_name()` returning `None`
- Interaction with other capabilities in `CombinedCapability`

**Note:** The auto-wrapping in Agent `__init__` (lines 373-375) means the capability is exercised indirectly by existing history processor tests, but the capability class itself has no direct test coverage.

### Goal

Bridge the existing history processor API with the capabilities system by wrapping history processors as capabilities.

### Description

`HistoryProcessorCapability[AgentDepsT]` capability:
- Accepts `processor: HistoryProcessor[AgentDepsT]`
- Calls `run_history_processor()` in its `before_model_request` hook to process messages before sending to the model
- `get_serialization_name()` returns `None` — opts out of spec-based construction (processors contain callable code)
- The Agent constructor auto-wraps `history_processors` arg into `HistoryProcessorCapability` instances for backward compatibility

### Considerations

- PR review noted duplicate validation between this capability and `_agent_graph.py` (empty messages, must-end-with-ModelRequest) — should keep validation only in the graph layer
- This maintains backward compatibility: users can still pass `history_processors=` to Agent and they'll be automatically wrapped

### Linked Issues

- Part of [#4303](https://github.com/pydantic/pydantic-ai/issues/4303)
- Related to [#4137](https://github.com/pydantic/pydantic-ai/issues/4137) — First-class context compaction API (compaction will likely be implemented as a capability using history processing)

---

## Issue 14: Durable execution refactor (unified `WrapperAgent` pattern)

**Type:** Refactoring (enabler)
**Can be done independently:** Yes — primarily a refactoring of existing integrations
**Files:** `pydantic_ai_slim/pydantic_ai/durable_exec/dbos/_agent.py`, `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_agent.py`, `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_agent.py`

### Implementation Status: COMPLETE

All three adapters fully implement the `WrapperAgent` pattern with no stubs, no TODOs, and no incomplete methods:

- **DBOSAgent:** Inherits `WrapperAgent`, creates `DBOSModel`, wraps toolsets via `dbosify_toolset`, `_dbos_overrides()` context manager, all run methods delegated. Runtime validation prevents non-DBOS models.
- **PrefectAgent:** Inherits `WrapperAgent`, creates `PrefectModel`, wraps toolsets via `_prefectify_toolset`, `_prefect_overrides()` context manager, `ContextVar` for flow detection, event stream handlers wrapped in tasks.
- **TemporalAgent:** Inherits `WrapperAgent`, creates `TemporalModel`, activity registration, `_temporal_overrides()` context manager, serialized run context, most restrictive (no `run_sync`/`run_stream`/`iter` in workflows).

**Test coverage:** Existing tests in `test_dbos.py` and `test_temporal.py` continue to pass.

### Goal

Refactor DBOS, Prefect, and Temporal durable execution agents to follow a consistent `WrapperAgent`-based pattern with unified model/toolset overriding via context managers.

### Description

All three durable execution integrations are refactored to follow the same structural template:

1. **Inherit from `WrapperAgent`** — Composition-based approach wrapping an `AbstractAgent`
2. **Store integration-specific models** — Each creates a wrapper model (DBOSModel, PrefectModel, TemporalModel)
3. **Wrap toolsets via visitor pattern** — `toolset.visit_and_replace()` to transform toolsets integration-specifically
4. **Context-based overrides** — A single `_overrides()` context manager centralizes model/toolset swapping
5. **Delegate run methods** — All public methods delegate to parent while activating overrides

Integration-specific behaviors remain:
- **DBOS:** Workflow decorators, no runtime model changes (determinism requirement)
- **Prefect:** `@flow`/`@task` decorators, `ContextVar` for flow detection
- **Temporal:** Activity registration, serialized run context, most restrictive (no `run_sync`/`run_stream`/`iter` in workflows)

### Considerations

- This refactoring aligns the durable execution agents with the `WrapperAgent` / `AbstractAgent` abstractions that the capabilities system builds on
- Each wrapper creates a specialized model instance for the integration's execution model
- Event stream handlers are adapted per-integration (DBOS: workflow context, Prefect: tasks, Temporal: activities)

### Linked Issues

- Supports [#4303](https://github.com/pydantic/pydantic-ai/issues/4303) — durable execution will eventually become a capability itself

---

## Issue 15: UI adapter enhancements (`toolsets`/`builtin_tools` params)

**Type:** Enhancement (enabler)
**Can be done independently:** Yes — additive parameters
**Files:** `pydantic_ai_slim/pydantic_ai/ui/_adapter.py`, `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/_adapter.py`

### Implementation Status: COMPLETE

**UI adapter (`_adapter.py`):**
- `run_stream_native()`, `run_stream()`, `dispatch_request()` all gain `toolsets` and `builtin_tools` parameters
- Parameters passed through to `agent.run_stream_events()`
- `StateHandler` protocol fully implemented

**Vercel AI SDK v6 (`vercel_ai/_adapter.py`):**
- `sdk_version` parameter (5 or 6) added to `from_request()` and `dispatch_request()`
- v6 enables deferred tool approval streaming
- Comprehensive message conversion (212-428 lines), dump (675-737 lines), metadata handling
- 2 design TODOs for citation support (out of scope)

**Test coverage:** Tested via integration in `test_ag_ui.py`.

### Goal

Enhance UI adapter methods to support passing additional toolsets and builtin tools per run, and add Vercel AI SDK v6 support.

### Description

**UIAdapter enhancements:**
- `toolsets` parameter (optional additional toolsets per run) added to `run_stream_native()`, `run_stream()`, and `dispatch_request()`
- `builtin_tools` parameter (optional additional built-in tools per run) added to the same methods

**Vercel AI SDK v6:**
- New `sdk_version` parameter with values `5` (default, backwards-compatible) and `6` (enables tool approval streaming for human-in-the-loop workflows)
- Added to `from_request()` and `dispatch_request()` methods

### Considerations

- These are additive, backward-compatible changes
- The `sdk_version=6` enables deferred tool approval handling in the Vercel adapter

### Linked Issues

- Supports [#4303](https://github.com/pydantic/pydantic-ai/issues/4303) — capabilities may provide toolsets/builtin_tools that need to flow through UI adapters

---

## Issue 16: `AgentSpec` serialization + `Agent.from_spec` + `Agent.from_file`

**Type:** New feature (serialization)
**Can be done independently:** No — depends on [1] (NamedSpec), [4] (TemplateStr), [7-13] (capabilities)
**Files:** `pydantic_ai_slim/pydantic_ai/agent/spec.py` (new), `pydantic_ai_slim/pydantic_ai/agent/__init__.py` (new methods)
**Existing issue:** [#4251](https://github.com/pydantic/pydantic-ai/issues/4251)

### Implementation Status: COMPLETE (code) / GOOD (tests)

**Code status — `agent/spec.py`:** Fully implemented:
- `AgentSpec(BaseModel)` with all fields (model, name, description, instructions, deps_schema, output_schema, model_settings, capabilities, retries, output_retries, end_strategy, tool_timeout, instrument, metadata, json_schema_path)
- `from_file(path, fmt)` — YAML/JSON loading with format auto-detection
- `to_file(path, fmt, schema_path, custom_capability_types)` — file saving with optional schema
- `model_json_schema_with_capabilities(custom_capability_types)` — JSON schema generation
- Helper functions: `_infer_fmt()`, `_get_capability_registry()`, `_build_capability_schema_types()`

**Code status — `Agent.from_spec()`:** Fully implemented (lines 465-691 of `agent/__init__.py`):
- Accepts `spec: dict[str, Any] | AgentSpec`
- Parameters: `deps_type`, `custom_capability_types`, plus all standard Agent params
- Implementation: builds template context -> validates spec -> resolves output_type (from arg, spec.output_schema via `StructuredDict`, or default `str`) -> builds capability registry -> merges instructions -> loads capabilities -> merges with passed capabilities -> constructs Agent
- `output_schema` support via `StructuredDict`
- `deps_schema` support for template validation context

**Code quality:** No TODOs. 1 `cast()` on line 139 (context dict access). All docstrings present.

**Test coverage (64 tests in `test_capabilities.py`):**
- Basic from_spec with capabilities, no capabilities, unknown capability, bad args
- Custom capability types, AgentSpec objects, output_type, output_schema
- Instructions merging, model_settings merging, all Agent params (retries, end_strategy, tool_timeout, instrument, metadata)
- JSON schema generation (default + custom), schema file persistence
- YAML and JSON file I/O, round-trips, `$schema` field handling
- CLI `load_agent` with YAML and JSON
- `from_spec` with deps_type and deps_schema

**Test coverage gaps:**
- `Agent.from_file()` convenience method is **not directly tested** (only `AgentSpec.from_file()` tested)
- File round-trip with capabilities in spec is **not tested** (only round-trips without capabilities)
- Complex capability compositions in from_spec (e.g., Instructions + ModelSettings + custom capability + Toolset) are **not tested**

### Goal

Enable agents to be defined declaratively via JSON, YAML, or Python dicts, and loaded from files — making agent definitions serializable and optimizable.

### Description

**`AgentSpec(BaseModel)`** — Pydantic model representing a serializable agent specification:
- `model` (required) — Model identifier string
- `name`, `description` — Optional agent metadata
- `instructions` — String, `TemplateStr`, or list
- `model_settings` — Dict of model settings
- `capabilities` — List of `CapabilitySpec` (alias for `NamedSpec`)
- `deps_schema`, `output_schema` — JSON schemas for deps/output types
- `retries`, `output_retries`, `end_strategy`, `tool_timeout`, `instrument`, `metadata`
- `json_schema_path` (`$schema`) — Optional JSON schema reference

**Key methods:**
- `AgentSpec.from_file(path, fmt)` — Load from YAML/JSON (auto-detects format from extension)
- `AgentSpec.to_file(path, fmt, schema_path, custom_capability_types)` — Save to file with optional schema
- `AgentSpec.model_json_schema_with_capabilities(custom_capability_types)` — Generate JSON schema

**New `Agent` methods:**
- `Agent.from_spec(spec, *, model, capabilities, custom_capability_types, deps_type, output_type, ...)` — Create agent from spec dict, `AgentSpec`, or file path. Parameters are additional to or take precedence over spec values.
- `Agent.from_file(path, ...)` — Convenience wrapper for `from_spec` with file path

**Capability registry:**
- `DEFAULT_CAPABILITY_TYPES` — Built-in capabilities (Instructions, ModelSettings, Thinking, WebSearch)
- `CAPABILITY_TYPES` — Full registry dict mapping names to classes
- `custom_capability_types` parameter allows users to register their own capability classes
- User capabilities can override defaults

### Considerations

- PR review flagged the inner `AgentSpec` class in `model_json_schema_with_capabilities` shadows the outer class — rename to `_AgentSpecSchema`
- PR review noted the generated JSON schema only includes `model` and `capabilities` fields, not all `AgentSpec` fields (instructions, settings, retries, etc.)
- PR review flagged unsafe `cast(ModelSettings, validated_spec.model_settings)` when value is `None`
- Supports the GEPA-style "agent optimization" use case where specs can be passed as runtime overrides
- `from_spec` behavior: instructions and capabilities are additive; model, model_settings, and other scalar fields take precedence from parameters
- `output_schema` enables defining output types via JSON schema for fully declarative agents

### Linked Issues

- [#4251](https://github.com/pydantic/pydantic-ai/issues/4251) — Agent Definition from JSON/YAML
- [#3179](https://github.com/pydantic/pydantic-ai/issues/3179) — Support for algorithmic optimizers (specs enable optimization)
- [#921](https://github.com/pydantic/pydantic-ai/issues/921) — Prompt management, versioning, and optimization

---

## Issue 17: CLI agent loading from spec files

**Type:** New feature
**Can be done independently:** No — depends on [16] (AgentSpec)
**Files:** `pydantic_ai_slim/pydantic_ai/_cli/__init__.py`

### Implementation Status: COMPLETE

**Code status:** Fully implemented:
- `load_agent()` detects YAML/JSON spec files and loads via `Agent.from_spec()`
- `SUPPORTED_BUILTIN_TOOLS` used for CLI tool ID resolution
- `model_settings` parameter added to `run_chat()` and passed to `agent.iter()`
- Returns `None` for non-existent files (graceful fallback)

**Code quality:** 1 `pyright: ignore` for dynamic import. No TODOs in core logic.

**Test coverage:** COMPLETE — `test_cli_load_agent_yaml`, `test_cli_load_agent_json`, `test_cli_load_agent_missing_file` in `test_capabilities.py`.

### Goal

Enable the CLI (`clai`) to load agents from YAML/JSON spec files, allowing agents to be defined declaratively and used via the command line.

### Description

The `load_agent` function in the CLI module gains support for:
- Detecting YAML/JSON agent spec files
- Loading agents via `Agent.from_spec()` / `Agent.from_file()`
- Using `SUPPORTED_BUILTIN_TOOLS` to determine available CLI tool IDs

### Considerations

- Returns `None` for non-existent files (graceful fallback)
- Both YAML and JSON formats are auto-detected from file extension

### Linked Issues

- Part of [#4251](https://github.com/pydantic/pydantic-ai/issues/4251)
- Part of [#4303](https://github.com/pydantic/pydantic-ai/issues/4303) (example from parent issue shows `clai tui agent.yml`)

---

## Appendix A: Critical Test Coverage Gaps

These are the most important missing tests that should be addressed:

### Must Fix (Critical)

1. **`after_model_request` hook — UNTESTED**
   - No test verifies the hook is called, that it can modify responses, or that `CombinedCapability` runs hooks in reverse order
   - This is a core feature of the capabilities system

2. **`HistoryProcessorCapability` — ZERO direct tests**
   - The capability wrapper class has no test coverage at all
   - Only exercised indirectly through Agent's auto-wrapping of `history_processors=`

3. **`before_model_request` hook mutation — UNTESTED**
   - Only pass-through (no-op) behavior tested
   - No test verifies a capability can modify messages, settings, or request parameters via this hook

4. **`CombinedCapability` merge logic — PARTIALLY TESTED**
   - `get_toolset()` merging of multiple toolsets: untested
   - `get_builtin_tools()` flattening from multiple capabilities: untested
   - Dynamic settings resolver function: untested

### Should Fix (Important)

5. **`WebSearch` integration test** — No test runs an agent with `WebSearch` and verifies the builtin tool is provided
6. **`Agent.from_file()` convenience method** — Not directly tested
7. **File round-trip with capabilities** — `to_file`/`from_file` not tested with capabilities in spec
8. **`Instructions` with `TemplateStr`** — No test of template strings through the Instructions capability

---

## Appendix B: Other Changes in the PR

### Minor fixes included in the PR

- **Bedrock `sanitize_tool_name` revert** — PR review identified that the merge accidentally removed the `sanitize_tool_name` call from Bedrock model (added in PR #4713). This should be restored.
- **OpenRouter provider** — Minor cleanup (2 lines removed)
- **Google model** — Minor fix (1 line changed)
- **Cohere embeddings** — Minor cleanup (1 line removed per cassette)

### CI/workflow changes

- `.github/workflows/bots.yml` — 9 lines changed
- `.github/workflows/pr-guard.yml` — 2 lines changed

### Dependency changes

- `pydantic-handlebars` added as optional dependency (for TemplateStr)
- `uv.lock` updated (30 lines changed)

---

## Appendix C: Gaps Identified from Related Issues

After reading all linked issues and PRs, several significant concerns emerged that are **not adequately addressed** by the current PR but must be considered for the capabilities design.

### C1: Toolset State Isolation (#4347) — Design gap in current Toolset capability

[#4347](https://github.com/pydantic/pydantic-ai/issues/4347) documents that static toolsets leak mutable state across runs, breaking the "agents are stateless" promise. This directly impacts Issue [12] (Toolset capability).

**The problem:** When a user passes a `Toolset(my_stateful_toolset)` capability, the toolset instance is shared across all runs. `DynamicToolset` (via factory) provides isolation, but users following the docs naturally use static instances.

**DouweM's design decisions (from issue comments):**
- `get_toolset()` on capabilities intentionally does **NOT** take `RunContext` — it is evaluated once at capability registration, not per-run
- A `per_run`/`stateful` flag on `AbstractToolset` is the preferred direction, with a `copy()` method for creating per-run instances
- Sub-agent isolation via a `for_sub_agent()` method on toolsets is being considered (Layer 3)
- First-class factory support in `toolsets=` is already shipped (`lambda ctx: BrowserToolset()` auto-wraps in `DynamicToolset`)

**Impact on Issue [12]:** The `Toolset` capability wrapper should document/address the state isolation question. Consider whether `Toolset(stateful_toolset)` should auto-wrap in a factory, or if the `per_run` flag on `AbstractToolset` is the right solution.

**Impact on Issue [7]:** `CombinedCapability.get_toolset()` needs to be aware of stateful toolsets. Since `get_toolset()` is called once at init, toolsets that need per-run freshness must either use `DynamicToolset`/factory or rely on the future `per_run` flag.

### C2: Missing Hook Surface Area (#2885) — `before/after_model_request` is insufficient

[#2885](https://github.com/pydantic/pydantic-ai/issues/2885) describes middleware needs that go well beyond the current `before_model_request`/`after_model_request` hooks on `AbstractCapability`:

**What the current PR provides:**
- `before_model_request(ctx, request_context)` — modify messages/settings/params before model call
- `after_model_request(ctx, response)` — modify model response after model call

**What #2885 users need but the PR does NOT provide:**

| Hook | Use Case | Status |
|------|----------|--------|
| `before_tool_call(ctx, tool_name, tool_args)` | Validate/modify tool arguments, block calls, inject PII replacements | NOT IN PR |
| `after_tool_call(ctx, tool_name, tool_args, tool_result)` | Validate/sanitize tool results, revert PII replacements, log calls | NOT IN PR |
| `on_agent_start(ctx)` | Initialize per-run middleware state, start timers | NOT IN PR |
| `on_agent_end(ctx, result)` | Cleanup, final validation, post-process output | NOT IN PR |
| Streaming hooks (`on_new_chunk`) | Throttle chunks, modify streaming output | NOT IN PR |

**Key use cases from the issue that require these missing hooks:**
- **PII Replacement Middleware:** Replace phone numbers/emails in `before_model_request`, revert them in `after_model_request` — this part works. But also need to sanitize tool call arguments (`before_tool_call`) and tool results (`after_tool_call`).
- **URL Shortener Middleware:** Replace long URLs before model sees them, restore in response — requires stateful pre/post with shared context.
- **JSON Fixer Middleware:** Fix malformed JSON in tool call arguments — requires `before_tool_call`.
- **FileSystem Middleware:** Adds tools + instructions + compaction logic + intercepts large tool outputs — combines multiple capability facets.

**Recommendation:** The `AbstractCapability` base class should be designed with future hook points in mind even if they're not implemented now. At minimum, document that `before_tool_call`/`after_tool_call` and `on_agent_start`/`on_agent_end` are planned additions.

### C3: Guardrails (#1197) — More than hooks, needs parallel execution

[#1197](https://github.com/pydantic/pydantic-ai/issues/1197) has 20+ comments and active community demand. Key insights:

**What guardrails need that hooks alone don't provide:**
- **Input guardrails running in PARALLEL** with the first model request (latency optimization) — OpenAI's SDK does `asyncio.gather(guardrail_check, model_request)` and cancels the model request if the guardrail trips
- **Tripwire exceptions** (`InputGuardrailTripwireTriggered`, `OutputGuardrailTripwireTriggered`) for control flow
- **Output guardrails** that run after final result is produced, before returning to caller
- **Per-tool-call guardrails** — blocking specific tool calls based on custom criteria (flagged by @blairhudson)

**PR #3938** (by @sarth6) implements basic input/output guardrails but DouweM confirmed it **won't land as-is** — guardrails will be tackled as capabilities via #4303.

**Impact on the capabilities design:**
- `before_model_request` can serve as input guardrail hook, but parallel execution with the model request requires different wiring than sequential hooks
- Output guardrails need a hook point after `result.data` is computed but before `agent.run()` returns — this is different from `after_model_request` which runs per-step, not per-run
- A `Guardrails` capability would need to provide both the guardrail functions AND configure their execution mode (parallel vs sequential)

### C4: Tool Output Validation (#4262) — Agent-side trust boundary

[#4262](https://github.com/pydantic/pydantic-ai/issues/4262) proposes strict runtime validation for untrusted tool outputs. DouweM confirmed: "This will be easy to build once #2885 lands as part of #4303."

**Two layers identified:**
1. **Tool-schema validation** — validate tool return against the tool's own schema (baseline)
2. **Agent-side strict validation** — agent author defines a stricter local output model with `max_length`, constrained fields, sanitization hooks before data enters LLM context

**Impact:** This needs `after_tool_call` hooks (see C2 above) to intercept and validate/sanitize tool outputs. The current `before_model_request` hook can only see the assembled message history, not individual tool returns.

### C5: Tool Budget Awareness (#4359) — Hooks-based capability

[#4359](https://github.com/pydantic/pydantic-ai/issues/4359) proposes automatic tool budget reminders. Both DouweM and @adtyavrdhn confirmed: "This sounds like a good candidate for a hooks-based Capability once #2885 lands as part of #4303."

**The pattern:** After tool calls complete, inject a `UserPromptPart` with budget information ("5/10 tool calls used, 5 remaining") so the model can self-regulate.

**Impact:** The `before_model_request` hook IS sufficient for this — `ctx` (the `RunContext` first argument) includes `ctx.usage.tool_calls` and `ctx.usage.input_tokens`. The capability can read these to decide whether to inject a budget reminder into `request_context.messages`. No new hooks are needed for this specific capability.

Note: `BeforeModelRequestContext` itself doesn't contain usage info, but the `RunContext` parameter does. This is an important distinction — the hook signature `before_model_request(ctx, request_context)` gives access to both runtime context (via `ctx`) and the modifiable request (via `request_context`).

### C6: MCP Server Instructions Injection (#4725)

[#4725](https://github.com/pydantic/pydantic-ai/issues/4725) requests that MCP servers' instructions (sent in the initialize response per the MCP spec) be auto-injected into the agent's system prompt.

**Impact:** An MCP-based capability could use `get_instructions()` to provide these instructions. This is a natural fit for the capabilities system — an MCP capability that provides both a toolset (via `get_toolset()`) and instructions (via `get_instructions()`).

---

## Appendix D: Future Capabilities Referenced in #4303

The parent issue [#4303](https://github.com/pydantic/pydantic-ai/issues/4303) envisions many more capabilities beyond what this PR implements. These are **not** in the current PR but are part of the broader vision.

For each, the table shows: whether it's a **Capability** (subclass of `AbstractCapability`), **Infrastructure** (enables capabilities), or a **Hook Extension** (adds new methods to `AbstractCapability`), and its serializability tier.

| Item | Category | Serialization | Interface Methods | Relevant Issue | Notes |
|---|---|---|---|---|---|
| **Context window on ModelProfile** | Infrastructure | N/A | N/A | [#4538](https://github.com/pydantic/pydantic-ai/issues/4538) | Adds `context_window` to `ModelProfile` from genai-prices + exposes `RunUsage` on `RunContext`. Prerequisite for Compaction — threshold detection needs `ctx.model.profile.context_window`. |
| **Python signatures for tools (Code Mode infra)** | Infrastructure | N/A | N/A | [PR #4755](https://github.com/pydantic/pydantic-ai/pull/4755) | Extracts function signatures from `@tool` definitions and JSON schemas so Code Mode can generate correct Python calls. Prerequisite for CodeMode capability. |
| **Compaction** | Capability | Tier S — `- Compaction: {strategy: smart, threshold: 80000}` | `before_model_request()` (history summarization) | [#4137](https://github.com/pydantic/pydantic-ai/issues/4137) | Detailed API proposal exists. Blocked by context window exposure (#4538) for threshold detection. |
| **Memory** | Capability | Tier P — `- Memory: {backend: sqlite}` | `get_toolset()` + `get_instructions()` + `before_model_request()` | — | Backend name serializable; embedding functions not |
| **CodeMode** | Capability | Tier P — `- CodeMode: {sandbox: docker}` | `get_toolset()` + `get_instructions()` | — | Sandbox type serializable. Blocked by Python signatures ([PR #4755](https://github.com/pydantic/pydantic-ai/pull/4755)) for generating correct tool calls. |
| **Approval** | Capability | Tier P — `- Approval: {mode: writes}` | Needs `before_tool_call` hook (Hook Extension required) | — | Approval callback not serializable |
| **Sessions** | Capability | Tier P — `- Session: {store: sqlite}` | Needs `on_agent_start`/`on_agent_end` hooks | — | Store type serializable |
| **DurableExecution** | Capability | Tier P — `- DurableExecution: temporal` | Complex (wraps model + toolsets) | — | PR refactors to WrapperAgent pattern first |
| **FileSystem** | Capability | Tier S — `- FileSystem: {root: ., ignore: [.git]}` | `get_toolset()` + `get_instructions()` | — | All config is paths/patterns |
| **Shell** | Capability | Tier S — `- Shell: {timeout: 60, confirm_destructive: true}` | `get_toolset()` + `get_instructions()` | — | All config is simple types |
| **WebFetch** | Capability | Tier S — `- WebFetch` | `get_toolset()` | — | — |
| **Todos** | Capability | Tier P — `- Todos: {backend: sqlite}` | `get_toolset()` | — | Backend serializable |
| **SubAgents** | Capability | Tier P — `- SubAgents: {agents_dir: ./agents}` | `get_toolset()` | — | Agent specs serializable (recursive) |
| **MCP** | Capability | Tier S — `- MCP: {config: .mcp.json}` | `get_toolset()` + `get_instructions()` | [#4725](https://github.com/pydantic/pydantic-ai/issues/4725) | MCP config is already JSON |
| **Skills** | Capability | Tier S — `- Skills: {dirs: [./skills]}` | `get_toolset()` + `get_instructions()` | — | Paths are strings; skill content is filesystem |
| **KnowsCurrentTime** | Capability | Tier S — `- KnowsCurrentTime: {tz: UTC}` | `get_instructions()` only | — | Purely config-driven |
| **Guardrails** | Capability | Tier P — `- InputGuardrail: {parallel: true}` | Needs `on_agent_start` + new guardrail hooks | [#1197](https://github.com/pydantic/pydantic-ai/issues/1197) | Validator callable not serializable |
| **Hook Extension** | Infrastructure | N/A | Extends `AbstractCapability` | [#2885](https://github.com/pydantic/pydantic-ai/issues/2885) | Adds `before/after_tool_call`, `on_agent_start/end` |
| **Builtin tool fallback** | Capability | Tier S — `- WebSearch: {fallback: tavily}` | `get_builtin_tools()` + `get_toolset()` | [#3212](https://github.com/pydantic/pydantic-ai/issues/3212) | — |
| **ToolBudget** | Capability | Tier P — `- ToolBudget: {limit: 10}` | `before_model_request()` (reads `ctx.usage.tool_calls`) | [#4359](https://github.com/pydantic/pydantic-ai/issues/4359) | Threshold serializable; custom formatter not |
| **ToolOutputValidation** | Capability | Tier N | Needs `after_tool_call` hook | [#4262](https://github.com/pydantic/pydantic-ai/issues/4262) | Validator is a Pydantic model (possibly serializable) |
| **ToolPolicy** | Capability | Tier P — `- ToolPolicy: {max_uses: 5}` | `before_tool_call` hook | [#3352](https://github.com/pydantic/pydantic-ai/issues/3352) | PR #3691 open |
| **Governance** | Capability | Tier P | Needs `before_tool_call` hook | [#4335](https://github.com/pydantic/pydantic-ai/issues/4335) | Policy model possibly serializable |
| **Toolset Lifecycle** | Infrastructure | N/A | Changes to `AbstractToolset` | [#4347](https://github.com/pydantic/pydantic-ai/issues/4347) | `per_run` flag, `copy()` method |

---

## Appendix E: Cross-Framework Research — Missing Capabilities

Comprehensive research across 10+ frameworks reveals capabilities that Pydantic AI does not currently have and that the current PR (#4640) does not address. These are organized by category, with the source framework(s) noted.

> **Methodology:** Research covered Mastra, CrewAI, AutoGen, Claude Agent SDK, LangGraph, OpenAI Agents SDK, Google ADK, Code Puppy, Pydantic Deep Agents, pai-agent-sdk, llm-do, and the [traits research report](https://github.com/pydantic/pydantic-ai/pull/4233). Only capabilities that are **not already in Pydantic AI or the current PR** are listed.

### E1: Missing Hook Surface Area

The current PR provides `before_model_request` and `after_model_request`. Every major framework provides significantly more hook points. The traits research report ([PR #4233](https://github.com/pydantic/pydantic-ai/pull/4233)) proposed 6 lifecycle hooks. Here's the complete gap:

| Hook | What It Does | Who Has It |
|------|-------------|-----------|
| **`before_tool_call`** | Validate/modify tool arguments, block calls, inject PII replacements, enforce approval | Google ADK, OpenAI SDK, Mastra, Claude SDK, traits report, pydantic-deepagents, Code Puppy |
| **`after_tool_call`** | Validate/sanitize tool results, revert PII, log calls, replace results | Google ADK, OpenAI SDK, Mastra, Claude SDK, traits report, pydantic-deepagents |
| **`on_agent_start`** | Initialize per-run state, start timers, validate input | Google ADK, OpenAI SDK, Mastra (`processInput`), traits report |
| **`on_agent_end`** | Cleanup, final output validation, post-processing | Google ADK, OpenAI SDK, Mastra (`processOutputResult`), traits report |
| **`on_llm_start` / `on_llm_end`** | Observe/modify individual LLM calls within multi-step runs | OpenAI SDK (`RunHooks`), Google ADK |
| **`on_handoff`** | Intercept agent delegation events | OpenAI SDK, Google ADK |
| **Per-step hooks** (`processInputStep`/`processOutputStep`) | Intercept each iteration of the agent loop, not just model calls | Mastra (unique — 5 lifecycle methods per processor) |
| **Streaming hooks** (`on_new_chunk`, `processOutputStream`) | Intercept/modify streaming tokens | Mastra, Vercel AI SDK, #2885 request |

**Traits report proposed hooks (not in current PR):**
- `before_tool_call(ctx, tool_name, tool_args)` — return None (proceed), dict (modified args), or False (block)
- `after_tool_call(ctx, tool_name, tool_args, result)` — return None (original) or replacement
- `on_agent_start(ctx)` / `on_agent_end(ctx)`
- `check_input(ctx, user_input)` / `check_output(ctx, output)` — guardrail hooks

### E2: Missing Capabilities — Safety & Guardrails

| Capability | Description | Source Frameworks |
|---|---|---|
| **Input Guardrails** | Validate user input before agent processes it. Can run in **parallel** with model request for zero added latency. Tripwire exceptions halt execution. | OpenAI SDK, Google ADK, Mastra, traits report |
| **Output Guardrails** | Validate agent's final output before returning to caller. Can transform, block, or warn. | OpenAI SDK, Google ADK, Mastra, traits report |
| **Tool Guardrails** | Validate before/after individual tool execution. Can skip, replace, or reject tool calls. | OpenAI SDK (`@tool_input_guardrail`, `@tool_output_guardrail`), Google ADK |
| **PII Detection/Redaction** | Detect and remove/replace personally identifiable information in messages and tool results. | Mastra (built-in `PIIDetector`), Google ADK (PII Redaction plugin) |
| **Prompt Injection Detection** | LLM-based classifier scanning for prompt injection, jailbreak attempts. | Mastra (`PromptInjectionDetector`), Google ADK (Gemini as Judge) |
| **Content Moderation** | Detect harmful content (hate, harassment, violence) in input and output. | Mastra (`ModerationProcessor`) |
| **System Prompt Leak Detection** | Detect and redact system prompts leaked in model responses. | Mastra (`SystemPromptScrubber`) |
| **Unicode Normalization** | Clean/normalize Unicode, standardize whitespace, remove problematic symbols. | Mastra (`UnicodeNormalizer`) |
| **GuardrailResult with Actions** | `halt` (stop), `transform` (replace), `warn` (log and continue) — richer than just block/allow. | Traits report, OpenAI SDK, Mastra |

### E3: Missing Capabilities — Memory & Context

| Capability | Description | Source Frameworks |
|---|---|---|
| **Working Memory** | Persistent structured user data (names, preferences, goals) across conversations. Template-based (markdown, replace) or schema-based (JSON, merge). | Mastra (`WorkingMemory`), LangGraph (semantic memory), traits report |
| **Semantic Recall** | Vector similarity search over past messages. Embeds conversation history and retrieves by meaning. | Mastra (`SemanticRecall`), LangGraph (long-term memory), CrewAI |
| **Observational Memory** | Background Observer/Reflector agents that compress conversation history into dated observation logs. 5-40x compression. Three-tier: recent messages → observations → reflections. | Mastra (unique — most sophisticated memory system found) |
| **Episodic Memory** | Past experiences stored as few-shot examples for future prompting. Agent learns from its own history. | LangGraph |
| **Procedural Memory** | Self-modifying prompts via reflection. Agent updates its own instructions based on experience. | LangGraph |
| **Cross-Thread Memory** | Persistent memory shared across different conversation sessions, scoped by user/org/namespace. | LangGraph (`store.put()`/`store.get()`/`store.search()`), Mastra, CrewAI, Google ADK (`user:` prefix) |
| **Session Persistence** | Save/restore full conversation state across process restarts. Multiple backends. | OpenAI SDK (6 session backends), LangGraph (checkpointing), Google ADK, Code Puppy, traits report |
| **Session Branching** | Fork conversation from a specific turn for A/B exploration. | OpenAI SDK (`AdvancedSQLiteSession`), Claude SDK (session forking), LangGraph (time travel) |
| **Encrypted Sessions** | Encrypt session data with TTL for compliance. | OpenAI SDK (`EncryptedSession`) |
| **Context Caching** | Reuse instructions/data across requests to reduce token cost. Configurable TTL. | Google ADK (`ContextCacheConfig`) |

### E4: Missing Capabilities — Agent Composition & Delegation

| Capability | Description | Source Frameworks |
|---|---|---|
| **Agent Handoff with Input Filters** | When transferring control between agents, modify/filter conversation history. Remove tool calls, trim context, inject metadata. | OpenAI SDK (`Handoff` with `input_filter`, `input_type`), traits report (`HandoffTrait`) |
| **Agent-as-Tool** | Run an agent as a tool — parent retains control, sub-agent response returned as tool result. Unlike full handoff. | OpenAI SDK (`agent.as_tool()`), Google ADK (`AgentTool`), llm-do |
| **Dynamic Agent Creation** | Agents can create new specialist agents at runtime based on task requirements. | pydantic-deepagents (`DynamicAgentRegistry`), llm-do, Code Puppy |
| **Supervisor Pattern** | Central agent coordinates delegation to specialized sub-agents. Manager plans, delegates, reviews quality. | CrewAI (hierarchical process), Mastra, LangGraph (supervisor), AutoGen (SelectorGroupChat) |
| **Swarm Pattern** | Decentralized agents with peer-to-peer handoff tools. No central orchestrator. ~40% faster than supervisor. | AutoGen (Swarm), LangGraph (swarm), OpenAI SDK (handoffs) |
| **Workflow Agents** | Deterministic orchestration: Sequential, Parallel, Loop agents without LLM overhead. | Google ADK (`SequentialAgent`, `ParallelAgent`, `LoopAgent`) |
| **Nested Handoff History** | Collapse prior agent transcripts into summarized messages during handoffs to preserve context window. | OpenAI SDK (`RunConfig.nest_handoff_history`) |

### E5: Missing Capabilities — Execution Control

| Capability | Description | Source Frameworks |
|---|---|---|
| **Composable Stop Conditions** | Pluggable predicates for when to stop: max steps, text detection, token usage, timeout, custom function. Composable via `&` (AND) and `|` (OR). | AutoGen (6+ conditions), Vercel AI SDK (`stopWhen`), traits report (`StopConditionTrait`) |
| **Tool Use Behavior** | Control what happens after tool calls: re-run LLM, stop on first tool, stop on specific tools, custom function. | OpenAI SDK (`tool_use_behavior`) |
| **Call Model Input Filter** | Modify/trim/inject into model input just before LLM call. Separate from `before_model_request` which sees the full request. | OpenAI SDK (`call_model_input_filter`) |
| **Error Handlers** | Map specific error kinds to controlled final output instead of exceptions. | OpenAI SDK (`error_handlers`) |
| **Fault-Tolerant Execution** | Resume from failures, re-running only failed nodes. Checkpointing at every step. | LangGraph |
| **Planning** | Explicit reasoning/planning step before task execution. Built-in planner or plan-act-reason cycles. | Google ADK (`BuiltInPlanner`, `PlanReActPlanner`), CrewAI (agent reasoning) |

### E6: Missing Capabilities — Tools & Execution

| Capability | Description | Source Frameworks |
|---|---|---|
| **Code Execution (Sandboxed)** | Run LLM-generated code in Docker/sandbox/E2B. Multiple safety modes. | Code Puppy, CrewAI, Google ADK, OpenAI SDK (`CodeInterpreterTool`), traits report (`PythonExecTrait`) |
| **Browser Automation** | Navigate, click, extract, screenshot via headless browser. | pai-agent-sdk, Code Puppy, traits report (`BrowserTrait`), Claude SDK (computer use) |
| **Tool Progress Streaming** | Tools emit incremental progress during execution via writer/stream objects. Supports transient (non-persisted) data. | Mastra (`context.writer`) |
| **Tool Caching** | Configurable per-tool caching of results to avoid redundant expensive calls. | CrewAI (`cache_function` on tools) |
| **Tool Approval (Suspend/Resume)** | Tool execution pauses for human approval, with support for clarification questions, automatic resumption from conversation, and sticky decisions. | Mastra (`requireApproval`, `suspend()`/`resume()`), OpenAI SDK (`needs_approval`), Google ADK (`LongRunningFunctionTool`) |
| **Deferred Tool Loading / Tool Search** | Load tool definitions lazily to preserve context window. Agent discovers tools on demand. ~85% token savings. | OpenAI SDK (`defer_loading=True` + `ToolSearchTool`), Claude SDK (Tool Search) |
| **Tool Authentication** | OAuth/API key flows managed by the framework. Tools request credentials, framework handles auth UI/flow. | Google ADK (`request_credential()`, `get_auth_response()`) |

### E7: Missing Capabilities — Knowledge & Instructions

| Capability | Description | Source Frameworks |
|---|---|---|
| **Skills System** | Reusable knowledge packages (markdown files with YAML frontmatter) that agents discover, load, and activate on demand. Progressive disclosure. | Claude SDK, Code Puppy, Google ADK (`SkillToolset`), Mastra, pydantic-deepagents, pai-agent-sdk, llm-do, traits report |
| **KnowsCurrentTime** | Dynamic instruction injecting current date/time with configurable timezone/format. | Traits report |
| **KnowsUserDetails** | Dynamic instruction injecting user name, preferences, role from a provider callable. | Traits report |
| **KnowsProjectContext** | Dynamic instruction injecting project description, tech stack from a context file (e.g., CLAUDE.md). | Traits report, Claude SDK (CLAUDE.md) |
| **Instruction Deduplication** | `instruction_group` property preventing duplicate instructions when multiple capabilities provide the same type. | Traits report, pai-agent-sdk (`group` field on `get_instruction()`) |
| **RAG / Knowledge Sources** | Attach document knowledge (PDF, CSV, JSON, web) to agents with vector store retrieval. | CrewAI, Mastra, Google ADK (Vertex AI Search) |
| **MCP Server Instructions** | Auto-inject MCP server's instructions into agent system prompt. | [#4725](https://github.com/pydantic/pydantic-ai/issues/4725) |

### E8: Missing Capabilities — Observability & Evaluation

| Capability | Description | Source Frameworks |
|---|---|---|
| **Sensitive Data Filtering in Traces** | Redact passwords, tokens, keys from observability traces. | Mastra (`SensitiveDataFilter`), OpenAI SDK (`trace_include_sensitive_data`) |
| **Live Evaluation / Scoring** | Run evaluation criteria (accuracy, safety, hallucination detection) during agent operation, not just post-hoc. | Mastra (live scorers), Google ADK (Gemini as Judge plugin) |
| **Cost/Token Budget Enforcement** | Hard budget limit with exception when exceeded. Per-run cost tracking with callbacks. | pydantic-deepagents (`cost_budget_usd`, `BudgetExceededError`) |

### E9: Missing Capabilities — Interaction

| Capability | Description | Source Frameworks |
|---|---|---|
| **User Interaction Tool** | Agent can ask the user clarifying questions mid-execution with structured options. | Code Puppy, Claude SDK (`AskUserQuestion`), traits report (`UserInteractionTrait`) |
| **Reasoning Transparency Tool** | Agent shares its reasoning process with the user as a tool call. | Code Puppy (`share_your_reasoning`), traits report (`ReasoningTrait`) |
| **Voice / Speech** | TTS, STT, speech-to-speech for voice-based agent interactions. | Mastra (12+ providers), OpenAI SDK (`VoicePipeline`), Google ADK (Gemini Live) |
| **Artifact Management** | Named, versioned binary data (files, images) associated with sessions or users. | Google ADK (artifacts with versioning, GCS backend), traits report (`ArtifactTrait`) |

### E10: Missing Infrastructure — Capability Composition

| Feature | Description | Source |
|---|---|---|
| **Dependency Resolution** (`requires`) | Capabilities declare prerequisite capability IDs. Topological sorting ensures correct initialization order. | Traits report |
| **Conflict Detection** (`conflicts_with`) | Capabilities declare mutually exclusive capabilities. Error at registration if conflicts detected. | Traits report |
| **Capability Presets / Bundles** | Pre-configured combinations: `coding_agent_capabilities()`, `research_agent_capabilities()`. | Traits report |
| **Capability Registry** | Named registration system for custom capabilities, enabling YAML specs to reference user-defined capabilities by name. | Traits report (Phase 4) |
| **Async Context Manager on Capabilities** | `__aenter__`/`__aexit__` for resource setup/cleanup (MCP servers, browser sessions, DB connections). | Traits report |
| **Global Plugins** | Reusable callback/hook modules registered once on the runner, applying to all agents/tools/LLM calls. Separate from per-agent capabilities. | Google ADK (`BasePlugin` with 12 hook methods) |
| **Processor Strategies** | Configurable action when a processor/guardrail triggers: `block`, `warn`, `detect`, `redact`, `rewrite`, `translate`. | Mastra |

### E11: Prioritized Recommendations

Based on frequency across frameworks, community demand, and fit with the `AbstractCapability` interface:

**Tier 1 — High demand, maps cleanly to current capability interface:**

| # | Item | Category | Serialization | Interface Methods | Blocked By |
|---|------|----------|---------------|-------------------|-----------|
| 1 | **Compaction** | Capability | Tier S | `before_model_request()` | Nothing — implementable today |
| 2 | **Skills system** | Capability | Tier S | `get_instructions()` + `get_toolset()` | Nothing — implementable today |
| 3 | **KnowsCurrentTime** | Capability | Tier S | `get_instructions()` | Nothing — implementable today |
| 4 | **ToolBudget awareness** | Capability | Tier P | `before_model_request()` (reads `ctx.usage`) | Nothing — implementable today |
| 5 | **FileSystem** | Capability | Tier S | `get_toolset()` + `get_instructions()` | Nothing — implementable today |
| 6 | **Shell** | Capability | Tier S | `get_toolset()` + `get_instructions()` | Nothing — implementable today |
| 7 | **MCP (tools + instructions)** | Capability | Tier S | `get_toolset()` + `get_instructions()` | Nothing — implementable today |

**Tier 2 — High demand, requires hook extension first:**

| # | Item | Category | Serialization | Blocked By |
|---|------|----------|---------------|-----------|
| 8 | **`before/after_tool_call` hooks** | Hook Extension | N/A | Changes to `AbstractCapability` + graph layer |
| 9 | **`on_agent_start`/`on_agent_end` hooks** | Hook Extension | N/A | Changes to `AbstractCapability` + graph layer |
| 10 | **Input/Output Guardrails** | Capability | Tier P | Needs #8 + #9 + parallel execution wiring |
| 11 | **Approval** | Capability | Tier P | Needs #8 (`before_tool_call`) |
| 12 | **Tool output validation** | Capability | Tier N | Needs #8 (`after_tool_call`) |
| 13 | **Session persistence** | Capability | Tier P | Needs #9 (`on_agent_start`/`on_agent_end`) |

**Tier 3 — Strong consensus, more complex integration:**

| # | Item | Category | Serialization | Notes |
|---|------|----------|---------------|-------|
| 14 | **Memory (working + semantic)** | Capability | Tier P | Multiple sub-types; complex but maps to interface |
| 15 | **Agent-as-Tool / SubAgent** | Capability | Tier P | Needs toolset isolation (#4347) consideration |
| 16 | **Code execution (sandboxed)** | Capability | Tier P | Docker/E2B; maps to `get_toolset()` |
| 17 | **Composable stop conditions** | NOT a capability | N/A | Needs different mechanism (Agent-level, not capability) |
| 18 | **Dependency resolution** | Infrastructure | N/A | `requires`/`conflicts_with` on `AbstractCapability` |
| 19 | **Planning** | Capability | Tier P | `get_instructions()` + `before_model_request()` |

**Key insight:** Items 1-7 are implementable TODAY with the current `AbstractCapability` interface — no hook extensions needed. This is the low-hanging fruit. Items 8-13 are blocked by the hook extension work. Item 17 (composable stop conditions) does NOT map to the capability interface — it would need to be an Agent-level or run-level feature instead.
