# Pydantic AI Harness

## Repository purpose

`pydantic-ai-harness` is the first-party capability library for Pydantic AI.

Pydantic AI core owns the primitive runtime: agent loop semantics, normalized
messages, model/provider/profile behavior, tool execution semantics, durable
execution primitives, and generic capability hooks.

Harness owns optional, batteries-included compositions built from those
primitives: coding-agent tools, guardrails, memory, context management, repo
tools, verification loops, skills, planning, sub-agents, and other reusable
agent behaviors.

When a change needs new core semantics, stop and propose the Pydantic AI core
change instead of reimplementing core behavior in harness.

## Vocabulary

- **Capability**: an `AbstractCapability` subclass that bundles tools, hooks, instructions, and model settings into a reusable unit. This is the core abstraction of pydantic-ai-harness.
- **Hook**: a lifecycle method on `AbstractCapability` that intercepts agent graph execution (e.g. `before_model_request`, `wrap_run`, `after_tool_execute`)
- **Toolset**: a collection of tools that a capability can provide to the agent
- **Guard**: a type of capability that validates inputs/outputs or controls tool access (e.g. `InputGuardrail`, `CostGuard`)
- **Harness**: this package -- a collection of pre-made capabilities for Pydantic AI.
- **AICA**: AI Code Assistant -- the automated agent that implements issues, reviews plans, and handles PR feedback
- **Ralph loop**: the state-machine-based workflow that drives AICA through phases (TRIAGE -> GOALS -> PLAN -> CODE -> VERIFY -> REVIEW -> PUBLISH)
- **DDD+ protocol**: classification system for PR review comments (do, dismiss, discuss, waiting, done)

## AICA preflight

Before implementing or reviewing a capability change:

1. Read `agent_docs/index.md`.
2. Read the linked `agent_docs/` guide for the task.
3. Read the public Pydantic AI docs for every integration point you touch:
   - capabilities: <https://pydantic.dev/docs/ai/core-concepts/capabilities/>
   - hooks: <https://pydantic.dev/docs/ai/core-concepts/hooks/>
   - toolsets: <https://pydantic.dev/docs/ai/tools-toolsets/toolsets/>
   - advanced tools: <https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/>
   - agents: <https://pydantic.dev/docs/ai/core-concepts/agent/>
   - testing: <https://pydantic.dev/docs/ai/guides/testing/>
4. Inspect the installed `pydantic_ai` package source for exact hook/toolset
   signatures when needed. Do not assume a contributor's local checkout layout.
5. Use `pydantic_ai_harness.code_mode` as the exemplar for capability shape,
   docs, tests, and public exports until another capability becomes a better
   example. `code_mode` is a released top-level capability; new capabilities
   start under `pydantic_ai_harness.experimental` (see
   `agent_docs/capability-authoring.md`, "Experimental Vs Released Exports").

## Capabilities API reference

When implementing a new capability, read the public Pydantic AI docs listed in
the [AICA preflight](#aica-preflight) above (capabilities, hooks, toolsets,
advanced tools, agents, testing), plus:

- <https://pydantic.dev/docs/ai/guides/extensibility/> -- publishing capabilities as packages, spec serialization
- Installed `pydantic_ai.capabilities` source -- `AbstractCapability`, hook signatures, and composition behavior
- Installed `pydantic_ai.toolsets` source -- `AbstractToolset`, `WrapperToolset`, and `ToolsetTool`

## Coding standards

- Python 3.10+ (target version for pyright and ruff)
- **pyright strict** mode -- no `Any` types, full type annotations
- **ruff**: line-length=120, single quotes, max-complexity=15
- **100% branch coverage** required -- `make testcov` runs the suite and fails if
  `coverage report` drops below the `fail_under` threshold (`make test` alone does
  not check coverage)
- docstrings use single backticks (markdown), not RST double backticks
- no typecasting (`as` in TypeScript, `cast()` in Python) -- use type narrowing instead
- prefer the most generic input types possible (reduce dependency chains)
- don't add comments that restate what the code does

## Writing style

Applies to docs, READMEs, docstrings, comments, commit messages, and PR text.

- No em-dashes (`—`). Use `--` for an aside or interruption, or split into two
  sentences. Em-dash-heavy prose reads as machine-generated.
- State facts, not sales copy. Cut marketing superlatives and hype ("blazingly
  fast", "battle-tested", "the single most expensive thing you can do",
  "footgun") and editorializing adjectives ("sprawling", "noisy", "silently").
- Avoid absolute claims ("never", "always", "guaranteed") unless they are
  literally true and load-bearing. Name the specific mechanism instead of the
  slogan.
- Use bold sparingly -- for the lead-in term of a list item, not to emphasize
  whole sentences.
- Document the why, the constraints, and the non-obvious. Don't restate what the
  code or signature already says.
- Prefer plain ASCII punctuation over decorative Unicode (arrows, fancy quotes)
  in prose and comments.

## Package management

- Use `uv` for all dependency operations
- Never edit `pyproject.toml` or `uv.lock` directly -- use `uv add`, `uv remove`
- External PRs that change dependencies are auto-closed by CI

## Commands

```bash
make format     # ruff format
make lint       # ruff check
make typecheck  # pyright strict
make test       # pytest
make testcov    # pytest with branch coverage
```

Always run `make lint && make typecheck && make test` before committing.

## File structure

```
pydantic_ai_harness/
  __init__.py          # public API re-exports (stable capabilities only)
  <capability>/        # a promoted, stable capability package
    __init__.py        # public exports for the capability
    _capability.py     # capability class (AbstractCapability subclass)
    _toolset.py        # toolset implementation
    README.md          # standalone docs for the capability
  experimental/        # new capabilities land here first
    __init__.py        # exports HarnessExperimentalWarning (does not warn on its own import)
    <capability>/      # same package shape; __init__ calls warn_experimental('<name>')
tests/
  conftest.py          # shared fixtures (TestModel, test_agent)
  <capability>/        # tests mirror source packages
    test_<capability>.py
```

New capabilities start under `experimental/` (see
`agent_docs/capability-authoring.md`, "Experimental Vs Released Exports"); they
graduate to a top-level package and a top-level re-export only once the API is
stable. Do not add placeholder template files. Start from the existing
`code_mode` package shape, then delete what the new capability does not need.

## Testing patterns

- Use `pydantic_ai.models.TestModel` for all tests (no real API calls)
- `ALLOW_MODEL_REQUESTS = False` is set globally in `conftest.py`
- Tests use `pytest-anyio` for async support
- Each capability test class follows: `TestCapabilityName` with methods `test_<scenario>`
- Prefer tests through `Agent(..., capabilities=[...])` when that is the public
  behavior. Use direct `Toolset`/`RunContext` tests for lower-level lifecycle,
  schema, retry, or wrapper behavior that is hard to isolate through `Agent`.

## Contributing rules for AICAs

- Never change `pyproject.toml` or `uv.lock` -- if a dependency is needed, open an issue
- Always link sources for any claims made during research
- Run `make lint && make typecheck && make test` before every commit
- Commit messages should summarize the "why", not the "what"
- All GitHub comments must start with "Claude here: "
