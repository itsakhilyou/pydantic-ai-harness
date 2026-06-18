# Testing Capabilities

Harness tests should exercise the behavior users rely on, not only private
helpers.

## Default Shape

- Use `pydantic_ai.models.TestModel` for model behavior.
- Keep real provider calls out of tests.
- Prefer `Agent(..., capabilities=[...])` tests for public behavior.
- Mirror source packages under `tests/<capability>/`.
- Use `pytest-anyio` for async capability/toolset behavior.

## Lower-Level Tests

Direct toolset tests are appropriate when you need to inspect:

- listed tools and schemas
- wrapper-toolset lifecycle
- retry behavior
- metadata and synthetic tool-call records
- `RunContext` or `ToolManager` interactions
- edge cases that are awkward to force through a full agent run

Use the `CodeMode` tests as the current reference for direct `RunContext` and
`ToolManager` setup.

## Coverage

The project enforces 100% branch coverage with `make testcov`. Tests for a new
capability should cover:

- default configuration
- important option combinations
- failure and retry paths
- composition with relevant Pydantic AI features
- docs examples when examples are executable

Use snapshots when behavior is protocol-shaped: messages, event streams,
schemas, telemetry spans, or structured tool metadata.

For a branch that genuinely cannot run in the test environment (an `except` for
an error the harness can't provoke, a defensive guard), mark it with a
`# pragma: no cover` comment rather than contorting a test to reach it. Reserve
this for truly unreachable paths -- a missing test is not an excuse to pragma.

## Commands

Run focused checks first, then broaden:

```bash
uv run pytest tests/<capability>
make lint
make typecheck
make test
make testcov
```
