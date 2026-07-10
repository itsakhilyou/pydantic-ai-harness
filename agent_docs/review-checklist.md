# Review Checklist

Use this before opening a PR or reviewing a capability change.

## Product Fit

- The capability has a clear user or dogfooding need.
- The behavior belongs in harness, not Pydantic AI core.
- The public API is small and named around user concepts.
- The capability composes with relevant existing capabilities.

## Implementation

- Public exports are intentional.
- Private helpers stay private.
- Types are precise; new public signatures do not use `Any`.
- No casts are used to paper over type design.
- The implementation uses Pydantic AI hooks/toolsets instead of duplicating core
  runtime behavior.
- Capability ordering is justified when present.
- Dependency changes were made through `uv` and have a clear reason.

## Stale Or Pre-Merge PRs

Run these checks when adopting, rebasing, or re-reviewing a PR that was opened
well before now, or that was built against unreleased Pydantic AI changes.

- Temporary `[tool.uv.sources]` pins to a branch or git ref are removed once the
  upstream change they waited on has landed in a released `pydantic-ai-slim`.
- Each upstream Pydantic AI PR or branch the change rode on has merged. Link the
  upstream PR and its merge state.
- The touched surface has not drifted: re-check the capability, hook, and toolset
  signatures it depends on against current main, not against the state at fork
  time.
- Behavior the PR worked around because a primitive was missing is reconsidered
  if that primitive now exists in core.

## Tests

- Tests cover the public `Agent(..., capabilities=[...])` path where possible.
- Lower-level tests cover lifecycle, schemas, retries, and metadata when needed.
- Error paths and important option combinations are covered.
- Relevant protocol-shaped output is snapshotted.
- `make lint`, `make typecheck`, and `make test` pass before handoff.

## Docs

Every released capability ships two hand-maintained docs that must stay in sync
with the code and with each other:

- the **README** next to the implementation (`pydantic_ai_harness/<capability>/README.md`),
  which serves GitHub and PyPI, and
- the **unified doc** on the docs site (`docs/capabilities/<capability>.md`, or
  `docs/experimental/<capability>.md` for experimental capabilities).

Checks:

- Both the README and the unified doc are updated for any user-facing change
  (public class, params, defaults, tool names, extras, safety semantics). A
  change reflected in only one of them is a defect, not a follow-up.
- The two do not contradict each other or the source on extras, option names,
  defaults, or safety caveats.
- Every snippet in both docs is runnable: all imports present, class/param names
  match the source, model ids unchanged from what the source uses.
- The unified doc ends with a `## API reference` section containing one or more
  `::: pydantic_ai_harness...` autodoc blocks covering the capability's public
  class(es) -- some capabilities (e.g. compaction, subagents) export several
  (auto-expanded from the docstring, not hand-written), uses relative `.md`
  links to other harness pages, and links Pydantic AI docs with root-relative
  internal links `/ai/<section>/<page>/` (verify the route resolves on the live
  `pydantic.dev/docs` site before using it).
- Docs explain composition constraints and safety implications.
- The PR links an issue.

This is the last documentation gate before merge. Run the `docs-parity-reviewer`
subagent (`.agents/agents/docs-parity-reviewer.md`) on the change as the final
review step; treat its blocking findings as merge blockers.
