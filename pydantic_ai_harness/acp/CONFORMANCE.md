# ACP conformance record

This adapter exposes a Pydantic AI agent as an ACP **agent** (server). This file records, per spec
page, which agent-binding normative clauses are pinned by a test, which are deliberately not
applicable, and which remain open. It exists so review can ask *"which clause has no test?"* rather
than *"do the tests pass?"* -- a clause with no row is a gap, not a silent pass.

Two test layers back it:

- **`tests/acp/test_conformance.py`** drives the adapter over a real in-memory wire
  (`tests/acp/_wire.py`): a `ClientSideConnection` talks to `acp.run_agent` across a
  `socket.socketpair`, so requests cross the SDK's JSON-RPC router (including unstable-method
  gating) and serialization, and updates arrive as bytes the client re-parses. This is the only
  layer that can see router-reachability and frame-byte bugs.
- The pre-existing unit and subprocess tests (`test_acp.py`, `test_content.py`, `test_native.py`,
  `test_persistence.py`, `test_models.py`) pin lower-level behavior (presenter mapping, chunk math,
  permission scoping, persistence) and end-to-end stdio.

Each conformance test names the clause it enforces, with an oracle built from the spec/input rather
than from the adapter's own output.

## Coverage by spec page

| Page | Agent-binding clauses | Status |
| --- | --- | --- |
| initialization | version echo / negotiate-down; advertise chosen version + capabilities; advertise only supported methods | **Covered** — `TestVersionNegotiation`, `TestCapabilityAdvertisement` (wire). Version echo asserts the *input* per in-range version, not a literal. |
| session-setup | unique session id; `session/load` replays the entire conversation incl. user turns; respond after replay; `cwd` is the resolution base | **Covered** — user-turn replay pinned at the wire (`TestSessionLoadReplay`) and unit (`test_persistence.py`); cwd rooting in `test_acp.py`. |
| prompt-turn | respond with a `StopReason`; tool-call `pending→in_progress→completed/failed` ordering; `cancelled` on cancel; final update precedes response | **Mostly covered** — stop reason, status lifecycle, cancel mapping tested. Open: G4, G5 below. |
| content | accept text + resource-link baseline; optional types gated by advertised `promptCapabilities` | **Covered (advertisement)** — `test_content.py` + advertisement tests. Inbound leniency is intentional (see N/A ledger). |
| tool-calls | id/title/kind/status/content/locations/rawInput/rawOutput semantics; full-ToolCall then ToolCallUpdate; diff fields | **Covered** — `test_acp.py` presenter + lifecycle tests. |
| authentication | advertise `authMethods` (empty ⇒ no auth); no-op authenticate | **Covered** — `TestCapabilityAdvertisement.test_no_authentication_is_advertised` (wire). |
| session-modes / -config-options / -list / -delete | advertise only when supported; reject unsupported method with `method_not_found` | **Covered** — advertisement-off asserted at the wire (`TestCapabilityAdvertisement`, `TestErrorCodes`); modes/config off at `new_session`. |
| transports / schema | UTF-8 + newline framing; no frame exceeding the client buffer; stdio entry | **Covered (bytes)** — large non-ASCII output reassembles intact through the 64 KiB-bounded client reader (`TestStreamedFrameBytes`). Open: G-stdout below. |
| extensibility | `method_not_found` for unknown ext request; ignore unknown ext notification | **Covered** — `test_acp.py`. |
| file-system / terminals | check client capability before use; release terminal once; kill-then-release on cancel | **Covered** — `test_native.py` (incl. failing-kill cancel edge). One intentional deviation, below. |
| agent-plan | report a plan if the model produces one (SHOULD) | **N/A** — Pydantic AI surfaces no structured plan event; nothing to map. |

## Open gaps (no test yet)

Tracked for follow-up; none is a known bug.

- **G4 — stop-reason fidelity.** Non-cancelled turns always report `end_turn`; ACP also names
  `max_tokens` and `refusal`. Verified actionable: `result.response.finish_reason` (`length` /
  `content_filter`) maps cleanly. MAY-level (collapsing is spec-conformant), deferred only because
  testing it needs a custom streamed model. Low effort when picked up.
- **G5 — update-before-response ordering on cancel.** Structurally guaranteed (the turn task is
  awaited before responding) but not asserted as an ordering invariant.
- **G-stdout — stdout purity.** No test asserts the agent process writes *only* framed ACP messages
  to stdout (a stray print/banner would corrupt the stream).
- **Partial-read / output-byte-limit params** (`fs/read` `line`/`limit`, `terminal/create`
  `outputByteLimit`) are unimplemented optional features, hence untested.

## Verified deviations

- **`additionalDirectories` accepted but not advertised** (session-setup). The adapter accepts and
  forwards `additional_directories` to the `session_config` hook but never advertises
  `sessionCapabilities.additionalDirectories`, so a conformant client never sends them. The SDK
  field is also marked UNSTABLE/not-in-spec, and advertising would over-promise a root boundary the
  adapter does not enforce. Low severity (no wrong behavior occurs). **Decision pending**: drop the
  param from the advertised surface, or document it as a non-advertised passthrough.
- **Read-only-fs client → local writes** (file-system). When a client advertises fs read but not
  write, the adapter still exposes a `write_file` tool that writes to local disk. It never calls the
  client's `fs/write_text_file` (so it honors the literal MUST NOT), but it diverges from the intent
  for a *remote* editor. This is an intentional, documented design choice (`_native.py`), pinned by
  its own test.

## N/A ledger (recorded so the matrix is complete)

- Client-bound MUSTs (the client verifies capabilities, restricts content to `promptCapabilities`,
  responds `cancelled` to pending permissions, etc.) are not the agent's obligation.
- Inbound content is decoded by runtime type, not gated on advertised `promptCapabilities` —
  intentional leniency; the spec puts content restriction on the client, and a model that cannot
  take a modality surfaces a normal error.
- `session/resume`, `session/fork`, `session/set_mode`, `session/set_config_option`,
  `session/list` are advertised off and rejected with `method_not_found`.
- stdio MCP servers are delegated to the `session_config` hook rather than connected by the adapter;
  MCP capabilities are advertised off by default and servers are rejected when no hook is installed.
