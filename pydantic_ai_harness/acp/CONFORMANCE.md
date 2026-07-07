# ACP conformance

This package exposes a Pydantic AI agent as an ACP ([Agent Client Protocol](https://agentclientprotocol.com))
agent. This page summarizes which parts of the protocol the adapter supports.

Tests live in `tests/acp/`. `test_conformance.py` drives the adapter over a real in-memory
connection -- through the SDK's JSON-RPC router and serialization, as an editor would -- and asserts
each behavior against the spec.

## Supported

- Protocol version negotiation.
- `session/new`, `session/prompt`, `session/cancel`, and streamed `session/update`.
- `session/load` with a `session_store` -- replays the full conversation, user turns included.
- `session/close`.
- `session/set_config_option` for the stable `model` config option when `models` are configured.
- Tool calls with rich presentation (kind, file locations, diffs) and the
  `pending → in_progress → completed/failed` status lifecycle.
- Human-in-the-loop approval via `session/request_permission`.
- Prompt capabilities advertised (text by default; image/audio/embedded opt-in). Inbound blocks
  are converted as-is -- the spec places the restriction on the client.
- Stop reasons: `end_turn`, `max_tokens` (model-reported, or a token usage limit),
  `refusal` (content filter), `max_turn_requests` (request/tool-call usage limits), and
  `cancelled`.
- Client-provided MCP servers via a `session_config` factory (advertise transports with
  `mcp_capabilities`).
- Per-turn token usage.

## Not supported

- `session/fork`, `session/resume`, `session/list`, and `session/set_mode` -- advertised off and
  rejected with `method_not_found`.
- Agent plans -- Pydantic AI surfaces no structured plan event to report.
- `additionalDirectories` (extra workspace roots beyond `cwd`) -- not advertised or consumed; would
  require multi-root filesystem support first.
- Partial file reads (`line`/`limit`) and terminal `outputByteLimit`.
- Connecting client-provided MCP servers out of the box. They are surfaced to a `session_config`
  to turn into toolsets; without one, a session request carrying MCP servers is rejected --
  including stdio servers, which the spec requires every agent to support. An agent meant for
  arbitrary editors should install a `session_config` that connects them.

## Notes

- A client that advertises filesystem reads but not writes gets editor-native reads, with writes to
  the local workspace disk -- coherent when the agent shares the editor's disk.
- A turn ended by a usage limit answers with its `max_tokens`/`max_turn_requests` stop reason but
  rolls back like a cancellation (the raising run's messages are not retrievable), so nothing is
  committed or persisted for it.
- A cancel that lands after the turn has already committed (during the store save) still answers
  `cancelled`, but with usage set: the turn's history and transcript are kept, and only the durable
  copy may lag until the next save.
