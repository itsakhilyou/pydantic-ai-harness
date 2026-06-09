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
- `session/set_model` when `models` are configured.
- Tool calls with rich presentation (kind, file locations, diffs) and the
  `pending → in_progress → completed/failed` status lifecycle.
- Human-in-the-loop approval via `session/request_permission`.
- Prompt content gated by advertised capabilities (text by default; image/audio/embedded opt-in).
- Client-provided MCP servers via a `session_config` factory (advertise transports with
  `mcp_capabilities`).
- Per-turn token usage.

## Not supported

- `session/fork`, `session/resume`, `session/list`, `session/set_mode`, and
  `session/set_config_option` -- advertised off and rejected with `method_not_found`.
- Agent plans -- Pydantic AI surfaces no structured plan event to report.
- `additionalDirectories` (extra workspace roots beyond `cwd`) -- not advertised or consumed; would
  require multi-root filesystem support first.
- Partial file reads (`line`/`limit`) and terminal `outputByteLimit`.

## Notes

- A client that advertises filesystem reads but not writes gets editor-native reads, with writes to
  the local workspace disk -- coherent when the agent shares the editor's disk.
- A completed turn reports `end_turn`; model-side `max_tokens` / `refusal` stop reasons are not yet
  distinguished.
