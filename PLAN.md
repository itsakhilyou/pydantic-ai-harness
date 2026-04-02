# ToolOrphanRepair Capability

## Problem

Multi-turn conversations with tools accumulate structurally invalid message history:

- **Orphaned tool calls**: A `ToolCallPart` in a `ModelResponse` whose result was never
  recorded (streaming timeout, deferred tool dropped). The next `ModelRequest` lacks a
  matching `ToolReturnPart`.
- **Orphaned builtin tool calls**: A `BuiltinToolCallPart` without a matching
  `BuiltinToolReturnPart` in the same `ModelResponse`.
- **Orphaned tool returns**: A `ToolReturnPart` or `RetryPromptPart` whose
  `tool_call_id` does not match any call in the preceding `ModelResponse`
  (frontend-generated IDs, mismatched call IDs from deferred tools).

Providers (especially Anthropic) reject structurally invalid history with 400 errors.
Once a conversation is poisoned, every subsequent run fails on the same history.

## Solution

A `ToolOrphanRepair` capability that hooks into `before_model_request` to sanitize
`request_context.messages` before each model call.

### Repair logic (single forward pass)

For each `ModelResponse` paired with the `ModelRequest` that follows it:

1. **Builtin call repair**: Inject synthetic `BuiltinToolReturnPart` for any
   `BuiltinToolCallPart` without a matching return in the same response.
2. **Regular call matching**: Collect `tool_call_id` values from `ToolCallPart` parts.
3. **Orphaned return stripping**: Remove `ToolReturnPart` / `RetryPromptPart` from the
   request whose `tool_call_id` is not in the call set.
4. **Orphaned call patching**: Inject synthetic `ToolReturnPart` for call IDs with no
   matching return or retry in the request.
5. **Empty request guard**: If stripping leaves only `SystemPromptPart` parts, insert a
   placeholder `UserPromptPart("Continue.")` to maintain alternation.

For a trailing `ModelResponse` with no following request:
- If it contains only unmatched tool calls, drop it entirely.
- If it has other content (text, builtin results), keep it but strip the tool calls.

### Configuration

- `orphan_call_content: str` -- content for synthetic returns (default: `"Tool call was not completed."`)
- `warn: bool` -- emit a `UserWarning` when repairs are made (default: `True`)

## References

- pydantic-ai #4728: "Built-in HistoryProcessor for orphaned tool call/result repair"
- pydantic-harness #82: "Tool Output Management"
