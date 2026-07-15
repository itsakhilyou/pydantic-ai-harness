"""Pin contract: mark message content that every shipped compaction strategy must preserve.

A *pin* is a lightweight, harness-side convention: content wrapped in a recognizable
`<pinned>` envelope that compaction treats as load-bearing -- it is never summarized away or
dropped, and if a strategy would have discarded it, the strategy re-injects it verbatim.
Planning's plan does not need pinning (it is re-injected ephemerally every request in
`wrap_model_request`, so it already survives compaction by construction); the motivating
consumers here are durable task state and a scratchpad the model wants guaranteed to ride
along -- see the README.

Marking mechanism / core gap: message-part types do not share a universal `metadata` field
(only `ToolReturnPart` has one today), so there is no clean per-part flag to set. Rather than
depend on a part subtype, pins use a sentinel-in-content envelope -- the same shape the
existing `LimitWarner`/summary markers already use. If core later grows a universal part-level
`metadata`/`pinned` seam, migrate the marker onto it.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    SystemPromptPart,
    UserPromptPart,
)

_PIN_OPEN = '<pinned>'
"""Sentinel opening the pinned envelope; also the detection prefix."""

_PIN_CLOSE = '</pinned>'


def pin(content: str) -> UserPromptPart:
    """Wrap *content* in a pinned envelope that every shipped compaction strategy preserves.

    The returned `UserPromptPart` can be placed in a `ModelRequest` in the run's message
    history (e.g. by a capability or by the user); compaction keeps it verbatim.
    """
    return UserPromptPart(content=f'{_PIN_OPEN}\n{content}\n{_PIN_CLOSE}')


def is_pinned(part: object) -> bool:
    """Return True if *part* is a text part carrying the pinned envelope."""
    return (
        isinstance(part, (UserPromptPart, SystemPromptPart))
        and isinstance(part.content, str)
        and part.content.startswith(_PIN_OPEN)
    )


def collect_pinned(messages: Sequence[ModelMessage]) -> list[ModelRequestPart]:
    """Return every pinned part found across *messages*, in order."""
    out: list[ModelRequestPart] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            out.extend(part for part in msg.parts if is_pinned(part))
    return out


def _leading_context_len(messages: Sequence[ModelMessage]) -> int:
    """Count the leading run of system-only requests (system prompts + a summary message)."""
    count = 0
    for msg in messages:
        if isinstance(msg, ModelRequest) and msg.parts and all(isinstance(p, SystemPromptPart) for p in msg.parts):
            count += 1
        else:
            break
    return count


def reinject_pinned(original: Sequence[ModelMessage], compacted: list[ModelMessage]) -> list[ModelMessage]:
    """Re-inject any pinned parts from *original* that *compacted* dropped.

    Pinned parts already present in *compacted* are left where they are; missing ones are
    gathered into a single `ModelRequest` placed right after any leading system/summary
    messages, so they sit near the top of the surviving history. A no-op when *original* has
    no pins or all pins survived, so it is always safe to call.
    """
    pinned = collect_pinned(original)
    if not pinned:
        return compacted
    surviving = collect_pinned(compacted)
    missing = [part for part in pinned if part not in surviving]
    if not missing:
        return compacted
    index = _leading_context_len(compacted)
    pin_message = ModelRequest(parts=list(missing))
    return [*compacted[:index], pin_message, *compacted[index:]]
