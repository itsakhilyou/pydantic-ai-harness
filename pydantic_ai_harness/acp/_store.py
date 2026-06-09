"""Pluggable persistence for ACP sessions, so `session/load` can reopen a past conversation.

Persistence is opt-in: pass a `SessionStore` to the adapter to advertise and support
`session/load`. [`InMemorySessionStore`][pydantic_ai_harness.acp.InMemorySessionStore] keeps
sessions for the process's lifetime; implement the protocol over a file or database for
durability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from pydantic_ai.messages import ModelMessage

from ._session import SessionUpdate


# The two views are stored separately because neither can be derived from the other: the messages
# lack a tool call's rendered title/diff, and the transcript lacks what the model saw.
@dataclass(frozen=True, kw_only=True)
class StoredSession:
    """A session's persisted state: the model's message history and the client-visible transcript.

    On `session/load`, `messages` is restored into the agent and `updates` is replayed verbatim
    to the client. Both hold Pydantic models, so a durable store can serialize them with Pydantic.
    """

    messages: list[ModelMessage] = field(default_factory=list[ModelMessage])
    updates: list[SessionUpdate] = field(default_factory=list[SessionUpdate])
    # The model selected via `session/set_model`, restored so a reopened session keeps it.
    model: str | None = None


class SessionStore(Protocol):
    """Where the adapter saves and restores sessions so `session/load` can reopen them.

    `save` is called after each committed turn (and once when the session is created); `load`
    returns a previously saved session or `None` if the id is unknown. The adapter does not catch
    exceptions raised by `save` or `load`: they propagate to the client as a request error on the
    operation that triggered them.
    """

    # Known shortcoming (not yet addressed): the adapter neither retries nor degrades on store
    # failures. A `save` that raises (a full disk, an unavailable database) fails its triggering
    # operation even though the turn already streamed and committed in memory; a `load` that raises
    # on a corrupt payload surfaces a raw error instead of the clean `invalid_params` an unknown id
    # gets. A durable store should own its durability errors until the adapter defines explicit
    # (likely non-fatal and surfaced) handling.

    async def save(self, session_id: str, session: StoredSession) -> None: ...  # pragma: no cover - protocol stub

    async def load(self, session_id: str) -> StoredSession | None: ...  # pragma: no cover - protocol stub


class InMemorySessionStore:
    """A [`SessionStore`][pydantic_ai_harness.acp.SessionStore] holding sessions in a dict.

    Sessions can be reopened within one process but do not survive a restart; back the store
    with a file or database for that.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, StoredSession] = {}

    async def save(self, session_id: str, session: StoredSession) -> None:
        self._sessions[session_id] = session

    async def load(self, session_id: str) -> StoredSession | None:
        return self._sessions.get(session_id)
