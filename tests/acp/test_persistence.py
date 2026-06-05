"""Tests for ACP session persistence (`session/load` via a `SessionStore`)."""

from __future__ import annotations

import acp
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.acp import InMemorySessionStore, PydanticAIACPAgent, StoredSession
from tests._acp_clients import RecordingClient  # pyright: ignore[reportMissingTypeStubs]

pytestmark = pytest.mark.anyio


def _adapter(store: InMemorySessionStore | None) -> PydanticAIACPAgent[None, str]:
    return PydanticAIACPAgent(Agent(TestModel(custom_output_text='hello')), session_store=store)


async def test_initialize_advertises_load_session_only_with_a_store() -> None:
    with_store = (await _adapter(InMemorySessionStore()).initialize(protocol_version=1)).agent_capabilities
    without_store = (await _adapter(None).initialize(protocol_version=1)).agent_capabilities
    assert with_store is not None and with_store.load_session is True
    assert without_store is not None and without_store.load_session is False


async def test_new_session_persists_an_empty_session() -> None:
    store = InMemorySessionStore()
    adapter = _adapter(store)
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    session = await adapter.new_session(cwd='/ws')
    # The session is stored on creation, so it can be reopened before its first turn.
    stored = await store.load(session.session_id)
    assert stored == StoredSession(messages=[], updates=[])


async def test_turn_persists_history_and_transcript() -> None:
    store = InMemorySessionStore()
    adapter = _adapter(store)
    client = RecordingClient()
    adapter.on_connect(client)
    await adapter.initialize(protocol_version=1)
    session = await adapter.new_session(cwd='/ws')
    await adapter.prompt(prompt=[acp.text_block('hi')], session_id=session.session_id)

    stored = await store.load(session.session_id)
    assert stored is not None
    assert len(stored.messages) > 0  # the model exchange was persisted
    # The transcript is exactly what the client was shown this turn.
    assert stored.updates == client.updates


async def test_load_session_restores_history_and_replays_transcript() -> None:
    store = InMemorySessionStore()
    adapter = _adapter(store)
    first = RecordingClient()
    adapter.on_connect(first)
    await adapter.initialize(protocol_version=1)
    session = await adapter.new_session(cwd='/ws')
    await adapter.prompt(prompt=[acp.text_block('hi')], session_id=session.session_id)
    shown = list(first.updates)

    # Reopen the session on a fresh connection, as an editor would after a restart.
    reopened = RecordingClient()
    adapter.on_connect(reopened)
    await adapter.load_session(cwd='/ws', session_id=session.session_id)

    # The transcript is replayed verbatim to the new client...
    assert reopened.updates == shown
    # ...and the model history is restored so the next turn continues the conversation.
    stored = await store.load(session.session_id)
    assert stored is not None
    assert adapter._sessions[session.session_id].history == stored.messages  # pyright: ignore[reportPrivateUsage]


async def test_load_unknown_session_is_rejected() -> None:
    adapter = _adapter(InMemorySessionStore())
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    with pytest.raises(acp.RequestError):
        await adapter.load_session(cwd='/ws', session_id='does-not-exist')


async def test_load_session_without_a_store_is_method_not_found() -> None:
    adapter = _adapter(None)
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    with pytest.raises(acp.RequestError):
        await adapter.load_session(cwd='/ws', session_id='whatever')
