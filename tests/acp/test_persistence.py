"""Tests for ACP session persistence (`session/load` via a `SessionStore`)."""

from __future__ import annotations

import acp
import pytest
from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.acp import InMemorySessionStore, PydanticAIACPAgent, StoredSession
from tests._acp_clients import RecordingClient  # pyright: ignore[reportMissingTypeStubs]

pytestmark = pytest.mark.anyio


def _adapter(store: InMemorySessionStore | None) -> PydanticAIACPAgent[None, str]:
    return PydanticAIACPAgent(Agent(TestModel(custom_output_text='hello')), session_store=store)


def test_stored_session_round_trips_through_pydantic() -> None:
    # A durable store serializes `StoredSession` with Pydantic; the whole `SessionUpdate` union
    # (not just the variants a turn happens to produce) must survive a JSON round-trip.
    original = StoredSession(
        messages=[
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[TextPart(content='yo')]),
        ],
        updates=[
            acp.update_user_message_text('hi'),
            acp.update_agent_message_text('yo'),
            acp.update_agent_thought_text('thinking'),
        ],
        model='openai:gpt-4o',
    )
    adapter = TypeAdapter(StoredSession)
    assert adapter.validate_json(adapter.dump_json(original)) == original


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
    # The transcript is the user's prompt (recorded for replay, never sent live -- the client
    # renders its own prompt) followed by exactly what the client was shown this turn.
    assert stored.updates == [acp.update_user_message_text('hi'), *client.updates]


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

    # The whole conversation is replayed to the new client: the user's turn (which the live
    # client rendered itself, so it was never sent as an update) followed by what was shown.
    assert reopened.updates == [acp.update_user_message_text('hi'), *shown]
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
