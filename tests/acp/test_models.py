"""Tests for ACP model selection (`session/set_model`)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import acp
import pytest
from acp import schema
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.acp import InMemorySessionStore, PydanticAIACPAgent
from tests._acp_clients import RecordingClient  # pyright: ignore[reportMissingTypeStubs]

pytestmark = pytest.mark.anyio


def _adapter(
    *, models: Sequence[KnownModelName | str] | Literal['all'] | None = None, store: InMemorySessionStore | None = None
) -> PydanticAIACPAgent[None, str]:
    return PydanticAIACPAgent(Agent(TestModel(custom_output_text='hi')), models=models, session_store=store)


async def _started(adapter: PydanticAIACPAgent[None, str]) -> str:
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    return (await adapter.new_session(cwd='/ws')).session_id


async def test_models_all_advertises_every_known_model() -> None:
    adapter = _adapter(models='all')
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    response = await adapter.new_session(cwd='/ws')
    assert response.models is not None
    ids = [m.model_id for m in response.models.available_models]
    assert len(ids) > 100  # the whole known set, not a curated handful
    assert 'openai:gpt-4o' in ids
    assert response.models.current_model_id == ids[0]  # first known model is the default


async def test_new_session_advertises_configured_models() -> None:
    adapter = _adapter(models=['openai:gpt-4o', 'test'])
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    response = await adapter.new_session(cwd='/ws')
    assert response.models is not None
    assert [m.model_id for m in response.models.available_models] == ['openai:gpt-4o', 'test']
    assert response.models.current_model_id == 'openai:gpt-4o'  # the first configured model is the default


async def test_new_session_without_models_advertises_none() -> None:
    adapter = _adapter()
    assert (await adapter.new_session(cwd='/ws')).models is None


async def test_set_model_updates_the_session_and_persists() -> None:
    store = InMemorySessionStore()
    adapter = _adapter(models=['openai:gpt-4o', 'test'], store=store)
    session_id = await _started(adapter)
    assert await adapter.set_session_model(model_id='test', session_id=session_id) == schema.SetSessionModelResponse()
    assert adapter._sessions[session_id].model == 'test'  # pyright: ignore[reportPrivateUsage]
    stored = await store.load(session_id)
    assert stored is not None and stored.model == 'test'


async def test_selected_model_applies_to_a_run() -> None:
    # 'test' resolves to TestModel, so the per-run override runs offline -- proving it reaches the run.
    adapter = _adapter(models=['test'])
    session_id = await _started(adapter)
    response = await adapter.prompt(prompt=[acp.text_block('hi')], session_id=session_id)
    assert response.stop_reason == 'end_turn'


async def test_selected_model_survives_reload() -> None:
    store = InMemorySessionStore()
    adapter = _adapter(models=['openai:gpt-4o', 'test'], store=store)
    session_id = await _started(adapter)
    await adapter.set_session_model(model_id='test', session_id=session_id)
    response = await adapter.load_session(cwd='/ws', session_id=session_id)
    assert adapter._sessions[session_id].model == 'test'  # pyright: ignore[reportPrivateUsage]
    assert response is not None and response.models is not None and response.models.current_model_id == 'test'


async def test_set_unknown_model_is_rejected() -> None:
    adapter = _adapter(models=['test'])
    session_id = await _started(adapter)
    with pytest.raises(acp.RequestError):
        await adapter.set_session_model(model_id='not-a-model', session_id=session_id)


async def test_set_model_for_unknown_session_is_rejected() -> None:
    adapter = _adapter(models=['test'])
    await _started(adapter)
    with pytest.raises(acp.RequestError):
        await adapter.set_session_model(model_id='test', session_id='no-such-session')


async def test_set_model_without_configured_models_is_method_not_found() -> None:
    adapter = _adapter()
    session_id = await _started(adapter)
    with pytest.raises(acp.RequestError):
        await adapter.set_session_model(model_id='test', session_id=session_id)
