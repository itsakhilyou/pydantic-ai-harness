"""Tests for the `ManagedSkills` capability (source package `pydantic_ai_harness.logfire`).

Shared fixtures (`anyio_backend`, Logfire configuration) live in `conftest.py`; the variable-naming
contract common to all managed-variable capabilities is covered in `test_managed_variable.py`, and
the optional-`name` derivation and auto-create plumbing more broadly in `test_nameless.py` /
`test_auto_create.py`. This module focuses on `ManagedSkills` resolving a skill list per run and
materializing each entry as a *deferred* capability: instructions-only, hidden behind the framework's
`load_capability` tool until the model asks for them (progressive disclosure via `defer_loading`).
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from logfire.testing import CaptureLogfire
from logfire.variables import LabeledValue, Rollout, Variable, VariableConfig, VariablesConfig
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

import pydantic_ai_harness.logfire._managed_variable as managed_variable
from pydantic_ai_harness import ManagedSkill, ManagedSkills
from pydantic_ai_harness.logfire import ManagedSkill as ManagedSkillFromPackage
from pydantic_ai_harness.logfire import ManagedSkills as ManagedSkillsFromPackage
from pydantic_ai_harness.logfire._managed_skills import _Skill

from ._helpers import variables_provider

pytestmark = pytest.mark.anyio

LOAD_CAPABILITY = 'load_capability'
CATALOG_PREFIX = 'The following capabilities are deferred and can be loaded using the `load_capability` tool:'


def _tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [p for m in messages if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)]


def _last_instructions(messages: list[ModelMessage]) -> str:
    return [m.instructions for m in messages if isinstance(m, ModelRequest) and m.instructions][-1]


# --- exports / construction -------------------------------------------------------------------


def test_reexported_from_top_level_and_package() -> None:
    assert ManagedSkills is ManagedSkillsFromPackage
    assert ManagedSkill is ManagedSkillFromPackage


def test_name_becomes_skill_variable_name() -> None:
    capability = ManagedSkills('support_agent')
    assert capability._variable.name == 'skill__support_agent'


def test_default_not_required() -> None:
    # `default` is optional -- an empty list means "no skills yet".
    capability = ManagedSkills('no_default')
    assert capability._variable.default == []


def test_prebuilt_variable_prefix_warning() -> None:
    with pytest.warns(UserWarning, match="'skill__' prefix is added automatically"):
        capability = ManagedSkills('skill__foo')
    assert capability._variable.name == 'skill__foo'


# --- `_Skill` deferred-capability wiring ------------------------------------------------------


def test_skill_capability_is_deferred_with_stable_id() -> None:
    skill = ManagedSkill(name='refunds', description='Refund policy.', instructions='Refund within 30 days.')
    cap = _Skill[None](skill)
    # Deferred, keyed by the skill's own name, with the description as the catalog blurb and the
    # instructions kept for post-load disclosure.
    assert cap.defer_loading is True
    assert cap.id == 'refunds'
    assert cap.get_description() == 'Refund policy.'
    assert cap.get_instructions() == 'Refund within 30 days.'


def test_materialize_skills_dedupes_last_wins_with_warning() -> None:
    capability = ManagedSkills('dupes')
    skills = [
        ManagedSkill(name='refunds', description='First.', instructions='first'),
        ManagedSkill(name='refunds', description='Second.', instructions='second'),
    ]
    with pytest.warns(UserWarning, match="Multiple managed skills are named 'refunds'; the last one wins"):
        children = capability._materialize_skills(skills)

    # A duplicate name collapses to one capability (the last), keeping ids unique for the run.
    assert [c.id for c in children] == ['refunds']
    assert children[0].get_description() == 'Second.'  # type: ignore[union-attr]


# --- progressive disclosure end-to-end --------------------------------------------------------


async def test_skill_hidden_until_loaded_then_revealed() -> None:
    capability = ManagedSkills(
        default=[ManagedSkill(name='refunds', description='Refund policy.', instructions='Refund within 30 days.')]
    )
    catalog_seen: list[str] = []
    loaded_instructions: list[object] = []

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        returns = _tool_returns(messages)
        instructions = _last_instructions(messages)
        if not any(p.tool_name == LOAD_CAPABILITY for p in returns):
            # Before loading: the catalog lists the skill, but its instructions are hidden.
            catalog_seen.append(instructions)
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY, args={'id': 'refunds'}, tool_call_id='l1')]
            )
        # After loading: the framework returns the skill's instructions in the load result.
        load_return = next(p for p in returns if p.tool_name == LOAD_CAPABILITY)
        loaded_instructions.append(load_return.content)
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn), name='support', capabilities=[capability])
    result = await agent.run('How do I get a refund?')

    catalog = catalog_seen[0]
    assert CATALOG_PREFIX in catalog
    assert '- refunds: Refund policy.' in catalog
    # The instructions are not in the prompt prefix before the model loads the skill.
    assert 'Refund within 30 days.' not in catalog
    # They are disclosed only via the `load_capability` result once the model asks for them.
    assert loaded_instructions == [{'instructions': 'Refund within 30 days.'}]
    assert result.output == 'done'
    assert capability._variable.name == 'skill__support'


async def test_multiple_skills_all_appear_in_catalog() -> None:
    capability = ManagedSkills(
        default=[
            ManagedSkill(name='refunds', description='Refund policy.', instructions='Refund within 30 days.'),
            ManagedSkill(name='shipping', description='Shipping help.', instructions='Ships in 3 days.'),
        ]
    )
    seen: list[str] = []

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        seen.append(_last_instructions(messages))
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn), name='support', capabilities=[capability])
    await agent.run('hello')

    catalog = seen[0]
    assert '- refunds: Refund policy.' in catalog
    assert '- shipping: Shipping help.' in catalog


# --- empty / invalid: run proceeds with no skills ---------------------------------------------


async def test_empty_list_adds_no_catalog() -> None:
    capability = ManagedSkills(default=[])
    seen: list[str | None] = []

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        seen.append(next((m.instructions for m in messages if isinstance(m, ModelRequest)), None))
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn), name='support', capabilities=[capability])
    result = await agent.run('hello')

    # No skills -> no deferred-capability catalog is installed and the run proceeds normally.
    assert seen[0] is None or CATALOG_PREFIX not in seen[0]
    assert result.output == 'done'


async def test_invalid_payload_falls_back_to_code(capfire: CaptureLogfire) -> None:
    capability = ManagedSkills('invalid_payload', label='production')
    seen: list[str | None] = []

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        seen.append(next((m.instructions for m in messages if isinstance(m, ModelRequest)), None))
        return ModelResponse(parts=[TextPart('done')])

    config = VariablesConfig(
        variables={
            'skill__invalid_payload': VariableConfig(
                name='skill__invalid_payload',
                # A skill requires `description`/`instructions`; a bare name fails validation.
                labels={'production': LabeledValue(version=1, serialized_value='[{"name": "x"}]')},
                rollout=Rollout(labels={'production': 1.0}),
                overrides=[],
            )
        }
    )
    with variables_provider(capfire, config):
        agent = Agent(FunctionModel(model_fn), capabilities=[capability])
        result = await agent.run('hello')

    # The bad remote value is rejected; the empty code default means no skills, and the run proceeds.
    assert seen[0] is None or CATALOG_PREFIX not in seen[0]
    assert result.output == 'done'


# --- optional name / auto-create --------------------------------------------------------------


async def test_derives_variable_from_agent_name() -> None:
    capability = ManagedSkills(
        default=[ManagedSkill(name='refunds', description='Refund policy.', instructions='Refund within 30 days.')]
    )
    # A plain model that ends the run (not `TestModel`, which would call the `load_capability` tool).
    agent = Agent(
        FunctionModel(lambda _m, _i: ModelResponse(parts=[TextPart('done')])),
        name='support_agent',
        capabilities=[capability],
    )

    assert capability._built_variable is None
    assert capability._name_omitted

    await agent.run('hello')

    assert capability._variable.name == 'skill__support_agent'


async def test_unknown_variable_is_auto_created(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch) -> None:
    managed_variable._reset_auto_create_guard()
    created: list[str] = []

    def record(variable: Variable[Any]) -> None:
        created.append(variable.name)

    monkeypatch.setattr(managed_variable, '_spawn_create', record)

    with variables_provider(capfire, VariablesConfig(variables={})):
        agent = Agent(TestModel(), name='autocreate', capabilities=[ManagedSkills()])
        await agent.run('hello')

    # The provider has no `skill__autocreate`, so it is auto-created under the derived name.
    assert created == ['skill__autocreate']


def test_no_stray_warnings_on_unique_skill_names() -> None:
    capability = ManagedSkills('unique')
    skills = [
        ManagedSkill(name='refunds', description='Refund policy.', instructions='a'),
        ManagedSkill(name='shipping', description='Shipping help.', instructions='b'),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        children = capability._materialize_skills(skills)
    assert [c.id for c in children] == ['refunds', 'shipping']
