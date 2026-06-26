"""Tests for the `ManagedAgentSpec` capability (source package `pydantic_ai_harness.logfire`).

Resolution runs against the code default (no Logfire provider) or against a
locally-overridden variable value. The build is then exercised against
TestModel so the assembled Agent actually runs end-to-end.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import logfire
import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness import ManagedAgentSpec
from pydantic_ai_harness.logfire import ManagedAgentSpec as ManagedAgentSpecFromPackage

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True, scope='module')
def _configure_logfire() -> None:
    """Configure Logfire once so variable resolution does not warn (warnings are errors)."""
    logfire.configure(send_to_logfire=False, console=False)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_build_falls_back_to_defaults_when_spec_unpublished() -> None:
    spec = ManagedAgentSpec('unpublished_slug', default_model=TestModel())

    agent = await spec.build()

    # The default-spec dict produces an Agent that runs with the default
    # model and no instructions / settings / tools / mcp servers.
    assert isinstance(agent, Agent)
    result = await agent.run('hello', model=TestModel())
    assert result.output == 'success (no tool calls)'


async def test_re_exported_at_package_root() -> None:
    assert ManagedAgentSpec is ManagedAgentSpecFromPackage


async def test_strips_prefix_with_warning() -> None:
    # Users sometimes copy the full `agentspec__` variable name out of the
    # UI; warn but accept rather than constructing the doubly-prefixed
    # variable `agentspec__agentspec__foo`.
    with pytest.warns(UserWarning, match='prefix is added automatically'):
        spec = ManagedAgentSpec('agentspec__leading_prefix')

    assert spec._variable.name == 'agentspec__leading_prefix'


async def test_invalid_name_rejected() -> None:
    with pytest.raises(ValueError, match='produces an invalid variable name'):
        ManagedAgentSpec('has spaces')


async def test_accepts_prebuilt_variable() -> None:
    var = logfire.var(name='agentspec__prebuilt', type=dict, default={})

    spec = ManagedAgentSpec(var, default_model=TestModel())
    agent = await spec.build()

    assert isinstance(agent, Agent)


async def test_override_value_flows_into_agent_kwargs() -> None:
    spec = ManagedAgentSpec(
        'override_slug',
        tools={'tool__weather': _fetch_weather},
        default_model=TestModel(call_tools=['_fetch_weather']),
    )

    with spec._variable.override(
        {
            'instructions': 'Be brief.',
            'model_settings': {'temperature': 0.4},
            'tools': [{'name': 'tool__weather', 'description': 'Look up weather'}],
        }
    ):
        agent = await spec.build()

    assert agent.model_settings == {'temperature': 0.4}
    result = await agent.run('What is the weather in Paris?')
    assert '_fetch_weather' in result.output


async def test_unmapped_tools_are_silently_skipped() -> None:
    # Spec lists two tools, caller registered one — only the registered
    # one reaches the Agent. Model just doesn't see the other.
    spec = ManagedAgentSpec(
        'skipped_tools_slug',
        tools={'tool__weather': _fetch_weather},
        default_model=TestModel(),
    )
    with spec._variable.override(
        {
            'tools': [
                {'name': 'tool__weather', 'description': 'weather'},
                {'name': 'tool__news', 'description': 'unmapped'},
            ],
        }
    ):
        agent = await spec.build()

    # No assertion on agent.tools (private), but the agent runs without raising.
    result = await agent.run('hello', model=TestModel())
    assert isinstance(result.output, str)


async def test_skills_fold_into_instructions() -> None:
    spec = ManagedAgentSpec('skills_slug', default_model=TestModel())

    with spec._variable.override(
        {
            'instructions': 'You are a helpful assistant.',
            'skills': [
                {'name': 'skill__refund', 'description': 'Refund eligibility lookup.'},
                {'name': 'skill__shipping', 'description': 'Shipping status.'},
            ],
        }
    ):
        agent = await spec.build()

    instructions = '\n'.join(i for i in (agent._instructions or []) if isinstance(i, str))
    assert 'You are a helpful assistant.' in instructions
    assert 'Available skills:' in instructions
    assert 'skill__refund: Refund eligibility lookup.' in instructions
    assert 'skill__shipping: Shipping status.' in instructions


async def test_skills_without_base_instructions_emit_catalog_only() -> None:
    spec = ManagedAgentSpec('skills_only_slug', default_model=TestModel())

    with spec._variable.override({'skills': [{'name': 'skill__refund', 'description': 'Refund lookup.'}]}):
        agent = await spec.build()

    instructions = '\n'.join(i for i in (agent._instructions or []) if isinstance(i, str))
    assert instructions.startswith('Available skills:')


async def test_capability_classes_instantiated_from_spec() -> None:
    instantiated: list[tuple[str, dict[str, Any]]] = []

    class StubCapability(AbstractCapability[Any]):
        def __init__(self, **config: Any) -> None:
            instantiated.append(('stub', config))

    spec = ManagedAgentSpec('caps_slug', capability_classes={'stub_cap': StubCapability}, default_model=TestModel())

    with spec._variable.override({'capabilities': [{'type': 'stub_cap', 'config': {'foo': 'bar'}}]}):
        await spec.build()

    assert instantiated == [('stub', {'foo': 'bar'})]


async def test_unregistered_capability_dropped_so_optional_deps_dont_break_callers() -> None:
    # Spec references `code_mode` but caller never installed the harness
    # extra, so they don't pass it in capability_classes. The build must
    # succeed (rather than raising) and just drop that capability.
    spec = ManagedAgentSpec('missing_cap_slug', default_model=TestModel())

    with spec._variable.override({'capabilities': [{'type': 'code_mode'}]}):
        agent = await spec.build()

    assert isinstance(agent, Agent)


async def test_capability_with_bad_config_raises_user_error() -> None:
    class StrictCapability(AbstractCapability[Any]):
        def __init__(self, allowed_arg: int) -> None:
            self.allowed_arg = allowed_arg  # pragma: no cover

    spec = ManagedAgentSpec(
        'strict_cap_slug', capability_classes={'strict': StrictCapability}, default_model=TestModel()
    )

    with spec._variable.override({'capabilities': [{'type': 'strict', 'config': {'unknown_arg': 1}}]}):
        with pytest.raises(UserError, match="capability 'strict' rejected config"):
            await spec.build()


async def test_capability_with_non_dict_config_dropped() -> None:
    # `config` of e.g. a string is malformed; the entry is dropped rather than
    # crashing the build. Capability class is never instantiated.
    instantiated: list[str] = []

    class NoArgCapability(AbstractCapability[Any]):
        def __init__(self) -> None:
            instantiated.append('called')  # pragma: no cover

    spec = ManagedAgentSpec(
        'bad_config_slug', capability_classes={'no_arg': NoArgCapability}, default_model=TestModel()
    )

    with spec._variable.override({'capabilities': [{'type': 'no_arg', 'config': 'not-a-dict'}]}):
        await spec.build()

    assert instantiated == []


async def test_malformed_entries_silently_skipped() -> None:
    # Defending against a hand-edited spec value — non-dict entries in the
    # tools/mcp/skills/capabilities lists shouldn't crash the build.
    spec = ManagedAgentSpec('malformed_slug', default_model=TestModel())

    with spec._variable.override(
        {
            'tools': ['not-a-dict', {'no_name': True}, {'name': 42}],
            'mcp_servers': [None, {'url': ''}, {'no_url': True}],
            'skills': [{'name': 'skill__x'}, 'string'],  # missing description → dropped
            'capabilities': [{'type': 42}, None, {'no_type': True}],
        }
    ):
        agent = await spec.build()

    # No skills with a usable description → no catalog appended.
    assert not agent._instructions


async def test_logfire_instance_ignored_with_prebuilt_variable_warns() -> None:
    # The Variable carries its own Logfire instance, so passing one explicitly
    # is ambiguous; warn rather than silently use the wrong instance.
    var = logfire.var(name='agentspec__instance_warn', type=dict, default={})

    with pytest.warns(UserWarning, match='`logfire_instance` is ignored'):
        ManagedAgentSpec(var, logfire_instance=logfire.DEFAULT_LOGFIRE_INSTANCE)


async def test_capability_without_config_instantiates_with_no_args() -> None:
    instantiated: list[str] = []

    class NoArgCapability(AbstractCapability[Any]):
        def __init__(self) -> None:
            instantiated.append('no-arg')

    spec = ManagedAgentSpec('noarg_cap_slug', capability_classes={'no_arg': NoArgCapability}, default_model=TestModel())

    # Entry omits `config` entirely; we should fall through to the `cls()` branch.
    with spec._variable.override({'capabilities': [{'type': 'no_arg'}]}):
        await spec.build()

    assert instantiated == ['no-arg']


async def test_capability_with_valid_config_instantiates() -> None:
    # Counter-test to the bad-config one above: when the config matches the
    # capability's __init__ signature, the instance reaches the Agent.
    instantiated: list[int] = []

    class StrictCapability(AbstractCapability[Any]):
        def __init__(self, allowed_arg: int) -> None:
            instantiated.append(allowed_arg)

    spec = ManagedAgentSpec(
        'valid_cap_slug', capability_classes={'strict': StrictCapability}, default_model=TestModel()
    )

    with spec._variable.override({'capabilities': [{'type': 'strict', 'config': {'allowed_arg': 42}}]}):
        await spec.build()

    assert instantiated == [42]


async def test_valid_mcp_server_reaches_toolset_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    # The harness's default install lacks the `[mcp]` extra, so `pydantic_ai.mcp`
    # would fail to import. Mock just the symbol we need so the wirer's
    # construction path runs end-to-end and the resulting toolset lands on the
    # Agent kwargs.
    fake_module = MagicMock()
    fake_toolset = MagicMock()
    fake_module.MCPToolset = fake_toolset
    monkeypatch.setitem(__import__('sys').modules, 'pydantic_ai.mcp', fake_module)

    spec = ManagedAgentSpec('mcp_slug', default_model=TestModel())
    with spec._variable.override(
        {
            'mcp_servers': [
                {'url': 'https://example.com/mcp', 'headers': {'Authorization': 'Bearer x'}},
                {'url': 'https://other.com/mcp'},  # no headers → None
            ],
        }
    ):
        await spec.build()

    # Both servers reached MCPToolset(); headers preserved per-entry.
    assert fake_toolset.call_args_list[0].args == ('https://example.com/mcp',)
    assert fake_toolset.call_args_list[0].kwargs == {'headers': {'Authorization': 'Bearer x'}}
    assert fake_toolset.call_args_list[1].args == ('https://other.com/mcp',)
    assert fake_toolset.call_args_list[1].kwargs == {'headers': None}


def _fetch_weather(city: str) -> str:
    return f'sunny and 72°F in {city}'
