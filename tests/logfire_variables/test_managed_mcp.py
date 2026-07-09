"""Tests for the `ManagedMCP` capability (source package `pydantic_ai_harness.logfire`).

Shared fixtures (`anyio_backend`, Logfire configuration) live in `conftest.py`; the variable-naming
contract common to all managed-variable capabilities is covered in `test_managed_variable.py`, and
the optional-`name` derivation and auto-create plumbing more broadly in `test_nameless.py` /
`test_auto_create.py`. This module focuses on `ManagedMCP` resolving a connection per run --
materializing a filtered, optionally prefixed MCP toolset from the resolved config, and degrading to
no MCP server when no URL is published. The MCP connection itself is never opened: `MCP._build_local`
is monkeypatched to a plain in-memory `FunctionToolset`, so the filtering and prefixing that
`ManagedMCP` assembles are exercised without a live server.
"""

from __future__ import annotations

from typing import Any

import pytest
from logfire.testing import CaptureLogfire
from logfire.variables import LabeledValue, Rollout, Variable, VariableConfig, VariablesConfig
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

import pydantic_ai_harness.logfire._managed_variable as managed_variable
from pydantic_ai_harness import ManagedMCP, ManagedMCPValue
from pydantic_ai_harness.logfire import ManagedMCP as ManagedMCPFromPackage
from pydantic_ai_harness.logfire import ManagedMCPValue as ManagedMCPValueFromPackage

from ._helpers import advertised, capture_tools, variables_provider

pytestmark = pytest.mark.anyio


def local_tool() -> str:
    """A local tool."""
    return 'local'


def search(query: str) -> str:  # pragma: no cover - definition-only, never executed
    """Search."""
    return query


def create(title: str) -> str:  # pragma: no cover - definition-only, never executed
    """Create."""
    return title


def fake_server_toolset() -> FunctionToolset[None]:
    """An in-memory stand-in for an MCP server's toolset, so no connection is ever opened."""
    return FunctionToolset[None](tools=[search, create])


@pytest.fixture
def stub_mcp_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the MCP capability's local-toolset builder with an in-memory `FunctionToolset`.

    `MCP` builds its local toolset from the URL at construction (`_build_local`); stubbing that seam
    lets `ManagedMCP` assemble and flow through a real toolset -- with the filter and prefix applied
    on top -- without importing `mcp` or opening a connection.
    """

    def build_local(self: MCP[Any], url: str) -> AbstractToolset[None]:
        return fake_server_toolset()

    monkeypatch.setattr(MCP, '_build_local', build_local)


# --- exports / construction -------------------------------------------------------------------


def test_reexported_from_top_level_and_package() -> None:
    assert ManagedMCP is ManagedMCPFromPackage
    assert ManagedMCPValue is ManagedMCPValueFromPackage


def test_name_becomes_mcp_variable_name() -> None:
    capability = ManagedMCP('github')
    assert capability._variable.name == 'mcp__github'


def test_default_not_required() -> None:
    # `default` is optional -- an empty value (no URL) means "no managed MCP server yet".
    capability = ManagedMCP('no_default')
    assert capability._variable.default == ManagedMCPValue()


def test_prebuilt_variable_prefix_warning() -> None:
    with pytest.warns(UserWarning, match="'mcp__' prefix is added automatically"):
        capability = ManagedMCP('mcp__foo')
    assert capability._variable.name == 'mcp__foo'


# --- materialization: filter + prefix ---------------------------------------------------------


async def test_materializes_filtered_prefixed_toolset(stub_mcp_connection: None) -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedMCP(default=ManagedMCPValue(url='https://mcp.example.com', tools=['search'], tool_prefix='gh'))
    agent = Agent(capture_tools(seen), name='gh_agent', capabilities=[capability])

    await agent.run('hello')

    # `tools=['search']` filters out `create`; `tool_prefix='gh'` namespaces what's left.
    assert advertised(seen) == {'gh_search': 'Search.'}


async def test_materializes_unfiltered_unprefixed_toolset(stub_mcp_connection: None) -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedMCP(default=ManagedMCPValue(url='https://mcp.example.com'))
    agent = Agent(capture_tools(seen), name='plain', capabilities=[capability])

    await agent.run('hello')

    # No filter, no prefix -> every server tool is advertised under its bare name.
    assert advertised(seen) == {'search': 'Search.', 'create': 'Create.'}


# --- no URL: degrade to no MCP server ---------------------------------------------------------


async def test_no_url_contributes_no_tools() -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedMCP(default=ManagedMCPValue())
    agent = Agent(capture_tools(seen), name='empty', tools=[local_tool], capabilities=[capability])

    result = await agent.run('hello')

    # No URL -> no MCP toolset materialized; only the agent's own tool is advertised and the run runs.
    assert advertised(seen) == {'local_tool': 'A local tool.'}
    assert result.output == 'done'


# --- optional name: derive from the agent's own name ------------------------------------------


async def test_derives_variable_from_agent_name(stub_mcp_connection: None) -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedMCP(default=ManagedMCPValue(url='https://mcp.example.com', tools=['search']))
    agent = Agent(capture_tools(seen), name='github', capabilities=[capability])

    # Nothing is built until the first run derives the name from the agent.
    assert capability._built_variable is None
    assert capability._name_omitted

    await agent.run('hello')

    assert capability._variable.name == 'mcp__github'
    assert advertised(seen) == {'search': 'Search.'}


# --- resolution / fallback --------------------------------------------------------------------


async def test_resolved_config_from_provider(capfire: CaptureLogfire, stub_mcp_connection: None) -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedMCP('remote', label='production')
    config = VariablesConfig(
        variables={
            'mcp__remote': VariableConfig(
                name='mcp__remote',
                labels={
                    'production': LabeledValue(
                        version=1,
                        serialized_value='{"url": "https://mcp.example.com", "tools": ["create"]}',
                    )
                },
                rollout=Rollout(labels={'production': 1.0}),
                overrides=[],
            )
        }
    )
    with variables_provider(capfire, config):
        agent = Agent(capture_tools(seen), capabilities=[capability])
        await agent.run('hello')

    # The remote connection resolves and its filter is applied.
    assert advertised(seen) == {'create': 'Create.'}


async def test_invalid_payload_falls_back_to_code(capfire: CaptureLogfire) -> None:
    seen: list[ToolDefinition] = []
    capability = ManagedMCP('invalid_payload', label='production')
    config = VariablesConfig(
        variables={
            'mcp__invalid_payload': VariableConfig(
                name='mcp__invalid_payload',
                # `tools` must be a list of strings; a number fails `ManagedMCPValue` validation.
                labels={'production': LabeledValue(version=1, serialized_value='{"tools": 123}')},
                rollout=Rollout(labels={'production': 1.0}),
                overrides=[],
            )
        }
    )
    with variables_provider(capfire, config):
        agent = Agent(capture_tools(seen), capabilities=[capability])
        result = await agent.run('hello')

    # The bad remote value is rejected; the empty code default has no URL, so no MCP server is added.
    assert seen == []
    assert result.output == 'done'


# --- auto-create ------------------------------------------------------------------------------


async def test_unknown_variable_is_auto_created(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch) -> None:
    managed_variable._reset_auto_create_guard()
    created: list[str] = []

    def record(variable: Variable[Any]) -> None:
        created.append(variable.name)

    monkeypatch.setattr(managed_variable, '_spawn_create', record)

    seen: list[ToolDefinition] = []
    with variables_provider(capfire, VariablesConfig(variables={})):
        agent = Agent(capture_tools(seen), name='autocreate', capabilities=[ManagedMCP()])
        await agent.run('hello')

    # The provider has no `mcp__autocreate`, so it is auto-created under the derived name.
    assert created == ['mcp__autocreate']
