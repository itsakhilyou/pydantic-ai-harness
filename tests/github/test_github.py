"""Tests for the GitHub capability."""

from __future__ import annotations

import sys

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FilteredToolset, PrefixedToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness import GitHub
from pydantic_ai_harness.github._capability import _PAT_ENV_VAR, _ToolNameFilter

pytestmark = pytest.mark.anyio


def _run_context() -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


def _server(capability: GitHub[None]) -> MCPServerStdio:
    """The MCP server the capability builds (before any filtering/prefixing wrappers)."""
    server = capability._build_server()
    assert isinstance(server, MCPServerStdio)
    return server


def _public_toolset(capability: GitHub[None]) -> AbstractToolset[None]:
    """The toolset registered with the agent, narrowed away from the callable form."""
    toolset = capability.get_toolset()
    assert toolset is not None and not callable(toolset)
    return toolset


@pytest.fixture(autouse=True)
def _clear_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests control the token explicitly; never let an ambient PAT leak in."""
    monkeypatch.delenv('GITHUB_PERSONAL_ACCESS_TOKEN', raising=False)
    monkeypatch.delenv('GITHUB_TOKEN', raising=False)


class TestTokenResolution:
    def test_explicit_token(self) -> None:
        assert _server(GitHub(token='explicit')).env == {_PAT_ENV_VAR: 'explicit'}

    def test_token_from_pat_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('GITHUB_PERSONAL_ACCESS_TOKEN', 'from-pat')
        assert _server(GitHub()).env == {_PAT_ENV_VAR: 'from-pat'}

    def test_token_from_github_token_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('GITHUB_TOKEN', 'from-token')
        assert _server(GitHub()).env == {_PAT_ENV_VAR: 'from-token'}

    def test_missing_token_raises(self) -> None:
        with pytest.raises(UserError, match='needs a token'):
            GitHub().get_toolset()


class TestServerSideOptions:
    def test_bare_defaults(self) -> None:
        server = _server(GitHub(token='t'))
        assert server.command == 'docker'
        assert server.env == {_PAT_ENV_VAR: 't'}
        assert server.args == ['run', '-i', '--rm', '-e', _PAT_ENV_VAR, 'ghcr.io/github/github-mcp-server']
        assert server.id == 'github'
        assert server.timeout == 30.0
        assert server.read_timeout == 300.0

    def test_toolsets_read_only_dynamic_host(self) -> None:
        server = _server(
            GitHub(
                token='t',
                toolsets=['repos', 'issues'],
                read_only=True,
                dynamic_toolsets=True,
                host='ghe.example.com',
            )
        )
        assert server.env == {
            _PAT_ENV_VAR: 't',
            'GITHUB_TOOLSETS': 'repos,issues',
            'GITHUB_READ_ONLY': '1',
            'GITHUB_DYNAMIC_TOOLSETS': '1',
            'GITHUB_HOST': 'ghe.example.com',
        }
        # Every env var the server reads is forwarded into the container with `-e`.
        assert server.env is not None
        for name in server.env:
            assert ['-e', name] == server.args[server.args.index(name) - 1 : server.args.index(name) + 1]

    def test_extra_env_merged_and_forwarded(self) -> None:
        server = _server(GitHub(token='t', env={'HTTP_PROXY': 'http://proxy:3128'}))
        assert server.env == {_PAT_ENV_VAR: 't', 'HTTP_PROXY': 'http://proxy:3128'}
        assert '-e' in server.args and 'HTTP_PROXY' in server.args

    def test_docker_overrides(self) -> None:
        server = _server(
            GitHub(
                token='t',
                docker_command='podman',
                docker_image='example.com/ghmcp:dev',
                docker_args=['-v', '/cache:/cache'],
                id='gh-1',
                init_timeout=10.0,
                read_timeout=60.0,
            )
        )
        assert server.command == 'podman'
        assert server.args[-1] == 'example.com/ghmcp:dev'
        assert server.args[-3:-1] == ['-v', '/cache:/cache']
        assert server.id == 'gh-1'
        assert server.timeout == 10.0
        assert server.read_timeout == 60.0


class TestToolFiltering:
    def test_bare_returns_unfiltered_server(self) -> None:
        assert isinstance(_public_toolset(GitHub[None](token='t')), MCPServerStdio)

    def test_allowed_tools_wrap(self) -> None:
        toolset = _public_toolset(GitHub[None](token='t', allowed_tools=['get_issue']))
        assert isinstance(toolset, FilteredToolset)

    def test_prefix_wraps_outermost(self) -> None:
        toolset = _public_toolset(GitHub[None](token='t', tool_prefix='gh'))
        assert isinstance(toolset, PrefixedToolset)
        assert toolset.prefix == 'gh'

    def test_prefix_and_filter_compose(self) -> None:
        toolset = _public_toolset(GitHub[None](token='t', tool_prefix='gh', allowed_tools=['get_issue']))
        assert isinstance(toolset, PrefixedToolset)
        assert isinstance(toolset.wrapped, FilteredToolset)

    async def test_allowed_filter_logic(self) -> None:
        f: _ToolNameFilter[None] = _ToolNameFilter(allowed=frozenset({'get_issue'}), denied=None, predicate=None)
        ctx = _run_context()
        assert f(ctx, ToolDefinition(name='get_issue')) is True
        assert f(ctx, ToolDefinition(name='create_issue')) is False

    async def test_denied_filter_logic(self) -> None:
        f: _ToolNameFilter[None] = _ToolNameFilter(allowed=None, denied=frozenset({'delete_repo'}), predicate=None)
        ctx = _run_context()
        assert f(ctx, ToolDefinition(name='delete_repo')) is False
        assert f(ctx, ToolDefinition(name='get_issue')) is True

    async def test_allow_and_deny_combine(self) -> None:
        f: _ToolNameFilter[None] = _ToolNameFilter(
            allowed=frozenset({'get_issue', 'create_issue'}),
            denied=frozenset({'create_issue'}),
            predicate=None,
        )
        ctx = _run_context()
        assert f(ctx, ToolDefinition(name='get_issue')) is True
        assert f(ctx, ToolDefinition(name='create_issue')) is False

    async def test_sync_predicate(self) -> None:
        f: _ToolNameFilter[None] = _ToolNameFilter(
            allowed=None, denied=None, predicate=lambda _ctx, td: td.name.startswith('get_')
        )
        ctx = _run_context()
        assert f(ctx, ToolDefinition(name='get_issue')) is True
        assert f(ctx, ToolDefinition(name='create_issue')) is False

    async def test_async_predicate(self) -> None:
        async def predicate(_ctx: RunContext[None], td: ToolDefinition) -> bool:
            return td.name == 'get_issue'

        f: _ToolNameFilter[None] = _ToolNameFilter(allowed=None, denied=None, predicate=predicate)
        ctx = _run_context()
        result = f(ctx, ToolDefinition(name='get_issue'))
        assert not isinstance(result, bool)
        assert await result is True

    def test_capability_filter_passed_through(self) -> None:
        toolset = _public_toolset(GitHub[None](token='t', tool_filter=lambda _ctx, _td: True))
        assert isinstance(toolset, FilteredToolset)


class TestAgentIntegration:
    def test_capability_registers_toolset(self) -> None:
        agent = Agent(TestModel(), capabilities=[GitHub(token='t', read_only=True)])
        leaves: list[type[AbstractToolset[None]]] = []

        def _visit(ts: AbstractToolset[None]) -> AbstractToolset[None]:
            leaves.append(type(ts))
            return ts

        for toolset in agent.toolsets:
            toolset.visit_and_replace(_visit)
        assert MCPServerStdio in leaves


class TestMissingExtra:
    def test_missing_mcp_extra_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, 'pydantic_ai.mcp', None)
        with pytest.raises(UserError, match='requires the `mcp` extra'):
            GitHub(token='t').get_toolset()
