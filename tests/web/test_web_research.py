"""Tests for the WebResearch capability and its toolset."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits

from pydantic_ai_harness import WebResearch
from pydantic_ai_harness.web import WebResearchToolset

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    # Tests run a nested Agent, which is asyncio-bound; pin the backend.
    return 'asyncio'


def build_run_context(model: Model | None = None, usage: RunUsage | None = None) -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=model or TestModel(),
        usage=usage or RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


def fake_search(query: str) -> list[dict[str, str]]:
    return [{'title': 'Result', 'url': 'http://example.com'}]


def fake_fetch(url: str) -> str:
    return f'content of {url}'


def make_toolset(
    *,
    research_model: Model | KnownModelName | None = None,
    search_tool: Tool[Any] | None = None,
    fetch_tool: Tool[Any] | None = None,
    instructions: str | None = None,
    research_usage_limits: UsageLimits | None = None,
) -> WebResearchToolset[None]:
    return WebResearchToolset[None](
        research_model=research_model,
        search_tool=search_tool,
        fetch_tool=fetch_tool,
        max_results=5,
        instructions=instructions,
        research_usage_limits=research_usage_limits,
        max_content_length=50_000,
        allow_local_urls=False,
        timeout=30,
        allowed_domains=None,
        blocked_domains=None,
    )


class TestWebResearchToolset:
    async def test_research_with_custom_tools(self) -> None:
        # Custom search/fetch tools: the nested TestModel calls them, then synthesizes.
        ts = make_toolset(
            research_model=TestModel(custom_output_text='FINDINGS'),
            search_tool=Tool(fake_search, name='search'),
            fetch_tool=Tool(fake_fetch, name='fetch'),
        )
        out = await ts.research(build_run_context(), 'what happened')
        assert out == 'FINDINGS'

    async def test_research_builds_default_tools(self) -> None:
        # search_tool/fetch_tool=None builds the DuckDuckGo + web_fetch defaults
        # (constructed, not invoked: call_tools=[] keeps the run offline).
        ts = make_toolset(research_model=TestModel(custom_output_text='FINDINGS', call_tools=[]))
        out = await ts.research(build_run_context(), 'query')
        assert out == 'FINDINGS'

    async def test_research_model_inherits_parent_model(self) -> None:
        ts = make_toolset(research_model=None)
        ctx = build_run_context(model=TestModel(custom_output_text='INHERITED', call_tools=[]))
        out = await ts.research(ctx, 'query')
        assert out == 'INHERITED'

    async def test_usage_aggregates_into_parent(self) -> None:
        ctx = build_run_context(usage=RunUsage())
        ts = make_toolset(research_model=TestModel(custom_output_text='FINDINGS', call_tools=[]))
        await ts.research(ctx, 'query')
        assert ctx.usage.requests >= 1

    async def test_usage_limits_bound_nested_loop(self) -> None:
        # request_limit=1 is exceeded once the nested agent needs a second request
        # (after the tool call) to synthesize its answer.
        ts = make_toolset(
            research_model=TestModel(custom_output_text='FINDINGS'),
            search_tool=Tool(fake_search, name='search'),
            fetch_tool=Tool(fake_fetch, name='fetch'),
            research_usage_limits=UsageLimits(request_limit=1),
        )
        with pytest.raises(UsageLimitExceeded):
            await ts.research(build_run_context(), 'query')

    def test_instructions_override(self) -> None:
        ts = make_toolset(
            research_model=TestModel(call_tools=[]),
            instructions='Custom research instructions.',
        )
        assert ts._research_instructions == 'Custom research instructions.'  # pyright: ignore[reportPrivateUsage]


class TestWebResearchCapability:
    def test_rejects_non_positive_timeout(self) -> None:
        with pytest.raises(ValueError, match='timeout must be a positive integer'):
            WebResearch(timeout=0)

    def test_rejects_non_positive_max_results(self) -> None:
        with pytest.raises(ValueError, match='max_results must be a positive integer'):
            WebResearch(max_results=0)

    def test_rejects_non_positive_max_content_length(self) -> None:
        with pytest.raises(ValueError, match='max_content_length must be a positive integer'):
            WebResearch(max_content_length=-1)

    def test_accepts_none_optionals(self) -> None:
        capability = WebResearch(max_results=None, max_content_length=None)
        assert capability.max_results is None
        assert capability.max_content_length is None

    async def test_agent_integration(self) -> None:
        capability = WebResearch(research_model=TestModel(custom_output_text='FINDINGS', call_tools=[]))
        agent: Agent[None, str] = Agent(
            TestModel(call_tools=['research'], custom_output_text='done'),
            capabilities=[capability],
        )
        result = await agent.run('research the topic')
        assert result.output == 'done'
