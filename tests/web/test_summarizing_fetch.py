"""Tests for the SummarizingFetch capability and its toolset."""

from __future__ import annotations

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness import SummarizingFetch
from pydantic_ai_harness.web import FetchedPage, Fetcher, Summarizer, SummarizingFetchToolset

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    # Tests run a nested Agent, which is asyncio-bound; pin the backend.
    return 'asyncio'


def build_run_context(model: Model | None = None, usage: RunUsage | None = None) -> RunContext[None]:
    """Build a `RunContext` for invoking the toolset directly."""
    return RunContext[None](
        deps=None,
        model=model or TestModel(),
        usage=usage or RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


def make_fetcher(result: FetchedPage | BinaryContent) -> Fetcher:
    async def _fetch(url: str) -> FetchedPage | BinaryContent:
        return result

    return _fetch


def make_toolset(
    *,
    fetcher: Fetcher | None = None,
    summarizer: Summarizer[None] | None = None,
    summarizer_model: Model | KnownModelName | None = None,
    summarize_threshold: int = 4000,
) -> SummarizingFetchToolset[None]:
    return SummarizingFetchToolset[None](
        fetcher=fetcher,
        summarizer=summarizer,
        summarizer_model=summarizer_model,
        summarize_threshold=summarize_threshold,
        max_content_length=50_000,
        allow_local_urls=False,
        timeout=30,
        allowed_domains=None,
        blocked_domains=None,
    )


class TestSummarizingFetchToolset:
    async def test_long_content_is_summarized(self) -> None:
        ts = make_toolset(
            fetcher=make_fetcher(FetchedPage(url='http://x', title='T', content='x' * 5000)),
            summarizer_model=TestModel(custom_output_text='SUMMARY'),
        )
        out = await ts.fetch_url(build_run_context(), 'http://x', 'what is x')
        assert out == 'SUMMARY'

    async def test_short_content_skips_summarizer(self) -> None:
        # Fast path: content <= threshold returns verbatim and the model is never called.
        ctx = build_run_context()
        ts = make_toolset(
            fetcher=make_fetcher(FetchedPage(url='http://x', title='T', content='short')),
            summarizer_model=TestModel(custom_output_text='SHOULD_NOT_APPEAR'),
        )
        out = await ts.fetch_url(ctx, 'http://x', 'q')
        assert out == 'short'
        assert ctx.usage.requests == 0

    async def test_binary_content_passthrough(self) -> None:
        binary = BinaryContent(data=b'%PDF-1.4', media_type='application/pdf')
        ctx = build_run_context()
        ts = make_toolset(
            fetcher=make_fetcher(binary),
            summarizer_model=TestModel(custom_output_text='SHOULD_NOT_APPEAR'),
        )
        out = await ts.fetch_url(ctx, 'http://x/doc.pdf', 'q')
        assert out is binary
        assert ctx.usage.requests == 0

    async def test_custom_summarizer_always_runs(self) -> None:
        # Even short content is passed to a custom summarizer (no fast-path skip).
        seen: dict[str, object] = {}

        async def summarizer(ctx: RunContext[None], page: FetchedPage, query: str) -> str:
            seen['title'] = page.title
            seen['query'] = query
            return 'CUSTOM'

        ts = make_toolset(
            fetcher=make_fetcher(FetchedPage(url='http://x', title='T', content='short')),
            summarizer=summarizer,
            summarizer_model=TestModel(custom_output_text='SHOULD_NOT_APPEAR'),
        )
        out = await ts.fetch_url(build_run_context(), 'http://x', 'my query')
        assert out == 'CUSTOM'
        assert seen == {'title': 'T', 'query': 'my query'}

    async def test_summarizer_model_inherits_parent_model(self) -> None:
        ts = make_toolset(
            fetcher=make_fetcher(FetchedPage(url='http://x', title='T', content='x' * 5000)),
            summarizer_model=None,
        )
        ctx = build_run_context(model=TestModel(custom_output_text='INHERITED'))
        out = await ts.fetch_url(ctx, 'http://x', 'q')
        assert out == 'INHERITED'

    async def test_usage_aggregates_into_parent(self) -> None:
        ctx = build_run_context(usage=RunUsage())
        ts = make_toolset(
            fetcher=make_fetcher(FetchedPage(url='http://x', title='T', content='x' * 5000)),
            summarizer_model=TestModel(custom_output_text='SUMMARY'),
        )
        await ts.fetch_url(ctx, 'http://x', 'q')
        assert ctx.usage.requests >= 1


class TestDefaultFetcher:
    """Cover the default fetcher that adapts core's web_fetch_tool (no network)."""

    async def test_default_fetcher_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pydantic_ai.common_tools import web_fetch as core_web_fetch

        async def fake_call(self: object, url: str) -> core_web_fetch.WebFetchResult:
            return core_web_fetch.WebFetchResult(url=url, title='Doc', content='c' * 5000)

        monkeypatch.setattr(core_web_fetch.WebFetchLocalTool, '__call__', fake_call)
        ts = make_toolset(fetcher=None, summarizer_model=TestModel(custom_output_text='SUMMARY'))
        out = await ts.fetch_url(build_run_context(), 'http://x', 'q')
        assert out == 'SUMMARY'

    async def test_default_fetcher_binary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pydantic_ai.common_tools import web_fetch as core_web_fetch

        binary = BinaryContent(data=b'\x89PNG', media_type='image/png')

        async def fake_call(self: object, url: str) -> BinaryContent:
            return binary

        monkeypatch.setattr(core_web_fetch.WebFetchLocalTool, '__call__', fake_call)
        ts = make_toolset(fetcher=None)
        out = await ts.fetch_url(build_run_context(), 'http://x/img.png', 'q')
        assert out is binary


class TestSummarizingFetchCapability:
    def test_rejects_non_positive_timeout(self) -> None:
        with pytest.raises(ValueError, match='timeout must be a positive integer'):
            SummarizingFetch(timeout=0)

    def test_rejects_negative_threshold(self) -> None:
        with pytest.raises(ValueError, match='summarize_threshold must be a non-negative integer'):
            SummarizingFetch(summarize_threshold=-1)

    def test_rejects_non_positive_max_content_length(self) -> None:
        with pytest.raises(ValueError, match='max_content_length must be a positive integer'):
            SummarizingFetch(max_content_length=0)

    def test_rejects_non_integer_timeout(self) -> None:
        with pytest.raises(ValueError, match='timeout must be a positive integer'):
            SummarizingFetch(timeout='30')  # type: ignore[arg-type]

    def test_accepts_none_max_content_length(self) -> None:
        assert SummarizingFetch(max_content_length=None).max_content_length is None

    async def test_agent_integration(self) -> None:
        capability = SummarizingFetch(
            fetcher=make_fetcher(FetchedPage(url='http://x', title='T', content='x' * 5000)),
            summarizer_model=TestModel(custom_output_text='SUMMARY'),
        )
        agent: Agent[None, str] = Agent(
            TestModel(call_tools=['fetch_url'], custom_output_text='done'),
            capabilities=[capability],
        )
        result = await agent.run('summarize http://x')
        assert result.output == 'done'
