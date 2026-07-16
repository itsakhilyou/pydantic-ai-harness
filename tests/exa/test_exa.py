"""Tests for the ExaSearch capability and ExaSearchToolset."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from exa_py.api import ContentsOptions, Result, SearchResponse, TextContentsOptions
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.exa import ExaSearch, ExaSearchToolset


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def _result(
    url: str = 'https://example.dev/page',
    *,
    title: str | None = 'Example page',
    text: str | None = 'Example page text.',
    author: str | None = None,
    published_date: str | None = None,
) -> Result:
    """A real `exa_py` result with the fields the toolset reads."""
    return Result(url=url, id=url, title=title, text=text, author=author, published_date=published_date)


def _response(*results: Result) -> SearchResponse[Result]:
    """A real `exa_py` response wrapping the given results."""
    return SearchResponse(results=list(results), resolved_search_type=None, auto_date=None)


@dataclass
class _FakeExaClient:
    """In-memory `ExaClient` double: canned responses, recorded call arguments."""

    search_response: SearchResponse[Result] = field(default_factory=_response)
    contents_response: SearchResponse[Result] = field(default_factory=_response)
    search_calls: list[tuple[str, ContentsOptions, int]] = field(default_factory=list[tuple[str, ContentsOptions, int]])
    get_contents_calls: list[tuple[str, TextContentsOptions]] = field(
        default_factory=list[tuple[str, TextContentsOptions]]
    )

    async def search(self, query: str, *, contents: ContentsOptions, num_results: int) -> SearchResponse[Result]:
        self.search_calls.append((query, contents, num_results))
        return self.search_response

    async def get_contents(self, urls: str, *, text: TextContentsOptions) -> SearchResponse[Result]:
        self.get_contents_calls.append((urls, text))
        return self.contents_response


def _toolset(client: _FakeExaClient, *, num_results: int = 5, max_text_chars: int = 10_000) -> ExaSearchToolset[None]:
    return ExaSearch[None](num_results=num_results, max_text_chars=max_text_chars, client=client).get_toolset()


class TestWebSearch:
    async def test_formats_results_with_metadata(self) -> None:
        client = _FakeExaClient(
            search_response=_response(
                _result('https://a.dev', title='A', text='alpha text', author='Ada', published_date='2026-07-01'),
                _result('https://b.dev', title=None, text=None),
            )
        )
        output = await _toolset(client).web_search('rust web frameworks')
        assert output == (
            "Found 2 results for 'rust web frameworks':\n\n"
            'Title: A\n'
            'URL: https://a.dev\n'
            'Published: 2026-07-01\n'
            'Author: Ada\n'
            '\n'
            'alpha text'
            '\n\n---\n\n'
            'Title: (untitled)\n'
            'URL: https://b.dev'
        )

    async def test_passes_num_results_and_contents_cap_to_client(self) -> None:
        client = _FakeExaClient(search_response=_response(_result()))
        await _toolset(client, num_results=3, max_text_chars=42).web_search('q')
        assert client.search_calls == [('q', {'text': {'max_characters': 42}}, 3)]

    async def test_no_results(self) -> None:
        client = _FakeExaClient()
        output = await _toolset(client).web_search('nothing to see')
        assert output == "No results found for 'nothing to see'."

    async def test_text_at_cap_is_not_truncated(self) -> None:
        client = _FakeExaClient(search_response=_response(_result(title='T', text='x' * 10)))
        output = await _toolset(client, max_text_chars=10).web_search('q')
        assert output == f"Found 1 result for 'q':\n\nTitle: T\nURL: https://example.dev/page\n\n{'x' * 10}"

    async def test_text_over_cap_is_truncated_keeping_the_head(self) -> None:
        client = _FakeExaClient(search_response=_response(_result(title='T', text='x' * 10 + 'TAIL')))
        output = await _toolset(client, max_text_chars=10).web_search('q')
        assert output == (
            f"Found 1 result for 'q':\n\nTitle: T\nURL: https://example.dev/page\n\n{'x' * 10}\n"
            '[... page text truncated at 10 characters]'
        )


class TestGetPage:
    async def test_returns_page_text(self) -> None:
        client = _FakeExaClient(contents_response=_response(_result('https://a.dev', title='A', text='alpha text')))
        output = await _toolset(client).get_page('https://a.dev')
        assert client.get_contents_calls == [('https://a.dev', {'max_characters': 10_000})]
        assert output == 'Title: A\nURL: https://a.dev\n\nalpha text'

    async def test_no_content_raises_model_retry(self) -> None:
        client = _FakeExaClient()
        with pytest.raises(ModelRetry, match='No content could be retrieved'):
            await _toolset(client).get_page('https://gone.dev')


class TestExaSearch:
    def test_default_client_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('EXA_API_KEY', raising=False)
        with pytest.raises(UserError, match='EXA_API_KEY'):
            ExaSearch[None]().get_toolset()

    def test_default_client_built_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('EXA_API_KEY', 'test-key')
        toolset = ExaSearch[None]().get_toolset()
        assert isinstance(toolset, ExaSearchToolset)

    def test_instructions_reference_the_tools(self) -> None:
        instructions = ExaSearch[None]().get_instructions()
        assert 'web_search' in instructions
        assert 'get_page' in instructions
        assert 'cite' in instructions

    async def test_agent_run_uses_both_tools_and_instructions(self) -> None:
        client = _FakeExaClient(
            search_response=_response(_result('https://a.dev', title='A', text='alpha')),
            contents_response=_response(_result('https://b.dev', title='B', text='beta')),
        )
        agent = Agent(TestModel(), capabilities=[ExaSearch(client=client)])

        result = await agent.run('Research something.')

        messages = result.all_messages()
        first = messages[0]
        assert isinstance(first, ModelRequest)
        assert first.instructions is not None
        assert 'web_search' in first.instructions

        calls = {
            part.tool_name: part.args_as_dict()
            for message in messages
            if isinstance(message, ModelResponse)
            for part in message.parts
            if isinstance(part, ToolCallPart)
        }
        returns = {
            part.tool_name: part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        }
        assert set(returns) == {'web_search', 'get_page'}
        query = calls['web_search']['query']
        assert returns['web_search'] == f'Found 1 result for {query!r}:\n\nTitle: A\nURL: https://a.dev\n\nalpha'
        assert returns['get_page'] == 'Title: B\nURL: https://b.dev\n\nbeta'
