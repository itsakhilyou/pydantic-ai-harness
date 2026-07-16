"""Tests for the Youdotcom capability and YoudotcomToolset."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.youdotcom import Youdotcom, YoudotcomToolset

# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _make_web_payload() -> dict[str, object]:
    """Build a minimal search API response with one web result."""
    return {
        'results': {
            'web': [
                {
                    'title': 'Example Page',
                    'url': 'https://example.com',
                    'description': 'An example page.',
                    'snippets': ['snippet one', 'snippet two'],
                    'thumbnail_url': 'https://example.com/thumb.png',
                    'favicon_url': 'https://example.com/favicon.ico',
                    'authors': ['Jane Doe'],
                    'page_age': '2025-01-15T10:30:00Z',
                }
            ],
            'news': [],
        }
    }


def _make_news_payload() -> dict[str, object]:
    """Build a minimal search API response with one news result."""
    return {
        'results': {
            'web': [],
            'news': [
                {
                    'title': 'Breaking News',
                    'url': 'https://news.example.com/story',
                    'description': 'Something happened.',
                    'page_age': '2025-06-01T12:00:00Z',
                }
            ],
        }
    }


def _make_livecrawl_payload() -> dict[str, object]:
    """Build a search API response with livecrawled content."""
    return {
        'results': {
            'web': [
                {
                    'title': 'Live Page',
                    'url': 'https://live.example.com',
                    'contents': {
                        'html': '<p>Hello</p>',
                        'markdown': 'Hello',
                    },
                }
            ],
            'news': [],
        }
    }


def _make_empty_search_payload() -> dict[str, object]:
    """Build an empty search API response."""
    return {'results': {'web': [], 'news': []}}


def _make_malformed_search_payload() -> dict[str, object]:
    """Build a search payload missing the results key entirely."""
    return {'unrelated': 'data'}


def _make_contents_payload() -> list[dict[str, object]]:
    """Build a Contents API response with one result."""
    return [
        {
            'url': 'https://example.com/page',
            'title': 'Example Page',
            'markdown': '# Example\n\nHello world.',
            'html': '<h1>Example</h1><p>Hello world.</p>',
            'metadata': {
                'site_name': 'Example',
                'favicon_url': 'https://example.com/favicon.ico',
            },
        }
    ]


def _make_contents_minimal_payload() -> list[dict[str, object]]:
    """Build a Contents API response with only required fields."""
    return [{'url': 'https://example.com', 'title': 'Minimal'}]


def _make_contents_partial_payload() -> list[dict[str, object]]:
    """Build a Contents API response where one URL failed (null content)."""
    return [
        {'url': 'https://ok.com', 'title': 'OK', 'markdown': 'content'},
        {'url': 'https://fail.com', 'title': 'Fail', 'html': None, 'markdown': None},
    ]


def _make_research_payload() -> dict[str, object]:
    """Build a Research API response."""
    return {
        'output': {
            'content': '## Answer\n\nSomething happened [[1, 2]].',
            'content_type': 'text',
            'sources': [
                {
                    'url': 'https://source1.com',
                    'title': 'Source 1',
                    'snippets': ['relevant excerpt'],
                },
                {
                    'url': 'https://source2.com',
                    'title': 'Source 2',
                },
            ],
        }
    }


def _make_research_minimal_payload() -> dict[str, object]:
    """Build a Research API response with minimal fields."""
    return {
        'output': {
            'content': 'Short answer.',
            'content_type': 'text',
            'sources': [{'url': 'https://src.com'}],
        }
    }


def _make_research_empty_payload() -> dict[str, object]:
    """Build a Research API response with no sources."""
    return {'output': {'content': 'No sources needed.', 'content_type': 'text', 'sources': []}}


def _make_finance_research_payload() -> dict[str, object]:
    """Build a Finance Research API response."""
    return {
        'output': {
            'content': 'Revenue grew 114% [[1]].',
            'content_type': 'text',
            'sources': [
                {
                    'url': 'https://sec.gov/filing',
                    'title': '10-K Filing',
                    'snippets': ['Total revenue $130.5B'],
                }
            ],
        }
    }


def _make_malformed_research_payload() -> dict[str, object]:
    """Build a research payload missing the output key entirely."""
    return {'error': 'something went wrong'}


def _make_research_structured_payload() -> dict[str, object]:
    """Build a Research API response with structured (object) output."""
    return {
        'output': {
            'content': {'answer': 'The sky is blue.', 'confidence': 0.95},
            'content_type': 'object',
            'sources': [{'url': 'https://source.com', 'title': 'Source'}],
        }
    }


# ---------------------------------------------------------------------------
# Search: result field integration tests
# ---------------------------------------------------------------------------


class TestSearchResultFields:
    """Exercise search result field mapping through the public search() tool."""

    @staticmethod
    def _toolset_with_payload(payload: dict[str, object]) -> tuple[YoudotcomToolset[None], httpx.AsyncClient]:
        """Create a toolset and its mock client backed by *payload*."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
        client = httpx.AsyncClient(transport=transport)
        return YoudotcomToolset(api_key='test', http_client=client), client

    async def test_web_result_all_fields(self) -> None:
        toolset, client = self._toolset_with_payload(_make_web_payload())
        try:
            results = await toolset.search('q')
            assert len(results) == 1
            r = results[0]
            assert r['title'] == 'Example Page'
            assert r['url'] == 'https://example.com'
            assert r.get('description') == 'An example page.'
            assert r.get('snippets') == ['snippet one', 'snippet two']
            assert r.get('thumbnail_url') == 'https://example.com/thumb.png'
            assert r.get('favicon_url') == 'https://example.com/favicon.ico'
            assert r.get('authors') == ['Jane Doe']
            assert r.get('page_age') == datetime(2025, 1, 15, 10, 30, tzinfo=timezone.utc)
        finally:
            await client.aclose()

    async def test_web_result_minimal_fields(self) -> None:
        payload: dict[str, object] = {'results': {'web': [{'title': 'T', 'url': 'https://x.com'}], 'news': []}}
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.search('q')
            assert len(results) == 1
            assert results[0]['title'] == 'T'
            assert results[0]['url'] == 'https://x.com'
            assert 'description' not in results[0]
            assert 'snippets' not in results[0]
            assert 'thumbnail_url' not in results[0]
            assert 'favicon_url' not in results[0]
            assert 'authors' not in results[0]
            assert 'page_age' not in results[0]
            assert 'contents' not in results[0]
        finally:
            await client.aclose()

    async def test_web_result_empty_description_not_included(self) -> None:
        payload: dict[str, object] = {
            'results': {'web': [{'title': 'T', 'url': 'https://x.com', 'description': ''}], 'news': []}
        }
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.search('q')
            assert 'description' not in results[0]
        finally:
            await client.aclose()

    async def test_web_result_empty_snippets_not_included(self) -> None:
        payload: dict[str, object] = {
            'results': {'web': [{'title': 'T', 'url': 'https://x.com', 'snippets': []}], 'news': []}
        }
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.search('q')
            assert 'snippets' not in results[0]
        finally:
            await client.aclose()

    async def test_news_result_no_web_fields(self) -> None:
        toolset, client = self._toolset_with_payload(_make_news_payload())
        try:
            results = await toolset.search('q')
            assert len(results) == 1
            assert results[0]['title'] == 'Breaking News'
            assert 'snippets' not in results[0]
            assert 'favicon_url' not in results[0]
            assert 'authors' not in results[0]
        finally:
            await client.aclose()

    async def test_livecrawl_both_formats(self) -> None:
        toolset, client = self._toolset_with_payload(_make_livecrawl_payload())
        try:
            results = await toolset.search('q')
            assert len(results) == 1
            assert results[0].get('contents') == {'html': '<p>Hello</p>', 'markdown': 'Hello'}
        finally:
            await client.aclose()

    async def test_livecrawl_html_only(self) -> None:
        payload: dict[str, object] = {
            'results': {
                'web': [{'title': 'T', 'url': 'https://x.com', 'contents': {'html': '<p>Hi</p>'}}],
                'news': [],
            }
        }
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.search('q')
            assert results[0].get('contents') == {'html': '<p>Hi</p>'}
            assert 'markdown' not in results[0].get('contents', {})
        finally:
            await client.aclose()

    async def test_livecrawl_markdown_only(self) -> None:
        payload: dict[str, object] = {
            'results': {
                'web': [{'title': 'T', 'url': 'https://x.com', 'contents': {'markdown': 'Hi'}}],
                'news': [],
            }
        }
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.search('q')
            assert results[0].get('contents') == {'markdown': 'Hi'}
            assert 'html' not in results[0].get('contents', {})
        finally:
            await client.aclose()

    async def test_livecrawl_empty_strings_not_included(self) -> None:
        payload: dict[str, object] = {
            'results': {
                'web': [{'title': 'T', 'url': 'https://x.com', 'contents': {'html': '', 'markdown': ''}}],
                'news': [],
            }
        }
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.search('q')
            assert 'contents' not in results[0]
        finally:
            await client.aclose()

    async def test_both_web_and_news(self) -> None:
        payload: dict[str, object] = {
            'results': {
                'web': [{'title': 'W', 'url': 'https://w.com'}],
                'news': [{'title': 'N', 'url': 'https://n.com'}],
            }
        }
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.search('q')
            assert len(results) == 2
            assert results[0]['title'] == 'W'
            assert results[1]['title'] == 'N'
        finally:
            await client.aclose()

    async def test_empty_results(self) -> None:
        toolset, client = self._toolset_with_payload(_make_empty_search_payload())
        try:
            results = await toolset.search('q')
            assert results == []
        finally:
            await client.aclose()

    async def test_malformed_payload_raises(self) -> None:
        """Missing results key raises ValidationError instead of returning empty list."""
        toolset, client = self._toolset_with_payload(_make_malformed_search_payload())
        try:
            with pytest.raises(ValidationError):
                await toolset.search('q')
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Contents: result field integration tests
# ---------------------------------------------------------------------------


class TestContentsResultFields:
    """Exercise contents result field mapping through the public extract_contents() tool."""

    @staticmethod
    def _toolset_with_payload(payload: list[dict[str, object]]) -> tuple[YoudotcomToolset[None], httpx.AsyncClient]:
        """Create a toolset and its mock client backed by *payload*."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
        client = httpx.AsyncClient(transport=transport)
        return YoudotcomToolset(api_key='test', http_client=client), client

    async def test_all_fields(self) -> None:
        toolset, client = self._toolset_with_payload(_make_contents_payload())
        try:
            results = await toolset.extract_contents(['https://example.com/page'])
            assert len(results) == 1
            r = results[0]
            assert r['url'] == 'https://example.com/page'
            assert r['title'] == 'Example Page'
            assert r.get('html') == '<h1>Example</h1><p>Hello world.</p>'
            assert r.get('markdown') == '# Example\n\nHello world.'
            assert r.get('metadata') == {
                'site_name': 'Example',
                'favicon_url': 'https://example.com/favicon.ico',
            }
        finally:
            await client.aclose()

    async def test_minimal_fields(self) -> None:
        toolset, client = self._toolset_with_payload(_make_contents_minimal_payload())
        try:
            results = await toolset.extract_contents(['https://example.com'])
            assert len(results) == 1
            assert results[0]['url'] == 'https://example.com'
            assert results[0]['title'] == 'Minimal'
            assert 'html' not in results[0]
            assert 'markdown' not in results[0]
            assert 'metadata' not in results[0]
        finally:
            await client.aclose()

    async def test_empty_html_not_included(self) -> None:
        payload: list[dict[str, object]] = [{'url': 'https://x.com', 'title': 'X', 'html': ''}]
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.extract_contents(['https://x.com'])
            assert 'html' not in results[0]
        finally:
            await client.aclose()

    async def test_empty_markdown_not_included(self) -> None:
        payload: list[dict[str, object]] = [{'url': 'https://x.com', 'title': 'X', 'markdown': ''}]
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.extract_contents(['https://x.com'])
            assert 'markdown' not in results[0]
        finally:
            await client.aclose()

    async def test_metadata_only_site_name(self) -> None:
        payload: list[dict[str, object]] = [
            {'url': 'https://x.com', 'title': 'X', 'metadata': {'site_name': 'Example'}}
        ]
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.extract_contents(['https://x.com'])
            assert results[0].get('metadata') == {'site_name': 'Example'}
            assert 'favicon_url' not in results[0].get('metadata', {})
        finally:
            await client.aclose()

    async def test_metadata_only_favicon(self) -> None:
        payload: list[dict[str, object]] = [
            {'url': 'https://x.com', 'title': 'X', 'metadata': {'favicon_url': 'https://x.com/fav.ico'}}
        ]
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.extract_contents(['https://x.com'])
            assert results[0].get('metadata') == {'favicon_url': 'https://x.com/fav.ico'}
            assert 'site_name' not in results[0].get('metadata', {})
        finally:
            await client.aclose()

    async def test_empty_metadata_not_included(self) -> None:
        payload: list[dict[str, object]] = [
            {'url': 'https://x.com', 'title': 'X', 'metadata': {'site_name': '', 'favicon_url': ''}}
        ]
        toolset, client = self._toolset_with_payload(payload)
        try:
            results = await toolset.extract_contents(['https://x.com'])
            assert 'metadata' not in results[0]
        finally:
            await client.aclose()

    async def test_empty_results(self) -> None:
        toolset, client = self._toolset_with_payload([])
        try:
            results = await toolset.extract_contents(['https://x.com'])
            assert results == []
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Research: result field integration tests
# ---------------------------------------------------------------------------


class TestResearchResultFields:
    """Exercise research result field mapping through the public research() tool."""

    @staticmethod
    def _toolset_with_payload(payload: dict[str, object]) -> tuple[YoudotcomToolset[None], httpx.AsyncClient]:
        """Create a toolset and its mock client backed by *payload*."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=payload))
        client = httpx.AsyncClient(transport=transport)
        return YoudotcomToolset(api_key='test', http_client=client), client

    async def test_full_response(self) -> None:
        toolset, client = self._toolset_with_payload(_make_research_payload())
        try:
            result = await toolset.research('What happened?')
            assert result['content'] == '## Answer\n\nSomething happened [[1, 2]].'
            assert result['content_type'] == 'text'
            assert len(result['sources']) == 2
            assert result['sources'][0]['url'] == 'https://source1.com'
            assert result['sources'][0].get('title') == 'Source 1'
            assert result['sources'][0].get('snippets') == ['relevant excerpt']
            assert result['sources'][1]['url'] == 'https://source2.com'
            assert result['sources'][1].get('title') == 'Source 2'
            assert 'snippets' not in result['sources'][1]
        finally:
            await client.aclose()

    async def test_source_empty_title_not_included(self) -> None:
        payload: dict[str, object] = {
            'output': {
                'content': 'A',
                'content_type': 'text',
                'sources': [{'url': 'https://x.com', 'title': ''}],
            }
        }
        toolset, client = self._toolset_with_payload(payload)
        try:
            result = await toolset.research('q')
            assert 'title' not in result['sources'][0]
        finally:
            await client.aclose()

    async def test_source_empty_snippets_not_included(self) -> None:
        payload: dict[str, object] = {
            'output': {
                'content': 'A',
                'content_type': 'text',
                'sources': [{'url': 'https://x.com', 'snippets': []}],
            }
        }
        toolset, client = self._toolset_with_payload(payload)
        try:
            result = await toolset.research('q')
            assert 'snippets' not in result['sources'][0]
        finally:
            await client.aclose()

    async def test_minimal_response(self) -> None:
        toolset, client = self._toolset_with_payload(_make_research_minimal_payload())
        try:
            result = await toolset.research('q')
            assert result['content'] == 'Short answer.'
            assert len(result['sources']) == 1
            assert result['sources'][0]['url'] == 'https://src.com'
            assert 'title' not in result['sources'][0]
            assert 'snippets' not in result['sources'][0]
        finally:
            await client.aclose()

    async def test_empty_sources(self) -> None:
        toolset, client = self._toolset_with_payload(_make_research_empty_payload())
        try:
            result = await toolset.research('q')
            assert result['sources'] == []
        finally:
            await client.aclose()

    async def test_structured_output(self) -> None:
        """Structured output returns content as a dict with content_type 'object'."""
        toolset, client = self._toolset_with_payload(_make_research_structured_payload())
        try:
            result = await toolset.research('q')
            assert result['content_type'] == 'object'
            assert result['content'] == {'answer': 'The sky is blue.', 'confidence': 0.95}
            assert len(result['sources']) == 1
            assert result['sources'][0]['url'] == 'https://source.com'
        finally:
            await client.aclose()

    async def test_malformed_payload_raises(self) -> None:
        """Missing output key raises ValidationError."""
        toolset, client = self._toolset_with_payload(_make_malformed_research_payload())
        try:
            with pytest.raises(ValidationError):
                await toolset.research('q')
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Search: parameter building tests
# ---------------------------------------------------------------------------


class TestBuildSearchParams:
    def test_query_always_present(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        params = toolset._build_search_params(
            query='hello',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['query'] == 'hello'
        assert len(params) == 1

    def test_configured_count_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', count=5)
        params = toolset._build_search_params(
            query='q',
            count=10,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['count'] == 5

    def test_llm_count_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        params = toolset._build_search_params(
            query='q',
            count=10,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['count'] == 10

    def test_offset_always_included(self) -> None:
        toolset = YoudotcomToolset(api_key='test', offset=20)
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['offset'] == 20

    def test_offset_never_from_llm(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert 'offset' not in params

    def test_configured_freshness_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', freshness='day')
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness='week',
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['freshness'] == 'day'

    def test_llm_freshness_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness='week',
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['freshness'] == 'week'

    def test_configured_country_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', country='US')
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country='GB',
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['country'] == 'US'

    def test_llm_country_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country='GB',
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['country'] == 'GB'

    def test_configured_language_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', language='EN')
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language='FR',
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['language'] == 'EN'

    def test_configured_safesearch_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', safesearch='strict')
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch='off',
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['safesearch'] == 'strict'

    def test_configured_livecrawl_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', livecrawl='all')
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl='web',
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['livecrawl'] == 'all'

    def test_configured_livecrawl_formats_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', livecrawl_formats=['html'])
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=['markdown'],
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['livecrawl_formats'] == ['html']

    def test_all_params_configured(self) -> None:
        toolset = YoudotcomToolset(
            api_key='test',
            count=5,
            offset=10,
            freshness='day',
            country='US',
            language='EN',
            safesearch='strict',
            livecrawl='all',
            livecrawl_formats=['markdown'],
            include_domains=['nytimes.com'],
            exclude_domains=['spam.com'],
            boost_domains=['good.com'],
            search_crawl_timeout=30,
        )
        params = toolset._build_search_params(
            query='q',
            count=100,
            freshness='year',
            country='GB',
            language='FR',
            safesearch='off',
            livecrawl='web',
            livecrawl_formats=['html'],
            include_domains=['bbc.com'],
            exclude_domains=['other.com'],
            boost_domains=['bad.com'],
            crawl_timeout=10,
        )
        assert params == {
            'query': 'q',
            'count': 5,
            'offset': 10,
            'freshness': 'day',
            'country': 'US',
            'language': 'EN',
            'safesearch': 'strict',
            'livecrawl': 'all',
            'livecrawl_formats': ['markdown'],
            'include_domains': 'nytimes.com',
            'exclude_domains': 'spam.com',
            'boost_domains': 'good.com',
            'crawl_timeout': 30,
        }

    def test_freshness_date_range_string(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness='2025-01-01to2025-06-01',
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['freshness'] == '2025-01-01to2025-06-01'

    def test_configured_include_domains_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', include_domains=['nytimes.com'])
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=['bbc.com'],
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['include_domains'] == 'nytimes.com'

    def test_llm_include_domains_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=['bbc.com', 'reuters.com'],
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['include_domains'] == 'bbc.com,reuters.com'

    def test_configured_exclude_domains_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', exclude_domains=['spam.com'])
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=['other.com'],
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['exclude_domains'] == 'spam.com'

    def test_llm_exclude_domains_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=['spam.com', 'ads.com'],
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['exclude_domains'] == 'spam.com,ads.com'

    def test_configured_boost_domains_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', boost_domains=['good.com'])
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=['other.com'],
            crawl_timeout=None,
        )
        assert params['boost_domains'] == 'good.com'

    def test_llm_boost_domains_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=['good.com', 'great.com'],
            crawl_timeout=None,
        )
        assert params['boost_domains'] == 'good.com,great.com'

    def test_no_domain_filters_when_none(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert 'include_domains' not in params
        assert 'exclude_domains' not in params
        assert 'boost_domains' not in params

    def test_configured_search_crawl_timeout_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', search_crawl_timeout=30)
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=10,
        )
        assert params['crawl_timeout'] == 30

    def test_llm_crawl_timeout_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=15,
        )
        assert params['crawl_timeout'] == 15

    def test_no_crawl_timeout_when_none(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert 'crawl_timeout' not in params

    def test_llm_livecrawl_formats_both(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=['html', 'markdown'],
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['livecrawl_formats'] == ['html', 'markdown']


# ---------------------------------------------------------------------------
# Contents: parameter building tests
# ---------------------------------------------------------------------------


class TestBuildContentsBody:
    def test_urls_always_present(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_contents_body(urls=['https://x.com'], formats=None, crawl_timeout=None)
        assert body['urls'] == ['https://x.com']
        assert len(body) == 1

    def test_configured_formats_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', contents_formats=['markdown'])
        body = toolset._build_contents_body(urls=['https://x.com'], formats=['html', 'metadata'], crawl_timeout=None)
        assert body['formats'] == ['markdown']

    def test_llm_formats_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_contents_body(urls=['https://x.com'], formats=['html'], crawl_timeout=None)
        assert body['formats'] == ['html']

    def test_configured_crawl_timeout_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', crawl_timeout=30)
        body = toolset._build_contents_body(urls=['https://x.com'], formats=None, crawl_timeout=10)
        assert body['crawl_timeout'] == 30

    def test_llm_crawl_timeout_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_contents_body(urls=['https://x.com'], formats=None, crawl_timeout=15)
        assert body['crawl_timeout'] == 15

    def test_max_age_always_included_when_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test', max_age=86400)
        body = toolset._build_contents_body(urls=['https://x.com'], formats=None, crawl_timeout=None)
        assert body['max_age'] == 86400

    def test_max_age_never_from_llm(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_contents_body(urls=['https://x.com'], formats=None, crawl_timeout=None)
        assert 'max_age' not in body

    def test_all_params_configured(self) -> None:
        toolset = YoudotcomToolset(
            api_key='test',
            contents_formats=['html', 'markdown'],
            crawl_timeout=20,
            max_age=3600,
        )
        body = toolset._build_contents_body(urls=['https://x.com'], formats=['metadata'], crawl_timeout=5)
        assert body == {
            'urls': ['https://x.com'],
            'formats': ['html', 'markdown'],
            'crawl_timeout': 20,
            'max_age': 3600,
        }


# ---------------------------------------------------------------------------
# Contents: integration tests
# ---------------------------------------------------------------------------


class TestContentsIntegration:
    async def test_contents_returns_results(self) -> None:
        """End-to-end contents extraction with a mock transport."""
        payload = _make_contents_payload()

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == 'POST'
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            results = await toolset.extract_contents(['https://example.com/page'])
            assert len(results) == 1
            assert results[0]['url'] == 'https://example.com/page'
            assert results[0]['title'] == 'Example Page'
            assert results[0].get('markdown') == '# Example\n\nHello world.'
        finally:
            await client.aclose()

    async def test_contents_with_configured_formats(self) -> None:
        """Configured formats are sent, not LLM-provided ones."""
        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_contents_minimal_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client, contents_formats=['markdown'])
        try:
            await toolset.extract_contents(['https://x.com'], formats=['html'])
            assert captured_body['formats'] == ['markdown']
        finally:
            await client.aclose()

    async def test_contents_partial_failure(self) -> None:
        """URLs that fail to crawl return with no content fields."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=_make_contents_partial_payload()))
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            results = await toolset.extract_contents(['https://ok.com', 'https://fail.com'])
            assert len(results) == 2
            assert results[0].get('markdown') == 'content'
            assert 'markdown' not in results[1]
            assert 'html' not in results[1]
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Research: parameter building tests
# ---------------------------------------------------------------------------


class TestBuildResearchBody:
    def test_input_always_present(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='question',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert body['input'] == 'question'
        assert len(body) == 1

    def test_configured_effort_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', research_effort='deep')
        body = toolset._build_research_body(
            input='q',
            research_effort='lite',
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert body['research_effort'] == 'deep'

    def test_llm_effort_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort='exhaustive',
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert body['research_effort'] == 'exhaustive'

    def test_source_control_not_included_when_none(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert 'source_control' not in body

    def test_configured_research_include_domains_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', research_include_domains=['arxiv.org'])
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=['bbc.com'],
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert body['source_control'] == {'include_domains': ['arxiv.org']}

    def test_llm_research_include_domains_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=['arxiv.org', 'nature.com'],
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert body['source_control'] == {'include_domains': ['arxiv.org', 'nature.com']}

    def test_configured_research_exclude_domains_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', research_exclude_domains=['spam.com'])
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=['other.com'],
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert body['source_control'] == {'exclude_domains': ['spam.com']}

    def test_llm_research_exclude_domains_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=['spam.com'],
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert body['source_control'] == {'exclude_domains': ['spam.com']}

    def test_configured_research_boost_domains_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', research_boost_domains=['good.com'])
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=['other.com'],
            freshness=None,
            country=None,
        )
        assert body['source_control'] == {'boost_domains': ['good.com']}

    def test_llm_research_boost_domains_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=['good.com'],
            freshness=None,
            country=None,
        )
        assert body['source_control'] == {'boost_domains': ['good.com']}

    def test_configured_research_freshness_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', research_freshness='day')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness='week',
            country=None,
        )
        assert body['source_control'] == {'freshness': 'day'}

    def test_llm_research_freshness_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness='month',
            country=None,
        )
        assert body['source_control'] == {'freshness': 'month'}

    def test_configured_research_country_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', research_country='US')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country='GB',
        )
        assert body['source_control'] == {'country': 'US'}

    def test_llm_research_country_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country='GB',
        )
        assert body['source_control'] == {'country': 'GB'}

    def test_source_control_only_includes_set_fields(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness='day',
            country='US',
        )
        assert body['source_control'] == {'freshness': 'day', 'country': 'US'}

    def test_output_schema_included_when_configured(self) -> None:
        schema: dict[str, object] = {'type': 'object', 'properties': {'answer': {'type': 'string'}}}
        toolset = YoudotcomToolset(api_key='test', output_schema=schema)
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert body['output_schema'] == schema

    def test_output_schema_not_included_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert 'output_schema' not in body


# ---------------------------------------------------------------------------
# Research: integration tests
# ---------------------------------------------------------------------------


class TestResearchIntegration:
    async def test_research_returns_result(self) -> None:
        """End-to-end research with a mock transport."""
        payload = _make_research_payload()

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == 'POST'
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            result = await toolset.research('What happened?')
            assert result['content'] == '## Answer\n\nSomething happened [[1, 2]].'
            assert result['content_type'] == 'text'
            assert len(result['sources']) == 2
        finally:
            await client.aclose()

    async def test_research_with_configured_effort(self) -> None:
        """Configured research_effort is sent, not LLM-provided."""
        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_research_empty_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client, research_effort='deep')
        try:
            await toolset.research('q', research_effort='lite')
            assert captured_body['research_effort'] == 'deep'
        finally:
            await client.aclose()

    async def test_research_with_configured_source_control(self) -> None:
        """Configured source_control fields are sent in the request body."""
        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_research_empty_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(
            api_key='test',
            http_client=client,
            research_include_domains=['arxiv.org'],
            research_freshness='day',
            research_country='US',
        )
        try:
            await toolset.research(
                'q',
                include_domains=['bbc.com'],
                freshness='week',
                country='GB',
            )
            assert captured_body['source_control'] == {
                'include_domains': ['arxiv.org'],
                'freshness': 'day',
                'country': 'US',
            }
        finally:
            await client.aclose()

    async def test_research_source_control_from_llm(self) -> None:
        """LLM-provided source_control fields are sent when not configured."""
        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_research_empty_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.research(
                'q',
                include_domains=['arxiv.org'],
                exclude_domains=['spam.com'],
                freshness='day',
                country='US',
            )
            assert captured_body['source_control'] == {
                'include_domains': ['arxiv.org'],
                'exclude_domains': ['spam.com'],
                'freshness': 'day',
                'country': 'US',
            }
        finally:
            await client.aclose()

    async def test_research_with_configured_output_schema(self) -> None:
        """Configured output_schema is included in the request body."""
        captured_body: dict[str, object] = {}
        schema: dict[str, object] = {'type': 'object', 'properties': {'answer': {'type': 'string'}}}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_research_structured_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client, output_schema=schema)
        try:
            await toolset.research('q')
            assert captured_body['output_schema'] == schema
        finally:
            await client.aclose()

    async def test_research_structured_output(self) -> None:
        """Research with structured output returns content as a dict with content_type 'object'."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=_make_research_structured_payload()))
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            result = await toolset.research('q')
            assert result['content_type'] == 'object'
            assert result['content'] == {'answer': 'The sky is blue.', 'confidence': 0.95}
            assert len(result['sources']) == 1
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Finance research: parameter building tests
# ---------------------------------------------------------------------------


class TestBuildFinanceResearchBody:
    def test_input_always_present(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_finance_research_body(input='question', research_effort=None)
        assert body['input'] == 'question'
        assert len(body) == 1

    def test_configured_effort_locks(self) -> None:
        toolset = YoudotcomToolset(api_key='test', finance_research_effort='exhaustive')
        body = toolset._build_finance_research_body(input='q', research_effort='deep')
        assert body['research_effort'] == 'exhaustive'

    def test_llm_effort_when_not_configured(self) -> None:
        toolset = YoudotcomToolset(api_key='test')
        body = toolset._build_finance_research_body(input='q', research_effort='exhaustive')
        assert body['research_effort'] == 'exhaustive'


# ---------------------------------------------------------------------------
# Finance research: integration tests
# ---------------------------------------------------------------------------


class TestFinanceResearchIntegration:
    async def test_finance_research_returns_result(self) -> None:
        """End-to-end finance research with a mock transport."""
        payload = _make_finance_research_payload()

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == 'POST'
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            result = await toolset.finance_research('NVDA revenue growth')
            assert result['content'] == 'Revenue grew 114% [[1]].'
            assert result['content_type'] == 'text'
            assert len(result['sources']) == 1
            assert result['sources'][0]['url'] == 'https://sec.gov/filing'
        finally:
            await client.aclose()

    async def test_finance_research_with_configured_effort(self) -> None:
        """Configured finance_research_effort is sent, not LLM-provided."""
        captured_body: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured_body.update(json.loads(request.content))
            return httpx.Response(200, json=_make_research_empty_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client, finance_research_effort='exhaustive')
        try:
            await toolset.finance_research('q', research_effort='deep')
            assert captured_body['research_effort'] == 'exhaustive'
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# HTTP helper tests
# ---------------------------------------------------------------------------


class TestNormalizeParam:
    def test_none(self) -> None:
        assert YoudotcomToolset._normalize_param(None) is None

    def test_str(self) -> None:
        assert YoudotcomToolset._normalize_param('hello') == 'hello'

    def test_int(self) -> None:
        assert YoudotcomToolset._normalize_param(42) == '42'


class TestHttpGet:
    async def test_with_custom_client(self) -> None:
        """_get uses the provided http_client."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=_make_web_payload()))
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test-key', http_client=client)
        try:
            response = await toolset._get('https://api.you.com/v1/search', {'query': 'test'})
            assert response.status_code == 200
            data = response.json()
            assert 'results' in data
        finally:
            await client.aclose()

    async def test_raises_on_http_error(self) -> None:
        """_get raises for non-2xx responses."""
        transport = httpx.MockTransport(lambda req: httpx.Response(403, json={'error': 'forbidden'}))
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='bad-key', http_client=client)
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await toolset._get('https://api.you.com/v1/search', {'query': 'test'})
        finally:
            await client.aclose()

    async def test_without_custom_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_get creates a new httpx.AsyncClient when no http_client is provided."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=_make_web_payload()))
        real_async_client = httpx.AsyncClient

        class _MockAsyncClient(real_async_client):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(transport=transport)  # type: ignore[arg-type]

        monkeypatch.setattr(httpx, 'AsyncClient', _MockAsyncClient)
        toolset = YoudotcomToolset(api_key='test')
        response = await toolset._get('https://api.you.com/v1/search', {'query': 'test'})
        assert response.status_code == 200


class TestHttpPost:
    async def test_with_custom_client(self) -> None:
        """_post uses the provided http_client."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=_make_research_payload()))
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test-key', http_client=client)
        try:
            response = await toolset._post('https://api.you.com/v1/research', {'input': 'test'})
            assert response.status_code == 200
        finally:
            await client.aclose()

    async def test_raises_on_http_error(self) -> None:
        """_post raises for non-2xx responses."""
        transport = httpx.MockTransport(lambda req: httpx.Response(422, json={'error': 'invalid'}))
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='bad-key', http_client=client)
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await toolset._post('https://api.you.com/v1/research', {'input': 'test'})
        finally:
            await client.aclose()

    async def test_without_custom_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_post creates a new httpx.AsyncClient when no http_client is provided."""
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=_make_research_empty_payload()))
        real_async_client = httpx.AsyncClient

        class _MockAsyncClient(real_async_client):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(transport=transport)  # type: ignore[arg-type]

        monkeypatch.setattr(httpx, 'AsyncClient', _MockAsyncClient)
        toolset = YoudotcomToolset(api_key='test')
        response = await toolset._post('https://api.you.com/v1/research', {'input': 'test'})
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Search: integration tests
# ---------------------------------------------------------------------------


class TestSearchIntegration:
    async def test_search_returns_results(self) -> None:
        """End-to-end search through the tool method with a mock transport."""
        payload = _make_web_payload()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            results = await toolset.search('example query')
            assert len(results) == 1
            assert results[0]['title'] == 'Example Page'
        finally:
            await client.aclose()

    async def test_search_with_configured_params(self) -> None:
        """Configured params are sent in the request, not LLM-provided ones."""
        captured_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            for key, value in request.url.params.multi_items():
                captured_params[key] = value
            return httpx.Response(200, json=_make_empty_search_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(
            api_key='test',
            http_client=client,
            count=5,
            freshness='day',
            country='US',
        )
        try:
            await toolset.search('q', count=100, freshness='year', country='GB')
            assert captured_params['count'] == '5'
            assert captured_params['freshness'] == 'day'
            assert captured_params['country'] == 'US'
        finally:
            await client.aclose()

    async def test_search_with_configured_domains(self) -> None:
        """Configured domain filters are sent as comma-separated strings."""
        captured_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            for key, value in request.url.params.multi_items():
                captured_params[key] = value
            return httpx.Response(200, json=_make_empty_search_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(
            api_key='test',
            http_client=client,
            include_domains=['nytimes.com', 'bbc.com'],
            exclude_domains=['spam.com'],
        )
        try:
            await toolset.search('q', include_domains=['bbc.com'], exclude_domains=['other.com'])
            assert captured_params['include_domains'] == 'nytimes.com,bbc.com'
            assert captured_params['exclude_domains'] == 'spam.com'
        finally:
            await client.aclose()

    async def test_search_with_llm_domains(self) -> None:
        """LLM-provided domain filters are sent as comma-separated strings."""
        captured_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            for key, value in request.url.params.multi_items():
                captured_params[key] = value
            return httpx.Response(200, json=_make_empty_search_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.search('q', include_domains=['nytimes.com', 'bbc.com'])
            assert captured_params['include_domains'] == 'nytimes.com,bbc.com'
        finally:
            await client.aclose()

    async def test_search_with_livecrawl_formats_list(self) -> None:
        """livecrawl_formats list is sent as repeated query params."""
        captured_formats: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_formats.extend(v for k, v in request.url.params.multi_items() if k == 'livecrawl_formats')
            return httpx.Response(200, json=_make_empty_search_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client)
        try:
            await toolset.search('q', livecrawl_formats=['html', 'markdown'])
            assert captured_formats == ['html', 'markdown']
        finally:
            await client.aclose()

    async def test_search_with_configured_crawl_timeout(self) -> None:
        """Configured search crawl_timeout is sent, not LLM-provided."""
        captured_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            for key, value in request.url.params.multi_items():
                captured_params[key] = value
            return httpx.Response(200, json=_make_empty_search_payload())

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(transport=transport)
        toolset = YoudotcomToolset(api_key='test', http_client=client, search_crawl_timeout=30)
        try:
            await toolset.search('q', crawl_timeout=10)
            assert captured_params['crawl_timeout'] == '30'
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Capability tests
# ---------------------------------------------------------------------------


class TestCapability:
    def test_get_toolset_returns_youdotcom_toolset(self) -> None:
        cap = Youdotcom(api_key='test')
        toolset = cap.get_toolset()
        assert isinstance(toolset, YoudotcomToolset)

    def test_capability_passes_search_params(self) -> None:
        cap = Youdotcom(
            api_key='k',
            count=3,
            offset=6,
            freshness='week',
            country='GB',
            language='EN',
            safesearch='moderate',
            livecrawl='web',
            livecrawl_formats=['html'],
            include_domains=['nytimes.com'],
            exclude_domains=['spam.com'],
            boost_domains=['good.com'],
            search_crawl_timeout=30,
        )
        toolset = cap.get_toolset()
        params = toolset._build_search_params(
            query='q',
            count=None,
            freshness=None,
            country=None,
            language=None,
            safesearch=None,
            livecrawl=None,
            livecrawl_formats=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            crawl_timeout=None,
        )
        assert params['count'] == 3
        assert params['offset'] == 6
        assert params['freshness'] == 'week'
        assert params['country'] == 'GB'
        assert params['language'] == 'EN'
        assert params['safesearch'] == 'moderate'
        assert params['livecrawl'] == 'web'
        assert params['livecrawl_formats'] == ['html']
        assert params['include_domains'] == 'nytimes.com'
        assert params['exclude_domains'] == 'spam.com'
        assert params['boost_domains'] == 'good.com'
        assert params['crawl_timeout'] == 30

    def test_capability_passes_contents_params(self) -> None:
        cap = Youdotcom(
            api_key='k',
            contents_formats=['markdown', 'metadata'],
            crawl_timeout=15,
            max_age=3600,
        )
        toolset = cap.get_toolset()
        body = toolset._build_contents_body(urls=['https://x.com'], formats=None, crawl_timeout=None)
        assert body['formats'] == ['markdown', 'metadata']
        assert body['crawl_timeout'] == 15
        assert body['max_age'] == 3600

    def test_capability_passes_research_params(self) -> None:
        cap = Youdotcom(
            api_key='k',
            research_effort='deep',
            research_include_domains=['arxiv.org'],
            research_exclude_domains=['spam.com'],
            research_boost_domains=['good.com'],
            research_freshness='week',
            research_country='US',
            output_schema={'type': 'object', 'properties': {}},
        )
        toolset = cap.get_toolset()
        body = toolset._build_research_body(
            input='q',
            research_effort=None,
            include_domains=None,
            exclude_domains=None,
            boost_domains=None,
            freshness=None,
            country=None,
        )
        assert body['research_effort'] == 'deep'
        assert body['source_control'] == {
            'include_domains': ['arxiv.org'],
            'exclude_domains': ['spam.com'],
            'boost_domains': ['good.com'],
            'freshness': 'week',
            'country': 'US',
        }
        assert body['output_schema'] == {'type': 'object', 'properties': {}}

    def test_capability_passes_finance_research_params(self) -> None:
        cap = Youdotcom(api_key='k', finance_research_effort='exhaustive')
        toolset = cap.get_toolset()
        body = toolset._build_finance_research_body(input='q', research_effort=None)
        assert body['research_effort'] == 'exhaustive'

    def test_capability_registers_all_four_tools(self) -> None:
        cap = Youdotcom(api_key='test')
        toolset = cap.get_toolset()
        assert 'you_search' in toolset.tools
        assert 'you_contents' in toolset.tools
        assert 'you_research' in toolset.tools
        assert 'you_finance_research' in toolset.tools

    def test_capability_with_agent(self) -> None:
        """The capability registers tools with an Agent."""
        cap = Youdotcom(api_key='test')
        agent: Agent[None, str] = Agent(
            TestModel(custom_output_text='done', call_tools=[]),
            capabilities=[cap],
            name='test-agent',
        )
        toolset = cap.get_toolset()
        assert 'you_search' in toolset.tools
        result = agent.run_sync('Search for test')
        assert result.output == 'done'
