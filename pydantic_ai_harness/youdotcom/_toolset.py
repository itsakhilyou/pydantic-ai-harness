"""You.com toolset providing web search, content extraction, and research via the You.com API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import httpx
from pydantic import BaseModel, TypeAdapter
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from typing_extensions import NotRequired, TypedDict

__all__ = (
    'ContentsFormat',
    'Country',
    'FinanceResearchEffort',
    'Freshness',
    'Language',
    'LiveCrawl',
    'LiveCrawlFormats',
    'ResearchEffort',
    'SafeSearch',
    'YouContentsMetadata',
    'YouContentsResult',
    'YouLivecrawlContents',
    'YouResearchResult',
    'YouResearchSource',
    'YouSearchResult',
    'YoudotcomToolset',
)

_YOU_SEARCH_URL = 'https://api.you.com/v1/search'
_YOU_CONTENTS_URL = 'https://ydc-index.io/v1/contents'
_YOU_RESEARCH_URL = 'https://api.you.com/v1/research'
_YOU_FINANCE_RESEARCH_URL = 'https://api.you.com/v1/finance_research'

Country = Literal[
    'AR',
    'AU',
    'AT',
    'BE',
    'BR',
    'CA',
    'CL',
    'DK',
    'FI',
    'FR',
    'DE',
    'HK',
    'IN',
    'ID',
    'IT',
    'JP',
    'KR',
    'MY',
    'MX',
    'NL',
    'NZ',
    'NO',
    'CN',
    'PL',
    'PT',
    'PH',
    'RU',
    'SA',
    'ZA',
    'ES',
    'SE',
    'CH',
    'TW',
    'TR',
    'GB',
    'US',
]
"""ISO 3166-1 alpha-2 country codes for geographic focus of search results."""

Language = Literal[
    'AR',
    'EU',
    'BN',
    'BG',
    'CA',
    'ZH-HANS',
    'ZH-HANT',
    'HR',
    'CS',
    'DA',
    'NL',
    'EN',
    'EN-GB',
    'ET',
    'FI',
    'FR',
    'GL',
    'DE',
    'EL',
    'GU',
    'HE',
    'HI',
    'HU',
    'IS',
    'IT',
    'JP',
    'KN',
    'KO',
    'LV',
    'LT',
    'MS',
    'ML',
    'MR',
    'NB',
    'PL',
    'PT-BR',
    'PT-PT',
    'PA',
    'RO',
    'RU',
    'SR',
    'SK',
    'SL',
    'ES',
    'SV',
    'TA',
    'TE',
    'TH',
    'TR',
    'UK',
    'VI',
]
"""BCP 47 language codes for search results."""

Freshness = Literal['day', 'week', 'month', 'year'] | str
"""Result freshness filter: a named interval or a date range ``YYYY-MM-DDtoYYYY-MM-DD``."""

SafeSearch = Literal['off', 'moderate', 'strict']
"""Content moderation level for search results."""

LiveCrawl = Literal['web', 'news', 'all']
"""Which sections to livecrawl for full page content."""

LiveCrawlFormats = Literal['html', 'markdown']
"""Format for livecrawled content."""

ContentsFormat = Literal['html', 'markdown', 'metadata']
"""Format for content extraction via the Contents API."""

ResearchEffort = Literal['lite', 'standard', 'deep', 'exhaustive']
"""Research depth level for the Research API."""

FinanceResearchEffort = Literal['deep', 'exhaustive']
"""Research depth level for the Finance Research API."""


class YouLivecrawlContents(TypedDict, total=False):
    """Contents of a page when livecrawl is enabled in search."""

    html: str
    """The HTML content of the page."""

    markdown: str
    """The Markdown content of the page."""


class YouSearchResult(TypedDict):
    """A single You.com search result.

    `title` and `url` are always present. All other fields are optional and
    depend on the API response and livecrawl settings.
    """

    title: str
    """The title of the search result."""

    url: str
    """The URL of the search result."""

    description: NotRequired[str]
    """A description or snippet of the search result."""

    snippets: NotRequired[list[str]]
    """Text snippets from the search result, providing a preview of the content."""

    thumbnail_url: NotRequired[str]
    """URL of the thumbnail image for the search result."""

    page_age: NotRequired[datetime]
    """The age or publication date of the search result (ISO 8601 format)."""

    favicon_url: NotRequired[str]
    """The URL of the favicon of the search result's domain."""

    contents: NotRequired[YouLivecrawlContents]
    """Contents of the page if livecrawl was enabled."""

    authors: NotRequired[list[str]]
    """An array of authors of the search result."""


class YouContentsMetadata(TypedDict, total=False):
    """Metadata about a web page from the Contents API."""

    site_name: str
    """The OpenGraph site name of the web page."""

    favicon_url: str
    """The URL of the favicon of the web page's domain."""


class YouContentsResult(TypedDict):
    """A single result from the Contents API.

    `url` and `title` are always present. Other fields depend on the requested
    formats and what the API could extract.
    """

    url: str
    """The webpage URL whose content was fetched."""

    title: str
    """The title of the web page."""

    html: NotRequired[str]
    """The HTML content of the web page, if requested and available."""

    markdown: NotRequired[str]
    """The Markdown content of the web page, if requested and available."""

    metadata: NotRequired[YouContentsMetadata]
    """Metadata about the web page, if requested."""


class YouResearchSource(TypedDict):
    """A source cited in a research result."""

    url: str
    """The URL of the source webpage."""

    title: NotRequired[str]
    """The title of the source webpage."""

    snippets: NotRequired[list[str]]
    """Relevant excerpts from the source page used in generating the answer."""


class YouResearchResult(TypedDict):
    """A result from the Research or Finance Research API."""

    content: str
    """The comprehensive response with inline citations (Markdown)."""

    content_type: str
    """The format of the content field ('text')."""

    sources: list[YouResearchSource]
    """Web sources used to generate the answer."""


# ---------------------------------------------------------------------------
# Internal parsing models -- search
# ---------------------------------------------------------------------------


class _RawLivecrawlContents(BaseModel):
    """Raw livecrawl contents field from the search API response."""

    html: str | None = None
    markdown: str | None = None

    def to_contents(self) -> YouLivecrawlContents | None:
        """Build a `YouLivecrawlContents` if either field is present."""
        if not self.html and not self.markdown:
            return None
        contents: YouLivecrawlContents = {}
        if self.html:
            contents['html'] = self.html
        if self.markdown:
            contents['markdown'] = self.markdown
        return contents


class _RawSearchResult(BaseModel):
    """Shared fields present in both web and news search results."""

    title: str
    url: str
    description: str | None = None
    thumbnail_url: str | None = None
    page_age: datetime | None = None
    contents: _RawLivecrawlContents | None = None

    def to_result(self) -> YouSearchResult:
        """Convert to a public `YouSearchResult` TypedDict."""
        result: YouSearchResult = {'title': self.title, 'url': self.url}
        if self.description:
            result['description'] = self.description
        if self.thumbnail_url:
            result['thumbnail_url'] = self.thumbnail_url
        if self.page_age is not None:
            result['page_age'] = self.page_age
        if self.contents is not None:
            contents = self.contents.to_contents()
            if contents is not None:
                result['contents'] = contents
        return result


class _RawWebResult(_RawSearchResult):
    """Web result with additional fields not present in news results."""

    snippets: list[str] | None = None
    favicon_url: str | None = None
    authors: list[str] | None = None

    def to_result(self) -> YouSearchResult:
        """Convert to a public `YouSearchResult`, adding web-only fields."""
        result = super().to_result()
        if self.snippets:
            result['snippets'] = self.snippets
        if self.favicon_url:
            result['favicon_url'] = self.favicon_url
        if self.authors:
            result['authors'] = self.authors
        return result


class _RawSearchResults(BaseModel):
    """The `results` object from the You.com search API response."""

    web: list[_RawWebResult] | None = None
    news: list[_RawSearchResult] | None = None


class _RawSearchResponse(BaseModel):
    """Top-level You.com search API response."""

    results: _RawSearchResults = _RawSearchResults()


# ---------------------------------------------------------------------------
# Internal parsing models -- contents
# ---------------------------------------------------------------------------


class _RawContentsMetadata(BaseModel):
    """Metadata field from the Contents API response."""

    site_name: str | None = None
    favicon_url: str | None = None

    def to_metadata(self) -> YouContentsMetadata | None:
        """Build a `YouContentsMetadata` if any field is present."""
        if not self.site_name and not self.favicon_url:
            return None
        metadata: YouContentsMetadata = {}
        if self.site_name:
            metadata['site_name'] = self.site_name
        if self.favicon_url:
            metadata['favicon_url'] = self.favicon_url
        return metadata


class _RawContentsItem(BaseModel):
    """A single item from the Contents API response array."""

    url: str
    title: str
    html: str | None = None
    markdown: str | None = None
    metadata: _RawContentsMetadata | None = None

    def to_result(self) -> YouContentsResult:
        """Convert to a public `YouContentsResult` TypedDict."""
        result: YouContentsResult = {'url': self.url, 'title': self.title}
        if self.html:
            result['html'] = self.html
        if self.markdown:
            result['markdown'] = self.markdown
        if self.metadata is not None:
            metadata = self.metadata.to_metadata()
            if metadata is not None:
                result['metadata'] = metadata
        return result


_ContentsResponseAdapter: TypeAdapter[list[_RawContentsItem]] = TypeAdapter(list[_RawContentsItem])


# ---------------------------------------------------------------------------
# Internal parsing models -- research / finance research
# ---------------------------------------------------------------------------


class _RawResearchSource(BaseModel):
    """A source cited in a research response."""

    url: str
    title: str | None = None
    snippets: list[str] | None = None

    def to_source(self) -> YouResearchSource:
        """Convert to a public `YouResearchSource` TypedDict."""
        source: YouResearchSource = {'url': self.url}
        if self.title:
            source['title'] = self.title
        if self.snippets:
            source['snippets'] = self.snippets
        return source


class _RawResearchOutput(BaseModel):
    """The `output` object from a research API response."""

    content: str
    content_type: str
    sources: list[_RawResearchSource] = []


class _RawResearchResponse(BaseModel):
    """Top-level research or finance research API response."""

    output: _RawResearchOutput = _RawResearchOutput(content='', content_type='text')


class YoudotcomToolset(FunctionToolset[AgentDepsT]):
    """Toolset exposing You.com web search, content extraction, and research tools.

    Provides four tools backed by the You.com API:

    - `you_search`: Web and news search with configurable filters.
    - `you_contents`: Extract clean HTML or Markdown from known URLs.
    - `you_research`: Deep research with cited, synthesized answers.
    - `you_finance_research`: Finance-focused research with cited answers.

    Configured parameters (set at construction time) are locked: the LLM cannot
    override them. Unconfigured parameters are exposed to the LLM with sensible
    defaults. `offset` and `max_age` are never exposed to the LLM -- they are
    always human-controlled.
    """

    def __init__(
        self,
        *,
        api_key: str,
        http_client: httpx.AsyncClient | None = None,
        # Search params
        count: int | None = None,
        offset: int | None = None,
        freshness: Freshness | None = None,
        country: Country | None = None,
        language: Language | None = None,
        safesearch: SafeSearch | None = None,
        livecrawl: LiveCrawl | None = None,
        livecrawl_formats: LiveCrawlFormats | None = None,
        # Contents params
        contents_formats: list[ContentsFormat] | None = None,
        crawl_timeout: int | None = None,
        max_age: int | None = None,
        # Research params
        research_effort: ResearchEffort | None = None,
        # Finance research params
        finance_research_effort: FinanceResearchEffort | None = None,
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self._http_client = http_client
        # Search
        self._count = count
        self._offset = offset
        self._freshness = freshness
        self._country = country
        self._language = language
        self._safesearch = safesearch
        self._livecrawl = livecrawl
        self._livecrawl_formats = livecrawl_formats
        # Contents
        self._contents_formats = contents_formats
        self._crawl_timeout = crawl_timeout
        self._max_age = max_age
        # Research
        self._research_effort = research_effort
        # Finance research
        self._finance_research_effort = finance_research_effort

        self.add_function(self.search, name='you_search')
        self.add_function(self.extract_contents, name='you_contents')
        self.add_function(self.research, name='you_research')
        self.add_function(self.finance_research, name='you_finance_research')

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        count: int | None = None,
        freshness: Freshness | None = None,
        country: str | None = None,
        language: str | None = None,
        safesearch: SafeSearch | None = None,
        livecrawl: LiveCrawl | None = None,
        livecrawl_formats: LiveCrawlFormats | None = None,
    ) -> list[YouSearchResult]:
        """Search the web and news using the You.com Search API.

        Args:
            query: The search query to execute.
            count: Maximum number of results per section (1-100). Only used if not
                configured at tool creation.
            freshness: Result freshness: 'day', 'week', 'month', 'year', or
                'YYYY-MM-DDtoYYYY-MM-DD'. Only used if not configured at tool creation.
            country: ISO 3166-1 alpha-2 country code (e.g. 'US', 'GB'). Only used if
                not configured at tool creation.
            language: BCP 47 language code (e.g. 'EN', 'FR'). Only used if not
                configured at tool creation.
            safesearch: Content moderation: 'off', 'moderate', or 'strict'. Only used
                if not configured at tool creation.
            livecrawl: Sections to livecrawl: 'web', 'news', or 'all'. Only used if
                not configured at tool creation.
            livecrawl_formats: Format for livecrawled content: 'html' or 'markdown'.
                Only used if not configured at tool creation.
        """
        params = self._build_search_params(
            query=query,
            count=count,
            freshness=freshness,
            country=country,
            language=language,
            safesearch=safesearch,
            livecrawl=livecrawl,
            livecrawl_formats=livecrawl_formats,
        )
        response = await self._get(_YOU_SEARCH_URL, params)
        return self._parse_search_results(_RawSearchResponse.model_validate(response.json()))

    async def extract_contents(
        self,
        urls: list[str],
        *,
        formats: list[ContentsFormat] | None = None,
        crawl_timeout: int | None = None,
    ) -> list[YouContentsResult]:
        """Extract clean HTML or Markdown content from web pages.

        Pass a list of URLs to fetch full page content for each. The API crawls
        them in parallel and returns clean, LLM-ready content.

        Args:
            urls: The URLs to fetch content from (max 10 per request).
            formats: Content formats to return: 'markdown', 'html', 'metadata'.
                Defaults to 'markdown' if not configured. Only used if not
                configured at tool creation.
            crawl_timeout: Per-URL timeout in seconds (1-60). Only used if not
                configured at tool creation.
        """
        body = self._build_contents_body(urls=urls, formats=formats, crawl_timeout=crawl_timeout)
        response = await self._post(_YOU_CONTENTS_URL, body)
        items = _ContentsResponseAdapter.validate_python(response.json())
        return [item.to_result() for item in items]

    async def research(
        self,
        input: str,
        *,
        research_effort: ResearchEffort | None = None,
    ) -> YouResearchResult:
        """Research a complex question and return a cited, synthesized answer.

        The Research API runs multiple searches, reads through sources, and
        synthesizes everything into a thorough, well-cited answer. Use it when a
        question is too complex for a simple lookup.

        Args:
            input: The research question (max 40,000 characters).
            research_effort: Depth of research: 'lite', 'standard', 'deep', or
                'exhaustive'. Only used if not configured at tool creation.
        """
        body = self._build_research_body(input=input, research_effort=research_effort)
        response = await self._post(_YOU_RESEARCH_URL, body)
        return self._parse_research_result(_RawResearchResponse.model_validate(response.json()))

    async def finance_research(
        self,
        input: str,
        *,
        research_effort: FinanceResearchEffort | None = None,
    ) -> YouResearchResult:
        """Research a financial question and return a cited, synthesized answer.

        The Finance Research API uses a finance-optimized index to research
        earnings, filings, market data, and financial news. Use it for company
        fundamentals, market trends, competitive analysis, or earnings summaries.

        Args:
            input: The financial research question (max 40,000 characters).
            research_effort: Depth of research: 'deep' or 'exhaustive'. Only used
                if not configured at tool creation.
        """
        body = self._build_finance_research_body(input=input, research_effort=research_effort)
        response = await self._post(_YOU_FINANCE_RESEARCH_URL, body)
        return self._parse_research_result(_RawResearchResponse.model_validate(response.json()))

    # ------------------------------------------------------------------
    # Parameter building
    # ------------------------------------------------------------------

    def _build_search_params(
        self,
        *,
        query: str,
        count: int | None,
        freshness: Freshness | None,
        country: str | None,
        language: str | None,
        safesearch: SafeSearch | None,
        livecrawl: LiveCrawl | None,
        livecrawl_formats: LiveCrawlFormats | None,
    ) -> dict[str, str | int]:
        """Merge configured search defaults with LLM-provided values.

        Configured values (set at construction) always win. `offset` is always
        included if set, regardless of LLM input.
        """
        params: dict[str, str | int] = {'query': query}

        effective_count = self._count if self._count is not None else count
        if effective_count is not None:
            params['count'] = effective_count
        if self._offset is not None:
            params['offset'] = self._offset

        effective_values: tuple[tuple[str, object | None], ...] = (
            ('freshness', self._freshness if self._freshness is not None else freshness),
            ('country', self._country if self._country is not None else country),
            ('language', self._language if self._language is not None else language),
            ('safesearch', self._safesearch if self._safesearch is not None else safesearch),
            ('livecrawl', self._livecrawl if self._livecrawl is not None else livecrawl),
            (
                'livecrawl_formats',
                self._livecrawl_formats if self._livecrawl_formats is not None else livecrawl_formats,
            ),
        )
        for key, value in effective_values:
            normalized = self._normalize_param(value)
            if normalized is not None:
                params[key] = normalized
        return params

    def _build_contents_body(
        self,
        *,
        urls: list[str],
        formats: list[ContentsFormat] | None,
        crawl_timeout: int | None,
    ) -> dict[str, object]:
        """Build the JSON body for a Contents API request.

        `max_age` is configured-only (never from the LLM). `formats` and
        `crawl_timeout` use configured values when set, otherwise LLM values.
        """
        body: dict[str, object] = {'urls': urls}

        effective_formats = self._contents_formats if self._contents_formats is not None else formats
        if effective_formats is not None:
            body['formats'] = effective_formats

        effective_timeout = self._crawl_timeout if self._crawl_timeout is not None else crawl_timeout
        if effective_timeout is not None:
            body['crawl_timeout'] = effective_timeout

        if self._max_age is not None:
            body['max_age'] = self._max_age

        return body

    def _build_research_body(
        self,
        *,
        input: str,
        research_effort: ResearchEffort | None,
    ) -> dict[str, object]:
        """Build the JSON body for a Research API request."""
        body: dict[str, object] = {'input': input}
        effective_effort = self._research_effort if self._research_effort is not None else research_effort
        if effective_effort is not None:
            body['research_effort'] = effective_effort
        return body

    def _build_finance_research_body(
        self,
        *,
        input: str,
        research_effort: FinanceResearchEffort | None,
    ) -> dict[str, object]:
        """Build the JSON body for a Finance Research API request."""
        body: dict[str, object] = {'input': input}
        effective_effort = (
            self._finance_research_effort if self._finance_research_effort is not None else research_effort
        )
        if effective_effort is not None:
            body['research_effort'] = effective_effort
        return body

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _get(self, url: str, params: dict[str, str | int]) -> httpx.Response:
        """Execute a GET request with the API key header."""
        headers = {'X-API-Key': self._api_key}
        if self._http_client is not None:
            response = await self._http_client.get(url, params=params, headers=headers)
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response

    async def _post(self, url: str, json_body: dict[str, object]) -> httpx.Response:
        """Execute a POST request with the API key header."""
        headers = {'X-API-Key': self._api_key}
        if self._http_client is not None:
            response = await self._http_client.post(url, json=json_body, headers=headers)
        else:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=json_body, headers=headers)
        response.raise_for_status()
        return response

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_search_results(self, response: _RawSearchResponse) -> list[YouSearchResult]:
        """Convert the parsed search response into a flat list of search results."""
        results: list[YouSearchResult] = []
        for item in response.results.web or []:
            results.append(item.to_result())
        for item in response.results.news or []:
            results.append(item.to_result())
        return results

    def _parse_research_result(self, response: _RawResearchResponse) -> YouResearchResult:
        """Convert the parsed research response into a public `YouResearchResult`."""
        output = response.output
        return {
            'content': output.content,
            'content_type': output.content_type,
            'sources': [s.to_source() for s in output.sources],
        }

    @staticmethod
    def _normalize_param(value: object | None) -> str | None:
        """Convert a parameter value to its string form for the API query string."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return str(value)
