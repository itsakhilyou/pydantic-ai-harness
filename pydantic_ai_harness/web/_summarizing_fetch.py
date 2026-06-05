"""`SummarizingFetch`: fetch a URL and return a query-relevant synopsis."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset, FunctionToolset


@dataclass(frozen=True)
class FetchedPage:
    """A web page fetched as text, the input a `Summarizer` compresses.

    Custom `Fetcher` implementations build this from whatever transport and
    HTML-to-markdown conversion they choose.
    """

    url: str
    """The URL that was fetched (after any redirects the fetcher followed)."""

    title: str
    """The page title, or an empty string if none was found."""

    content: str
    """The page content as text (markdown by default)."""


Fetcher = Callable[[str], Awaitable['FetchedPage | BinaryContent']]
"""Fetches a URL, returning text as a `FetchedPage` or `BinaryContent` for binary responses.

The default fetcher composes pydantic-ai core's SSRF-protected `web_fetch_tool`.
A custom fetcher owns its own SSRF safety -- see the package README.
"""

Summarizer = Callable[[RunContext[AgentDepsT], 'FetchedPage', str], Awaitable[str]]
"""Compresses a `FetchedPage` against the caller's query into a synopsis.

Receives the run context (for usage aggregation and model access), the fetched
page, and the caller's query. The default implementation runs a sub-agent.
"""


_SUMMARIZE_INSTRUCTIONS = (
    'You compress a fetched web page for another agent. Return only the information '
    'relevant to the query, faithfully and concisely. Preserve key facts, code, '
    'identifiers, and exact quotes verbatim. Do not add information that is not present '
    'on the page; if the page does not answer the query, say so.'
)


def _build_summarize_prompt(page: FetchedPage, query: str) -> str:
    return f'URL: {page.url}\nTitle: {page.title}\n\nQuery: {query}\n\nPage content:\n{page.content}'


def _build_default_fetcher(
    *,
    max_content_length: int | None,
    allow_local_urls: bool,
    timeout: int,
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
) -> Fetcher:
    """Adapt core's SSRF-protected, markdownify-based fetch into a `Fetcher`."""
    # `WebFetchLocalTool` is module-public but not in web_fetch.py's `__all__`. We use it
    # directly (rather than `web_fetch_tool(...).function`) for a precisely typed
    # `WebFetchResult | BinaryContent` return; the factory's `.function` is `Any`-typed.
    try:
        from pydantic_ai.common_tools.web_fetch import WebFetchLocalTool
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'SummarizingFetch without a custom `fetcher` requires the `web-fetch` dependency. '
            'Install it with: pip install "pydantic-ai-harness[summarizing-fetch]"'
        ) from e

    core_fetch = WebFetchLocalTool(
        max_content_length=max_content_length,
        allow_local_urls=allow_local_urls,
        timeout=timeout,
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
    )

    async def default_fetcher(url: str) -> FetchedPage | BinaryContent:
        result = await core_fetch(url)
        if isinstance(result, BinaryContent):
            return result
        return FetchedPage(url=result['url'], title=result['title'], content=result['content'])

    return default_fetcher


class SummarizingFetchToolset(FunctionToolset[AgentDepsT]):
    """Exposes a single `fetch_url` tool that returns compressed, query-relevant content."""

    def __init__(
        self,
        *,
        fetcher: Fetcher | None,
        summarizer: Summarizer[AgentDepsT] | None,
        summarizer_model: Model | KnownModelName | None,
        summarize_threshold: int,
        max_content_length: int | None,
        allow_local_urls: bool,
        timeout: int,
        allowed_domains: list[str] | None,
        blocked_domains: list[str] | None,
    ) -> None:
        super().__init__()
        self._fetcher: Fetcher = fetcher or _build_default_fetcher(
            max_content_length=max_content_length,
            allow_local_urls=allow_local_urls,
            timeout=timeout,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )
        self._summarizer = summarizer
        self._summarizer_model = summarizer_model
        self._summarize_threshold = summarize_threshold
        self.add_function(self.fetch_url, name='fetch_url')

    async def fetch_url(self, ctx: RunContext[AgentDepsT], url: str, query: str) -> str | BinaryContent:
        """Fetch a web page and return only the parts relevant to `query`.

        The page is fetched and compressed to a query-focused synopsis so the full
        page never floods the conversation. Binary responses (PDFs, images) are
        returned unchanged for the model to process natively.

        Args:
            ctx: The agent run context (injected by the agent).
            url: The URL to fetch.
            query: What you want from the page. Drives what the synopsis keeps.
        """
        page = await self._fetcher(url)
        if isinstance(page, BinaryContent):
            return page
        if self._summarizer is not None:
            return await self._summarizer(ctx, page, query)
        if len(page.content) <= self._summarize_threshold:
            return page.content
        return await self._summarize_with_model(ctx, page, query)

    async def _summarize_with_model(self, ctx: RunContext[AgentDepsT], page: FetchedPage, query: str) -> str:
        """Default summarizer: a sub-agent compresses the page against the query.

        Uses `summarizer_model` when set, otherwise the parent run's model (`ctx.model`).
        The sub-agent runs with its own usage so its request budget is independent of the
        parent; its token usage is then aggregated into the parent run.
        """
        model = self._summarizer_model or ctx.model
        agent = Agent(model, output_type=str, instructions=_SUMMARIZE_INSTRUCTIONS)
        result = await agent.run(_build_summarize_prompt(page, query))
        ctx.usage.incr(result.usage())
        return result.output


@dataclass
class SummarizingFetch(AbstractCapability[AgentDepsT]):
    """Fetch a URL and return a query-relevant synopsis instead of the full page.

    Modern coding agents avoid flooding their context with raw pages by compressing
    each fetch to just what the caller asked for. This capability adds a `fetch_url`
    tool that fetches via pydantic-ai core (SSRF-protected, HTML-to-markdown), then
    runs a sub-agent to compress the page against the caller's query.

    Every step is pluggable with a sensible default:

    - `fetcher`: how a URL becomes text. Default composes core's `web_fetch_tool`.
      Swap in any service or library (the markdown-conversion seam).
    - `summarizer` / `summarizer_model`: how the page is compressed.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import SummarizingFetch

    agent = Agent('openai:gpt-5', capabilities=[SummarizingFetch()])
    ```

    The default `fetcher` requires the `web-fetch` dependency:

    ```bash
    pip install "pydantic-ai-harness[summarizing-fetch]"
    ```
    """

    summarizer_model: Model | KnownModelName | None = None
    """Model for the summarizing sub-agent.

    `None` (default) inherits the parent run's model via `ctx.model` at call time.
    Inheriting an expensive parent model makes every fetch costly -- pass a cheap,
    fast model here to control cost. Ignored when `summarizer` is set.
    """

    summarizer: Summarizer[AgentDepsT] | None = None
    """Custom compression step `(ctx, page, query) -> str`.

    When set, it always runs (the `summarize_threshold` fast path does not apply) and
    `summarizer_model` is ignored. Use it for a custom prompt, a non-LLM compressor, or
    an external summarization service.
    """

    fetcher: Fetcher | None = None
    """Custom `(url) -> FetchedPage | BinaryContent` fetcher.

    `None` (default) composes core's SSRF-protected `web_fetch_tool`. A custom fetcher
    owns its own SSRF safety and makes the fetch-related fields below inert.
    """

    summarize_threshold: int = 4000
    """Pages whose content is at most this many characters skip the default summarizer
    and are returned as-is. Does not apply when a custom `summarizer` is set."""

    max_content_length: int | None = 50_000
    """Max characters fetched before summarization (default fetcher only). `None` for no limit."""

    allow_local_urls: bool = False
    """Allow fetching private/local IP addresses (default fetcher only)."""

    timeout: int = 30
    """Fetch timeout in seconds (default fetcher only)."""

    allowed_domains: list[str] | None = None
    """Only fetch from these domains, exact hostname match (default fetcher only)."""

    blocked_domains: list[str] | None = None
    """Never fetch from these domains, exact hostname match (default fetcher only)."""

    def __post_init__(self) -> None:
        # Dataclass annotations are advisory; a config-driven caller could pass bad values.
        positive: dict[str, Any] = {'timeout': self.timeout}
        if self.max_content_length is not None:
            positive['max_content_length'] = self.max_content_length
        for name, value in positive.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer, got {value!r}')
        threshold: Any = self.summarize_threshold
        if not isinstance(threshold, int) or threshold < 0:
            raise ValueError(f'summarize_threshold must be a non-negative integer, got {threshold!r}')

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        """Build the `fetch_url` toolset."""
        return SummarizingFetchToolset[AgentDepsT](
            fetcher=self.fetcher,
            summarizer=self.summarizer,
            summarizer_model=self.summarizer_model,
            summarize_threshold=self.summarize_threshold,
            max_content_length=self.max_content_length,
            allow_local_urls=self.allow_local_urls,
            timeout=self.timeout,
            allowed_domains=self.allowed_domains,
            blocked_domains=self.blocked_domains,
        )
