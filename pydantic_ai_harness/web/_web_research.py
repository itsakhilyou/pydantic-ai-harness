"""`WebResearch`: delegate a search-then-fetch research loop to a sub-agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.tools import AgentDepsT, Tool
from pydantic_ai.toolsets import AgentToolset, FunctionToolset
from pydantic_ai.usage import UsageLimits

_RESEARCH_INSTRUCTIONS = (
    'You are a web research assistant working on behalf of another agent. Search the '
    'web, then fetch the most relevant pages, then synthesize a concise, accurate answer '
    'to the query. Cite every claim with its source URL. If the available sources are '
    'insufficient to answer, say so rather than guessing.'
)


def _build_default_search_tool(max_results: int | None) -> Tool[Any]:
    try:
        from pydantic_ai.common_tools.duckduckgo import (
            duckduckgo_search_tool,  # pyright: ignore[reportUnknownVariableType]
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'WebResearch without a custom `search_tool` requires the `web-research` dependency. '
            'Install it with: pip install "pydantic-ai-harness[web-research]"'
        ) from e
    return duckduckgo_search_tool(max_results=max_results)


def _build_default_fetch_tool(
    *,
    max_content_length: int | None,
    allow_local_urls: bool,
    timeout: int,
    allowed_domains: list[str] | None,
    blocked_domains: list[str] | None,
) -> Tool[Any]:
    try:
        from pydantic_ai.common_tools.web_fetch import web_fetch_tool
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            'WebResearch without a custom `fetch_tool` requires the `web-research` dependency. '
            'Install it with: pip install "pydantic-ai-harness[web-research]"'
        ) from e
    return web_fetch_tool(
        max_content_length=max_content_length,
        allow_local_urls=allow_local_urls,
        timeout=timeout,
        allowed_domains=allowed_domains,
        blocked_domains=blocked_domains,
    )


class WebResearchToolset(FunctionToolset[AgentDepsT]):
    """Exposes a single `research` tool backed by a nested search+fetch agent."""

    def __init__(
        self,
        *,
        research_model: Model | KnownModelName | None,
        search_tool: Tool[Any] | None,
        fetch_tool: Tool[Any] | None,
        max_results: int | None,
        instructions: str | None,
        research_usage_limits: UsageLimits | None,
        max_content_length: int | None,
        allow_local_urls: bool,
        timeout: int,
        allowed_domains: list[str] | None,
        blocked_domains: list[str] | None,
    ) -> None:
        super().__init__()
        self._research_model = research_model
        self._research_instructions = instructions or _RESEARCH_INSTRUCTIONS
        self._usage_limits = research_usage_limits
        self._search_tool = search_tool or _build_default_search_tool(max_results)
        self._fetch_tool = fetch_tool or _build_default_fetch_tool(
            max_content_length=max_content_length,
            allow_local_urls=allow_local_urls,
            timeout=timeout,
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )
        self.add_function(self.research, name='research')

    async def research(self, ctx: RunContext[AgentDepsT], query: str) -> str:
        """Research a question on the web and return synthesized, cited findings.

        A nested agent searches, fetches the most relevant pages, and synthesizes an
        answer with source URLs. The multi-step loop stays inside the sub-agent, so the
        caller's context only ever sees the final synthesis.

        Args:
            ctx: The agent run context (injected by the agent).
            query: The research question to answer.
        """
        model = self._research_model or ctx.model
        agent = Agent(
            model,
            output_type=str,
            tools=[self._search_tool, self._fetch_tool],
            instructions=self._research_instructions,
        )
        # Own usage so `research_usage_limits` bounds the nested loop independently of the
        # parent run; token usage is aggregated back into the parent afterwards.
        result = await agent.run(query, usage_limits=self._usage_limits)
        ctx.usage.incr(result.usage())
        return result.output


@dataclass
class WebResearch(AbstractCapability[AgentDepsT]):
    """Answer a question by delegating a search-then-fetch loop to a sub-agent.

    A web fetch usually follows a search, and that multi-step loop bloats the caller's
    context with intermediate results. This capability adds one `research` tool: it runs
    a nested agent equipped with search and fetch tools, which searches, reads the most
    relevant pages, and returns only a synthesized, cited answer. The caller's context
    never sees the intermediate pages.

    Both tools are pluggable with sensible defaults:

    - `search_tool`: default is DuckDuckGo. Pass any `Tool` (e.g. Tavily, Exa) to swap it.
    - `fetch_tool`: default composes core's SSRF-protected `web_fetch_tool`.

    Custom `search_tool`/`fetch_tool` run inside the nested research agent, which does not
    inherit the parent run's `deps` -- a custom tool that reads `ctx.deps` gets `None`.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import WebResearch

    agent = Agent('openai:gpt-5', capabilities=[WebResearch()])
    ```

    The default tools require the `web-research` dependency:

    ```bash
    pip install "pydantic-ai-harness[web-research]"
    ```
    """

    research_model: Model | KnownModelName | None = None
    """Model for the research sub-agent.

    `None` (default) inherits the parent run's model via `ctx.model` at call time.
    Inheriting an expensive parent model makes research costly -- pass a cheaper model
    here, and bound the loop with `research_usage_limits`.
    """

    search_tool: Tool[Any] | None = None
    """Search tool for the sub-agent. `None` (default) uses DuckDuckGo.

    Pass any `Tool` -- e.g. `tavily_search_tool(...)` or `exa_search_tool(...)` -- to swap
    the backend. When set, the `web-research` search dependency is not imported.
    """

    fetch_tool: Tool[Any] | None = None
    """Fetch tool for the sub-agent. `None` (default) composes core's `web_fetch_tool`.

    When set, the fetch-related fields below are inert and the default fetch dependency
    is not imported.
    """

    max_results: int | None = 5
    """Max search results for the default search tool. Inert when `search_tool` is set."""

    instructions: str | None = None
    """Override the research sub-agent's synthesis instructions. `None` uses the default."""

    research_usage_limits: UsageLimits | None = None
    """Bound the nested research loop's cost (request/token/tool-call limits)."""

    max_content_length: int | None = 50_000
    """Max characters per fetched page (default fetch tool only). `None` for no limit."""

    allow_local_urls: bool = False
    """Allow fetching private/local IP addresses (default fetch tool only)."""

    timeout: int = 30
    """Fetch timeout in seconds (default fetch tool only)."""

    allowed_domains: list[str] | None = None
    """Only fetch from these domains, exact hostname match (default fetch tool only)."""

    blocked_domains: list[str] | None = None
    """Never fetch from these domains, exact hostname match (default fetch tool only)."""

    def __post_init__(self) -> None:
        # Dataclass annotations are advisory; a config-driven caller could pass bad values.
        positive: dict[str, Any] = {'timeout': self.timeout}
        if self.max_results is not None:
            positive['max_results'] = self.max_results
        if self.max_content_length is not None:
            positive['max_content_length'] = self.max_content_length
        for name, value in positive.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer, got {value!r}')

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        """Build the `research` toolset."""
        return WebResearchToolset[AgentDepsT](
            research_model=self.research_model,
            search_tool=self.search_tool,
            fetch_tool=self.fetch_tool,
            max_results=self.max_results,
            instructions=self.instructions,
            research_usage_limits=self.research_usage_limits,
            max_content_length=self.max_content_length,
            allow_local_urls=self.allow_local_urls,
            timeout=self.timeout,
            allowed_domains=self.allowed_domains,
            blocked_domains=self.blocked_domains,
        )
