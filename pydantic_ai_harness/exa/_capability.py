"""Exa search capability that gives an agent web research tools."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.exa._toolset import ExaClient, ExaSearchToolset

_INSTRUCTIONS = (
    'You have web research tools backed by the Exa search API. Start broad: use `web_search` '
    'to survey several sources on a topic, then use `get_page` to read the most promising '
    'URLs in full before drawing conclusions. Prefer primary sources, and cite the URLs of '
    'the pages you relied on in your answer.'
)


@dataclass
class ExaSearch(AbstractCapability[AgentDepsT]):
    """Web research for agents, backed by the [Exa](https://exa.ai) search API.

    Adds two tools: `web_search`, which returns search results together with
    page text, and `get_page`, which retrieves the text of a specific URL.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.exa import ExaSearch

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ExaSearch()])
    ```

    Authentication comes from the `EXA_API_KEY` environment variable by
    default; pass `client` to configure it explicitly.
    """

    num_results: int = 5
    """Number of results `web_search` returns per query."""

    max_text_chars: int = 10_000
    """Maximum characters of page text returned per result.

    Sent to Exa as the contents cap and re-enforced when tool output is
    formatted, so page text stays bounded even with a custom `client`.
    """

    client: ExaClient | None = None
    """Exa client to use; when `None`, an `exa_py.AsyncExa` is built from `EXA_API_KEY`.

    Any object satisfying the `ExaClient` protocol works: use it to pass an API
    key explicitly, point at a different base URL, or substitute a fake in tests.
    """

    def get_instructions(self) -> str:
        """Static research guidance: search wide, read the promising pages in full, cite URLs."""
        return _INSTRUCTIONS

    def get_toolset(self) -> ExaSearchToolset[AgentDepsT]:
        """Build the toolset providing the `web_search` and `get_page` tools."""
        return ExaSearchToolset[AgentDepsT](
            client=self.client,
            num_results=self.num_results,
            max_text_chars=self.max_text_chars,
        )
