# Web

Context-efficient web access for agents. Pydantic AI core already ships plain fetch and
search ([`WebFetch`][web-fetch] / `WebSearch`); this package adds the layer core lacks — a
**sub-agent that reads the page so the caller's context doesn't have to**.

| Capability | What it adds |
|---|---|
| `SummarizingFetch` | Fetch one URL → a sub-agent compresses it to a query-relevant synopsis |
| `WebResearch` | One `research(query)` tool → a sub-agent runs the whole search→fetch→synthesize loop |

[web-fetch]: https://pydantic.dev/docs/ai/core-concepts/capabilities/

## SummarizingFetch

Dumping a whole page into context is expensive and noisy. `SummarizingFetch` adds a
`fetch_url(url, query)` tool that fetches the page (core's SSRF-protected, HTML-to-markdown
`web_fetch_tool` by default), then runs a sub-agent to return only the query-relevant
synopsis.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SummarizingFetch

agent = Agent('openai:gpt-5', capabilities=[SummarizingFetch()])
```

Install the default fetcher's dependency:

```bash
pip install "pydantic-ai-harness[summarizing-fetch]"
```

**Cost:** `summarizer_model` defaults to `None`, inheriting the parent run's model via
`ctx.model`. On an expensive parent that adds up — pass a cheap model:
`SummarizingFetch(summarizer_model='anthropic:claude-haiku-4-5')`. The sub-agent's token
usage is aggregated into the parent run.

**Pluggable seams (sensible defaults):**

| Field | Default | Swap it for |
|---|---|---|
| `fetcher` | core `web_fetch_tool` (SSRF + markdownify) | any service/library that turns a URL into text |
| `summarizer` | a sub-agent over `summarizer_model` | a custom prompt, non-LLM compressor, external API |
| `summarizer_model` | the parent run's model | a cheaper/faster model |
| `summarize_threshold` | `4000` chars | tune the fast-path size (pages at/below it return verbatim) |

`summarize_threshold` applies only to the default summarizer; a custom `summarizer` always
runs. Binary responses (PDFs, images) pass through unchanged.

### Custom fetcher: swap the markdown conversion

The `fetcher` is the seam for a different transport or HTML-to-markdown strategy. A `fetcher`
is `async (url: str) -> FetchedPage | BinaryContent`. This example uses
[Jina Reader](https://jina.ai/reader/); because the service fetches the page, SSRF is handled
on its side:

```python
import httpx
from pydantic_ai_harness import SummarizingFetch
from pydantic_ai_harness.web import FetchedPage


async def jina_fetcher(url: str) -> FetchedPage:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f'https://r.jina.ai/{url}', headers={'Accept': 'text/markdown'})
        response.raise_for_status()
        return FetchedPage(url=url, title='', content=response.text)


capability = SummarizingFetch(fetcher=jina_fetcher)
```

> **SSRF safety:** a custom fetcher that downloads URLs *itself* (e.g. `httpx` + a local
> markdown library) is responsible for its own SSRF protection — rejecting
> private/loopback/link-local addresses and cloud-metadata endpoints. The default fetcher
> gets this from Pydantic AI core. Prefer a service-based fetcher (like above) or reuse a
> vetted SSRF guard when fetching locally.

## WebResearch

A fetch usually follows a search, and running that loop in the main agent floods its context
with intermediate results. `WebResearch` adds one `research(query)` tool that runs a nested
agent (search + fetch tools), which searches, reads the most relevant pages, and returns only
a synthesized, cited answer.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import WebResearch

agent = Agent('openai:gpt-5', capabilities=[WebResearch()])
```

Install the default search + fetch dependency:

```bash
pip install "pydantic-ai-harness[web-research]"
```

**Cost & bounding:** `research_model` defaults to the parent run's model. The loop can make
several requests, so on an expensive parent it adds up — pass a cheaper model and bound it:

```python
from pydantic_ai.usage import UsageLimits

WebResearch(
    research_model='anthropic:claude-haiku-4-5',
    research_usage_limits=UsageLimits(request_limit=8),
)
```

**Pluggable seams (sensible defaults):**

| Field | Default | Swap it for |
|---|---|---|
| `search_tool` | DuckDuckGo | any `Tool` — Tavily, Exa, your own |
| `fetch_tool` | core `web_fetch_tool` (SSRF + markdownify) | any fetch `Tool` |
| `research_model` | the parent run's model | a cheaper/faster model |
| `instructions` | a synthesize-and-cite prompt | your own research instructions |
| `research_usage_limits` | unbounded | a `UsageLimits` cap on the nested loop |

### Custom search backend

`search_tool` accepts any Pydantic AI `Tool`. To use Tavily instead of DuckDuckGo
(`pip install "pydantic-ai-harness[web-research-tavily]"`):

```python
from pydantic_ai.common_tools.tavily import tavily_search_tool
from pydantic_ai_harness import WebResearch

capability = WebResearch(search_tool=tavily_search_tool(api_key='...'))
```

Custom `search_tool`/`fetch_tool` run inside the nested research agent, which does **not**
inherit the parent run's `deps`.

## When to use core instead

For a plain fetch or search where the **main agent** should read the results directly, use
pydantic-ai core's `WebFetch` / `WebSearch` — not this package. These capabilities are for
when a **sub-agent** should read and compress first.
