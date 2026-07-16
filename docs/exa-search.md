---
title: Exa Search
description: Give a Pydantic AI agent web research tools backed by the Exa search API -- search with page text and full-page retrieval, with output capped to fit model context.
---

# Exa Search

`ExaSearch` gives an agent web research tools backed by the
[Exa](https://exa.ai) search API: search that returns page text alongside each
hit, and full-page retrieval for digging into a specific URL. Page text is
capped per result so tool output fits the model's context.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/exa/)

## The problem

Search tools that return only titles and snippets force a second round of
fetching before the agent can judge a source. Wiring a search API together with
a page fetcher, capping page text so tool output doesn't overwhelm the model's
context, and prompting the agent to research methodically is boilerplate every
research agent reinvents.

`ExaSearch` bundles that plumbing into a single
[capability](/ai/core-concepts/capabilities/): two research tools, a per-result
page-text cap, and short research guidance in the system prompt.

## Usage

Install the `exa` extra and set the `EXA_API_KEY` environment variable (create
a key at <https://dashboard.exa.ai>):

```bash
uv add "pydantic-ai-harness[exa]"
```

Then pass `ExaSearch` to an `Agent` via the `capabilities` parameter:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.exa import ExaSearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ExaSearch()])

result = agent.run_sync('What changed in the latest stable Python release?')
print(result.output)
```

## Tools

`ExaSearch` contributes two tools to the agent:

| Tool | Purpose |
|---|---|
| `web_search` | Search the web and return the top `num_results` pages, each with title, URL, and page text. |
| `get_page` | Retrieve the text contents of one specific URL. |

Page text is capped at `max_text_chars` characters per result: the cap is sent
to Exa as the contents limit and re-enforced when tool output is formatted, so
text stays bounded even with a custom client. When text is cut, the **head** is
kept (a page's lead carries the substance) and a
`[... page text truncated at N characters]` marker is appended.

A URL that returns no content surfaces to the model as a
[`ModelRetry`](/ai/tools-toolsets/tools-advanced/#tool-retries) rather than a
hard error: the run continues and the model can correct the URL or pick another
page.

## Instructions

`ExaSearch` contributes short research guidance to the system prompt: search
wide with `web_search` first, read the most promising pages in full with
`get_page` before drawing conclusions, prefer primary sources, and cite the
URLs relied on.

## Configuration

Every field of `ExaSearch` with its default:

```python
from pydantic_ai_harness.exa import ExaSearch

ExaSearch(
    num_results=5,          # results per web_search call
    max_text_chars=10_000,  # page-text cap per result, in characters
    client=None,            # ExaClient -- None builds exa_py.AsyncExa from EXA_API_KEY
)
```

## Custom client

The default client is `exa_py.AsyncExa`, configured from the `EXA_API_KEY`
environment variable; when the variable is missing, construction fails with a
setup hint. Pass any object satisfying the `ExaClient` protocol -- the subset of
`AsyncExa` the toolset calls -- to configure authentication or the base URL
explicitly, or to substitute a fake in tests:

```python
from exa_py import AsyncExa

from pydantic_ai_harness.exa import ExaSearch

ExaSearch(client=AsyncExa(api_key='...'))
```

The API may change between releases while the capability settles; breaking
changes ship deprecation warnings where practical.

## Further reading

- [Pydantic AI capabilities](/ai/core-concepts/capabilities/)
- [Toolsets](/ai/tools-toolsets/toolsets/)
- [Exa API documentation](https://docs.exa.ai)

## API reference

::: pydantic_ai_harness.exa.ExaSearch
