# Markdownify

Convert HTML to Markdown so the model works against compact, readable text
instead of raw markup.

## The problem

Scrapers, HTTP tools, and email/CMS APIs hand back HTML. Feeding raw HTML to a
model wastes tokens on tags and attributes and makes the content harder to
reason about. Every agent ends up reinventing an HTML-to-Markdown step.

## The solution

`Markdownify` adds an `html_to_markdown` tool backed by the
[`markdownify`](https://github.com/matthewwithanm/python-markdownify) library.
The model passes an HTML fragment or document and gets Markdown back. Conversion
options are fixed on the capability, so the model only supplies the HTML.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Markdownify

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[Markdownify(heading_style='atx')],
)

result = agent.run_sync('Fetch the article and give me the key points.')
print(result.output)
```

## Installation

`markdownify` is an optional dependency:

```bash
pip install "pydantic-ai-harness[markdownify]"
```

Without it, building the agent succeeds but the first conversion raises
`ImportError` with this install hint.

## Tool

| Tool | Purpose |
|---|---|
| `html_to_markdown(html)` | Convert an HTML fragment or document to Markdown. |

Output longer than `max_output_chars` is truncated, keeping the head and
appending a marker.

## Options

All options are set on the capability and apply to every call.

| Field | Effect | Default |
|---|---|---|
| `heading_style` | `atx` (`#`), `atx_closed`, `underlined`, or `setext`. | `atx` |
| `bullets` | Bullet characters cycled by nesting depth. | `'*+-'` |
| `strip` | Tag names to remove, keeping their text. | `None` |
| `convert` | Allowlist of tag names to convert; others are stripped. | `None` |
| `autolinks` | Emit `<url>` when an anchor's text equals its href. | `True` |
| `wrap` / `wrap_width` | Wrap paragraph text at a column width. | `False` / `80` |
| `escape_asterisks` | Escape literal `*`. | `True` |
| `escape_underscores` | Escape literal `_`. | `True` |
| `escape_misc` | Escape other Markdown-significant characters. | `False` |
| `strong_em_symbol` | `*` or `_` for emphasis. | `'*'` |
| `newline_style` | `spaces` or `backslash` for `<br>`. | `'spaces'` |
| `code_language` | Default language label on fenced code blocks. | `''` |
| `default_title` | Use a link/image URL as its title when absent. | `False` |
| `max_output_chars` | Cap on returned Markdown. | `50_000` |

`strip` and `convert` are mutually exclusive -- markdownify treats them as
opposite ends of the same allow/deny choice, so setting both raises
`ValueError` at construction.

`heading_style` defaults to `atx` here (the markdownify default is
`underlined`), because `#`-prefixed headings are the more common Markdown
convention.

## Agent spec (YAML/JSON)

`Markdownify` works with Pydantic AI's
[agent spec](https://ai.pydantic.dev/agent-spec/):

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - Markdownify:
      heading_style: atx
      strip: ['script', 'style']
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Markdownify

agent = Agent.from_file('agent.yaml', custom_capability_types=[Markdownify])
```

## See also

- [`Docling`](../docling/) converts whole documents (PDF, DOCX, PPTX, images)
  rather than HTML strings.

## Further reading

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Toolsets](https://ai.pydantic.dev/toolsets/)
- [markdownify](https://github.com/matthewwithanm/python-markdownify)
