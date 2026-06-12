# Docling

Convert documents -- PDF, DOCX, PPTX, XLSX, HTML, images, and more -- into text
the model can read, using [Docling](https://github.com/docling-project/docling).

## The problem

Agents are routinely handed a PDF report, a Word doc, or a slide deck and asked
to summarize, extract, or answer questions. Models can't read those binary
formats directly, and rolling per-format extraction (PDF text layers, tables,
OCR for scans) is a project of its own.

## The solution

`Docling` adds a `convert_document` tool. The model passes a local path or URL
and gets back the document as Markdown (or text, HTML, DocTags, or JSON).
Docling handles layout, tables, and figures, and falls back to OCR for scanned
input.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Docling

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[Docling()],
)

result = agent.run_sync('Summarize ./reports/q3.pdf in five bullets.')
print(result.output)
```

## Installation

`docling` is an optional dependency:

```bash
pip install "pydantic-ai-harness[docling]"
```

Without it, building the agent succeeds but the first conversion raises
`ImportError` with this install hint.

## Tool

| Tool | Purpose |
|---|---|
| `convert_document(source)` | Convert a local path or URL into the configured output format. |

The first call builds a Docling `DocumentConverter`, which loads conversion
pipelines and can be slow; later calls reuse it. Conversion runs in a worker
thread so the agent's event loop is not blocked. A missing path or unreadable
source is surfaced to the model as a retry so it can supply a different source.

## Options

| Field | Effect | Default |
|---|---|---|
| `output_format` | `markdown`, `text`, `html`, `doctags`, or `json`. | `markdown` |
| `max_num_pages` | Cap on pages converted. `None` converts all. | `None` |
| `page_range` | Inclusive 1-based `(start, end)` range. `None` converts all. | `None` |
| `max_output_chars` | Cap on returned text. | `50_000` |

## A note on URLs

`convert_document` accepts URLs, and Docling fetches them. If the agent runs
untrusted instructions, treat the source as attacker-controlled: a fetched URL
can reach internal services the host can see. Restrict the source (validate it
in a tool wrapper or run behind egress controls) when that matters.

## Agent spec (YAML/JSON)

`Docling` works with Pydantic AI's
[agent spec](https://ai.pydantic.dev/agent-spec/):

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - Docling:
      output_format: markdown
      max_num_pages: 20
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Docling

agent = Agent.from_file('agent.yaml', custom_capability_types=[Docling])
```

## See also

- [`Markdownify`](../markdownify/) converts HTML strings to Markdown without
  Docling's heavier document pipeline.

## Further reading

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Toolsets](https://ai.pydantic.dev/toolsets/)
- [Docling](https://github.com/docling-project/docling)
