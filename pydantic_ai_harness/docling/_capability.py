"""Docling capability that converts documents to text for agents."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.docling._toolset import DoclingToolset, OutputFormat


@dataclass
class Docling(AbstractCapability[AgentDepsT]):
    """Document conversion for agents, backed by the `docling` library.

    Gives the agent a `convert_document` tool that accepts a local path or URL
    and returns the document (PDF, DOCX, PPTX, XLSX, HTML, images, and others)
    rendered as Markdown by default.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import Docling

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[Docling()])
    ```

    Requires the optional `docling` dependency
    (`pip install "pydantic-ai-harness[docling]"`). The first conversion builds a
    `DocumentConverter`, which loads conversion pipelines and can be slow; the
    converter is then reused for the lifetime of the toolset.
    """

    output_format: OutputFormat = 'markdown'
    """Format the converted document is exported as.

    - `markdown` (default): structured Markdown.
    - `text`: plain text.
    - `html`: HTML.
    - `doctags`: Docling's DocTags markup.
    - `json`: the document's dictionary representation, JSON-encoded.
    """

    max_num_pages: int | None = None
    """Maximum number of pages to convert. `None` converts all pages."""

    page_range: tuple[int, int] | None = None
    """Inclusive 1-based `(start, end)` page range to convert. `None` converts all pages."""

    max_output_chars: int = 50_000
    """Maximum characters returned to the model; longer output is truncated."""

    def get_toolset(self) -> DoclingToolset[AgentDepsT]:
        """Build the toolset exposing `convert_document`."""
        return DoclingToolset[AgentDepsT](
            output_format=self.output_format,
            max_output_chars=self.max_output_chars,
            max_num_pages=self.max_num_pages,
            page_range=self.page_range,
        )
