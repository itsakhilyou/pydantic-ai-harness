"""Markdownify capability that converts HTML to Markdown for agents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.markdownify._toolset import MarkdownifyToolset


@dataclass
class Markdownify(AbstractCapability[AgentDepsT]):
    """HTML-to-Markdown conversion for agents, backed by the `markdownify` library.

    Gives the agent an `html_to_markdown` tool. Useful when a tool or scrape
    returns raw HTML and the model works better against compact Markdown.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import Markdownify

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[Markdownify()])
    ```

    Requires the optional `markdownify` dependency
    (`pip install "pydantic-ai-harness[markdownify]"`).
    """

    heading_style: Literal['atx', 'atx_closed', 'underlined', 'setext'] = 'atx'
    """Markdown heading style. `atx` uses `#` prefixes (the markdownify default is `underlined`)."""

    bullets: str = '*+-'
    """Characters cycled through for unordered-list bullets at successive nesting depths."""

    strip: Sequence[str] | None = None
    """Tag names to remove while keeping their text. Mutually exclusive with `convert`."""

    convert: Sequence[str] | None = None
    """Allowlist of tag names to convert; all others are stripped. Mutually exclusive with `strip`."""

    autolinks: bool = True
    """Emit `<url>` autolinks when an anchor's text equals its href."""

    wrap: bool = False
    """Wrap paragraph text at `wrap_width` columns."""

    wrap_width: int = 80
    """Column width used when `wrap` is enabled."""

    escape_asterisks: bool = True
    """Escape literal `*` so it is not interpreted as emphasis."""

    escape_underscores: bool = True
    """Escape literal `_` so it is not interpreted as emphasis."""

    escape_misc: bool = False
    """Escape other Markdown-significant characters (e.g. `#`, `-`, `>`)."""

    strong_em_symbol: Literal['*', '_'] = '*'
    """Symbol used for bold/italic emphasis."""

    newline_style: Literal['spaces', 'backslash'] = 'spaces'
    """How a `<br>` is rendered: two trailing spaces or a trailing backslash."""

    code_language: str = ''
    """Default language label applied to fenced code blocks."""

    default_title: bool = False
    """Use a link/image's URL as its title when no title attribute is present."""

    max_output_chars: int = 50_000
    """Maximum characters of Markdown returned to the model; longer output is truncated."""

    def get_toolset(self) -> MarkdownifyToolset[AgentDepsT]:
        """Build the toolset exposing `html_to_markdown`."""
        return MarkdownifyToolset[AgentDepsT](
            heading_style=self.heading_style,
            bullets=self.bullets,
            strip=list(self.strip) if self.strip is not None else None,
            convert=list(self.convert) if self.convert is not None else None,
            autolinks=self.autolinks,
            wrap=self.wrap,
            wrap_width=self.wrap_width,
            escape_asterisks=self.escape_asterisks,
            escape_underscores=self.escape_underscores,
            escape_misc=self.escape_misc,
            strong_em_symbol=self.strong_em_symbol,
            newline_style=self.newline_style,
            code_language=self.code_language,
            default_title=self.default_title,
            max_output_chars=self.max_output_chars,
        )
