"""Markdownify toolset -- converts HTML to Markdown via the `markdownify` library."""

from __future__ import annotations

from typing import Protocol

from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset


class _MarkdownifyFn(Protocol):
    """Callable shape of `markdownify.markdownify`."""

    def __call__(self, html: str, /, **options: object) -> str: ...  # pragma: no cover


def _load_markdownify() -> _MarkdownifyFn:
    """Import `markdownify.markdownify`, raising a clear error when the extra is missing.

    The import lives behind a function so the package stays importable without
    the optional dependency installed, and the converter is resolved lazily on
    first conversion rather than at agent construction.
    """
    try:
        from markdownify import markdownify  # pyright: ignore
    except ImportError as e:
        raise ImportError(
            'markdownify is required for the Markdownify capability. '
            'Install it with: pip install "pydantic-ai-harness[markdownify]"'
        ) from e
    return markdownify  # pyright: ignore  # pragma: no cover


class MarkdownifyToolset(FunctionToolset[AgentDepsT]):
    """Exposes a single `html_to_markdown` tool backed by `markdownify`.

    Conversion options are fixed at construction time and applied to every call,
    so the model only supplies the HTML to convert.
    """

    def __init__(
        self,
        *,
        heading_style: str,
        bullets: str,
        strip: list[str] | None,
        convert: list[str] | None,
        autolinks: bool,
        wrap: bool,
        wrap_width: int,
        escape_asterisks: bool,
        escape_underscores: bool,
        escape_misc: bool,
        strong_em_symbol: str,
        newline_style: str,
        code_language: str,
        default_title: bool,
        max_output_chars: int,
    ) -> None:
        super().__init__()
        if strip is not None and convert is not None:
            raise ValueError('Specify strip or convert, not both -- markdownify treats them as mutually exclusive.')
        self._options: dict[str, object] = {
            'heading_style': heading_style,
            'bullets': bullets,
            'autolinks': autolinks,
            'wrap': wrap,
            'wrap_width': wrap_width,
            'escape_asterisks': escape_asterisks,
            'escape_underscores': escape_underscores,
            'escape_misc': escape_misc,
            'strong_em_symbol': strong_em_symbol,
            'newline_style': newline_style,
            'code_language': code_language,
            'default_title': default_title,
        }
        if strip is not None:
            self._options['strip'] = strip
        if convert is not None:
            self._options['convert'] = convert
        self._max_output_chars = max_output_chars

        self.add_function(self.html_to_markdown, name='html_to_markdown')

    def _truncate(self, text: str) -> str:
        """Cap output at `max_output_chars`, keeping the head and flagging the cut."""
        if len(text) <= self._max_output_chars:
            return text
        return text[: self._max_output_chars] + f'\n\n[... output truncated to {self._max_output_chars} chars ...]'

    async def html_to_markdown(self, html: str) -> str:
        """Convert an HTML fragment or document into Markdown.

        Args:
            html: The HTML source to convert.

        Returns:
            The Markdown rendering of the input, truncated to the configured cap.
        """
        markdownify = _load_markdownify()
        return self._truncate(markdownify(html, **self._options))
