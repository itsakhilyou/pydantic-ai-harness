"""Docling toolset -- converts documents (PDF, DOCX, PPTX, HTML, images) to text."""

from __future__ import annotations

import functools
import json
import sys
from collections.abc import Awaitable, Callable
from typing import Concatenate, Literal, ParamSpec, Protocol

import anyio.to_thread
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from typing_extensions import assert_never

_P = ParamSpec('_P')

OutputFormat = Literal['markdown', 'text', 'html', 'json', 'doctags']
"""Export formats Docling can render a converted document into."""


class _DoclingDocument(Protocol):
    """Subset of `DoclingDocument` the toolset exports through."""

    def export_to_markdown(self) -> str: ...  # pragma: no cover
    def export_to_text(self) -> str: ...  # pragma: no cover
    def export_to_html(self) -> str: ...  # pragma: no cover
    def export_to_doctags(self) -> str: ...  # pragma: no cover
    def export_to_dict(self) -> dict[str, object]: ...  # pragma: no cover


class _ConversionResult(Protocol):
    """Subset of Docling's `ConversionResult` the toolset reads."""

    @property
    def document(self) -> _DoclingDocument: ...  # pragma: no cover


class _DocumentConverter(Protocol):
    """Subset of Docling's `DocumentConverter` the toolset drives."""

    def convert(  # pragma: no cover
        self, source: str, *, max_num_pages: int, page_range: tuple[int, int]
    ) -> _ConversionResult: ...


def _build_converter() -> _DocumentConverter:
    """Construct a Docling `DocumentConverter`, raising a clear error when the extra is missing.

    The import and converter build are deferred to first use: constructing a
    `DocumentConverter` loads conversion pipelines, so doing it lazily keeps
    agent construction cheap and the package importable without the dependency.
    """
    try:
        from docling.document_converter import DocumentConverter  # pyright: ignore
    except ImportError as e:
        raise ImportError(
            'docling is required for the Docling capability. '
            'Install it with: pip install "pydantic-ai-harness[docling]"'
        ) from e
    return DocumentConverter()  # pyright: ignore  # pragma: no cover


def _recoverable(
    fn: Callable[Concatenate[DoclingToolset, _P], Awaitable[str]],
) -> Callable[Concatenate[DoclingToolset, _P], Awaitable[str]]:
    """Surface a missing or unreadable source as `ModelRetry` so the agent can correct it.

    pyai only feeds `ModelRetry` back to the model as a retry prompt; any other
    exception aborts the whole run. A bad path is something the model can fix by
    supplying a different source, so it is surfaced as a retry.
    """

    @functools.wraps(fn)
    async def wrapper(self: DoclingToolset, *args: _P.args, **kwargs: _P.kwargs) -> str:
        try:
            return await fn(self, *args, **kwargs)
        except (FileNotFoundError, ValueError) as e:
            raise ModelRetry(str(e)) from e

    return wrapper


class DoclingToolset(FunctionToolset[AgentDepsT]):
    """Exposes a `convert_document` tool backed by Docling.

    Accepts a local path or URL and returns the document rendered in the
    configured format. The underlying `DocumentConverter` is built once on first
    use and reused across calls, since it loads conversion pipelines.
    """

    def __init__(
        self,
        *,
        output_format: OutputFormat,
        max_output_chars: int,
        max_num_pages: int | None,
        page_range: tuple[int, int] | None,
    ) -> None:
        super().__init__()
        self._output_format: OutputFormat = output_format
        self._max_output_chars = max_output_chars
        self._max_num_pages = max_num_pages
        self._page_range = page_range
        self._converter: _DocumentConverter | None = None

        self.add_function(self.convert_document, name='convert_document')

    def _get_converter(self) -> _DocumentConverter:
        if self._converter is None:
            self._converter = _build_converter()
        return self._converter

    def _export(self, document: _DoclingDocument) -> str:
        fmt = self._output_format
        if fmt == 'markdown':
            return document.export_to_markdown()
        if fmt == 'text':
            return document.export_to_text()
        if fmt == 'html':
            return document.export_to_html()
        if fmt == 'doctags':
            return document.export_to_doctags()
        if fmt == 'json':
            return json.dumps(document.export_to_dict(), indent=2)
        assert_never(fmt)

    def _convert_sync(self, source: str) -> str:
        converter = self._get_converter()
        max_pages = self._max_num_pages if self._max_num_pages is not None else sys.maxsize
        page_range = self._page_range if self._page_range is not None else (1, sys.maxsize)
        result = converter.convert(source, max_num_pages=max_pages, page_range=page_range)
        return self._export(result.document)

    def _truncate(self, text: str) -> str:
        """Cap output at `max_output_chars`, keeping the head and flagging the cut."""
        if len(text) <= self._max_output_chars:
            return text
        return text[: self._max_output_chars] + f'\n\n[... output truncated to {self._max_output_chars} chars ...]'

    @_recoverable
    async def convert_document(self, source: str) -> str:
        """Convert a document into text.

        Args:
            source: A local file path or URL to a supported document
                (PDF, DOCX, PPTX, XLSX, HTML, Markdown, images, and others).

        Returns:
            The document rendered in the configured output format, truncated to
            the configured cap.
        """
        text = await anyio.to_thread.run_sync(self._convert_sync, source)
        return self._truncate(text)
