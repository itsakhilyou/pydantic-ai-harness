"""Tests for the `Docling` capability and `DoclingToolset`.

`docling` is an optional dependency and is not installed in CI, so the real
`DocumentConverter` is replaced with a fake via `monkeypatch` for the conversion
paths, and the loader's `ImportError` path is asserted directly.
"""

from __future__ import annotations

import sys

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness import Docling
from pydantic_ai_harness.docling import DoclingToolset, OutputFormat
from pydantic_ai_harness.docling import _toolset as toolset_module

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class _FakeDocument:
    def export_to_markdown(self) -> str:
        return '# Title\n\nbody'

    def export_to_text(self) -> str:
        return 'Title body'

    def export_to_html(self) -> str:
        return '<h1>Title</h1>'

    def export_to_doctags(self) -> str:
        return '<doctag>Title</doctag>'

    def export_to_dict(self) -> dict[str, object]:
        return {'name': 'Title'}


class _FakeResult:
    def __init__(self, document: _FakeDocument) -> None:
        self.document = document


class _FakeConverter:
    """Records `convert` calls and returns a fixed fake document."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, tuple[int, int]]] = []
        self.build_count = 0

    def convert(self, source: str, *, max_num_pages: int, page_range: tuple[int, int]) -> _FakeResult:
        self.calls.append((source, max_num_pages, page_range))
        return _FakeResult(_FakeDocument())


def _install_fake(monkeypatch: pytest.MonkeyPatch, fake: _FakeConverter) -> None:
    """Make `_build_converter` count builds and hand back the shared fake."""

    def build() -> _FakeConverter:
        fake.build_count += 1
        return fake

    monkeypatch.setattr(toolset_module, '_build_converter', build)


class TestDoclingCapability:
    def test_default_construction(self) -> None:
        cap = Docling()
        assert cap.output_format == 'markdown'
        assert cap.max_num_pages is None
        assert cap.page_range is None
        assert cap.max_output_chars == 50_000

    def test_get_toolset_returns_toolset(self) -> None:
        toolset = Docling().get_toolset()
        assert isinstance(toolset, DoclingToolset)
        assert 'convert_document' in toolset.tools


class TestDoclingConversion:
    @pytest.mark.parametrize(
        ('output_format', 'expected'),
        [
            ('markdown', '# Title\n\nbody'),
            ('text', 'Title body'),
            ('html', '<h1>Title</h1>'),
            ('doctags', '<doctag>Title</doctag>'),
            ('json', '{\n  "name": "Title"\n}'),
        ],
    )
    async def test_export_formats(
        self, monkeypatch: pytest.MonkeyPatch, output_format: OutputFormat, expected: str
    ) -> None:
        fake = _FakeConverter()
        _install_fake(monkeypatch, fake)
        toolset = Docling(output_format=output_format).get_toolset()

        assert await toolset.convert_document('doc.pdf') == expected

    async def test_defaults_convert_all_pages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeConverter()
        _install_fake(monkeypatch, fake)
        toolset = Docling().get_toolset()

        await toolset.convert_document('doc.pdf')

        source, max_pages, page_range = fake.calls[0]
        assert source == 'doc.pdf'
        assert max_pages == sys.maxsize
        assert page_range == (1, sys.maxsize)

    async def test_page_limits_are_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeConverter()
        _install_fake(monkeypatch, fake)
        toolset = Docling(max_num_pages=3, page_range=(2, 5)).get_toolset()

        await toolset.convert_document('doc.pdf')

        _, max_pages, page_range = fake.calls[0]
        assert max_pages == 3
        assert page_range == (2, 5)

    async def test_converter_is_built_once_and_reused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeConverter()
        _install_fake(monkeypatch, fake)
        toolset = Docling().get_toolset()

        await toolset.convert_document('a.pdf')
        await toolset.convert_document('b.pdf')

        assert fake.build_count == 1
        assert len(fake.calls) == 2

    async def test_output_is_truncated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeConverter()
        _install_fake(monkeypatch, fake)
        toolset = Docling(max_output_chars=4).get_toolset()

        result = await toolset.convert_document('doc.pdf')

        assert result.startswith('# Ti')
        assert 'truncated to 4 chars' in result


class TestDoclingRecoverableErrors:
    async def test_missing_source_becomes_model_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def build() -> _FakeConverter:
            return _RaisingConverter()  # pyright: ignore[reportReturnType]

        monkeypatch.setattr(toolset_module, '_build_converter', build)
        toolset = Docling().get_toolset()

        with pytest.raises(ModelRetry, match='no such document'):
            await toolset.convert_document('missing.pdf')


class _RaisingConverter:
    def convert(self, source: str, *, max_num_pages: int, page_range: tuple[int, int]) -> _FakeResult:
        raise FileNotFoundError('no such document')


class TestDoclingLoader:
    def test_missing_dependency_raises_with_install_hint(self) -> None:
        with pytest.raises(ImportError, match='docling is required'):
            toolset_module._build_converter()


class TestDoclingAgentIntegration:
    async def test_tool_runs_through_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeConverter()
        _install_fake(monkeypatch, fake)
        model = TestModel(custom_output_text='ok', call_tools=['convert_document'])
        agent: Agent[None, str] = Agent(model, capabilities=[Docling()])

        result = await agent.run('convert the document')

        assert result.output == 'ok'
        assert len(fake.calls) == 1
