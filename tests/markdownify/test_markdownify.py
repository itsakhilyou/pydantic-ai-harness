"""Tests for the `Markdownify` capability and `MarkdownifyToolset`.

`markdownify` is an optional dependency and is not installed in CI, so the real
converter is replaced with a fake via `monkeypatch` for the conversion paths,
and the loader's `ImportError` path is asserted directly.
"""

from __future__ import annotations

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness import Markdownify
from pydantic_ai_harness.markdownify import MarkdownifyToolset
from pydantic_ai_harness.markdownify import _toolset as toolset_module

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class _CapturingMarkdownify:
    """Stand-in for `markdownify.markdownify` that records its arguments."""

    def __init__(self, return_value: str = '# heading') -> None:
        self.return_value = return_value
        self.html: str | None = None
        self.options: dict[str, object] = {}

    def __call__(self, html: str, /, **options: object) -> str:
        self.html = html
        self.options = options
        return self.return_value


def _install_fake(monkeypatch: pytest.MonkeyPatch, fake: _CapturingMarkdownify) -> None:
    monkeypatch.setattr(toolset_module, '_load_markdownify', lambda: fake)


class TestMarkdownifyCapability:
    def test_default_construction(self) -> None:
        cap = Markdownify()
        assert cap.heading_style == 'atx'
        assert cap.bullets == '*+-'
        assert cap.strip is None
        assert cap.convert is None
        assert cap.max_output_chars == 50_000

    def test_get_toolset_returns_toolset(self) -> None:
        toolset = Markdownify().get_toolset()
        assert isinstance(toolset, MarkdownifyToolset)
        assert 'html_to_markdown' in toolset.tools

    async def test_conversion_passes_configured_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cap = Markdownify(heading_style='underlined', wrap=True, wrap_width=40, code_language='python')
        toolset = cap.get_toolset()
        fake = _CapturingMarkdownify('converted')
        _install_fake(monkeypatch, fake)

        result = await toolset.html_to_markdown('<h1>Hi</h1>')

        assert result == 'converted'
        assert fake.html == '<h1>Hi</h1>'
        assert fake.options['heading_style'] == 'underlined'
        assert fake.options['wrap'] is True
        assert fake.options['wrap_width'] == 40
        assert fake.options['code_language'] == 'python'
        # Not configured, so omitted entirely (markdownify treats strip/convert as exclusive).
        assert 'strip' not in fake.options
        assert 'convert' not in fake.options


class TestMarkdownifyToolset:
    async def test_strip_is_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        toolset = Markdownify(strip=['script', 'style']).get_toolset()
        fake = _CapturingMarkdownify()
        _install_fake(monkeypatch, fake)

        await toolset.html_to_markdown('<p>x</p>')

        assert fake.options['strip'] == ['script', 'style']
        assert 'convert' not in fake.options

    async def test_convert_is_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        toolset = Markdownify(convert=['p', 'a']).get_toolset()
        fake = _CapturingMarkdownify()
        _install_fake(monkeypatch, fake)

        await toolset.html_to_markdown('<p>x</p>')

        assert fake.options['convert'] == ['p', 'a']
        assert 'strip' not in fake.options

    def test_strip_and_convert_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match='not both'):
            Markdownify(strip=['script'], convert=['p']).get_toolset()

    async def test_output_is_truncated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        toolset = Markdownify(max_output_chars=10).get_toolset()
        fake = _CapturingMarkdownify('x' * 50)
        _install_fake(monkeypatch, fake)

        result = await toolset.html_to_markdown('<p>x</p>')

        assert result.startswith('x' * 10)
        assert 'truncated to 10 chars' in result

    async def test_output_under_cap_is_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        toolset = Markdownify(max_output_chars=100).get_toolset()
        fake = _CapturingMarkdownify('short')
        _install_fake(monkeypatch, fake)

        assert await toolset.html_to_markdown('<p>x</p>') == 'short'


class TestMarkdownifyLoader:
    def test_missing_dependency_raises_with_install_hint(self) -> None:
        with pytest.raises(ImportError, match='markdownify is required'):
            toolset_module._load_markdownify()


class TestMarkdownifyAgentIntegration:
    async def test_tool_runs_through_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _CapturingMarkdownify('# converted')
        _install_fake(monkeypatch, fake)
        model = TestModel(custom_output_text='ok', call_tools=['html_to_markdown'])
        agent: Agent[None, str] = Agent(model, capabilities=[Markdownify()])

        result = await agent.run('convert this html')

        assert result.output == 'ok'
        assert fake.html is not None
