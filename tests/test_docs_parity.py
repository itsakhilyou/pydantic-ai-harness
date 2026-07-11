"""Keep the README honest about what ships.

Every capability package must document itself with a `README.md` and be linked
from the top-level `README.md`. A capability cannot land without showing up in
the docs, so the "what's available today" tables cannot silently fall behind the
code. This is the mechanical half of docs parity; the semantic half (does the
prose match the code as written) is a review-time concern, not a unit test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
_PACKAGE = _ROOT / 'pydantic_ai_harness'

# The `experimental` package is a namespace/warning shim, not a capability, so it
# has no standalone README and is not listed in the top-level tables.
_NAMESPACE_PACKAGES = {_PACKAGE / 'experimental'}


def _is_deprecation_shim(package: Path) -> bool:
    """A package left at a moved capability's old path re-exports it and calls `warn_moved`.

    Such shims carry no docs of their own, so they are excluded from the capability tables.
    """
    return 'warn_moved(' in (package / '__init__.py').read_text(encoding='utf-8')


def _capability_packages() -> list[Path]:
    """Directories that are importable packages and represent a capability's public surface."""
    candidates: list[Path] = []
    for parent in (_PACKAGE, _PACKAGE / 'experimental'):
        for child in sorted(parent.iterdir()):
            if not child.is_dir() or child.name.startswith(('_', '.')):
                continue
            # A non-package dir under the capability roots does not occur in a clean tree, so this guard stays uncovered.
            if not (child / '__init__.py').exists():  # pragma: no cover
                continue
            if child in _NAMESPACE_PACKAGES or _is_deprecation_shim(child):
                continue
            candidates.append(child)
    return candidates


_CAPABILITY_PACKAGES = _capability_packages()


def test_capability_packages_discovered() -> None:
    # Guard against the discovery silently finding nothing (e.g. a moved package root),
    # which would make the parametrized checks below vacuously pass.
    assert len(_CAPABILITY_PACKAGES) >= 10


@pytest.mark.parametrize('package', _CAPABILITY_PACKAGES, ids=lambda p: str(p.relative_to(_ROOT)))
def test_capability_has_readme(package: Path) -> None:
    readme = package / 'README.md'
    assert readme.exists(), (
        f'{package.relative_to(_ROOT)} is an importable capability package but has no README.md. '
        'Add one (start from an existing capability README) so its public surface is documented.'
    )


@pytest.mark.parametrize('package', _CAPABILITY_PACKAGES, ids=lambda p: str(p.relative_to(_ROOT)))
def test_capability_linked_from_top_readme(package: Path) -> None:
    top_readme = (_ROOT / 'README.md').read_text(encoding='utf-8')
    link_target = f'{package.relative_to(_ROOT).as_posix()}/'
    assert link_target in top_readme, (
        f'{package.relative_to(_ROOT)} is not linked from the top-level README.md. '
        f'Add a row for it (linking `{link_target}`) to the "What\'s available today" or "Roadmap" tables '
        'so the README stays in step with the code.'
    )


# --- Unified-docs page checks (docs/*.md) -----------------------------------
#
# The flat pages under `docs/` render on the unified site. These mechanical
# checks encode the capability-authoring rules agreed in the 2026-07-10 team
# sync: purpose-first leads, a source link on every page, names that match the
# capability, and no leftover "experimental" framing on graduated capabilities.
# ACP is the one page that stays experimental.

_DOCS_DIR = _ROOT / 'docs'
_NON_CAPABILITY_PAGES = {'index.md', 'mutation-testing.md'}
_ACP_PAGE = 'acp.md'

_SOURCE_LINK = 'github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/'
# Framing that must not appear on a graduated (non-ACP) capability page.
_EXPERIMENTAL_MARKERS = ('HarnessExperimentalWarning', 'removed in any release', '!!! warning "Experimental')
# Lifecycle hook names must not lead a page -- mechanism goes below the purpose.
_LEAD_HOOK_NAMES = ('before_model_request', 'after_model_request', 'before_tool_execute', 'after_tool_execute')
# ClassName-style headings are a smell, except where the class name IS the name.
_ALLOWED_CLASSNAME_HEADINGS = {'FileSystem'}
_FORBIDDEN_HEADINGS = {'overflow', 'authoring', 'overflow capability', 'compaction capabilities'}


def _capability_doc_pages() -> list[Path]:
    return [p for p in sorted(_DOCS_DIR.glob('*.md')) if p.name not in _NON_CAPABILITY_PAGES]


_CAPABILITY_DOC_PAGES = _capability_doc_pages()


def _strip_frontmatter(text: str) -> str:
    if text.startswith('---\n'):
        end = text.find('\n---', 4)
        if end != -1:
            return text[end + 4 :]
    return text


def _heading_problem(h1: str) -> str | None:
    """Return why an H1 fails the name rule, or None if it is fine."""
    name = h1[2:].strip() if h1.startswith('# ') else h1.strip()
    if name.lower() in _FORBIDDEN_HEADINGS:
        return f'"{name}" is a short/legacy form -- use the full capability name'
    for word in name.split():
        if word in _ALLOWED_CLASSNAME_HEADINGS:
            continue
        if re.search(r'[a-z][A-Z]', word):
            return f'"{name}" is ClassName-style ("{word}") -- use spaced words'
    return None


def _h1(text: str) -> str:
    for line in _strip_frontmatter(text).splitlines():
        if line.startswith('# '):
            return line
    return ''


def _lead_paragraph(text: str) -> str:
    """The first prose paragraph after the H1, skipping links, notes, and admonitions."""
    lines = _strip_frontmatter(text).splitlines()
    start = next((i + 1 for i, ln in enumerate(lines) if ln.startswith('# ')), 0)
    collected: list[str] = []
    in_fence = False
    for line in lines[start:]:
        stripped = line.strip()
        if in_fence:
            if stripped.startswith('```'):
                in_fence = False
            continue
        if not collected:
            if not stripped:
                continue
            if stripped.startswith('```'):
                in_fence = True
                continue
            # Skip non-prose preamble: headings, blockquotes, admonitions,
            # links/images, tables, and indented admonition bodies.
            if stripped.startswith(('#', '>', '!!!', '[', '![', '|')) or line.startswith('    '):
                continue
            collected.append(stripped)
        elif not stripped:
            break
        else:
            collected.append(stripped)
    return ' '.join(collected)


def test_capability_doc_pages_discovered() -> None:
    # Guard against a moved docs root making every check below vacuously pass.
    assert len(_CAPABILITY_DOC_PAGES) >= 12


@pytest.mark.parametrize('page', _CAPABILITY_DOC_PAGES, ids=lambda p: p.name)
def test_doc_page_links_its_source(page: Path) -> None:
    text = page.read_text(encoding='utf-8')
    assert _SOURCE_LINK in text, (
        f'{page.relative_to(_ROOT)} has no source-code link. Add a link to '
        f'`https://{_SOURCE_LINK}<module>/` so a reading agent can verify behavior.'
    )


@pytest.mark.parametrize('page', _CAPABILITY_DOC_PAGES, ids=lambda p: p.name)
def test_doc_page_heading_matches_capability(page: Path) -> None:
    problem = _heading_problem(_h1(page.read_text(encoding='utf-8')))
    assert problem is None, f'{page.relative_to(_ROOT)}: {problem}'


@pytest.mark.parametrize('page', _CAPABILITY_DOC_PAGES, ids=lambda p: p.name)
def test_doc_page_lead_is_purpose_first(page: Path) -> None:
    lead = _lead_paragraph(page.read_text(encoding='utf-8'))
    hit = next((h for h in _LEAD_HOOK_NAMES if h in lead), None)
    assert hit is None, (
        f'{page.relative_to(_ROOT)}: opening paragraph names the `{hit}` hook. '
        'Lead with the purpose (what it is for, when to use it); move mechanism lower.'
    )


@pytest.mark.parametrize(
    'page',
    [p for p in _CAPABILITY_DOC_PAGES if p.name != _ACP_PAGE],
    ids=lambda p: p.name,
)
def test_graduated_doc_page_has_no_experimental_framing(page: Path) -> None:
    text = page.read_text(encoding='utf-8')
    hit = next((m for m in _EXPERIMENTAL_MARKERS if m in text), None)
    assert hit is None, (
        f'{page.relative_to(_ROOT)}: graduated capability still carries experimental framing ({hit!r}). '
        'Only ACP keeps an experimental note; soften the rest to the README stability note.'
    )


@pytest.mark.parametrize('package', _CAPABILITY_PACKAGES, ids=lambda p: str(p.relative_to(_ROOT)))
def test_capability_readme_heading_matches_capability(package: Path) -> None:
    problem = _heading_problem(_h1((package / 'README.md').read_text(encoding='utf-8')))
    assert problem is None, f'{package.relative_to(_ROOT) / "README.md"}: {problem}'
