"""Optional ripgrep-backed content search for the filesystem toolset.

When an `rg` binary is on `PATH`, the toolset can delegate `search_files` to
ripgrep, which is much faster than the pure-Python walker on large trees.
ripgrep is confined to the root: it runs with the root as its working directory,
searches a target inside it, and never follows symlinks. The toolset re-checks
containment and applies its own dotfile / allow / deny / protected / include_glob
filters to every result, so both backends produce the same output.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated, Literal

from anyio import run_process
from pydantic import BaseModel, Field, TypeAdapter


class RipgrepError(RuntimeError):
    """ripgrep failed to run a search (e.g. a pattern its engine rejects).

    The toolset catches this and falls back to the pure-Python search, so the
    backend choice never changes results.
    """


def ripgrep_available() -> bool:
    """Whether an `rg` binary is available on `PATH`."""
    return shutil.which('rg') is not None


def resolve_ripgrep_enabled(use_ripgrep: bool | None) -> bool:
    """Decide whether the ripgrep backend is active.

    `None` auto-detects via `ripgrep_available`; `True` requires it and raises if
    it is missing; `False` disables it.
    """
    if use_ripgrep is False:
        return False
    available = ripgrep_available()
    if use_ripgrep and not available:
        raise RuntimeError(
            "use_ripgrep=True but no 'rg' binary was found on PATH. Install the "
            "'ripgrep' extra (pydantic-ai-harness[ripgrep]) or ripgrep itself."
        )
    return available


class _RgText(BaseModel):
    text: str


class _RgMatchData(BaseModel):
    path: _RgText
    line_number: int


class _RgMatch(BaseModel):
    type: Literal['match']
    data: _RgMatchData


class _RgEvent(BaseModel):
    """Any non-match event (begin, end, summary, context); fields are ignored."""

    type: str


# ripgrep emits one JSON object per line; only `match` events carry results.
_RG_ROW: TypeAdapter[_RgMatch | _RgEvent] = TypeAdapter(
    Annotated[_RgMatch | _RgEvent, Field(union_mode='left_to_right')]
)


async def ripgrep_file_matches(*, root: Path, target: Path, pattern: str) -> list[tuple[Path, list[int]]]:
    """Return `(path_relative_to_root, line_numbers)` for each file with matches.

    `--no-ignore` matches the pure-Python walker, which ignores `.gitignore`;
    hidden files stay skipped by ripgrep's defaults. `--sort=path` gives the same
    deterministic ordering as the fallback's `sorted(...)`. Exit code 1 means no
    matches; anything other than 0 or 1 means ripgrep itself failed.
    """
    command = ['rg', '--json', '--no-ignore', '--sort=path', '-e', pattern, '--', str(target)]
    process = await run_process(command, cwd=root, check=False)
    if process.returncode not in (0, 1):
        raise RipgrepError(process.stderr.decode('utf-8', errors='replace').strip())

    matches: dict[Path, list[int]] = {}
    for line in process.stdout.splitlines():
        row = _RG_ROW.validate_json(line)
        if isinstance(row, _RgMatch):
            matches.setdefault(Path(row.data.path.text), []).append(row.data.line_number)
    return list(matches.items())
