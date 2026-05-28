"""Shared output-truncation helpers for the execution-environment toolset.

Truncation is presentation, not backend truth, so it lives in the capability layer
(here), never in `environments/`. Ported from pi-mono's `truncate.ts`
(see `agent_docs/pi-prompts.md` for attribution).
"""

from dataclasses import dataclass
from typing import Literal, TypeAlias

# Two independent caps; whichever is hit first wins. Mirrors pi's defaults so the tool
# descriptions (which quote these numbers to the model) stay truthful.
DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB

# Defined once, reused on the dataclass field and the local in truncate_head.
TruncatedBy: TypeAlias = Literal['lines', 'bytes'] | None


@dataclass(kw_only=True, frozen=True)
class TruncationResult:
    """What fit under the caps, and why we stopped. The caller turns this into a note."""

    truncated_lines: list[str]
    truncated_by: TruncatedBy = None
    # A single line wider than the byte cap can't be shown partially; this lets the
    # caller emit a "line too big" message instead of a normal continuation note.
    first_line_exceeded: bool = False

    @property
    def truncated(self) -> bool:
        # Derived, never stored, so it can't disagree with truncated_by.
        return self.truncated_by is not None


def format_size(num_bytes: int) -> str:
    """Render a byte count the way the continuation notes show it to the model."""
    if num_bytes < 1024:
        return f'{num_bytes}B'
    if num_bytes < 1024 * 1024:
        return f'{num_bytes / 1024:.1f}KB'
    return f'{num_bytes / (1024 * 1024):.1f}MB'


def truncate(lines: list[str], direction: Literal['head', 'tail'] = 'head') -> TruncationResult:
    """Keep the lines that fit under both caps; never emit a partial line.

    `head` keeps the first lines (file reads, top-down); `tail` keeps the last (shell output, where
    errors and the exit status live). We reverse up front so the rest of the function -- including the
    first-line guard -- always operates on "the line we'd keep first" regardless of direction.
    """
    if direction == 'tail':
        lines = lines[::-1]

    # A line wider than the byte cap can't be shown partially (we never split a line), so keep nothing
    # and flag it -- the caller reports the line's size and omits it. Checked after the reverse so for
    # `tail` this probes the last (newest) line, not the oldest.
    if lines and len(lines[0].encode('utf-8')) > DEFAULT_MAX_BYTES:
        return TruncationResult(truncated_lines=[], truncated_by='bytes', first_line_exceeded=True)

    kept: list[str] = []
    running_byte_size = 0
    truncated_by: TruncatedBy = None

    for line in lines:
        if len(kept) >= DEFAULT_MAX_LINES:
            truncated_by = 'lines'
            break
        # +1 for the '\n' that '\n'.join inserts before every line except the first,
        # so the budget matches the bytes actually emitted.
        cost = len(line.encode('utf-8')) + (1 if kept else 0)
        if running_byte_size + cost > DEFAULT_MAX_BYTES:
            truncated_by = 'bytes'
            break
        kept.append(line)
        running_byte_size += cost

    # Loop completed without breaking => kept everything => truncated_by stays None.
    if direction == 'tail':
        kept = kept[::-1]
    return TruncationResult(truncated_lines=kept, truncated_by=truncated_by)


def truncate_output(text: str, direction: Literal['head', 'tail'] = 'tail') -> str:
    """Cap free-form tool output (e.g. shell) and mark it when anything was dropped.

    Unlike `read_file`, this output isn't line-addressable, so the model gets a marker rather than a
    continuation offset. Defaults to `tail`: command errors and exit status live at the end. Hides the
    split/join round-trip so callers stay one line.
    """
    result = truncate(text.split('\n'), direction=direction)
    body = '\n'.join(result.truncated_lines)
    if not result.truncated:
        return body
    marker = f'[... output truncated to the last {format_size(DEFAULT_MAX_BYTES)} ...]'
    return f'{marker}\n{body}' if direction == 'tail' else f'{body}\n{marker}'
