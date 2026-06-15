"""Storage backend for spilled tool outputs.

`OverflowStore` is a narrow protocol: persist a payload under a key, read it back by
handle. `LocalFileStore` is the dependency-free default -- it writes each payload to a
file under a per-run directory. The handle is backend-addressable (a relative key), not
an absolute local path, so a durable backend (Temporal, a blob store) can resolve the
same handle in another process. This is the seam for consuming the core queryable-file
primitive (pydantic-ai #4352 / `ExecutionEnvironment`) once it lands.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class OverflowStore(Protocol):
    """Persist and retrieve spilled tool-output payloads.

    `write` takes a caller-chosen `key` and returns a `handle`. The handle is the only
    thing a later `read` needs, so it must be self-contained (a backend can encode the
    run, the call, and the retry into it). Implementations may return the key unchanged.
    """

    async def write(self, key: str, data: bytes) -> str:
        """Persist `data` under `key` and return a handle that `read` accepts."""
        ...  # pragma: no cover

    async def read(self, handle: str) -> bytes:
        """Return the payload previously stored for `handle`.

        Raise `FileNotFoundError` (or another `OSError`) when the handle is unknown.
        """
        ...  # pragma: no cover


_UNSAFE_SEGMENT = re.compile(r'[^A-Za-z0-9._-]+')


def _safe_segment(segment: str) -> str:
    """Make one path segment filesystem-safe without collapsing distinct keys.

    Empty or dot-only segments are replaced so a handle can never escape the root.
    """
    cleaned = _UNSAFE_SEGMENT.sub('_', segment)
    if cleaned in ('', '.', '..'):
        return '_'
    return cleaned


@dataclass
class LocalFileStore:
    """Dependency-free `OverflowStore` that writes each payload to a local file.

    The handle equals the key: a relative `run_id/tool_call_id.retry` path under
    `base_dir`. Files are kept after the run (a later `read_tool_result` may need them);
    a durable backend owns its own lifecycle. Reads load the whole file -- fine for the
    interim local store, and the seam to swap in a slicing backend stays the protocol.
    """

    base_dir: Path | None = None
    """Root directory for spilled files. Defaults to a temp subdirectory."""

    _root: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._root = (
            self.base_dir if self.base_dir is not None else Path(tempfile.gettempdir()) / 'pyai_harness_overflow'
        )

    def _path(self, key: str) -> Path:
        segments = [_safe_segment(part) for part in key.split('/') if part]
        if not segments:
            segments = ['_']
        return self._root.joinpath(*segments)

    async def write(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    async def read(self, handle: str) -> bytes:
        return self._path(handle).read_bytes()
