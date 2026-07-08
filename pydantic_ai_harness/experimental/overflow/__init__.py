"""Overflow capability: reduce oversized tool returns at production time.

`OverflowingToolOutput` intercepts a tool return when it is produced and reduces it --
truncating, spilling to a queryable file, or summarizing -- so an oversized payload does
not persist in history and get re-sent on every later model request. Combine the three
modes through an ordered list of size `bands`.

Spilled payloads are queried on demand through two registered tools: `grep_tool_result` to
search a payload and `read_tool_result` to read a line or byte range of it. The
`OverflowStore` protocol is the seam for a durable backend (the local-file default ships
for single-process runs).
"""

from pydantic_ai_harness.experimental._warn import warn_experimental
from pydantic_ai_harness.experimental.overflow._bands import (
    Action,
    Band,
    Passthrough,
    Spill,
    Summarize,
    SummarizeFunc,
    Truncate,
)
from pydantic_ai_harness.experimental.overflow._capability import OverflowingToolOutput
from pydantic_ai_harness.experimental.overflow._markers import GREP_TOOL_NAME, READ_TOOL_NAME
from pydantic_ai_harness.experimental.overflow._payload import TruncationStrategy
from pydantic_ai_harness.experimental.overflow._store import LocalFileStore, OverflowStore

warn_experimental('overflow')

__all__ = [
    'GREP_TOOL_NAME',
    'READ_TOOL_NAME',
    'Action',
    'Band',
    'LocalFileStore',
    'OverflowStore',
    'OverflowingToolOutput',
    'Passthrough',
    'Spill',
    'Summarize',
    'SummarizeFunc',
    'Truncate',
    'TruncationStrategy',
]
