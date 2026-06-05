"""Coerce tool input/output into JSON-able payloads sized for ACP `session/update` notifications."""

from __future__ import annotations

import json

from pydantic_core import to_jsonable_python

# ACP stdio is newline-delimited JSON, and clients read it with a bounded buffer (asyncio's
# default `StreamReader` limit is 64 KiB). A single oversized notification overruns that buffer
# and drops the connection, so long text is split across several updates. The cap is in
# characters and kept well below the byte limit to leave headroom for multi-byte text, JSON
# escaping, and the rest of the notification envelope.
MAX_TEXT_UPDATE_CHARS = 8 * 1024

# Tool input/output is sent whole in one notification (it cannot be chunked across updates like
# streamed text), so an oversized payload is truncated to keep the notification under the buffer.
MAX_RAW_FIELD_CHARS = 16 * 1024


def jsonable(value: object) -> object:
    """Coerce arbitrary tool input/output into something an ACP `session/update` can serialize."""
    # `bytes_mode='base64'` keeps raw (non-UTF-8) bytes from raising; `fallback=str` covers types
    # pydantic cannot otherwise serialize.
    return to_jsonable_python(value, fallback=str, bytes_mode='base64')


def bounded_jsonable(value: object) -> object:
    """`jsonable`, but replace an oversized payload with a truncated marker (see `MAX_RAW_FIELD_CHARS`)."""
    payload = jsonable(value)
    serialized = json.dumps(payload)
    if len(serialized) <= MAX_RAW_FIELD_CHARS:
        return payload
    return {'truncated': True, 'original_length': len(serialized), 'preview': serialized[:MAX_RAW_FIELD_CHARS]}
