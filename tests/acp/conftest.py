"""Shared fixtures for the ACP adapter tests."""

from __future__ import annotations

import importlib.util

import pytest

# The `agent-client-protocol` dependency is gated on the `acp` extra, so slim CI runs
# (no extras) can't import these modules. Ignore them at collection; `test_packaging.py`
# stays collected because it checks package metadata, which holds on base installs too.
if importlib.util.find_spec('acp') is None:
    collect_ignore = [
        'test_acp.py',
        'test_content.py',
        'test_models.py',
        'test_native.py',
        'test_persistence.py',
    ]


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'
