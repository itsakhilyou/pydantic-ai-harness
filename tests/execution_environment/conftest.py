"""Shared fixtures for execution_environment capability tests."""

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'
