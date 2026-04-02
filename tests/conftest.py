from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pydantic_ai.models
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

# Prevent accidental real model requests during tests.
pydantic_ai.models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture
def test_model() -> TestModel:
    """A fresh ``TestModel`` instance for each test."""
    return TestModel()


@pytest.fixture
def test_agent(test_model: TestModel) -> Agent[None, str]:
    """A minimal agent wired to ``TestModel`` for capability tests."""
    return Agent(test_model, name='test-agent')


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Convenience alias for ``tmp_path`` (useful for store / session tests)."""
    return tmp_path


@pytest.fixture
def allow_model_requests() -> Iterator[None]:
    """Temporarily allow real model requests within a test."""
    with pydantic_ai.models.override_allow_model_requests(True):
        yield


@pytest.fixture(scope='module', params=['asyncio'])
def anyio_backend(request: pytest.FixtureRequest) -> str:
    """Override anyio backend to asyncio-only.

    Pydantic AI uses ``asyncio.gather`` internally (e.g. in capabilities/combined.py)
    which is incompatible with the Trio event loop.
    """
    return request.param  # type: ignore[return-value]
