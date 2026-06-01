from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

import pydantic_ai_harness


def test_import():
    assert pydantic_ai_harness.__doc__ is not None
    assert isinstance(pydantic_ai_harness.__all__, list)


def test_lazy_exports_resolve():
    """Every name in `__all__` resolves through the lazy `__getattr__`, and unknown names raise."""
    from pydantic_ai_harness.code_mode import CodeMode
    from pydantic_ai_harness.execution_environment import ExecutionEnvironment

    assert pydantic_ai_harness.CodeMode is CodeMode
    assert pydantic_ai_harness.ExecutionEnvironment is ExecutionEnvironment
    assert set(pydantic_ai_harness.__all__) == {'CodeMode', 'ExecutionEnvironment'}

    with pytest.raises(AttributeError):
        pydantic_ai_harness.DoesNotExist  # pyright: ignore[reportAttributeAccessIssue]


def test_environments_lazy_exports_resolve():
    """`DockerEnvironment` resolves lazily so importing the `environments` package (e.g. for
    `LocalEnvironment`) never pulls the optional `docker` dependency. Unknown names raise."""
    import pydantic_ai_harness.environments as environments

    with pytest.raises(AttributeError):
        environments.DoesNotExist  # pyright: ignore[reportAttributeAccessIssue]

    pytest.importorskip('docker')  # the optional `docker` extra; skipped on the slim CI leg
    from pydantic_ai_harness.environments.docker import DockerEnvironment

    assert environments.DockerEnvironment is DockerEnvironment


def test_test_model_fixture(test_model: TestModel):
    assert isinstance(test_model, TestModel)


def test_test_agent_fixture(test_agent: Agent[None, str]):
    assert test_agent.name == 'test-agent'


def test_tmp_dir_fixture(tmp_dir: Path):
    assert tmp_dir.is_dir()


async def test_allow_model_requests(allow_model_requests: None):
    import pydantic_ai.models

    assert pydantic_ai.models.ALLOW_MODEL_REQUESTS is True
