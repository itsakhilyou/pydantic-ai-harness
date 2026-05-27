from pathlib import Path

import pytest
from pydantic_ai import Agent, ModelResponse, TextPart
from pydantic_ai.messages import ModelMessage, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.environments.local import LocalEnvironment
from pydantic_ai_harness.execution_env import ExecutionEnv


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_execution_env_capability_read_file(tmp_path: Path) -> None:
    # Let us write a file into the path first
    file_name = 'test.txt'
    (tmp_path / file_name).write_text('Hello, world!')

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        already_read = any(
            isinstance(part, ToolReturnPart) and part.tool_name == 'read_file' for msg in messages for part in msg.parts
        )
        if already_read:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='read_file', args={'path': file_name})])

    agent = Agent(
        FunctionModel(model_fn), capabilities=[ExecutionEnv(environment=LocalEnvironment(root=str(tmp_path)))]
    )

    result = await agent.run(
        f'Read the file {file_name} and return the contents.',
    )

    returns = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'read_file'
    ]

    assert returns == ['Hello, world!']
