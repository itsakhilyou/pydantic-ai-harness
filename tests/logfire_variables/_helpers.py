"""Plain test helpers shared by the managed-variable capability test modules.

Fixtures live in `conftest.py`; this module holds importable functions (tool bodies, the
tool-capturing model, and the local-provider context manager) so the capability test files can
share them without copy-paste.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import logfire
from logfire.testing import CaptureLogfire
from logfire.variables import VariablesConfig
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition


def get_weather(city: str) -> str:
    return f'sunny in {city}'


def get_forecast(city: str) -> str:
    return f'forecast for {city}'


def capture_tools(seen: list[ToolDefinition]) -> FunctionModel:
    """A model that records the advertised function tools it is shown, then ends the run."""

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(info.function_tools)
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


def advertised(seen: list[ToolDefinition]) -> dict[str, str | None]:
    """The advertised `{name: description}` of each captured tool definition."""
    return {td.name: td.description for td in seen}


@contextmanager
def variables_provider(capfire: CaptureLogfire, variables_config: VariablesConfig) -> Generator[None]:
    """Reconfigure Logfire with a local variables provider for the duration of the block.

    Restores the module's baseline configuration on exit so the change does not leak into other
    tests in this module (or any module collected after it).
    """
    logfire.configure(
        send_to_logfire=False,
        console=False,
        variables=logfire.LocalVariablesOptions(config=variables_config),
        additional_span_processors=[SimpleSpanProcessor(capfire.exporter)],
    )
    try:
        yield
    finally:
        logfire.configure(send_to_logfire=False, console=False)
