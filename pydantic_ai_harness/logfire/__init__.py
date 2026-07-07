"""Logfire-backed capabilities: drive agent configuration from Logfire managed variables."""

from pydantic_ai_harness.logfire._managed_agent_spec import ManagedAgent, ManagedAgentSpec
from pydantic_ai_harness.logfire._managed_prompt import ManagedPrompt
from pydantic_ai_harness.logfire._managed_settings import ManagedModelSettings, ManagedSettings, ManagedSettingsValue
from pydantic_ai_harness.logfire._managed_tool_definitions import ManagedToolDefinitions, ToolDefinitionOverride

__all__ = [
    'ManagedAgent',
    'ManagedAgentSpec',
    'ManagedModelSettings',
    'ManagedPrompt',
    'ManagedSettings',
    'ManagedSettingsValue',
    'ManagedToolDefinitions',
    'ToolDefinitionOverride',
]
