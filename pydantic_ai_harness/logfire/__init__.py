"""Logfire-backed capabilities: drive agent configuration from Logfire managed variables."""

from pydantic_ai_harness.logfire._managed_agent_spec import ManagedAgent, ManagedAgentSpec
from pydantic_ai_harness.logfire._managed_mcp import ManagedMCP, ManagedMCPValue
from pydantic_ai_harness.logfire._managed_prompt import ManagedPrompt
from pydantic_ai_harness.logfire._managed_settings import ManagedSettings, ManagedSettingsValue
from pydantic_ai_harness.logfire._managed_skills import ManagedSkill, ManagedSkills
from pydantic_ai_harness.logfire._managed_tool_definitions import ManagedToolDefinitions, ToolDefinitionOverride
from pydantic_ai_harness.logfire._managed_variable import resolution_reason

__all__ = [
    'ManagedAgent',
    'ManagedAgentSpec',
    'ManagedMCP',
    'ManagedMCPValue',
    'ManagedPrompt',
    'ManagedSettings',
    'ManagedSettingsValue',
    'ManagedSkill',
    'ManagedSkills',
    'ManagedToolDefinitions',
    'ToolDefinitionOverride',
    'resolution_reason',
]
