"""Logfire-backed capabilities: drive agent configuration from Logfire managed variables."""

from pydantic_ai_harness.logfire._managed_agent_spec import ManagedAgentSpec
from pydantic_ai_harness.logfire._managed_prompt import ManagedPrompt

__all__ = ['ManagedAgentSpec', 'ManagedPrompt']
