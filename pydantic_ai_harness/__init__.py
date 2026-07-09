"""The batteries for your Pydantic AI agent -- the official capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .filesystem import FileSystem
    from .logfire import (
        ManagedAgent,
        ManagedAgentSpec,
        ManagedMCP,
        ManagedMCPValue,
        ManagedPrompt,
        ManagedSettings,
        ManagedSkill,
        ManagedSkills,
        ManagedToolDefinitions,
    )
    from .shell import LLM_API_KEY_ENV_PATTERNS, Shell

__all__ = [
    'CodeMode',
    'FileSystem',
    'LLM_API_KEY_ENV_PATTERNS',
    'ManagedAgent',
    'ManagedAgentSpec',
    'ManagedMCP',
    'ManagedMCPValue',
    'ManagedPrompt',
    'ManagedSettings',
    'ManagedSkill',
    'ManagedSkills',
    'ManagedToolDefinitions',
    'Shell',
]


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    if name == 'FileSystem':
        from .filesystem import FileSystem

        return FileSystem
    if name == 'ManagedAgent':
        from .logfire import ManagedAgent

        return ManagedAgent
    if name == 'ManagedAgentSpec':
        from .logfire import ManagedAgentSpec

        return ManagedAgentSpec
    if name == 'ManagedMCP':
        from .logfire import ManagedMCP

        return ManagedMCP
    if name == 'ManagedMCPValue':
        from .logfire import ManagedMCPValue

        return ManagedMCPValue
    if name == 'ManagedPrompt':
        from .logfire import ManagedPrompt

        return ManagedPrompt
    if name == 'ManagedSettings':
        from .logfire import ManagedSettings

        return ManagedSettings
    if name == 'ManagedSkill':
        from .logfire import ManagedSkill

        return ManagedSkill
    if name == 'ManagedSkills':
        from .logfire import ManagedSkills

        return ManagedSkills
    if name == 'ManagedToolDefinitions':
        from .logfire import ManagedToolDefinitions

        return ManagedToolDefinitions
    if name == 'Shell':
        from .shell import Shell

        return Shell
    if name == 'LLM_API_KEY_ENV_PATTERNS':
        from .shell import LLM_API_KEY_ENV_PATTERNS

        return LLM_API_KEY_ENV_PATTERNS
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
