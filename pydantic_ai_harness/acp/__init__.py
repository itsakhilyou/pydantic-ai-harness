"""Expose a Pydantic AI agent over the Agent Client Protocol (ACP).

ACP lets terminal UIs and editors (such as Zed and Toad) drive a coding agent over stdio
JSON-RPC. [`PydanticAIACPAgent`][pydantic_ai_harness.acp.PydanticAIACPAgent] adapts a
Pydantic AI [`Agent`][pydantic_ai.Agent] to that interface, and
[`run_acp_stdio`][pydantic_ai_harness.acp.run_acp_stdio] serves it over stdio.
"""

from ._adapter import PydanticAIACPAgent
from ._content import PromptContentBlock
from ._native import AcpFileSystemToolset, AcpTerminalToolset, acp_filesystem, acp_terminal
from ._permission import PermissionPolicy, ToolCallPermission
from ._present import (
    ToolCallContent,
    ToolCallPresentation,
    ToolCallPresenter,
    chain_presenters,
    default_coding_presenter,
)
from ._server import run_acp_stdio, run_acp_stdio_sync
from ._session import AcpSession, AcpSessionConfig, McpServers, SessionConfigFunc, SessionUpdate
from ._store import InMemorySessionStore, SessionStore, StoredSession

__all__ = [
    'AcpFileSystemToolset',
    'AcpSession',
    'AcpSessionConfig',
    'AcpTerminalToolset',
    'InMemorySessionStore',
    'McpServers',
    'PermissionPolicy',
    'PromptContentBlock',
    'PydanticAIACPAgent',
    'SessionConfigFunc',
    'SessionStore',
    'SessionUpdate',
    'StoredSession',
    'ToolCallContent',
    'ToolCallPermission',
    'ToolCallPresentation',
    'ToolCallPresenter',
    'acp_filesystem',
    'acp_terminal',
    'chain_presenters',
    'default_coding_presenter',
    'run_acp_stdio',
    'run_acp_stdio_sync',
]
