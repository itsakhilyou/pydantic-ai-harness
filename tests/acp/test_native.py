"""Tests for the ACP-client-backed filesystem and terminal toolsets (`_native.py`)."""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest
from acp import Client, schema
from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.acp import (
    AcpFileSystemToolset,
    AcpSession,
    AcpTerminalToolset,
    acp_filesystem,
    acp_terminal,
)
from tests._acp_clients import RecordingClient  # pyright: ignore[reportMissingTypeStubs]

pytestmark = pytest.mark.anyio


def _ctx() -> RunContext[None]:
    return RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=1)


def _session(client: Client, capabilities: schema.ClientCapabilities | None) -> AcpSession:
    return AcpSession(
        cwd='/ws',
        mcp_servers=[],
        client_capabilities=capabilities,
        client=client,
        session_id='sid',
    )


def _fs_caps(*, read: bool, write: bool) -> schema.ClientCapabilities:
    return schema.ClientCapabilities(fs=schema.FileSystemCapabilities(read_text_file=read, write_text_file=write))


# --- Filesystem toolset ----------------------------------------------------


async def test_read_file_reads_through_the_client() -> None:
    client = RecordingClient({'/ws/a.py': 'hello'})
    ts = AcpFileSystemToolset[None](client=client, session_id='sid')
    assert await ts.read_file('/ws/a.py') == 'hello'
    assert client.reads == [('/ws/a.py', 'sid')]  # path and session id reached the client unchanged


async def test_write_file_writes_through_the_client() -> None:
    client = RecordingClient()
    ts = AcpFileSystemToolset[None](client=client, session_id='sid')
    result = await ts.write_file('/ws/b.py', 'data')
    assert client.writes == [('/ws/b.py', 'data', 'sid')]
    assert client.files['/ws/b.py'] == 'data'
    assert '/ws/b.py' in result  # confirmation names the path so the model knows the write landed


async def test_filesystem_registers_read_file_and_write_file_tools() -> None:
    # The tool names match the local FileSystem capability so the default presenter renders them.
    ts = AcpFileSystemToolset[None](client=RecordingClient(), session_id='sid')
    assert set(await ts.get_tools(_ctx())) == {'read_file', 'write_file'}


async def test_acp_filesystem_builds_a_working_toolset_when_fs_is_advertised() -> None:
    client = RecordingClient({'/ws/a.py': 'hi'})
    toolset = acp_filesystem(_session(client, _fs_caps(read=True, write=True)))
    assert isinstance(toolset, AcpFileSystemToolset)
    assert await toolset.read_file('/ws/a.py') == 'hi'  # the built toolset routes through the same client


async def test_acp_filesystem_read_only_client_reads_via_acp_and_writes_locally(tmp_path: Path) -> None:
    # A read-only client keeps editor-native reads, but writes go to the local workspace disk
    # rather than the client (coherent only when the agent shares that disk -- see the helper docs).
    client = RecordingClient({'notes.txt': 'hello'})
    session = _session(client, _fs_caps(read=True, write=False))
    session = AcpSession(
        cwd=str(tmp_path),
        mcp_servers=session.mcp_servers,
        client_capabilities=session.client_capabilities,
        client=client,
        session_id=session.session_id,
    )
    toolset = acp_filesystem(session)
    assert isinstance(toolset, AcpFileSystemToolset)

    assert await toolset.read_file('notes.txt') == 'hello'
    assert client.reads == [('notes.txt', 'sid')]  # the read routed through the editor
    await toolset.write_file('out.txt', 'data')
    assert client.writes == []  # the client was never asked to write
    assert (tmp_path / 'out.txt').read_text() == 'data'  # the write landed on local disk


@pytest.mark.parametrize(
    'capabilities',
    [
        pytest.param(None, id='no-capabilities'),
        pytest.param(schema.ClientCapabilities(), id='no-fs'),
        pytest.param(_fs_caps(read=False, write=True), id='no-read'),
    ],
)
def test_acp_filesystem_returns_none_when_no_readable_filesystem(
    capabilities: schema.ClientCapabilities | None,
) -> None:
    assert acp_filesystem(_session(RecordingClient(), capabilities)) is None


# --- Terminal toolset ------------------------------------------------------


async def test_run_command_creates_a_terminal_in_the_session_cwd() -> None:
    client = RecordingClient(output='ok')
    await AcpTerminalToolset[None](client=client, session_id='sid', cwd='/ws').run_command('ls')
    assert client.created == [('ls', '/ws')]


async def test_terminal_registers_run_command_tool() -> None:
    ts = AcpTerminalToolset[None](client=RecordingClient(), session_id='sid')
    assert set(await ts.get_tools(_ctx())) == {'run_command'}


@pytest.mark.parametrize(
    'kwargs, expected',
    [
        pytest.param({'output': 'hello', 'exit_code': 0}, 'hello', id='success'),
        pytest.param({'output': 'boom', 'exit_code': 2}, 'boom\n[exited with code 2]', id='nonzero-exit'),
        pytest.param(
            {'output': 'gone', 'exit_code': None, 'signal': 'SIGKILL'},
            'gone\n[terminated by signal SIGKILL]',
            id='signal',
        ),
        pytest.param(
            {'output': 'partial', 'truncated': True, 'no_exit_status': True},
            'partial\n[output truncated]',
            id='truncated-no-status',
        ),
    ],
)
async def test_run_command_formats_output_and_releases(kwargs: dict[str, object], expected: str) -> None:
    client = RecordingClient(**kwargs)  # pyright: ignore[reportArgumentType]
    result = await AcpTerminalToolset[None](client=client, session_id='sid', cwd='/ws').run_command('cmd')
    assert result == expected
    assert client.released == ['term-1']  # the terminal is always released


async def test_run_command_kills_and_releases_the_terminal_on_cancel() -> None:
    client = RecordingClient(block_exit=True)
    ts = AcpTerminalToolset[None](client=client, session_id='sid', cwd='/ws')
    async with anyio.create_task_group() as tg:
        tg.start_soon(ts.run_command, 'sleep 100')
        await client.create_event.wait()  # the terminal exists and the command is running
        tg.cancel_scope.cancel()
    assert client.killed == ['term-1']  # killed before unwinding
    assert client.released == ['term-1']  # and released so it is not left behind


async def test_run_command_cancelled_during_create_still_kills_the_terminal() -> None:
    client = RecordingClient(block_create=True)
    ts = AcpTerminalToolset[None](client=client, session_id='sid', cwd='/ws')
    async with anyio.create_task_group() as tg:
        tg.start_soon(ts.run_command, 'sleep 100')
        await client.create_event.wait()  # the create request is in flight; no terminal id yet
        tg.cancel_scope.cancel()
        client.release_create.set()  # the client answers the create only after the cancellation
    # The request was already on the wire, so the client started the command regardless; the
    # late-learned terminal must still be killed and released, not leaked running in the editor.
    assert client.killed == ['term-1']
    assert client.released == ['term-1']


async def test_run_command_cancel_survives_a_failing_kill() -> None:
    class _KillRaises(RecordingClient):
        async def kill_terminal(
            self, session_id: str, terminal_id: str, **kwargs: object
        ) -> schema.KillTerminalResponse | None:
            self.killed.append(terminal_id)
            raise RuntimeError('client kill failed')

    client = _KillRaises(block_exit=True)
    ts = AcpTerminalToolset[None](client=client, session_id='sid', cwd='/ws')
    async with anyio.create_task_group() as tg:
        tg.start_soon(ts.run_command, 'sleep 100')
        await client.create_event.wait()
        tg.cancel_scope.cancel()
    # The kill failure is suppressed, so cancellation still unwinds cleanly and the terminal is
    # still released rather than leaked.
    assert client.killed == ['term-1']
    assert client.released == ['term-1']


async def test_acp_terminal_builds_a_toolset_when_terminal_is_advertised() -> None:
    client = RecordingClient(output='hi')
    toolset = acp_terminal(_session(client, schema.ClientCapabilities(terminal=True)))
    assert isinstance(toolset, AcpTerminalToolset)
    assert await toolset.run_command('echo hi') == 'hi'  # the built toolset routes through the same client


@pytest.mark.parametrize(
    'capabilities',
    [
        pytest.param(None, id='no-capabilities'),
        pytest.param(schema.ClientCapabilities(terminal=False), id='terminal-false'),
        pytest.param(schema.ClientCapabilities(), id='terminal-unset'),
    ],
)
def test_acp_terminal_returns_none_when_terminal_is_unsupported(
    capabilities: schema.ClientCapabilities | None,
) -> None:
    assert acp_terminal(_session(RecordingClient(), capabilities)) is None
