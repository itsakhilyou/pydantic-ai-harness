"""A shared in-memory ACP `Client` for the adapter and toolset tests.

`Client` has no abstract methods at runtime, but pyright treats them as abstract (the SDK marks
them so), so a test client must still define the whole surface. `RecordingClient` implements the
filesystem and terminal methods over in-memory state, records the `session/update`s it receives,
and stubs the rest -- covering what `test_native`, `test_persistence`, and `test_models` each need.
"""

from __future__ import annotations

import asyncio

from acp import Client, RequestError, schema


class RecordingClient(Client):
    """An ACP client driving in-memory filesystem/terminal state and recording session updates."""

    def __init__(
        self,
        files: dict[str, str] | None = None,
        *,
        output: str = '',
        truncated: bool = False,
        exit_code: int | None = 0,
        signal: str | None = None,
        no_exit_status: bool = False,
        block_exit: bool = False,
        block_create: bool = False,
    ) -> None:
        self.updates: list[object] = []
        self.files: dict[str, str] = dict(files or {})
        self.reads: list[tuple[str, str]] = []
        self.writes: list[tuple[str, str, str]] = []
        self._output = output
        self._truncated = truncated
        self._exit_code = exit_code
        self._signal = signal
        self._no_exit_status = no_exit_status
        self._block_exit = block_exit
        self._block_create = block_create
        self.release_create = asyncio.Event()
        self.exit_event = asyncio.Event()
        self.created: list[tuple[str, str | None]] = []
        self.killed: list[str] = []
        self.released: list[str] = []
        self._terminals = 0
        self.create_event = asyncio.Event()

    async def session_update(self, session_id: str, update: object, **kwargs: object) -> None:
        self.updates.append(update)

    # --- filesystem -----------------------------------------------------------------------

    async def read_text_file(
        self, path: str, session_id: str, limit: int | None = None, line: int | None = None, **kwargs: object
    ) -> schema.ReadTextFileResponse:
        self.reads.append((path, session_id))
        return schema.ReadTextFileResponse(content=self.files[path])

    async def write_text_file(
        self, content: str, path: str, session_id: str, **kwargs: object
    ) -> schema.WriteTextFileResponse | None:
        self.files[path] = content
        self.writes.append((path, content, session_id))
        return None

    # --- terminal -------------------------------------------------------------------------

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[schema.EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: object,
    ) -> schema.CreateTerminalResponse:
        self._terminals += 1
        self.created.append((command, cwd))
        self.create_event.set()
        if self._block_create:
            await self.release_create.wait()
        return schema.CreateTerminalResponse(terminal_id=f'term-{self._terminals}')

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: object
    ) -> schema.WaitForTerminalExitResponse:
        self.exit_event.set()
        if self._block_exit:
            await asyncio.Event().wait()  # block until cancelled
        return schema.WaitForTerminalExitResponse(exit_code=self._exit_code, signal=self._signal)

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs: object
    ) -> schema.TerminalOutputResponse:
        status = (
            None if self._no_exit_status else schema.TerminalExitStatus(exit_code=self._exit_code, signal=self._signal)
        )
        return schema.TerminalOutputResponse(output=self._output, truncated=self._truncated, exit_status=status)

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: object
    ) -> schema.KillTerminalResponse | None:
        self.killed.append(terminal_id)
        return None

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: object
    ) -> schema.ReleaseTerminalResponse | None:
        self.released.append(terminal_id)
        return None

    # --- unused surface (present only to satisfy the interface) ----------------------------

    def on_connect(self, conn: object) -> None:
        return None  # pragma: no cover - unused

    async def request_permission(
        self,
        options: list[schema.PermissionOption],
        session_id: str,
        tool_call: schema.ToolCallUpdate,
        **kwargs: object,
    ) -> schema.RequestPermissionResponse:
        raise NotImplementedError  # pragma: no cover - unused

    async def ext_method(self, method: str, params: dict[str, object]) -> dict[str, object]:
        raise RequestError.method_not_found(method)  # pragma: no cover - unused

    async def ext_notification(self, method: str, params: dict[str, object]) -> None:
        return None  # pragma: no cover - unused
