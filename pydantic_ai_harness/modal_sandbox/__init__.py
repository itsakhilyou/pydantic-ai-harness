"""Modal sandbox capability: gives agents an isolated cloud sandbox to work in.

`ModalSandboxCapability` is the supported entry point; build an agent with it and use its
tools. `ModalSandboxSession` and `ModalSandboxToolset` are lower-level building
blocks exposed for advanced use. They are kept deliberately separate -- the
session owns the sandbox mechanism (running commands, file access, lifecycle) and
the toolset owns how that is presented to the model -- so the sandbox internals
can change without disturbing the capability or its tool surface. Treat the
lower-level pieces as more likely to change than `ModalSandboxCapability` itself.
"""

from pydantic_ai_harness.modal_sandbox._capability import ModalSandboxCapability
from pydantic_ai_harness.modal_sandbox._session import (
    ModalSandboxError,
    ModalSandboxSession,
    ModalSandboxTerminalError,
    ModalSandboxUnavailableError,
)
from pydantic_ai_harness.modal_sandbox._toolset import ModalSandboxToolset

__all__ = [
    'ModalSandboxCapability',
    'ModalSandboxError',
    'ModalSandboxSession',
    'ModalSandboxTerminalError',
    'ModalSandboxToolset',
    'ModalSandboxUnavailableError',
]
