"""File-level checkpoints via shadow git (private, not re-exported at top level).

Snapshots the project's files into a git repository separate from the user's own
`.git` before a mutating tool runs, so file damage is restorable. Conversation
rewind/fork is out of scope for v1 -- that pairs with the branchable
session-history track (harness issue #321).
"""

from pydantic_ai_harness.experimental._warn import warn_experimental
from pydantic_ai_harness.experimental.checkpoints._capability import (
    DEFAULT_BASH_TOOLS,
    DEFAULT_MUTATING_TOOLS,
    Checkpoints,
    CheckpointWarning,
)
from pydantic_ai_harness.experimental.checkpoints._shadow import (
    Checkpoint,
    CheckpointError,
    CheckpointStore,
)
from pydantic_ai_harness.experimental.checkpoints._toolset import RestoreCheckpointToolset

warn_experimental('checkpoints')

__all__ = [
    'DEFAULT_BASH_TOOLS',
    'DEFAULT_MUTATING_TOOLS',
    'Checkpoint',
    'CheckpointError',
    'CheckpointStore',
    'CheckpointWarning',
    'Checkpoints',
    'RestoreCheckpointToolset',
]
