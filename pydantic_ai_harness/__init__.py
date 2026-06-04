"""Pydantic AI capability library."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_mode import CodeMode
    from .dynamic_workflow import DynamicWorkflow, WorkflowAgent, WorkflowResourceLimits
    from .filesystem import FileSystem
    from .logfire import ManagedPrompt
    from .shell import Shell

__all__ = [
    'CodeMode',
    'DynamicWorkflow',
    'FileSystem',
    'ManagedPrompt',
    'Shell',
    'WorkflowAgent',
    'WorkflowResourceLimits',
]


def __getattr__(name: str) -> object:
    if name == 'CodeMode':
        from .code_mode import CodeMode

        return CodeMode
    if name == 'DynamicWorkflow':
        from .dynamic_workflow import DynamicWorkflow

        return DynamicWorkflow
    if name == 'WorkflowAgent':
        from .dynamic_workflow import WorkflowAgent

        return WorkflowAgent
    if name == 'WorkflowResourceLimits':  # pragma: no cover
        from .dynamic_workflow import WorkflowResourceLimits

        return WorkflowResourceLimits
    if name == 'FileSystem':
        from .filesystem import FileSystem

        return FileSystem
    if name == 'ManagedPrompt':
        from .logfire import ManagedPrompt

        return ManagedPrompt
    if name == 'Shell':
        from .shell import Shell

        return Shell
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
