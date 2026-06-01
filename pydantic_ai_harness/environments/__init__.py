"""Execution environments for Pydantic AI."""

from typing import TYPE_CHECKING

from .abstract import AbstractEnvironment, AbstractFile, AbstractMatch
from .local import LocalEnvironment

if TYPE_CHECKING:
    from .docker import DockerEnvironment

__all__ = ['AbstractEnvironment', 'AbstractFile', 'AbstractMatch', 'DockerEnvironment', 'LocalEnvironment']


def __getattr__(name: str) -> object:
    # `DockerEnvironment` is resolved lazily so importing this package (e.g. for
    # `LocalEnvironment`) does not require the optional `docker` dependency.
    if name == 'DockerEnvironment':
        from .docker import DockerEnvironment

        return DockerEnvironment
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
