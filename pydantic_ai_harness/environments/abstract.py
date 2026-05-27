"""Abstract base class for all execution environments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(kw_only=True)
class AbstractEnvironment(ABC):
    """Abstract base class for all execution environments."""

    root: str

    @abstractmethod
    async def read_file(self, path: str) -> bytes:
        """Read a file from the environment."""
        raise NotImplementedError

    @abstractmethod
    async def write_file(self, path: str, data: bytes) -> None:
        """Write a file to the environment."""
        raise NotImplementedError
