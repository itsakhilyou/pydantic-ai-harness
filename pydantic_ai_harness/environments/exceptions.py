"""Exceptions for the execution environments."""


class ExecutionEnvironmentError(Exception):
    """Base class for all execution environment errors."""


class PathEscapeError(ExecutionEnvironmentError):
    """Path escape error."""


class EnvFileNotFoundError(ExecutionEnvironmentError):
    """File not found in the environment."""


class EnvFilePermissionError(ExecutionEnvironmentError):
    """File permission error."""


class EnvFileIsADirectoryError(ExecutionEnvironmentError):
    """File is a directory."""


class EnvFileNotADirectoryError(ExecutionEnvironmentError):
    """File is not a directory."""


class EnvFileTooLargeError(ExecutionEnvironmentError):
    """File too large."""


class EnvReadError(ExecutionEnvironmentError):
    """Unexpected I/O failure during a non-mutating operation (e.g. `read_file`, `ls`).

    The catch-all for any OS error a read-shaped operation raises that is not one of the
    specific subclasses above. Nothing changed on disk.
    """


class EnvWriteError(ExecutionEnvironmentError):
    """Unexpected I/O failure during a mutating operation (e.g. `write_file`).

    The catch-all for any OS error a write-shaped operation raises that is not one of the
    specific subclasses above. State may have been partially changed.
    """
