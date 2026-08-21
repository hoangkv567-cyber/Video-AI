"""Thin, injectable subprocess runner for ffmpeg/ffprobe.

Command *builders* elsewhere in this package are pure functions returning
``list[str]`` argv; only this module touches ``subprocess``. Tests inject a
fake ``Runner`` and never require the ffmpeg/ffprobe binaries.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

_STDERR_TAIL_CHARS = 4000


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Runner(Protocol):
    """Anything that can execute an argv and report the outcome."""

    def run(self, argv: Sequence[str]) -> CommandResult: ...


class FFmpegError(RuntimeError):
    """A media command exited non-zero. Carries a stderr tail for diagnostics."""

    def __init__(self, argv: Sequence[str], returncode: int, stderr: str):
        self.argv = tuple(argv)
        self.returncode = returncode
        self.stderr_tail = stderr[-_STDERR_TAIL_CHARS:]
        program = self.argv[0] if self.argv else "?"
        super().__init__(f"command {program} failed with exit code {returncode}")


def check(result: CommandResult) -> CommandResult:
    """Raise FFmpegError on non-zero exit; return the result unchanged otherwise."""
    if result.returncode != 0:
        raise FFmpegError(result.argv, result.returncode, result.stderr)
    return result


@dataclass(frozen=True)
class SubprocessRunner:
    """Default production runner. Never used in unit tests (binaries absent)."""

    timeout_seconds: float = 1800.0

    def run(self, argv: Sequence[str]) -> CommandResult:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            check=False,
        )
        return CommandResult(tuple(argv), proc.returncode, proc.stdout or "", proc.stderr or "")
