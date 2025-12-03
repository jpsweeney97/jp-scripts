"""Types and data structures for constitutional compliance checking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class ViolationType(Enum):
    """Types of constitutional violations."""

    SYNC_SUBPROCESS = auto()  # subprocess.run without async wrapping
    BARE_EXCEPT = auto()  # except: without specific exception
    SHELL_TRUE = auto()  # shell=True in subprocess calls
    UNTYPED_ANY = auto()  # Any type without type: ignore comment
    SYNC_OPEN = auto()  # open() in async context without to_thread
    OS_SYSTEM = auto()  # os.system() usage (always forbidden)
    DESTRUCTIVE_FS = auto()  # Destructive filesystem call without safety override
    DYNAMIC_EXECUTION = auto()  # eval/exec/dynamic imports without safety override
    SECRET_LEAK = auto()  # Secret or token detected in diff
    PROCESS_EXIT = auto()  # sys.exit(), quit(), exit()
    DEBUG_LEFTOVER = auto()  # breakpoint(), pdb.set_trace(), ipdb.set_trace()
    SYNTAX_ERROR = auto()  # Python syntax error prevents AST analysis
    SECURITY_BYPASS = auto()  # Agent attempted to add # safety: checked override


@dataclass(frozen=True)
class Violation:
    """A single constitutional violation."""

    type: ViolationType
    file: Path
    line: int
    column: int
    message: str
    suggestion: str
    severity: str  # "error" | "warning"
    fatal: bool = False  # Fatal violations block patch application


__all__ = [
    "Violation",
    "ViolationType",
]
