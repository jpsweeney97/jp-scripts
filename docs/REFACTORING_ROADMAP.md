# Refactoring Roadmap: Architecture Consolidation

**Created:** 2025-12-01
**Status:** NOT STARTED
**Last Updated:** 2025-12-01

---

## Progress Tracker

### Overall Status
- [x] **Phase 1:** Extract AST skeleton to `analysis/skeleton.py` (COMPLETED - 42a742c)
- [x] **Phase 2:** Decompose `core/system.py` into `core/sys/` package (COMPLETED - 28ae0fc)
- [x] **Phase 3:** Merge `engine/` into `agent/` (COMPLETED - 577e588)

### Current Position
**Phase:** ALL COMPLETE
**Step:** N/A
**Blocked:** No

---

## Critical Notes (From Gap Analysis)

### Discovered Issues
1. **`AUDIT_PREFIX`** is in `engine/tool_executor.py:29`, NOT in `system.py`
2. **Two `run_safe_shell` functions exist:**
   - `system.py:274-308` - returns `Result[CommandResult, SystemResourceError]`
   - `engine/tool_executor.py:86-154` - returns `str` (different API!)
3. **Nested helpers in `get_file_skeleton`** - `_node_length`, `_doc_expr`, `_skeletonize_function`, `_skeletonize_class` are defined INSIDE the function (lines 300-355), must be extracted as module-level
4. **`DEFAULT_MODEL_CONTEXT_LIMIT`** - imported from `jpscripts.ai.tokens` (line 29 of context_gatherer.py)
5. **Internal helpers needed for tests** - `_clean_json_payload` and `_extract_balanced_json` from response_handler.py are tested in `test_json_extraction.py`, so they must be exported from `agent/parsing.py` and included in `__all__`

### Verification Commands (Run Before Each Phase)
```bash
# Find ALL imports of a module (run before updating consumers)
grep -rn "jpscripts.core.system\|from jpscripts.core import system" src/ tests/
grep -rn "jpscripts.engine\|from jpscripts import engine" src/ tests/

# Check for dynamic imports
grep -rn "importlib\|__import__" src/jpscripts/

# Type check after each phase
mypy src/jpscripts/analysis/skeleton.py  # Phase 1
mypy src/jpscripts/core/sys/             # Phase 2
mypy src/jpscripts/agent/                # Phase 3

# Star import test
python -c "from jpscripts.core.sys import *; print('OK')"
python -c "from jpscripts.agent import *; print('OK')"
```

---

## Phase 1: Extract AST Logic from context_gatherer.py

**Status:** COMPLETED
**Risk Level:** LOW
**Estimated Files:** 3

### Goal
Move AST skeleton extraction functions from `core/context_gatherer.py` to `analysis/skeleton.py` to improve separation of concerns.

### Pre-conditions
- [ ] 1.0.1: Run baseline tests: `pytest tests/unit/test_context.py -v`
- [ ] 1.0.2: Verify no dynamic imports: `grep -rn "importlib.*context_gatherer\|__import__.*context" src/`

### Step 1.1: Create analysis/skeleton.py
**Status:** NOT STARTED

**Action:** Create new file `src/jpscripts/analysis/skeleton.py`

**Exact content to extract from context_gatherer.py:**
```
Line 40:      SYNTAX_WARNING constant
Lines 300-303: _node_length() - NESTED inside get_file_skeleton, extract as module-level
Lines 305-308: _doc_expr() - NESTED inside get_file_skeleton, extract as module-level
Lines 310-330: _skeletonize_function() - NESTED inside get_file_skeleton, extract as module-level
Lines 332-355: _skeletonize_class() - NESTED inside get_file_skeleton, extract as module-level
Lines 401-405: _line_offsets() - already module-level
Lines 408-413: _is_parseable() - already module-level
Lines 416-472: _fallback_read() - already module-level
Lines 283-390: get_file_skeleton() - main function (refactor to take source string)
```

**Required imports for skeleton.py:**
```python
from __future__ import annotations

import ast

from jpscripts.core.console import get_logger

logger = get_logger(__name__)

# Constant
SYNTAX_WARNING = "# [WARN] Syntax error detected. AST features disabled.\n"
```

**Signature change:**
- Original: `get_file_skeleton(path: Path, *, limit: int = DEFAULT_MODEL_CONTEXT_LIMIT) -> str`
- New: `get_file_skeleton(source: str, *, limit: int = 1_000_000) -> str`
- Remove file reading (lines 295-298), caller provides source string
- Default limit uses literal `1_000_000` (avoids import from ai.tokens)

**Function structure in skeleton.py:**
```python
def _node_length(node: ast.AST) -> int:
    """Calculate line span of an AST node."""
    start = getattr(node, "lineno", 0)
    end = getattr(node, "end_lineno", start)
    return max(end - start + 1, 0)

def _doc_expr(raw: str | None) -> ast.Expr | None:
    """Convert docstring to AST expression node."""
    if raw is None:
        return None
    return ast.Expr(value=ast.Constant(value=raw))

def _skeletonize_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.FunctionDef | ast.AsyncFunctionDef:
    # ... (extract lines 310-330)

def _skeletonize_class(node: ast.ClassDef) -> ast.ClassDef:
    # ... (extract lines 332-355)

def _line_offsets(text: str) -> list[int]:
    # ... (copy lines 401-405)

def _is_parseable(snippet: str) -> bool:
    # ... (copy lines 408-413)

def _fallback_read(text: str, limit: int, error: SyntaxError | None) -> str:
    # ... (copy lines 416-472)

def get_file_skeleton(source: str, *, limit: int = 1_000_000) -> str:
    """Return AST skeleton of Python source code.

    Args:
        source: Python source code as string
        limit: Maximum output length (default 1M chars)
    """
    # Remove file reading, start at line 300 logic
    # ... rest of function
```

**Verification:**
- [ ] File created with all functions
- [ ] No syntax errors: `python -c "from jpscripts.analysis.skeleton import get_file_skeleton"`
- [ ] Type check: `mypy src/jpscripts/analysis/skeleton.py`

### Step 1.2: Update analysis/__init__.py
**Status:** NOT STARTED

**Action:** Add exports to `src/jpscripts/analysis/__init__.py`

**Add these lines:**
```python
from jpscripts.analysis.skeleton import SYNTAX_WARNING, get_file_skeleton
```

**Add to `__all__` list:**
```python
"SYNTAX_WARNING",
"get_file_skeleton",
```

**Verification:**
- [ ] Import works: `python -c "from jpscripts.analysis import get_file_skeleton"`

### Step 1.3: Update context_gatherer.py
**Status:** NOT STARTED

**Action:** Modify `src/jpscripts/core/context_gatherer.py`

**Changes:**
1. Add import at top: `from jpscripts.analysis.skeleton import SYNTAX_WARNING, get_file_skeleton as _get_skeleton`
2. Remove the following functions (delete entirely):
   - `_node_length`
   - `_doc_expr`
   - `_skeletonize_function`
   - `_skeletonize_class`
   - `_fallback_read`
   - `_line_offsets`
   - `_is_parseable`
   - The original `get_file_skeleton` function body
3. Remove `SYNTAX_WARNING` constant definition
4. Remove `import ast` if no longer needed
5. Create wrapper function:
```python
def get_file_skeleton(path: Path, *, limit: int = DEFAULT_MODEL_CONTEXT_LIMIT) -> str:
    """Return AST skeleton of a Python file. Delegates to analysis.skeleton."""
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return _get_skeleton(source, limit=limit)
```

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.core.context_gatherer import get_file_skeleton"`
- [ ] Function works: quick manual test

### Step 1.4: Run Tests
**Status:** NOT STARTED

**Action:** Run test suite for context module

**Commands:**
```bash
pytest tests/unit/test_context.py -v
```

**Expected outcome:** All tests pass

**If tests fail:**
- Check import paths
- Verify function signatures match
- Check that `limit` parameter is passed correctly

### Step 1.5: Commit and Push
**Status:** NOT STARTED

**Action:** Commit Phase 1 changes

**Commands:**
```bash
git add src/jpscripts/analysis/skeleton.py
git add src/jpscripts/analysis/__init__.py
git add src/jpscripts/core/context_gatherer.py
git status  # Verify only expected files
git commit -m "refactor: extract AST skeleton logic to analysis/skeleton.py

- Move get_file_skeleton and helper functions to analysis/skeleton.py
- Update context_gatherer.py to delegate to new module
- Improves separation of concerns (AST parsing vs file I/O)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
git push
```

**Verification:**
- [ ] Commit successful
- [ ] Push successful

---

## Phase 2: Decompose core/system.py into core/sys/ Package

**Status:** NOT STARTED
**Risk Level:** MEDIUM
**Estimated Files:** 14

### Goal
Split the monolithic `core/system.py` (490 LOC) into a `core/sys/` package with specialized modules for process management, audio, networking, command execution, and package management.

### Pre-conditions
- [ ] 2.0.1: Phase 1 completed and pushed
- [ ] 2.0.2: Read `src/jpscripts/core/system.py` to identify exact line ranges for each function group
- [ ] 2.0.3: Run baseline tests: `pytest tests/unit/test_system_commands.py tests/unit/test_dry_run_and_ast.py -v`

### Step 2.1: Create core/sys/ directory structure
**Status:** NOT STARTED

**Action:** Create directory and empty files

**Commands:**
```bash
mkdir -p src/jpscripts/core/sys
touch src/jpscripts/core/sys/__init__.py
touch src/jpscripts/core/sys/execution.py
touch src/jpscripts/core/sys/process.py
touch src/jpscripts/core/sys/audio.py
touch src/jpscripts/core/sys/network.py
touch src/jpscripts/core/sys/package.py
```

**Verification:**
- [ ] Directory exists: `ls src/jpscripts/core/sys/`

### Step 2.2: Create sys/execution.py
**Status:** NOT STARTED

**Action:** Create `src/jpscripts/core/sys/execution.py`

**IMPORTANT:** `AUDIT_PREFIX` is NOT in system.py - it's in `engine/tool_executor.py:29`. Do NOT include it here.

**Content to move from system.py (with exact line numbers):**
- Lines 46-51: `CommandResult` dataclass
- Lines 53-59: `SandboxProtocol` protocol class
- Lines 62-95: `LocalSandbox` class
- Lines 98-168: `DockerSandbox` class
- Lines 171-174: `get_sandbox()` function
- Lines 274-308: `run_safe_shell()` function (returns `Result[CommandResult, SystemResourceError]`)

**Required imports (copy from system.py lines 10-29):**
```python
from __future__ import annotations

import asyncio
import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from jpscripts.core.command_validation import CommandVerdict, validate_command
from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger
from jpscripts.core.result import Err, Ok, Result, SystemResourceError
from jpscripts.core.runtime import get_runtime

logger = get_logger(__name__)
```

**Exports (NO AUDIT_PREFIX):**
```python
__all__ = [
    "CommandResult",
    "DockerSandbox",
    "LocalSandbox",
    "SandboxProtocol",
    "get_sandbox",
    "run_safe_shell",
]
```

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.core.sys.execution import run_safe_shell, CommandResult"`
- [ ] Type check: `mypy src/jpscripts/core/sys/execution.py`

### Step 2.3: Create sys/process.py
**Status:** NOT STARTED

**Action:** Create `src/jpscripts/core/sys/process.py`

**Content to move from system.py (with exact line numbers):**
- Lines 34-44: `ProcessInfo` dataclass (includes `label` property)
- Lines 177-181: `_format_cmdline()` helper function
- Lines 184-226: `find_processes()` async function
- Lines 229-267: `kill_process_async()` async function
- Lines 270-271: `kill_process()` sync wrapper function

**Required imports:**
```python
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import psutil

from jpscripts.core.console import get_logger
from jpscripts.core.result import Err, Ok, Result, SystemResourceError
from jpscripts.core.runtime import get_runtime

logger = get_logger(__name__)
```

**Cross-dependency:** `kill_process_async` calls `get_sandbox()` (line 237).
**Solution:** Import from sibling: `from jpscripts.core.sys.execution import DockerSandbox, get_sandbox`

**Exports:**
```python
__all__ = [
    "ProcessInfo",
    "find_processes",
    "kill_process",
    "kill_process_async",
]
```

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.core.sys.process import ProcessInfo, find_processes"`
- [ ] Type check: `mypy src/jpscripts/core/sys/process.py`

### Step 2.4: Create sys/audio.py
**Status:** NOT STARTED

**Action:** Create `src/jpscripts/core/sys/audio.py`

**Content to move from system.py (with exact line numbers):**
- Lines 311-337: `get_audio_devices()` async function
- Lines 340-369: `set_audio_device()` async function

**Required imports:**
```python
from __future__ import annotations

import asyncio
import shutil

from jpscripts.core.console import get_logger
from jpscripts.core.result import Err, Ok, Result, SystemResourceError

logger = get_logger(__name__)
```

**Exports:**
```python
__all__ = [
    "get_audio_devices",
    "set_audio_device",
]
```

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.core.sys.audio import get_audio_devices"`
- [ ] Type check: `mypy src/jpscripts/core/sys/audio.py`

### Step 2.5: Create sys/network.py
**Status:** NOT STARTED

**Action:** Create `src/jpscripts/core/sys/network.py`

**Content to move from system.py (with exact line numbers):**
- Lines 372-398: `get_ssh_hosts()` async function
- Lines 401-429: `run_temp_server()` async function

**Required imports:**
```python
from __future__ import annotations

import asyncio
import functools
import http.server
from collections.abc import Callable
from pathlib import Path

from jpscripts.core.console import get_logger
from jpscripts.core.result import Err, Ok, Result, SystemResourceError

logger = get_logger(__name__)
```

**Exports:**
```python
__all__ = [
    "get_ssh_hosts",
    "run_temp_server",
]
```

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.core.sys.network import get_ssh_hosts"`
- [ ] Type check: `mypy src/jpscripts/core/sys/network.py`

### Step 2.6: Create sys/package.py
**Status:** NOT STARTED

**Action:** Create `src/jpscripts/core/sys/package.py`

**Content to move from system.py (with exact line numbers):**
- Lines 432-460: `search_brew()` async function
- Lines 463-489: `get_brew_info()` async function

**Required imports:**
```python
from __future__ import annotations

import asyncio
import shutil

from jpscripts.core.console import get_logger
from jpscripts.core.result import Err, Ok, Result, SystemResourceError

logger = get_logger(__name__)
```

**Exports:**
```python
__all__ = [
    "get_brew_info",
    "search_brew",
]
```

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.core.sys.package import search_brew"`
- [ ] Type check: `mypy src/jpscripts/core/sys/package.py`

### Step 2.7: Create sys/__init__.py with re-exports
**Status:** NOT STARTED

**Action:** Create `src/jpscripts/core/sys/__init__.py`

**IMPORTANT:** NO `AUDIT_PREFIX` - it's not in system.py!

**Content:**
```python
"""System utilities package.

Organized submodules:
- execution: Command execution, sandboxes, run_safe_shell
- process: Process management (find, kill)
- audio: Audio device control (macOS)
- network: SSH hosts, temp HTTP server
- package: Homebrew utilities
"""

from jpscripts.core.sys.audio import get_audio_devices, set_audio_device
from jpscripts.core.sys.execution import (
    CommandResult,
    DockerSandbox,
    LocalSandbox,
    SandboxProtocol,
    get_sandbox,
    run_safe_shell,
)
from jpscripts.core.sys.network import get_ssh_hosts, run_temp_server
from jpscripts.core.sys.package import get_brew_info, search_brew
from jpscripts.core.sys.process import (
    ProcessInfo,
    find_processes,
    kill_process,
    kill_process_async,
)

__all__ = [
    # execution (NO AUDIT_PREFIX)
    "CommandResult",
    "DockerSandbox",
    "LocalSandbox",
    "SandboxProtocol",
    "get_sandbox",
    "run_safe_shell",
    # process
    "ProcessInfo",
    "find_processes",
    "kill_process",
    "kill_process_async",
    # audio
    "get_audio_devices",
    "set_audio_device",
    # network
    "get_ssh_hosts",
    "run_temp_server",
    # package
    "get_brew_info",
    "search_brew",
]
```

**Verification:**
- [ ] All imports work: `python -c "from jpscripts.core.sys import ProcessInfo, run_safe_shell, get_audio_devices"`
- [ ] Star import works: `python -c "from jpscripts.core.sys import *; print('OK')"`

### Step 2.8: Update consumer - engine/tool_executor.py
**Status:** NOT STARTED

**Action:** Update imports in `src/jpscripts/engine/tool_executor.py`

**IMPORTANT:** This file has its OWN `run_safe_shell` function (lines 86-154) that is DIFFERENT from `system.py`'s version!
- `tool_executor.run_safe_shell` returns `str`
- `system.run_safe_shell` returns `Result[CommandResult, SystemResourceError]`

**Change line 22 from:**
```python
from jpscripts.core.system import CommandResult, get_sandbox
```

**Change to:**
```python
from jpscripts.core.sys import CommandResult, get_sandbox
```

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.engine.tool_executor import execute_tool"`
- [ ] This file will be moved to agent/tools.py in Phase 3

### Step 2.9: Update consumer - agent/ops.py
**Status:** NOT STARTED

**Action:** Update imports in `src/jpscripts/agent/ops.py`

**Change line 17 from:**
```python
from jpscripts.core.system import run_safe_shell
```

**Change to:**
```python
from jpscripts.core.sys import run_safe_shell
```

**Note:** This imports the Result-returning `run_safe_shell` from core.sys (NOT the str-returning version from tool_executor).

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.agent.ops import apply_patch_file"`

### Step 2.10: Update consumer - swarm/agent_adapter.py
**Status:** NOT STARTED

**Action:** Update imports in `src/jpscripts/swarm/agent_adapter.py`

**Change line 22 from:**
```python
from jpscripts.core.system import run_safe_shell
```

**Change to:**
```python
from jpscripts.core.sys import run_safe_shell
```

**Note:** This imports the Result-returning `run_safe_shell` from core.sys.

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.swarm.agent_adapter import SwarmAgentAdapter"`

### Step 2.11: Update consumer - commands/evolve.py
**Status:** NOT STARTED

**Action:** Update imports in `src/jpscripts/commands/evolve.py`

**Change line 38 from:**
```python
from jpscripts.core.system import run_safe_shell
```

**Change to:**
```python
from jpscripts.core.sys import run_safe_shell
```

**Note:** This imports the Result-returning `run_safe_shell` from core.sys.

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.commands.evolve import evolve"`

### Step 2.12: Update consumer - commands/system.py
**Status:** NOT STARTED

**Action:** Update imports in `src/jpscripts/commands/system.py`

**Change line 28 from:**
```python
from jpscripts.core import system as system_core
```

**Change to:**
```python
from jpscripts.core import sys as system_core
```

**Note:** All attribute accesses like `system_core.find_processes` will continue to work.

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.commands.system import app"`

### Step 2.13: Update consumer - mcp/tools/system.py
**Status:** NOT STARTED

**Action:** Update imports in `src/jpscripts/mcp/tools/system.py`

**Change line 11 from:**
```python
from jpscripts.core import system as system_core
```

**Change to:**
```python
from jpscripts.core import sys as system_core
```

**Note:** Line 14 (`from jpscripts.engine import AUDIT_PREFIX, run_safe_shell`) will be updated later in Phase 3 Step 3.19.

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.mcp.tools.system import list_processes"`

### Step 2.14: Update consumer - tests/unit/test_system_commands.py
**Status:** NOT STARTED

**Action:** Update imports in `tests/unit/test_system_commands.py`

**Change line 24 from:**
```python
from jpscripts.core.system import ProcessInfo
```

**Change to:**
```python
from jpscripts.core.sys import ProcessInfo
```

**Verification:**
- [ ] No syntax errors: `python -c "import tests.unit.test_system_commands"`

### Step 2.15: Update consumer - tests/unit/test_dry_run_and_ast.py
**Status:** NOT STARTED

**Action:** Update imports in `tests/unit/test_dry_run_and_ast.py`

**Change line 7 from:**
```python
from jpscripts.core import system
```

**Change to:**
```python
from jpscripts.core import sys as system
```

**Note:** Aliasing as `system` preserves all existing attribute accesses in the test file.

**Verification:**
- [ ] No syntax errors: `python -c "import tests.unit.test_dry_run_and_ast"`

### Step 2.16: Delete old system.py
**Status:** NOT STARTED

**Action:** Delete `src/jpscripts/core/system.py`

**Command:**
```bash
rm src/jpscripts/core/system.py
```

**Verification:**
- [ ] File no longer exists: `ls src/jpscripts/core/system.py` should fail

### Step 2.17: Run Tests
**Status:** NOT STARTED

**Action:** Run test suite

**Commands:**
```bash
pytest tests/unit/test_system_commands.py tests/unit/test_dry_run_and_ast.py -v
pytest tests/ -v  # Full suite to catch any missed imports
```

**Expected outcome:** All tests pass

**If tests fail:**
- Check for any imports from `jpscripts.core.system` that were missed
- Run `grep -r "from jpscripts.core.system" src/ tests/` to find any remaining
- Run `grep -r "from jpscripts.core import system" src/ tests/` to find module imports

### Step 2.18: Commit and Push
**Status:** NOT STARTED

**Action:** Commit Phase 2 changes

**Commands:**
```bash
git add src/jpscripts/core/sys/
git add src/jpscripts/engine/tool_executor.py
git add src/jpscripts/agent/ops.py
git add src/jpscripts/swarm/agent_adapter.py
git add src/jpscripts/commands/evolve.py
git add src/jpscripts/commands/system.py
git add src/jpscripts/mcp/tools/system.py
git add tests/unit/test_system_commands.py
git add tests/unit/test_dry_run_and_ast.py
git rm src/jpscripts/core/system.py
git status  # Verify changes
git commit -m "refactor: decompose core/system.py into core/sys/ package

- Create core/sys/ package with specialized modules:
  - execution.py: CommandResult, sandboxes, run_safe_shell
  - process.py: ProcessInfo, find_processes, kill_process
  - audio.py: get_audio_devices, set_audio_device
  - network.py: get_ssh_hosts, run_temp_server
  - package.py: search_brew, get_brew_info
- Update all consumers to import from new locations
- Delete monolithic system.py

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
git push
```

**Verification:**
- [ ] Commit successful
- [ ] Push successful

---

## Phase 3: Merge engine/ into agent/

**Status:** NOT STARTED
**Risk Level:** HIGH
**Estimated Files:** 20+

### Goal
Consolidate the `engine/` package into `agent/` to resolve the confusing separation and circular dependency. The merged package will have clear module boundaries with descriptive names.

### Pre-conditions
- [ ] 3.0.1: Phase 1 completed and pushed
- [ ] 3.0.2: Phase 2 completed and pushed
- [ ] 3.0.3: Read all engine/ files to understand current structure
- [ ] 3.0.4: Read agent/__init__.py and agent/types.py
- [ ] 3.0.5: Run baseline tests: `pytest tests/ -v`

### Step 3.1: Create agent/models.py (merge engine/models.py + agent/types.py)
**Status:** NOT STARTED

**Action:** Create `src/jpscripts/agent/models.py`

**Content to include:**
1. All content from `engine/models.py`:
   - `MemoryProtocol` protocol
   - `ResponseT` TypeVar
   - `Message` dataclass
   - `ToolCall` Pydantic model
   - `AgentResponse` Pydantic model
   - `PreparedPrompt` dataclass
   - `AgentTraceStep` dataclass
   - `SafetyLockdownError` exception

2. All content from `agent/types.py`:
   - `SecurityError` exception
   - `EventKind` enum
   - `AgentEvent` dataclass
   - `RepairLoopConfig` dataclass
   - Type aliases

**Required imports:** Merge imports from both files, deduplicate.

**Exports:** Combined `__all__` from both files.

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.agent.models import Message, PreparedPrompt, RepairLoopConfig"`

### Step 3.2: Create agent/parsing.py (from engine/response_handler.py)
**Status:** NOT STARTED

**Action:** Create `src/jpscripts/agent/parsing.py`

**Content:** Copy entire content from `engine/response_handler.py`

**Update imports:**
```python
# Change from:
from jpscripts.engine.models import AgentResponse
# To:
from jpscripts.agent.models import AgentResponse
```

**Key exports (add to `__all__`):**
- `parse_agent_response()` - main function
- `_clean_json_payload()` - **MUST export** - tested in test_json_extraction.py
- `_extract_balanced_json()` - **MUST export** - tested in test_json_extraction.py

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.agent.parsing import parse_agent_response"`
- [ ] Internal helpers exported: `python -c "from jpscripts.agent.parsing import _clean_json_payload, _extract_balanced_json"`

### Step 3.3: Create agent/circuit.py (from engine/safety_monitor.py)
**Status:** NOT STARTED

**Action:** Create `src/jpscripts/agent/circuit.py`

**Content:** Copy entire content from `engine/safety_monitor.py`

**Update imports:**
```python
# Change from:
from jpscripts.engine.models import SafetyLockdownError
# To:
from jpscripts.agent.models import SafetyLockdownError
```

**Key exports:**
- `enforce_circuit_breaker()`
- `_estimate_token_usage()`

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.agent.circuit import enforce_circuit_breaker"`

### Step 3.4: Create agent/tracing.py (from engine/trace_recorder.py)
**Status:** NOT STARTED

**Action:** Create `src/jpscripts/agent/tracing.py`

**Content:** Copy entire content from `engine/trace_recorder.py`

**Update imports:**
```python
# Change from:
from jpscripts.engine.models import AgentTraceStep, Message, ToolCall
# To:
from jpscripts.agent.models import AgentTraceStep, Message, ToolCall
```

**Key exports:**
- `TraceRecorder` class
- `_get_tracer()` (if exposed)

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.agent.tracing import TraceRecorder"`

### Step 3.5: Create agent/governance.py (from engine/governance_enforcer.py)
**Status:** NOT STARTED

**Action:** Create `src/jpscripts/agent/governance.py`

**Content:** Copy entire content from `engine/governance_enforcer.py`

**Update imports:**
```python
# Change from:
from jpscripts.engine.models import Message, PreparedPrompt
# To:
from jpscripts.agent.models import Message, PreparedPrompt
```

**Key exports:**
- `enforce_governance()`

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.agent.governance import enforce_governance"`

### Step 3.6: Create agent/tools.py (from engine/tool_executor.py)
**Status:** NOT STARTED

**Action:** Create `src/jpscripts/agent/tools.py`

**Content:** Copy entire content from `engine/tool_executor.py` (163 LOC)

**IMPORTANT - Two Different `run_safe_shell` Functions:**
There are TWO `run_safe_shell` functions with different APIs:
1. **`core/system.py:274-308`** → `core/sys/execution.py` (Phase 2)
   - Returns `Result[CommandResult, SystemResourceError]`
   - Takes optional `env` kwarg
   - Used by agent/ops.py, swarm/agent_adapter.py, commands/evolve.py
2. **`engine/tool_executor.py:86-154`** → `agent/tools.py` (THIS step)
   - Returns `str` (simplified output)
   - No `env` parameter
   - Returns error messages as plain strings
   - Used by MCP tools and direct shell execution

**Keep BOTH functions** - they serve different purposes:
- `core.sys.run_safe_shell` for Result-pattern error handling
- `agent.tools.run_safe_shell` for simplified string output

**Update imports:**
```python
# Change from:
from jpscripts.engine.models import ToolCall
from jpscripts.core.system import CommandResult, get_sandbox
# To:
from jpscripts.agent.models import ToolCall
from jpscripts.core.sys import CommandResult, get_sandbox  # Updated in Phase 2
```

**Also update:**
```python
# Change from:
from .safety_monitor import enforce_circuit_breaker
# To:
from .circuit import enforce_circuit_breaker
```

**Key exports:**
- `execute_tool()`
- `AUDIT_PREFIX` (defined on line 29)
- `run_safe_shell` (the simplified str-returning version)
- `load_template_environment()`

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.agent.tools import execute_tool, AUDIT_PREFIX, run_safe_shell"`
- [ ] Type check: `mypy src/jpscripts/agent/tools.py`

### Step 3.7: Create agent/engine.py (from engine/__init__.py)
**Status:** NOT STARTED

**Action:** Create `src/jpscripts/agent/engine.py`

**Content:** Extract `AgentEngine` class from `engine/__init__.py`

**Update imports:**
```python
# Change from engine.* to agent.*:
from jpscripts.agent.models import (
    AgentResponse,
    AgentTraceStep,
    MemoryProtocol,
    Message,
    PreparedPrompt,
    ResponseT,
    ToolCall,
)
from jpscripts.agent.governance import enforce_governance
from jpscripts.agent.parsing import parse_agent_response
from jpscripts.agent.circuit import _estimate_token_usage, enforce_circuit_breaker
from jpscripts.agent.tools import execute_tool
from jpscripts.agent.tracing import TraceRecorder, _get_tracer
```

**IMPORTANT - Preserve lazy import pattern:**
```python
# Inside _infer_files_touched method, keep as lazy import:
from jpscripts.agent.patching import extract_patch_paths
```

**Key exports:**
- `AgentEngine` class

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.agent.engine import AgentEngine"`

### Step 3.8: Update agent/__init__.py
**Status:** NOT STARTED

**Action:** Update `src/jpscripts/agent/__init__.py`

**Remove lines 61+ (the import block from engine):**
```python
# Remove these imports from engine:
from jpscripts.engine import (
    PreparedPrompt,
    parse_agent_response,
    # ... any other engine imports
)
```

**Replace with imports from new agent modules:**
```python
# Add imports from new agent modules:
from jpscripts.agent.circuit import enforce_circuit_breaker
from jpscripts.agent.engine import AgentEngine
from jpscripts.agent.governance import enforce_governance
from jpscripts.agent.models import (
    AgentEvent,
    AgentResponse,
    AgentTraceStep,
    EventKind,
    MemoryProtocol,
    Message,
    PreparedPrompt,
    RepairLoopConfig,
    SafetyLockdownError,
    SecurityError,
    ToolCall,
)
from jpscripts.agent.parsing import parse_agent_response
from jpscripts.agent.tools import AUDIT_PREFIX, execute_tool, run_safe_shell
from jpscripts.agent.tracing import TraceRecorder
```

**Also remove line 19** (if present):
```python
# Remove:
from jpscripts.agent.types import ...
```
(since types.py is deleted and content merged into models.py)

**Update `__all__`:** Add all new exports including `run_safe_shell` (the str-returning version).

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.agent import AgentEngine, parse_agent_response, PreparedPrompt"`
- [ ] Star import works: `python -c "from jpscripts.agent import *; print('OK')"`

### Step 3.9: Update agent/execution.py
**Status:** NOT STARTED

**Action:** Update imports in `src/jpscripts/agent/execution.py`

**Change line 41 (multi-line import) from:**
```python
from jpscripts.engine import (
    AgentEngine,
    AgentResponse,
    Message,
    PreparedPrompt,
    ToolCall,
    parse_agent_response,
)
```

**Change to:**
```python
from jpscripts.agent.engine import AgentEngine
from jpscripts.agent.models import AgentResponse, Message, PreparedPrompt, ToolCall
from jpscripts.agent.parsing import parse_agent_response
```

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.agent.execution import RepairLoopOrchestrator"`

### Step 3.10: Update agent/prompting.py
**Status:** NOT STARTED

**Action:** Update imports in `src/jpscripts/agent/prompting.py`

**Change line 33 from:**
```python
from jpscripts.engine import AgentResponse, PreparedPrompt
```

**Change to:**
```python
from jpscripts.agent.models import AgentResponse, PreparedPrompt
```

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.agent.prompting import prepare_agent_prompt"`

### Step 3.11: Delete agent/types.py
**Status:** NOT STARTED

**Action:** Delete `src/jpscripts/agent/types.py` (content merged into models.py)

**Command:**
```bash
rm src/jpscripts/agent/types.py
```

**Verification:**
- [ ] File no longer exists

### Step 3.12: Verify commands/agent.py (no changes needed)
**Status:** NOT STARTED

**Action:** Verify imports in `src/jpscripts/commands/agent.py`

**Line 24 already imports from `jpscripts.agent`:**
```python
from jpscripts.agent import (
    PreparedPrompt,
    RepairLoopConfig,
    RepairLoopOrchestrator,
    parse_agent_response,
    prepare_agent_prompt,
)
```

**No changes needed** - this file already imports from the agent package, not engine.

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.commands.agent import agent"`

### Step 3.13: Update external consumer - commands/handbook.py
**Status:** NOT STARTED

**Action:** Update imports in `src/jpscripts/commands/handbook.py`

**Change line 34 from:**
```python
from jpscripts.engine import AUDIT_PREFIX, run_safe_shell
```

**Change to:**
```python
from jpscripts.agent.tools import AUDIT_PREFIX, run_safe_shell
```

**Note:** This file uses the simplified str-returning `run_safe_shell` (line 740), so import from `agent.tools`, NOT `core.sys`.

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.commands.handbook import app"`

### Step 3.14: Update external consumer - commands/trace.py
**Status:** NOT STARTED

**Action:** Update imports in `src/jpscripts/commands/trace.py`

**Change line 30 from:**
```python
from jpscripts.engine import AgentEngine, AgentTraceStep, Message, PreparedPrompt
```

**Change to:**
```python
from jpscripts.agent import AgentEngine, AgentTraceStep, Message, PreparedPrompt
```

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.commands.trace import app"`

### Step 3.15: Update external consumer - core/replay.py
**Status:** NOT STARTED

**Action:** Update imports in `src/jpscripts/core/replay.py`

**Change line 17 from:**
```python
from jpscripts.engine import AgentTraceStep
```

**Change to:**
```python
from jpscripts.agent import AgentTraceStep
```

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.core.replay import replay_trace"`

### Step 3.16: Update external consumer - core/team.py
**Status:** NOT STARTED

**Action:** Update imports in `src/jpscripts/core/team.py`

**Change line 36 from:**
```python
from jpscripts.engine import AgentEngine, Message, PreparedPrompt
```

**Change to:**
```python
from jpscripts.agent import AgentEngine, Message, PreparedPrompt
```

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.core.team import TeamOrchestrator"`

### Step 3.17: Update external consumer - swarm/controller.py
**Status:** NOT STARTED

**Action:** Update imports in `src/jpscripts/swarm/controller.py`

**Change line 19 from:**
```python
from jpscripts.engine import PreparedPrompt
```

**Change to:**
```python
from jpscripts.agent import PreparedPrompt
```

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.swarm.controller import SwarmController"`

### Step 3.18: Update external consumer - swarm/agent_adapter.py
**Status:** NOT STARTED

**Action:** Update imports in `src/jpscripts/swarm/agent_adapter.py`

**Change line 23 from:**
```python
from jpscripts.engine import Message, PreparedPrompt, ToolCall, parse_agent_response
```

**Change to:**
```python
from jpscripts.agent import Message, PreparedPrompt, ToolCall, parse_agent_response
```

**Note:** Line 22 (`from jpscripts.core.system import run_safe_shell`) was already updated in Phase 2 Step 2.10.

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.swarm.agent_adapter import SwarmAgentAdapter"`

### Step 3.19: Update external consumer - mcp/tools/system.py
**Status:** NOT STARTED

**Action:** Update imports in `src/jpscripts/mcp/tools/system.py`

**Change line 14 from:**
```python
from jpscripts.engine import AUDIT_PREFIX, run_safe_shell
```

**Change to:**
```python
from jpscripts.agent.tools import AUDIT_PREFIX, run_safe_shell
```

**Note:** This file uses the simplified str-returning `run_safe_shell` (line 58), so import from `agent.tools`, NOT `core.sys`.

**Verification:**
- [ ] No syntax errors: `python -c "from jpscripts.mcp.tools.system import list_processes"`

### Step 3.20: Update test files
**Status:** NOT STARTED

**Action:** Update all test files importing from engine

**Test files with exact line numbers:**

1. **tests/unit/test_robust_json.py** (line 13):
   ```python
   # Change from:
   from jpscripts.engine import (
   # To:
   from jpscripts.agent import (
   ```

2. **tests/unit/test_json_extraction.py** (line 7):
   ```python
   # Change from:
   from jpscripts.engine import _clean_json_payload, _extract_balanced_json
   # To:
   from jpscripts.agent.parsing import _clean_json_payload, _extract_balanced_json
   ```
   **Note:** These are internal helpers; verify they're exported from parsing.py

3. **tests/unit/test_trace_rotation.py** (line 10):
   ```python
   # Change from:
   from jpscripts.engine import TraceRecorder
   # To:
   from jpscripts.agent import TraceRecorder
   ```

4. **tests/unit/test_agent_orchestrator.py** (line 24):
   ```python
   # Change from:
   from jpscripts.engine import PreparedPrompt
   # To:
   from jpscripts.agent import PreparedPrompt
   ```

5. **tests/integration/test_swarm_real.py** (line 16):
   ```python
   # Change from:
   from jpscripts.engine import PreparedPrompt
   # To:
   from jpscripts.agent import PreparedPrompt
   ```

**Verification:**
- [ ] No test file has imports from `jpscripts.engine`: `grep -r "from jpscripts.engine" tests/`
- [ ] All tests pass: `pytest tests/ -v`

### Step 3.21: Delete engine/ package
**Status:** NOT STARTED

**Action:** Delete entire `src/jpscripts/engine/` directory

**Command:**
```bash
rm -rf src/jpscripts/engine/
```

**Verification:**
- [ ] Directory no longer exists: `ls src/jpscripts/engine/` should fail

### Step 3.22: Run Full Test Suite
**Status:** NOT STARTED

**Action:** Run complete test suite

**Commands:**
```bash
pytest tests/ -v
make test  # Includes linting
```

**Expected outcome:** All tests pass

**If tests fail:**
- Check for any remaining `from jpscripts.engine` imports
- Run `grep -r "from jpscripts.engine" src/ tests/`
- Check for circular import issues
- Verify lazy import in agent/engine.py is correct

### Step 3.23: Commit and Push
**Status:** NOT STARTED

**Action:** Commit Phase 3 changes

**Commands:**
```bash
git add src/jpscripts/agent/
git add src/jpscripts/commands/
git add src/jpscripts/core/replay.py
git add src/jpscripts/core/team.py
git add src/jpscripts/swarm/
git add src/jpscripts/mcp/tools/system.py
git add tests/
git rm -rf src/jpscripts/engine/
git status  # Verify changes
git commit -m "refactor: merge engine/ package into agent/

- Create new agent modules with descriptive names:
  - engine.py: AgentEngine class
  - models.py: All data types and protocols (merged with types.py)
  - parsing.py: Response parsing (from response_handler.py)
  - governance.py: Constitutional compliance
  - circuit.py: Circuit breaker (from safety_monitor.py)
  - tools.py: Tool execution
  - tracing.py: Trace recording
- Update all consumers to import from agent package
- Delete engine/ package entirely
- Resolves circular dependency between engine and agent

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
git push
```

**Verification:**
- [ ] Commit successful
- [ ] Push successful

---

## Post-Refactoring Verification

### Final Checks
- [ ] `make test` passes
- [ ] `jp agent --help` works
- [ ] `jp process-kill --help` works
- [ ] `jp audioswap --help` works
- [ ] No imports from `jpscripts.core.system` anywhere
- [ ] No imports from `jpscripts.engine` anywhere

### Update Documentation
- [ ] Update `docs/ARCHITECTURE.md` if it references old module paths
- [ ] Update any other docs referencing old import paths

---

## Rollback Procedures

### If Phase 1 fails:
```bash
git checkout HEAD -- src/jpscripts/analysis/
git checkout HEAD -- src/jpscripts/core/context_gatherer.py
```

### If Phase 2 fails:
```bash
git checkout HEAD -- src/jpscripts/core/
git checkout HEAD -- src/jpscripts/engine/tool_executor.py
git checkout HEAD -- src/jpscripts/agent/ops.py
git checkout HEAD -- src/jpscripts/swarm/agent_adapter.py
git checkout HEAD -- src/jpscripts/commands/
git checkout HEAD -- src/jpscripts/mcp/tools/system.py
git checkout HEAD -- tests/unit/test_system_commands.py
git checkout HEAD -- tests/unit/test_dry_run_and_ast.py
```

### If Phase 3 fails:
```bash
git checkout HEAD -- src/jpscripts/agent/
git checkout HEAD -- src/jpscripts/engine/
git checkout HEAD -- src/jpscripts/commands/
git checkout HEAD -- src/jpscripts/core/replay.py
git checkout HEAD -- src/jpscripts/core/team.py
git checkout HEAD -- src/jpscripts/swarm/
git checkout HEAD -- src/jpscripts/mcp/tools/system.py
git checkout HEAD -- tests/
```
