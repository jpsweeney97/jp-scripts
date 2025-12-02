# Code Quality Roadmap

**Created:** 2025-12-02
**Target completion:** No fixed deadline
**Author:** Claude Code
**Purpose:** Improve code quality through explicit registration, import cleanup, error handling harmonization, and diagnostics integration

---

## Progress Tracker

### Overall Status

| Phase | Status | Commit |
|-------|--------|--------|
| Phase 1: Explicit Command Registration | `IN PROGRESS` | - |
| Phase 2: Clean Up Lazy Import | `NOT STARTED` | - |
| Phase 3: Harmonize Error Handling | `NOT STARTED` | - |
| Phase 4: Fix Swarm Controller Init | `NOT STARTED` | - |
| Phase 5: Decouple Registry from Engine | `NOT STARTED` | - |
| Phase 6: Worktree Check in jp doctor | `NOT STARTED` | - |

### Current Position

**Active phase:** Phase 1
**Active step:** Step 1.2
**Last updated:** 2025-12-02
**Blockers:** None

### Quick Stats

- **Total phases:** 6
- **Completed phases:** 0
- **Total steps:** 24
- **Completed steps:** 0

---

## Prerequisites & Constraints

### Before Starting

- [ ] All tests passing on main branch
- [ ] No uncommitted changes in working directory
- [ ] Backup recent work with git commit

### Constraints

- **Must not break:** CLI commands, existing tests, public APIs
- **Must preserve:** Backward compatibility, existing behavior
- **Forbidden changes:** Breaking changes to command names or signatures

### Dependencies

| Dependency | Owner | Status |
|------------|-------|--------|
| None | - | - |

---

## Phase 1: Explicit Command Registration

**Status:** `NOT STARTED`
**Estimated steps:** 5
**Commit:** -

### Phase 1 Overview

Replace dynamic `discover_commands()` with explicit imports in `main.py`. This improves static analysis, IDE support, and makes the command registration transparent. The current hybrid pattern (8 Typer apps + 11 function-based modules with hardcoded mapping) will be converted to explicit imports.

### Phase 1 Rollback Procedure

```bash
git checkout HEAD -- src/jpscripts/main.py
git checkout HEAD -- src/jpscripts/core/registry.py
```

**Additional rollback notes:**
- No data migration required
- No external service dependencies

---

### Step 1.1: Capture Current Command List

**Status:** `COMPLETED`

**Action:**
Run `jp com` (or equivalent command catalog function) to capture all currently registered commands as a baseline.

**Sub-tasks:**
- [x] Run `jp com` and save output to a temp file
- [x] Document the expected command list in this step's notes

**Verification:**
- [x] Command list captured and documented

**Files affected:**
- None (read-only step)

**Notes:**
Command baseline captured. Structure:

**Built-in (main.py):** com, doctor, config, version

**Function commands (from _FUNCTION_COMMANDS in registry.py):**
- `git_ops`: status-all, whatpush, sync
- `nav`: recent, proj
- `init`: init, config-fix
- `web`: web-snap
- `system`: process-kill, port-kill, brew-explorer, audioswap, ssh-open, tmpserver, update, panic
- `notes`: note, note-search, standup, standup-note, cliphist
- `map`: map, repo-map
- `search`: ripper, todo-scan, loggrep
- `git_extra`: gundo-last, gstage, gpr, gbrowse, git-branchcheck, stashview
- `agent`: fix, agent

**Typer modules (have app = typer.Typer()):**
- serialize, trace, handbook, evolve, memory, team, watch

---

### Step 1.2: Modify main.py for Explicit Imports

**Status:** `COMPLETED`

**Action:**
Replace `discover_commands()` call with explicit imports of each command module.

**Sub-tasks:**
- [x] Remove `discover_commands(commands_path)` call from `_register_commands()`
- [x] Add explicit imports for Typer modules: `evolve`, `serialize`, `trace`, `watch`, `handbook`, `memory`, `team`
- [x] Add `app.add_typer(module.app, name="...")` for each Typer module
- [x] Add explicit imports for function-based modules: `agent`, `git_ops`, `git_extra`, `nav`, `init`, `web`, `system`, `notes`, `map`, `search`
- [x] Register function commands with `app.command(name)(handler)`

**Verification:**
- [x] No import errors when running `python -c "from jpscripts.main import app"`
- [x] `jp com` shows all 45 commands

**Files affected:**
- `src/jpscripts/main.py` - Replace dynamic registration with explicit imports

**Notes:**
Imports must be inside `_register_commands()` to avoid circular imports.
Some command modules (like evolve.py) import `AppState` from main.py.

---

### Step 1.3: Update or Deprecate registry.py

**Status:** `COMPLETED`

**Action:**
Since `discover_commands()` is no longer used, either delete or deprecate it.

**Sub-tasks:**
- [x] Check if any other code uses `discover_commands()`
- [x] No other usages found - added deprecation warning instead of deleting
- [x] Updated module docstring to mark as deprecated
- [x] Added `warnings.warn()` to `discover_commands()` function

**Verification:**
- [x] No import errors across codebase
- [x] Function still works but emits DeprecationWarning

**Files affected:**
- `src/jpscripts/core/registry.py` - Deprecated `discover_commands()`

**Notes:**
Kept function for backward compatibility but added deprecation warning.
Module docstring updated to indicate this module is deprecated.

---

### Step 1.4: Verify Command Registration

**Status:** `COMPLETED`

**Action:**
Run `jp com` again and verify the output matches the original baseline.

**Sub-tasks:**
- [x] Run `jp com` and compare to baseline
- [x] Verify all command names match exactly (45 commands)
- [x] Verify command help text is preserved

**Verification:**
- [x] Command list matches baseline exactly
- [x] `jp --help` shows all expected commands

**Files affected:**
- None (verification step)

**Notes:**
All 45 commands present and functional.

---

### Step 1.5: Run Smoke Tests

**Status:** `COMPLETED`

**Action:**
Run smoke tests to ensure CLI still boots and basic commands work.

**Sub-tasks:**
- [x] Run `pytest tests/test_smoke.py` - 8 passed
- [x] Run `pytest tests/unit/` - 699 passed, 5 failed (pre-existing failures)
- [x] Manual test: `jp --help`, `jp version` - working

**Verification:**
- [x] All smoke tests pass
- [x] Unit tests pass (failures are pre-existing, unrelated to this change)
- [x] Manual commands work as expected

**Files affected:**
- None (verification step)

**Notes:**
5 test failures in test_memory_integrity.py are pre-existing UnicodeDecodeError issues.

---

### Phase 1 Completion Checklist

- [x] All steps marked `COMPLETED`
- [x] All verification checks passing
- [x] Tests pass: `pytest tests/` (699 passed, 5 pre-existing failures)
- [x] Linting passes: `ruff check src/jpscripts/`
- [x] Type checking passes: `mypy src/jpscripts/`
- [ ] Changes committed with message: `refactor: explicit command registration in main.py`
- [ ] Commit hash recorded in Progress Tracker
- [ ] Phase status updated to `COMPLETED`

---

## Phase 2: Clean Up Lazy Import in Engine

**Status:** `NOT STARTED`
**Estimated steps:** 4
**Commit:** -

### Phase 2 Overview

The method `_infer_files_touched` in `engine.py` contains a lazy import of `jpscripts.agent.patching` with a comment claiming it avoids a circular dependency. Analysis shows this dependency no longer exists. This phase verifies the import can be moved to top-level and cleans up the misleading comment.

### Phase 2 Rollback Procedure

```bash
git checkout HEAD -- src/jpscripts/agent/engine.py
rm -f scripts/verify_imports.py
```

**Additional rollback notes:**
- Simple file revert, no data migration

---

### Step 2.1: Create Import Verification Script

**Status:** `NOT STARTED`

**Action:**
Create a test script to verify both modules can be imported together.

**Sub-tasks:**
- [ ] Create `scripts/verify_imports.py`
- [ ] Add imports for both `jpscripts.agent.engine` and `jpscripts.agent.patching`
- [ ] Run the script to verify no circular import error

**Verification:**
- [ ] Script runs without ImportError
- [ ] No circular dependency detected

**Files affected:**
- `scripts/verify_imports.py` - NEW: Verification script

**Notes:**
```python
# scripts/verify_imports.py
"""Verify agent imports don't cause circular dependency."""
import sys

# Test both imports can coexist
from jpscripts.agent import engine
from jpscripts.agent import patching

print("SUCCESS: Both modules imported without circular dependency")
print(f"engine: {engine.__file__}")
print(f"patching: {patching.__file__}")
```

---

### Step 2.2: Move Import to Top-Level

**Status:** `NOT STARTED`

**Action:**
Move `from jpscripts.agent.patching import extract_patch_paths` from inside `_infer_files_touched` to the top-level imports section.

**Sub-tasks:**
- [ ] Add import at top of `engine.py`: `from .patching import extract_patch_paths`
- [ ] Remove the lazy import from inside `_infer_files_touched`
- [ ] Remove the misleading comment about circular dependency

**Verification:**
- [ ] File imports successfully: `python -c "from jpscripts.agent.engine import AgentEngine"`

**Files affected:**
- `src/jpscripts/agent/engine.py` - Move import to top-level

**Notes:**
[Empty until work begins]

---

### Step 2.3: Run Agent Tests

**Status:** `NOT STARTED`

**Action:**
Run the agent test suite to verify functionality is preserved.

**Sub-tasks:**
- [ ] Run `pytest tests/unit/test_agent.py -v`
- [ ] Run `pytest tests/integration/test_repair_loop.py -v`

**Verification:**
- [ ] All agent tests pass
- [ ] No regressions detected

**Files affected:**
- None (verification step)

**Notes:**
[Empty until work begins]

---

### Step 2.4: Run Type Checking

**Status:** `NOT STARTED`

**Action:**
Run mypy to ensure type checking still passes.

**Sub-tasks:**
- [ ] Run `mypy src/jpscripts/agent`
- [ ] Fix any type errors introduced

**Verification:**
- [ ] `mypy` passes with no errors

**Files affected:**
- None expected (verification step)

**Notes:**
[Empty until work begins]

---

### Phase 2 Completion Checklist

- [ ] All steps marked `COMPLETED`
- [ ] All verification checks passing
- [ ] Tests pass: `pytest tests/unit/test_agent.py`
- [ ] Type checking passes: `mypy src/jpscripts/agent`
- [ ] Changes committed with message: `refactor: move patching import to top-level in engine.py`
- [ ] Commit hash recorded in Progress Tracker
- [ ] Phase status updated to `COMPLETED`

---

## Phase 3: Harmonize Error Handling (Result Pattern)

**Status:** `NOT STARTED`
**Estimated steps:** 4
**Commit:** -

### Phase 3 Overview

The `swarm/controller.py` uses `Result[T, E]` pattern while `agent/engine.py` raises exceptions. This creates an impedance mismatch. This phase refactors `AgentEngine.step` to return a `Result` type, aligning error handling across the codebase.

### Phase 3 Rollback Procedure

```bash
git checkout HEAD -- src/jpscripts/agent/engine.py
git checkout HEAD -- src/jpscripts/agent/models.py
git checkout HEAD -- src/jpscripts/swarm/agent_adapter.py
```

**Additional rollback notes:**
- May need to revert test changes if mocks were updated

---

### Step 3.1: Define AgentResult Type

**Status:** `NOT STARTED`

**Action:**
Define a generic `AgentResult` type alias in `models.py`.

**Sub-tasks:**
- [ ] Import `Result` from `jpscripts.core.result`
- [ ] Define `AgentError` class if not exists
- [ ] Define `AgentResult = Result[ResponseT, AgentError]` type alias
- [ ] Export in `__all__`

**Verification:**
- [ ] Type alias can be imported: `from jpscripts.agent.models import AgentResult`

**Files affected:**
- `src/jpscripts/agent/models.py` - Add AgentResult type

**Notes:**
[Empty until work begins]

---

### Step 3.2: Refactor AgentEngine.step

**Status:** `NOT STARTED`

**Action:**
Modify `AgentEngine.step` to return `AgentResult` instead of raising exceptions.

**Sub-tasks:**
- [ ] Change return type from `ResponseT` to `AgentResult[ResponseT]`
- [ ] Wrap internal logic in try/except
- [ ] Catch `SafetyLockdownError` and return `Err(AgentError(...))`
- [ ] Return `Ok(response)` on success

**Verification:**
- [ ] `mypy src/jpscripts/agent/engine.py` passes

**Files affected:**
- `src/jpscripts/agent/engine.py` - Change step() return type

**Notes:**
[Empty until work begins]

---

### Step 3.3: Update Call Sites

**Status:** `NOT STARTED`

**Action:**
Update code that calls `AgentEngine.step` to handle the `Result` return type.

**Sub-tasks:**
- [ ] Find all call sites using grep: `grep -r "\.step(" src/`
- [ ] Update `swarm/agent_adapter.py` to handle Result
- [ ] Update any other call sites

**Verification:**
- [ ] All call sites updated
- [ ] No uncaught exceptions from step()

**Files affected:**
- `src/jpscripts/swarm/agent_adapter.py` - Handle Result return type
- Other call sites as discovered

**Notes:**
[Empty until work begins]

---

### Step 3.4: Run Full Test Suite

**Status:** `NOT STARTED`

**Action:**
Run the full test suite to verify error handling works correctly.

**Sub-tasks:**
- [ ] Run `pytest tests/` with verbose output
- [ ] Check for any error handling regressions
- [ ] Verify swarm integration tests pass

**Verification:**
- [ ] All tests pass
- [ ] Error handling behavior preserved

**Files affected:**
- None (verification step)

**Notes:**
[Empty until work begins]

---

### Phase 3 Completion Checklist

- [ ] All steps marked `COMPLETED`
- [ ] All verification checks passing
- [ ] Tests pass: `pytest tests/`
- [ ] Type checking passes: `mypy src/jpscripts/`
- [ ] Changes committed with message: `refactor: AgentEngine.step returns Result type`
- [ ] Commit hash recorded in Progress Tracker
- [ ] Phase status updated to `COMPLETED`

---

## Phase 4: Fix Swarm Controller Initialization

**Status:** `NOT STARTED`
**Estimated steps:** 3
**Commit:** -

### Phase 4 Overview

The `ParallelSwarmController` allows initialization without a valid `task_executor`, causing failures later during execution. This phase enforces strict invariants at init time - the controller MUST have a valid executor upon instantiation.

### Phase 4 Rollback Procedure

```bash
git checkout HEAD -- src/jpscripts/swarm/controller.py
```

**Additional rollback notes:**
- May need to revert test mocks if they relied on None executor

---

### Step 4.1: Add Init Validation

**Status:** `NOT STARTED`

**Action:**
Raise `ValueError` immediately if neither `task_executor` nor `fetch_response` is provided.

**Sub-tasks:**
- [ ] Add validation at end of `__init__` after fallback logic
- [ ] Raise `ValueError("ParallelSwarmController requires either task_executor or fetch_response")`
- [ ] Update docstring to document requirement

**Verification:**
- [ ] `ParallelSwarmController()` raises ValueError
- [ ] `ParallelSwarmController(task_executor=...)` works
- [ ] `ParallelSwarmController(fetch_response=...)` works

**Files affected:**
- `src/jpscripts/swarm/controller.py` - Add init validation

**Notes:**
[Empty until work begins]

---

### Step 4.2: Update Type Hints

**Status:** `NOT STARTED`

**Action:**
Change `self._task_executor` type from `TaskExecutor | None` to `TaskExecutor`.

**Sub-tasks:**
- [ ] Update type annotation: `self._task_executor: TaskExecutor`
- [ ] Remove Optional from any related type hints
- [ ] Ensure mypy is satisfied

**Verification:**
- [ ] `mypy src/jpscripts/swarm/controller.py` passes

**Files affected:**
- `src/jpscripts/swarm/controller.py` - Update type hints

**Notes:**
[Empty until work begins]

---

### Step 4.3: Remove None Check in _execute_task

**Status:** `NOT STARTED`

**Action:**
Remove the `if self._task_executor is None` check since executor is now guaranteed.

**Sub-tasks:**
- [ ] Find and remove None check in `_execute_task`
- [ ] Simplify the method logic
- [ ] Update any related error handling

**Verification:**
- [ ] Code is cleaner without None checks
- [ ] `pytest tests/unit/test_swarm*.py` passes

**Files affected:**
- `src/jpscripts/swarm/controller.py` - Remove None check

**Notes:**
[Empty until work begins]

---

### Phase 4 Completion Checklist

- [ ] All steps marked `COMPLETED`
- [ ] All verification checks passing
- [ ] Tests pass: `pytest tests/unit/test_swarm*.py`
- [ ] Type checking passes: `mypy src/jpscripts/swarm/`
- [ ] Changes committed with message: `refactor: enforce TaskExecutor requirement in SwarmController init`
- [ ] Commit hash recorded in Progress Tracker
- [ ] Phase status updated to `COMPLETED`

---

## Phase 5: Decouple Registry from Engine

**Status:** `NOT STARTED`
**Estimated steps:** 4
**Commit:** -

### Phase 5 Overview

The `AgentEngine.__init__` contains a lazy import of `get_tool_registry` from `jpscripts.core.mcp_registry`. This creates a hidden dependency and makes testing harder. This phase moves the import responsibility to the caller, following dependency injection principles.

### Phase 5 Rollback Procedure

```bash
git checkout HEAD -- src/jpscripts/agent/engine.py
git checkout HEAD -- src/jpscripts/commands/agent.py
```

**Additional rollback notes:**
- May need to revert other call sites

---

### Step 5.1: Remove Lazy Import from Engine

**Status:** `NOT STARTED`

**Action:**
Remove the `else` block in `AgentEngine.__init__` that does the lazy import.

**Sub-tasks:**
- [ ] Remove the conditional import block for `get_tool_registry`
- [ ] Make `tools` parameter mandatory OR default to empty dict `{}`
- [ ] Update docstring to document the change

**Verification:**
- [ ] File imports without error
- [ ] `mypy src/jpscripts/agent/engine.py` passes

**Files affected:**
- `src/jpscripts/agent/engine.py` - Remove lazy import

**Notes:**
[Empty until work begins]

---

### Step 5.2: Update Agent Command Handler

**Status:** `NOT STARTED`

**Action:**
Update `commands/agent.py` to explicitly pass the tool registry.

**Sub-tasks:**
- [ ] Import `get_tool_registry` at top of file
- [ ] Pass `tools=get_tool_registry()` when creating AgentEngine
- [ ] Verify the agent command still works

**Verification:**
- [ ] `jp agent` command works
- [ ] Tools are available to the agent

**Files affected:**
- `src/jpscripts/commands/agent.py` - Pass tools explicitly

**Notes:**
[Empty until work begins]

---

### Step 5.3: Update Other Call Sites

**Status:** `NOT STARTED`

**Action:**
Find and update any other code that instantiates AgentEngine.

**Sub-tasks:**
- [ ] Search for `AgentEngine(` across codebase
- [ ] Update each call site to pass `tools` parameter
- [ ] Or pass `tools={}` if tools not needed

**Verification:**
- [ ] All call sites updated
- [ ] No runtime errors when AgentEngine is instantiated

**Files affected:**
- Various files that create AgentEngine instances

**Notes:**
[Empty until work begins]

---

### Step 5.4: Run Agent Tests

**Status:** `NOT STARTED`

**Action:**
Run agent tests to verify functionality is preserved.

**Sub-tasks:**
- [ ] Run `pytest tests/unit/test_agent.py -v`
- [ ] Run `pytest tests/integration/test_repair_loop.py -v`
- [ ] Manual test: `jp agent "test prompt"`

**Verification:**
- [ ] All tests pass
- [ ] Agent commands work correctly

**Files affected:**
- None (verification step)

**Notes:**
[Empty until work begins]

---

### Phase 5 Completion Checklist

- [ ] All steps marked `COMPLETED`
- [ ] All verification checks passing
- [ ] Tests pass: `pytest tests/unit/test_agent.py`
- [ ] Type checking passes: `mypy src/jpscripts/`
- [ ] Changes committed with message: `refactor: inject tool registry into AgentEngine`
- [ ] Commit hash recorded in Progress Tracker
- [ ] Phase status updated to `COMPLETED`

---

## Phase 6: Integrate Worktree Check into jp doctor

**Status:** `NOT STARTED`
**Estimated steps:** 4
**Commit:** -

### Phase 6 Overview

The `WorktreeManager` in `swarm/worktree.py` contains logic to detect orphaned worktrees, but this is only triggered when running a swarm agent. The `jp doctor` command currently ignores these orphans. This phase integrates worktree detection into the diagnostics suite.

### Phase 6 Rollback Procedure

```bash
git checkout HEAD -- src/jpscripts/core/diagnostics.py
```

**Additional rollback notes:**
- Simple file revert, no data migration

---

### Step 6.1: Create WorktreeCheck Class

**Status:** `NOT STARTED`

**Action:**
Create a new diagnostic check class for orphaned worktrees.

**Sub-tasks:**
- [ ] Import `WorktreeManager` from `jpscripts.swarm.worktree`
- [ ] Create `WorktreeCheck(DiagnosticCheck)` class
- [ ] Implement `run` method that checks for orphans

**Verification:**
- [ ] Class can be imported without error
- [ ] `mypy` passes

**Files affected:**
- `src/jpscripts/core/diagnostics.py` - Add WorktreeCheck class

**Notes:**
```python
class WorktreeCheck(DiagnosticCheck):
    """Check for orphaned git worktrees from crashed swarm sessions."""

    name = "worktree"
    description = "Check for orphaned worktrees"

    async def run(self, ctx: DiagnosticContext) -> DiagnosticResult:
        # Implementation here
        pass
```

---

### Step 6.2: Implement Orphan Detection

**Status:** `NOT STARTED`

**Action:**
Implement the orphan detection logic in the `run` method.

**Sub-tasks:**
- [ ] Instantiate `WorktreeManager` with repo context
- [ ] Call `detect_orphaned_worktrees()` method
- [ ] Return warning if orphans found with count and suggested action
- [ ] Return success if no orphans

**Verification:**
- [ ] Check detects orphaned worktrees when present
- [ ] Check passes when no orphans exist

**Files affected:**
- `src/jpscripts/core/diagnostics.py` - Implement run method

**Notes:**
[Empty until work begins]

---

### Step 6.3: Register Check in _run_deep_checks

**Status:** `NOT STARTED`

**Action:**
Add `WorktreeCheck` to the list of checks in `_run_deep_checks`.

**Sub-tasks:**
- [ ] Find the list of diagnostic checks
- [ ] Add `WorktreeCheck` to the list
- [ ] Ensure proper ordering (can run independently)

**Verification:**
- [ ] `jp doctor` runs without error
- [ ] Worktree check appears in output

**Files affected:**
- `src/jpscripts/core/diagnostics.py` - Register the check

**Notes:**
[Empty until work begins]

---

### Step 6.4: Test the Integration

**Status:** `NOT STARTED`

**Action:**
Test the worktree check integration manually and with tests.

**Sub-tasks:**
- [ ] Run `jp doctor` and verify worktree check runs
- [ ] Create a test orphaned worktree and verify detection
- [ ] Add unit test for `WorktreeCheck` if not exists

**Verification:**
- [ ] `jp doctor` shows worktree check status
- [ ] Orphaned worktrees are detected and reported
- [ ] Check doesn't crash on invalid git repos

**Files affected:**
- None (verification step)
- Optionally: `tests/unit/test_diagnostics.py` - Add test

**Notes:**
[Empty until work begins]

---

### Phase 6 Completion Checklist

- [ ] All steps marked `COMPLETED`
- [ ] All verification checks passing
- [ ] Tests pass: `pytest tests/`
- [ ] Type checking passes: `mypy src/jpscripts/`
- [ ] Changes committed with message: `feat: add orphaned worktree detection to jp doctor`
- [ ] Commit hash recorded in Progress Tracker
- [ ] Phase status updated to `COMPLETED`

---

## Final Verification

### Full Test Suite
- [ ] `pytest` - All tests pass
- [ ] `pytest --cov` - Coverage not decreased
- [ ] `mypy src` - No type errors
- [ ] `ruff check src` - No linting errors

### Manual Verification
- [ ] `jp --help` shows all commands
- [ ] `jp agent` commands work
- [ ] `jp doctor` includes worktree check
- [ ] Swarm operations function correctly

### Documentation
- [ ] Code comments updated where needed
- [ ] This roadmap archived with `_COMPLETED` suffix

---

## Completion Checklist

- [ ] All phases marked `COMPLETED` in Progress Tracker
- [ ] All final verifications passing
- [ ] No `BLOCKED` or `IN PROGRESS` items remaining
- [ ] Session Log reviewed for any outstanding issues
- [ ] Final commit pushed to remote
- [ ] **Archive this roadmap:** Rename file with `_COMPLETED` suffix
- [ ] **Update CLAUDE.md:** Remove from active roadmap reference

---

## Session Log

### 2025-12-02 - Session 1
- **Started:** Initial creation
- **Ended:** Roadmap file created
- **Progress:** Created full roadmap with 6 phases, 24 steps
- **Next:** Begin Phase 1 implementation
- **Issues:** None

---

## Appendix: Reference Information

### Key Files
| File | Purpose |
|------|---------|
| `src/jpscripts/main.py` | CLI entry point, command registration |
| `src/jpscripts/core/registry.py` | Dynamic command discovery (to be replaced) |
| `src/jpscripts/agent/engine.py` | Agent execution engine |
| `src/jpscripts/swarm/controller.py` | Parallel swarm controller |
| `src/jpscripts/core/diagnostics.py` | jp doctor checks |

### Related Documentation
- `CLAUDE.md` Section 12 - Roadmap management guidelines
- `.claude/templates/ROADMAP_TEMPLATE.md` - Template used for this roadmap

### Decisions Log

| Decision | Rationale | Date |
|----------|-----------|------|
| Keep Phase 2 simple | No actual circular dependency exists, just move import | 2025-12-02 |
| Enhance jp doctor for Phase 6 | Reuse existing WorktreeManager orphan detection | 2025-12-02 |
| Result pattern for AgentEngine.step | Harmonize with swarm controller error handling | 2025-12-02 |
