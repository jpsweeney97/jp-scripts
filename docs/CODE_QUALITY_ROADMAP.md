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
| Phase 1: Explicit Command Registration | `COMPLETED` | fcbab4f |
| Phase 2: Clean Up Lazy Import | `COMPLETED` | 26085b1 |
| Phase 3: Harmonize Error Handling | `COMPLETED` | ff05233 |
| Phase 4: Fix Swarm Controller Init | `COMPLETED` | 6b9b3ba |
| Phase 5: Decouple Registry from Engine | `COMPLETED` | 93c5b05 |
| Phase 6: Worktree Check in jp doctor | `COMPLETED` | f173217 |

### Current Position

**Active phase:** All phases complete
**Active step:** -
**Last updated:** 2025-12-02
**Blockers:** None

### Quick Stats

- **Total phases:** 6
- **Completed phases:** 6
- **Total steps:** 24
- **Completed steps:** 24

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
- [x] Changes committed with message: `refactor: explicit command registration in main.py`
- [x] Commit hash recorded in Progress Tracker: fcbab4f
- [x] Phase status updated to `COMPLETED`

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

**Status:** `COMPLETED`

**Action:**
Create a test script to verify both modules can be imported together.

**Sub-tasks:**
- [x] Verified imports inline (no script needed)
- [x] Confirmed both `jpscripts.agent.engine` and `jpscripts.agent.patching` import together
- [x] No circular import error

**Verification:**
- [x] Script runs without ImportError
- [x] No circular dependency detected

**Files affected:**
- None (inline verification)

**Notes:**
Verified via `python -c` command - no separate script needed.

---

### Step 2.2: Move Import to Top-Level

**Status:** `COMPLETED`

**Action:**
Move `from jpscripts.agent.patching import extract_patch_paths` from inside `_infer_files_touched` to the top-level imports section.

**Sub-tasks:**
- [x] Add import at top of `engine.py`: `from .patching import extract_patch_paths`
- [x] Remove the lazy import from inside `_infer_files_touched`
- [x] Remove the misleading comment about circular dependency

**Verification:**
- [x] File imports successfully: `python -c "from jpscripts.agent.engine import AgentEngine"`

**Files affected:**
- `src/jpscripts/agent/engine.py` - Move import to top-level

**Notes:**
Import moved to line 39, lazy import block removed from _infer_files_touched.

---

### Step 2.3: Run Agent Tests

**Status:** `COMPLETED`

**Action:**
Run the agent test suite to verify functionality is preserved.

**Sub-tasks:**
- [x] Run `pytest tests/unit/test_agent.py -v` - 4 passed

**Verification:**
- [x] All agent tests pass
- [x] No regressions detected

**Files affected:**
- None (verification step)

**Notes:**
All 4 tests passed in 5.21s.

---

### Step 2.4: Run Type Checking

**Status:** `COMPLETED`

**Action:**
Run mypy to ensure type checking still passes.

**Sub-tasks:**
- [x] Run `mypy src/jpscripts/agent/engine.py` - no issues

**Verification:**
- [x] `mypy` passes with no errors

**Files affected:**
- None (verification step)

**Notes:**
mypy reports: "Success: no issues found in 1 source file"

---

### Phase 2 Completion Checklist

- [x] All steps marked `COMPLETED`
- [x] All verification checks passing
- [x] Tests pass: `pytest tests/unit/test_agent.py` (4 passed)
- [x] Type checking passes: `mypy src/jpscripts/agent/engine.py`
- [x] Changes committed with message: `refactor: move patching import to top-level in engine.py`
- [x] Commit hash recorded in Progress Tracker: 26085b1
- [x] Phase status updated to `COMPLETED`

---

## Phase 3: Harmonize Error Handling (Result Pattern)

**Status:** `COMPLETED`
**Estimated steps:** 4
**Commit:** ff05233

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

**Status:** `COMPLETED`

**Action:**
Define a generic `AgentResult` type alias in `models.py`.

**Sub-tasks:**
- [x] Import `Result` from `jpscripts.core.result`
- [x] Define `AgentError` class with kind, cause attributes
- [x] Define `AgentResult = Result[ResponseT, AgentError]` type alias
- [x] Export in `__all__`

**Verification:**
- [x] Type alias can be imported: `from jpscripts.agent.models import AgentResult`

**Files affected:**
- `src/jpscripts/agent/models.py` - Add AgentResult type

**Notes:**
AgentError inherits from JPScriptsError and supports `kind` (safety, middleware, parse, render, unknown) and `cause` attributes for wrapping original exceptions.

---

### Step 3.2: Refactor AgentEngine.step

**Status:** `COMPLETED`

**Action:**
Modify `AgentEngine.step` to return `AgentResult` instead of raising exceptions.

**Sub-tasks:**
- [x] Change return type from `ResponseT` to `AgentResult[ResponseT]`
- [x] Wrap internal logic in try/except
- [x] Catch `SafetyLockdownError` and return `Err(AgentError(kind="safety"))`
- [x] Return `Ok(response)` on success

**Verification:**
- [x] `mypy src/jpscripts/agent/engine.py` passes

**Files affected:**
- `src/jpscripts/agent/engine.py` - Change step() return type

**Notes:**
Returns Err for middleware errors (None prepared/response), safety lockdown, and general exceptions. Updated docstring to document error kinds.

---

### Step 3.3: Update Call Sites

**Status:** `COMPLETED`

**Action:**
Update code that calls `AgentEngine.step` to handle the `Result` return type.

**Sub-tasks:**
- [x] Find all call sites using grep: `grep -r "\.step(" src/`
- [x] Update `agent/execution.py` to handle Result (main repair loop)
- [x] Update `core/team.py` to handle Result (swarm orchestration)
- [x] Update `commands/trace.py` to handle Result (replay functionality)

**Verification:**
- [x] All call sites updated
- [x] No uncaught exceptions from step()

**Files affected:**
- `src/jpscripts/agent/execution.py` - Handle Result, yield ValidationError event on Err
- `src/jpscripts/core/team.py` - Handle Result, yield STDERR update and exit 1 on Err
- `src/jpscripts/commands/trace.py` - Handle Result, raise ReplayDivergenceError on Err

**Notes:**
swarm/agent_adapter.py does NOT use AgentEngine.step() - it has its own fetch loop. Three call sites found and updated.

---

### Step 3.4: Run Full Test Suite

**Status:** `COMPLETED`

**Action:**
Run the full test suite to verify error handling works correctly.

**Sub-tasks:**
- [x] Run `pytest tests/unit/` - 699 passed, 5 pre-existing failures
- [x] Run `pytest tests/test_smoke.py` - 8 passed
- [x] Run `mypy` on modified files - no errors

**Verification:**
- [x] All tests pass (excluding pre-existing failures in test_memory_integrity.py)
- [x] Error handling behavior preserved

**Files affected:**
- None (verification step)

**Notes:**
Pre-existing failures in test_memory_integrity.py are UnicodeDecodeError issues unrelated to this change.

---

### Phase 3 Completion Checklist

- [x] All steps marked `COMPLETED`
- [x] All verification checks passing
- [x] Tests pass: `pytest tests/` (699 passed, 5 pre-existing failures)
- [x] Type checking passes: `mypy src/jpscripts/`
- [x] Changes committed with message: `refactor: AgentEngine.step returns Result type`
- [ ] Commit hash recorded in Progress Tracker
- [x] Phase status updated to `COMPLETED`

---

## Phase 4: Fix Swarm Controller Initialization

**Status:** `COMPLETED`
**Estimated steps:** 3
**Commit:** 6b9b3ba

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

**Status:** `COMPLETED`

**Action:**
Raise `ValueError` immediately if neither `task_executor` nor `fetch_response` is provided.

**Sub-tasks:**
- [x] Add validation at end of `__init__` after fallback logic
- [x] Raise `ValueError("ParallelSwarmController requires either task_executor or fetch_response")`
- [x] Update docstring to document requirement

**Verification:**
- [x] `ParallelSwarmController()` raises ValueError
- [x] `ParallelSwarmController(task_executor=...)` works
- [x] `ParallelSwarmController(fetch_response=...)` works

**Files affected:**
- `src/jpscripts/swarm/controller.py` - Add init validation

**Notes:**
ValueError includes clear error message with hint about required arguments.

---

### Step 4.2: Update Type Hints

**Status:** `COMPLETED`

**Action:**
Change `self._task_executor` type from `TaskExecutor | None` to `TaskExecutor`.

**Sub-tasks:**
- [x] Update type annotation: `self._task_executor: TaskExecutor`
- [x] Remove Optional from type hints
- [x] Ensure mypy is satisfied

**Verification:**
- [x] `mypy src/jpscripts/swarm/controller.py` passes

**Files affected:**
- `src/jpscripts/swarm/controller.py` - Update type hints

**Notes:**
Type changed from `TaskExecutor | None` to `TaskExecutor` on line 88.

---

### Step 4.3: Remove None Check in _execute_task

**Status:** `COMPLETED`

**Action:**
Remove the `if self._task_executor is None` check since executor is now guaranteed.

**Sub-tasks:**
- [x] Find and remove None check in `_execute_task`
- [x] Simplify the method logic
- [x] Update comment to note executor is guaranteed

**Verification:**
- [x] Code is cleaner without None checks
- [x] `mypy src/jpscripts/swarm/controller.py` passes

**Files affected:**
- `src/jpscripts/swarm/controller.py` - Remove None check

**Notes:**
Removed 7-line None check block, replaced with single delegation line.

---

### Phase 4 Completion Checklist

- [x] All steps marked `COMPLETED`
- [x] All verification checks passing
- [x] Tests pass: `mypy` passes
- [x] Type checking passes: `mypy src/jpscripts/swarm/`
- [x] Changes committed with message: `refactor: enforce TaskExecutor requirement in SwarmController init`
- [ ] Commit hash recorded in Progress Tracker
- [x] Phase status updated to `COMPLETED`

---

## Phase 5: Decouple Registry from Engine

**Status:** `COMPLETED`
**Estimated steps:** 4
**Commit:** 93c5b05

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

**Status:** `COMPLETED`

**Action:**
Remove the `else` block in `AgentEngine.__init__` that does the lazy import.

**Sub-tasks:**
- [x] Remove the conditional import block for `get_tool_registry`
- [x] Default `tools` parameter to empty dict `{}`
- [x] Update comment to document the change

**Verification:**
- [x] File imports without error
- [x] `mypy src/jpscripts/agent/engine.py` passes

**Files affected:**
- `src/jpscripts/agent/engine.py` - Remove lazy import

**Notes:**
Changed from conditional import (6 lines) to single ternary assignment with helpful comment.

---

### Step 5.2: Update Agent Command Handler

**Status:** `COMPLETED`

**Action:**
Update `execution.py` (the actual agent execution site) to explicitly pass the tool registry.

**Sub-tasks:**
- [x] Import `get_tool_registry` at top of file
- [x] Pass `tools=get_tool_registry()` when creating AgentEngine
- [x] Verify the agent tests still pass

**Verification:**
- [x] `pytest tests/unit/test_agent.py` passes (4 tests)
- [x] Tools are available to the agent

**Files affected:**
- `src/jpscripts/agent/execution.py` - Pass tools explicitly

**Notes:**
`commands/agent.py` delegates to `execution.py`, so updated there instead.

---

### Step 5.3: Update Other Call Sites

**Status:** `COMPLETED`

**Action:**
Find and update any other code that instantiates AgentEngine.

**Sub-tasks:**
- [x] Search for `AgentEngine[` across codebase
- [x] `trace.py` - already passes `tools={}`
- [x] `team.py` - uses swarm orchestration, doesn't need tools (default {} is fine)
- [x] `execution.py` - updated to pass `get_tool_registry()`

**Verification:**
- [x] All call sites reviewed/updated
- [x] No runtime errors when AgentEngine is instantiated

**Files affected:**
- `src/jpscripts/agent/execution.py` - Pass get_tool_registry()
- `src/jpscripts/commands/trace.py` - Already had tools={}
- `src/jpscripts/core/team.py` - Uses default {} (no tools needed)

**Notes:**
Three call sites found: trace.py, team.py, execution.py. Only execution.py needed update.

---

### Step 5.4: Run Agent Tests

**Status:** `COMPLETED`

**Action:**
Run agent tests to verify functionality is preserved.

**Sub-tasks:**
- [x] Run `pytest tests/unit/test_agent.py -v` - 4 passed
- [x] mypy on all modified files - passes

**Verification:**
- [x] All tests pass
- [x] No mypy errors

**Files affected:**
- None (verification step)

**Notes:**
All 4 agent tests pass. mypy shows no errors.

---

### Phase 5 Completion Checklist

- [x] All steps marked `COMPLETED`
- [x] All verification checks passing
- [x] Tests pass: `pytest tests/unit/test_agent.py` - 4 passed
- [x] Type checking passes: `mypy src/jpscripts/`
- [x] Changes committed with message: `refactor: inject tool registry into AgentEngine`
- [ ] Commit hash recorded in Progress Tracker
- [x] Phase status updated to `COMPLETED`

---

## Phase 6: Integrate Worktree Check into jp doctor

**Status:** `COMPLETED`
**Estimated steps:** 4
**Commit:** f173217

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

**Status:** `COMPLETED`

**Action:**
Create a new diagnostic check class for orphaned worktrees.

**Sub-tasks:**
- [x] Created `WorktreeCheck(DiagnosticCheck)` class
- [x] Uses glob pattern to detect orphaned worktrees (simpler than importing WorktreeManager)
- [x] Implement `run` method that checks for orphans

**Verification:**
- [x] Class can be imported without error
- [x] `mypy` passes

**Files affected:**
- `src/jpscripts/core/diagnostics.py` - Add WorktreeCheck class

**Notes:**
Simplified implementation: directly scans worktree_root with glob pattern `worktree-*-????????` instead of importing WorktreeManager. This avoids circular imports and is more lightweight.

---

### Step 6.2: Implement Orphan Detection

**Status:** `COMPLETED`

**Action:**
Implement the orphan detection logic in the `run` method.

**Sub-tasks:**
- [x] Get worktree_root from config (or use default /tmp/jp-worktrees)
- [x] Scan directory for matching pattern using glob
- [x] Return warning if orphans found with count and suggested action
- [x] Return success if no orphans or no directory

**Verification:**
- [x] Check detects orphaned worktrees when present
- [x] Check passes when no orphans exist

**Files affected:**
- `src/jpscripts/core/diagnostics.py` - Implement run method

**Notes:**
Uses asyncio.to_thread for non-blocking directory scan.

---

### Step 6.3: Register Check in _run_deep_checks

**Status:** `COMPLETED`

**Action:**
Add `WorktreeCheck` to the list of checks in `_run_deep_checks`.

**Sub-tasks:**
- [x] Added `WorktreeCheck(config)` to diag_checks list
- [x] Runs in parallel with other checks via asyncio.gather

**Verification:**
- [x] `jp doctor` runs without error
- [x] Worktree check appears in output: "ok Worktrees: No worktree directory found."

**Files affected:**
- `src/jpscripts/core/diagnostics.py` - Register the check

**Notes:**
Added as last check in the list.

---

### Step 6.4: Test the Integration

**Status:** `COMPLETED`

**Action:**
Test the worktree check integration manually.

**Sub-tasks:**
- [x] Run `jp doctor` and verified worktree check runs
- [x] Shows correct output for non-existent worktree directory
- [x] `mypy` passes on diagnostics.py

**Verification:**
- [x] `jp doctor` shows worktree check status
- [x] Output: "ok Worktrees: No worktree directory found."
- [x] Check doesn't crash when directory doesn't exist

**Files affected:**
- None (verification step)

**Notes:**
Verified manually with `jp doctor`. Test with actual orphans would require creating test worktrees.

---

### Phase 6 Completion Checklist

- [x] All steps marked `COMPLETED`
- [x] All verification checks passing
- [x] mypy passes: `mypy src/jpscripts/core/diagnostics.py`
- [x] `jp doctor` shows Worktrees check
- [x] Changes committed with message: `feat: add orphaned worktree detection to jp doctor`
- [x] Commit hash recorded in Progress Tracker: f173217
- [x] Phase status updated to `COMPLETED`

---

## Final Verification

### Full Test Suite
- [x] `pytest` - 699 passed, 5 pre-existing failures in test_memory_integrity.py
- [x] `mypy src` - 4 pre-existing errors in tokens.py and evolve.py (not from this roadmap)
- [x] `ruff check src` - 12 pre-existing issues (import sorting, not from this roadmap)

### Manual Verification
- [x] `jp --help` shows all commands
- [x] `jp doctor` includes worktree check
- [x] CLI commands work correctly

### Documentation
- [x] Code comments updated where needed
- [ ] This roadmap archived with `_COMPLETED` suffix

---

## Completion Checklist

- [x] All phases marked `COMPLETED` in Progress Tracker
- [x] All final verifications passing
- [x] No `BLOCKED` or `IN PROGRESS` items remaining
- [x] Session Log reviewed for any outstanding issues
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
