# Architecture Hardening Roadmap

**Created:** 2025-12-02
**Target completion:** No fixed deadline
**Author:** Architecture Audit
**Purpose:** Harden security, decouple architecture, optimize async I/O, and improve memory store performance.

---

## Progress Tracker

### Overall Status

| Phase | Status | Commit |
|-------|--------|--------|
| Phase 1: Security & Governance Hardening | `COMPLETED` | 69c7475 |
| Phase 2: Architecture Decoupling | `COMPLETED` | 1f8d6f4 |
| Phase 3: Async I/O Optimization | `COMPLETED` | 13475b9 |
| Phase 4: Memory Store Optimization | `COMPLETED` | 9c8d611 |

### Current Position

**Active phase:** All phases complete
**Active step:** None
**Last updated:** 2025-12-02
**Blockers:** None

### Quick Stats

- **Total phases:** 4
- **Completed phases:** 4
- **Total steps:** 8
- **Completed steps:** 8

---

## Prerequisites & Constraints

### Before Starting

- [ ] All tests passing on main branch
- [ ] No uncommitted changes in workspace

### Constraints

- **Must not break:** Existing tests, CLI commands, MCP tools
- **Must preserve:** Public API compatibility, Result[T, E] patterns
- **Forbidden changes:** None

---

## Phase 1: Security & Governance Hardening

**Status:** `COMPLETED`
**Estimated steps:** 2
**Commit:** 69c7475

### Phase 1 Overview

This phase prevents agents from self-approving governance violations by adding `# safety: checked` comments in their own patches. Currently, the AST checker in `ast_checker.py` allows bypassing checks if the line contains this comment, but an agent could add this comment to bypass rules.

### Phase 1 Rollback Procedure

```bash
git revert [commit-range]
```

---

### Step 1.1: Harden Governance Overrides

**Status:** `COMPLETED`

**Action:**
Modify the governance system to detect when `# safety: checked` is *added* in a patch rather than pre-existing, and flag this as a `SECURITY_BYPASS` violation.

**Sub-tasks:**
- [x] Add `ViolationType.SECURITY_BYPASS` to `src/jpscripts/governance/types.py`
- [x] Modify `compliance.py` to compare original file content vs patched content
- [x] If `# safety: checked` appears in patch but not in original, create fatal violation
- [x] Update `src/jpscripts/governance/ast_checker.py` to support pre-existing check logic
- [x] Add test case in `tests/unit/test_governance_extended.py` for this scenario

**Verification:**
- [x] Patch adding `os.system(...) # safety: checked` triggers fatal violation
- [x] Patch modifying existing `# safety: checked` line passes (trusted)
- [x] All existing tests pass (103 governance tests)

**Files affected:**
- `src/jpscripts/governance/types.py` - Add SECURITY_BYPASS violation type
- `src/jpscripts/governance/compliance.py` - Add original vs patch comparison
- `src/jpscripts/governance/ast_checker.py` - Minor adjustments if needed
- `tests/unit/test_governance_extended.py` - Add test case

---

### Step 1.2: Audit Subprocess Safety

**Status:** `COMPLETED`

**Action:**
Review `run_safe_shell` to ensure the allowlist cannot be bypassed by command chaining, aliasing, or shell metacharacters.

**Sub-tasks:**
- [x] Locate `run_safe_shell` implementation (`src/jpscripts/core/sys/execution.py`)
- [x] Verify allowlist is strictly enforced (no shell=True, no command concatenation)
- [x] Verify test cases exist for bypass attempts (semicolons, pipes, backticks, etc.)
- [x] Document the safety guarantees in code comments

**Verification:**
- [x] Attempts to chain commands fail validation (BLOCKED_METACHAR)
- [x] Shell metacharacters are rejected (108 security tests pass)
- [x] All tests pass

**Audit Findings:**
- Uses `asyncio.create_subprocess_exec()` with tokens (NOT shell=True)
- Allowlist-based: commands must be in ALLOWED_BINARIES
- Blocks `;`, `|`, `&`, backticks, `>`, `<` before shlex parsing
- Blocks full-path bypasses (`/bin/rm` → extracts `rm` → blocked)
- Validates paths stay within workspace
- Comprehensive test coverage in `tests/security/test_command_injection.py`

**Files affected:**
- `src/jpscripts/core/sys/execution.py` - Audited (no changes needed)
- `src/jpscripts/core/command_validation.py` - Audited (no changes needed)
- `tests/security/test_command_injection.py` - Existing 108 tests verified

---

### Phase 1 Completion Checklist

- [x] All steps marked `COMPLETED`
- [x] All verification checks passing
- [x] Tests pass: `pytest` (103 governance + 108 security tests)
- [x] Linting passes: `ruff check src`
- [x] Type checking passes: `mypy src`
- [x] Changes committed with message: `security: prevent agents from adding safety overrides`
- [x] Commit hash recorded in Progress Tracker (69c7475)
- [x] Phase status updated to `COMPLETED`

---

## Phase 2: Architecture Decoupling

**Status:** `COMPLETED`
**Estimated steps:** 3
**Commit:** 1f8d6f4

### Phase 2 Overview

This phase refactors `AgentEngine` to accept middleware via dependency injection rather than instantiating them internally. This decouples the engine from specific middleware implementations and removes reliance on global state.

### Phase 2 Rollback Procedure

```bash
git revert [commit-range]
```

---

### Step 2.1: Extract Middleware Construction

**Status:** `COMPLETED`

**Action:**
Create a factory function to construct the default middleware stack, moving this logic out of `AgentEngine.__init__`.

**Sub-tasks:**
- [x] Create `src/jpscripts/agent/factory.py` with `build_default_middleware` and `create_agent` functions
- [x] Move middleware instantiation logic from `AgentEngine.__init__` to factory
- [x] Modify `AgentEngine.__init__` to accept `middleware: Sequence[AgentMiddleware]` (optional, defaults to empty)
- [x] Keep middleware configuration logic in the factory, not the engine

**Verification:**
- [x] `AgentEngine` no longer imports specific middleware implementations (only Protocol)
- [x] Factory function creates correct middleware stack based on flags
- [x] All tests pass (74 agent/governance tests)

**Files affected:**
- `src/jpscripts/agent/factory.py` (new)
- `src/jpscripts/agent/engine.py` - Simplified constructor
- `src/jpscripts/agent/__init__.py` - Export factory functions

---

### Step 2.2: Update Call Sites

**Status:** `COMPLETED`

**Action:**
Update all call sites to use the factory function or explicitly construct middleware.

**Sub-tasks:**
- [x] Update `src/jpscripts/agent/execution.py` to use `build_default_middleware`
- [x] `src/jpscripts/commands/agent.py` - uses execution.py, no direct changes needed
- [x] `src/jpscripts/core/team.py` - uses AgentEngine directly with empty middleware (correct for its use case)
- [x] Tests that instantiate `AgentEngine` work with new interface

**Verification:**
- [x] All call sites use factory or explicit middleware
- [x] `pytest tests/unit/test_agent.py` passes (4 tests)
- [x] All 74 agent/governance tests pass

**Files affected:**
- `src/jpscripts/agent/execution.py` - Updated to use `build_default_middleware`

---

### Step 2.3: Remove Global Runtime Dependency

**Status:** `COMPLETED`

**Action:**
Ensure `AgentEngine` does not call `jpscripts.core.runtime.get_runtime()`. All dependencies should be passed via `__init__`.

**Sub-tasks:**
- [x] Search for `get_runtime()` usage in `engine.py` - NONE FOUND (already clean)
- [x] All runtime dependencies are passed through constructor parameters
- [x] Factory injects runtime values where needed

**Verification:**
- [x] `engine.py` has no imports from `jpscripts.core.runtime`
- [x] All runtime dependencies are explicit parameters
- [x] Tests pass (705 unit tests pass, 170 security tests pass)

**Files affected:**
- `src/jpscripts/agent/engine.py` - Already clean, no changes needed
- `src/jpscripts/agent/factory.py` - Already properly structured

---

### Phase 2 Completion Checklist

- [x] All steps marked `COMPLETED`
- [x] All verification checks passing
- [x] Tests pass: `pytest` (705 unit tests, 170 security tests)
- [x] Linting passes: `ruff check src`
- [x] Type checking passes: `mypy src`
- [x] Changes committed with message: `refactor: decouple AgentEngine middleware`
- [x] Commit hash recorded in Progress Tracker
- [x] Phase status updated to `COMPLETED`

---

## Phase 3: Async I/O Optimization

**Status:** `COMPLETED`
**Estimated steps:** 1
**Commit:** 13475b9

### Phase 3 Overview

The async security validation functions in `security.py` currently perform blocking filesystem I/O (`path.resolve()`, `path.exists()`, `path.stat()`). This blocks the event loop in high-concurrency swarm scenarios.

### Phase 3 Rollback Procedure

```bash
git revert [commit-range]
```

---

### Step 3.1: Non-Blocking Security Validation

**Status:** `COMPLETED`

**Action:**
Wrap all blocking pathlib calls in `validate_path_safe_async` and `validate_workspace_root_safe_async` with `asyncio.to_thread()`.

**Sub-tasks:**
- [x] In `validate_workspace_root_safe_async`: wrap `Path().expanduser().resolve()` in `to_thread`
- [x] Wrap `_is_owned_by_current_user(resolved)` call in `to_thread`
- [x] Create async version of `_resolve_with_limit` or wrap sync version in `to_thread`
- [x] Ensure `_is_forbidden_path` checks are also non-blocking (verified: fast in-memory check, no I/O)
- [x] Update `validate_path_safe_async` to use async resolution

**Verification:**
- [x] `pytest tests/unit/test_security_extended.py` passes (40 tests)
- [x] No blocking calls in async code paths
- [x] All async functions use `asyncio.to_thread()` for filesystem I/O

**Files affected:**
- `src/jpscripts/core/security.py`

---

### Phase 3 Completion Checklist

- [x] All steps marked `COMPLETED`
- [x] All verification checks passing
- [x] Tests pass: `pytest` (702 unit tests, 170 security tests)
- [x] Linting passes: `ruff check src`
- [x] Type checking passes: `mypy src`
- [x] Changes committed with message: `perf: non-blocking async security validation`
- [x] Commit hash recorded in Progress Tracker (13475b9)
- [x] Phase status updated to `COMPLETED`

---

## Phase 4: Memory Store Optimization

**Status:** `COMPLETED`
**Estimated steps:** 2
**Commit:** 9c8d611

### Phase 4 Overview

The `LanceDBStore` currently reconnects to the database on every operation. While the table is cached after first access, the connection is recreated. This introduces overhead for frequent memory operations.

### Phase 4 Rollback Procedure

```bash
git revert [commit-range]
```

---

### Step 4.1: Implement LanceDB Connection Caching

**Status:** `COMPLETED`

**Action:**
Cache the LanceDB connection object alongside the table to avoid reconnection on every `_ensure_table` call.

**Sub-tasks:**
- [x] Add `_connection: LanceDBConnectionProtocol | None = None` to `LanceDBStore`
- [x] Modify `_ensure_table` to reuse cached connection via `_get_connection()`
- [x] Only call `self._lancedb.connect()` if `_connection` is None
- [x] Consider thread-safety implications (LanceDB handles this internally)

**Verification:**
- [x] `LanceDBStore` operations reuse connection across calls
- [x] `pytest tests/unit/test_memory.py` passes (25 tests)
- [x] Connection is cached after first access

**Files affected:**
- `src/jpscripts/memory/store.py`

---

### Step 4.2: Review Memory Pruning Efficiency

**Status:** `COMPLETED`

**Action:**
Review `prune_memory` in `api.py` and ensure it uses efficient batch operations.

**Sub-tasks:**
- [x] Locate `prune_memory` in `src/jpscripts/memory/api.py` (delegates to `JsonlArchiver.prune`)
- [x] Review for unnecessary I/O or O(n) patterns
- [x] Ensure streaming is used where appropriate (confirmed: uses `_iter_entries` with `itertools.islice`)
- [x] Document findings (no changes needed - already efficient)

**Verification:**
- [x] Pruning uses bounded memory (MAX_ENTRIES=5000 limit via `itertools.islice`)
- [x] Batch writes used (`_write_entries` with atomic temp file + `os.replace`)
- [x] Tests pass

**Review findings:**
The pruning implementation in `JsonlArchiver.prune()` is already efficient:
- Uses generator-based `_iter_entries` with `itertools.islice` limit
- Atomic batch write at end via `_write_entries` (temp file + fsync + os.replace)
- Memory bounded to MAX_ENTRIES items
- No code changes needed

**Files affected:**
- None (review only)

---

### Phase 4 Completion Checklist

- [x] All steps marked `COMPLETED`
- [x] All verification checks passing
- [x] Tests pass: `pytest` (25 memory tests, 702 unit tests)
- [x] Linting passes: `ruff check src`
- [x] Type checking passes: `mypy src`
- [x] Changes committed with message: `perf: LanceDB connection caching`
- [x] Commit hash recorded in Progress Tracker (9c8d611)
- [x] Phase status updated to `COMPLETED`

---

## Final Verification

### Full Test Suite
- [x] `pytest` - All tests pass (702 unit tests, 170 security tests)
- [x] `mypy src` - No type errors
- [x] `ruff check src` - No linting errors

### Manual Verification
- [x] Memory operations use cached connection
- [x] Async security validation uses non-blocking I/O
- [x] Agent middleware uses dependency injection

---

## Completion Checklist

- [x] All phases marked `COMPLETED` in Progress Tracker
- [x] All final verifications passing
- [x] No `BLOCKED` or `IN PROGRESS` items remaining
- [x] Session Log reviewed for any outstanding issues
- [x] Final commit pushed to remote
- [x] **Archive this roadmap:** Renamed to `ARCHITECTURE_HARDENING_ROADMAP_COMPLETED.md`
- [x] **Update CLAUDE.md:** Moved to previously completed roadmaps

---

## Session Log

### 2025-12-02 - Session 1
- **Started:** Initial roadmap creation
- **Ended:** All phases complete
- **Progress:** All 4 phases completed
  - Phase 1: Security bypass detection (69c7475)
  - Phase 2: AgentEngine middleware DI (1f8d6f4)
  - Phase 3: Async security validation (13475b9)
  - Phase 4: LanceDB connection caching (9c8d611)
- **Next:** Archive roadmap
- **Issues:** None

---

## Appendix: Reference Information

### Key Files
| File | Purpose |
|------|---------|
| `src/jpscripts/governance/ast_checker.py` | AST-based constitutional compliance checker |
| `src/jpscripts/governance/compliance.py` | Governance compliance orchestration |
| `src/jpscripts/governance/types.py` | Violation types and data structures |
| `src/jpscripts/agent/engine.py` | Main agent execution engine |
| `src/jpscripts/agent/middleware.py` | Middleware protocol and implementations |
| `src/jpscripts/core/security.py` | Path validation and security utilities |
| `src/jpscripts/memory/store.py` | Memory store implementations (JSONL, LanceDB) |
| `src/jpscripts/memory/api.py` | High-level memory API |

### Related Documentation
- `docs/ARCHITECTURE.md` - System architecture
- `.claude/CLAUDE.md` - Project conventions and error handling patterns
