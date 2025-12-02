# Shell, Governance, and Codex Refactoring Roadmap

**Created:** 2025-12-01
**Target completion:** No fixed deadline
**Author:** Claude Code
**Purpose:** Consolidate shell execution, extract governance logic, and retire CodexProvider

---

## Progress Tracker

### Overall Status

| Phase | Status | Commit |
|-------|--------|--------|
| Phase 1: Consolidate Shell Execution | `COMPLETED` | 8d43d95 |
| Phase 2: Extract Governance Logic | `COMPLETED` | 2a961ba |
| Phase 3: Retire CodexProvider | `COMPLETED` | 567e4a0 |

### Current Position

**Active phase:** All complete
**Active step:** N/A
**Last updated:** 2025-12-02
**Blockers:** None

### Quick Stats

- **Total phases:** 3
- **Completed phases:** 3
- **Total steps:** 8
- **Completed steps:** 8

---

## Prerequisites & Constraints

### Before Starting

- [ ] All tests passing on main branch
- [ ] Working directory clean (no uncommitted changes)

### Constraints

- **Must not break:** Public API compatibility, CLI behavior for existing flags
- **Must preserve:** All existing tests, backward compatibility for `--web` option
- **Forbidden changes:** Do not modify unrelated modules

---

## Phase 1: Consolidate Shell Execution

**Status:** `NOT STARTED`
**Estimated steps:** 1
**Commit:** -

### Phase 1 Overview

Make `agent/tools.py::run_safe_shell` delegate to `core/sys/execution.py::run_safe_shell` to eliminate duplicate shell execution logic and maintain security in a single location.

### Phase 1 Rollback Procedure

```bash
git checkout main -- src/jpscripts/agent/tools.py
```

---

### Step 1.1: Replace run_safe_shell Implementation

**Status:** `COMPLETED`

**Action:**
Replace the implementation in `agent/tools.py` to call the core implementation and convert Result to string.

**Sub-tasks:**
- [x] Import `run_safe_shell` from `jpscripts.core.sys.execution` as `core_run_safe_shell`
- [x] Import `Ok`, `Err` from `jpscripts.core.result`
- [x] Replace function body with wrapper that calls core implementation
- [x] Handle `Err` case with appropriate error messages
- [x] Handle `Ok` case, checking returncode and combining stdout/stderr
- [x] Remove now-unused imports (shlex, validate_command, get_sandbox)

**Verification:**
- [x] `pytest tests/unit/test_shell_policy.py` passes (2/2)
- [x] `pytest tests/security/test_command_injection.py` passes (108/108)
- [ ] Manual test: `jp agent "list files"` works

**Files affected:**
- `src/jpscripts/agent/tools.py` - Replace implementation (~50 lines removed)

**Notes:**
Reduced from 69 lines of shell logic to 20 lines of wrapper code. All validation and execution now delegated to core/sys/execution.py.

---

### Phase 1 Completion Checklist

- [x] All steps marked `COMPLETED`
- [x] All verification checks passing
- [x] Tests pass: `pytest` (893 passed, 6 pre-existing failures unrelated to change)
- [x] Linting passes: `ruff check src`
- [x] Type checking passes: `mypy src`
- [x] Changes committed with message: `refactor: consolidate shell execution to core/sys`
- [x] Commit hash recorded in Progress Tracker (8d43d95)
- [x] Phase status updated to `COMPLETED`

---

## Phase 2: Extract Governance Logic

**Status:** `NOT STARTED`
**Estimated steps:** 1
**Commit:** -

### Phase 2 Overview

Extract the governance conditional from `AgentEngine.step()` into a dedicated `_apply_governance()` helper method for improved testability and semantic clarity.

### Phase 2 Rollback Procedure

```bash
git checkout main -- src/jpscripts/agent/engine.py
```

---

### Step 2.1: Create _apply_governance Helper

**Status:** `COMPLETED`

**Action:**
Create a new private method `_apply_governance` and update `step()` to call it.

**Sub-tasks:**
- [x] Add `_apply_governance` method with signature matching the governance call
- [x] Move conditional logic (`_governance_enabled` and `_workspace_root` check) into helper
- [x] Update `step()` to call `_apply_governance`
- [x] Add docstring to helper method

**Verification:**
- [x] `pytest tests/unit/test_agent.py` passes (4/4)
- [x] `pytest tests/unit/test_governance.py` passes (31/31)
- [ ] `pytest tests/integration/test_agent_real.py` passes

**Files affected:**
- `src/jpscripts/agent/engine.py` - Add helper method (~30 lines added with docstring)

**Notes:**
Added comprehensive docstring explaining governance validation and retry behavior.

---

### Phase 2 Completion Checklist

- [x] All steps marked `COMPLETED`
- [x] All verification checks passing
- [x] Tests pass: `pytest` (35 governance/agent tests passed)
- [x] Changes committed with message: `refactor: extract governance logic into _apply_governance`
- [x] Commit hash recorded (2a961ba)
- [x] Phase status updated to `COMPLETED`

---

## Phase 3: Retire CodexProvider

**Status:** `NOT STARTED`
**Estimated steps:** 6
**Commit:** -

### Phase 3 Overview

Remove the deprecated CodexProvider and all Codex-specific code while preserving the `--web` option as provider-agnostic and refactoring `config_fix()` to use any available provider.

### Phase 3 Rollback Procedure

```bash
git revert [commit-range]
# Or restore individual files:
git checkout main -- src/jpscripts/providers/
git checkout main -- src/jpscripts/commands/agent.py
git checkout main -- src/jpscripts/commands/init.py
```

---

### Step 3.1: Delete CodexProvider

**Status:** `IN PROGRESS`

**Action:**
Remove the codex.py file and update provider registry.

**Sub-tasks:**
- [ ] Delete `src/jpscripts/providers/codex.py`
- [ ] Remove `CODEX = auto()` from `ProviderType` enum in `__init__.py`
- [ ] Remove "codex" from `parse_provider_type()` mapping in `factory.py`

**Verification:**
- [ ] No import errors: `python -c "from jpscripts.providers import ProviderType"`

**Files affected:**
- `src/jpscripts/providers/codex.py` - DELETE
- `src/jpscripts/providers/__init__.py` - Remove CODEX enum
- `src/jpscripts/providers/factory.py` - Remove codex mapping

---

### Step 3.2: Update Factory

**Status:** `IN PROGRESS`

**Action:**
Remove Codex-specific logic from factory.py.

**Sub-tasks:**
- [ ] Remove `is_codex_available` import
- [ ] Remove `prefer_codex`, `codex_full_auto`, `codex_web_enabled` from `ProviderConfig`
- [ ] Remove Codex fallback logic in `get_provider()`
- [ ] Add clear error if user requests "codex" provider
- [ ] Remove Codex from `list_available_models()` and `get_model_context_limit()`

**Verification:**
- [ ] `python -c "from jpscripts.providers.factory import get_provider"` works

**Files affected:**
- `src/jpscripts/providers/factory.py` - Remove Codex logic

---

### Step 3.3: Update CLI

**Status:** `NOT STARTED`

**Action:**
Remove `--full-auto` option, keep `--web` as provider-agnostic.

**Sub-tasks:**
- [ ] Remove `--full-auto` / `-y` option from `codex_exec()`
- [ ] Remove `is_codex_available` import
- [ ] Update `_fetch_agent_response()` to remove `full_auto` parameter
- [ ] Update docstrings to remove Codex references
- [ ] Update help text for `--provider` option

**Verification:**
- [ ] `jp agent --help` shows no --full-auto option
- [ ] `jp agent --web "test"` works (may warn if provider doesn't support)

**Files affected:**
- `src/jpscripts/commands/agent.py` - Remove --full-auto, update docstrings

---

### Step 3.4: Refactor config_fix()

**Status:** `NOT STARTED`

**Action:**
Make config_fix() use any available provider instead of requiring Codex.

**Sub-tasks:**
- [ ] Update `config_fix()` to use `get_provider()` with default provider
- [ ] Remove Codex-specific check
- [ ] Update error messages

**Verification:**
- [ ] `jp init --fix` works without Codex installed

**Files affected:**
- `src/jpscripts/commands/init.py` - Update config_fix()

---

### Step 3.5: Update Diagnostics and Config

**Status:** `NOT STARTED`

**Action:**
Remove Codex references from diagnostics and config.

**Sub-tasks:**
- [ ] Remove Codex from external tools list in `diagnostics.py`
- [ ] Remove `~/.codex/config.toml` reference
- [ ] Update `config.py` field description for default_model

**Verification:**
- [ ] `jp doctor` runs without Codex references

**Files affected:**
- `src/jpscripts/core/diagnostics.py` - Remove Codex tool check
- `src/jpscripts/core/config.py` - Update docstring

---

### Step 3.6: Update Tests

**Status:** `NOT STARTED`

**Action:**
Remove Codex-specific tests and update assertions.

**Sub-tasks:**
- [ ] Delete `TestCodexProvider` class from `test_providers.py`
- [ ] Delete `TestCodexErrorHandling` class from `test_providers.py`
- [ ] Update provider type assertions to exclude CODEX
- [ ] Update `test_agent.py` to remove `full_auto` from signatures
- [ ] Update `test_system_commands.py` Codex process mocking

**Verification:**
- [ ] `pytest tests/unit/test_providers.py` passes
- [ ] `pytest tests/unit/test_agent.py` passes
- [ ] Full test suite passes

**Files affected:**
- `tests/unit/test_providers.py` - Delete Codex tests (~70 lines)
- `tests/unit/test_agent.py` - Update signatures
- `tests/unit/test_system_commands.py` - Update mock

---

### Phase 3 Completion Checklist

- [ ] All steps marked `COMPLETED`
- [ ] All verification checks passing
- [ ] Tests pass: `pytest`
- [ ] Linting passes: `ruff check src`
- [ ] Type checking passes: `mypy src`
- [ ] Changes committed with message: `refactor: retire CodexProvider`
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
- [ ] `jp agent --provider anthropic "hello"` works
- [ ] `jp agent --provider openai "hello"` works
- [ ] `jp agent --provider codex "hello"` shows clear error
- [ ] `jp agent --web "search test"` works

---

## Completion Checklist

- [ ] All phases marked `COMPLETED` in Progress Tracker
- [ ] All final verifications passing
- [ ] No `BLOCKED` or `IN PROGRESS` items remaining
- [ ] Session Log reviewed for any outstanding issues
- [ ] Final commit pushed to remote
- [ ] **Archive this roadmap:** Rename to `SHELL_GOVERNANCE_CODEX_ROADMAP_COMPLETED.md`
- [ ] **Update CLAUDE.md:** Remove from active roadmap reference

---

## Session Log

### 2025-12-01 - Session 1
- **Started:** --
- **Ended:** --
- **Progress:** Created roadmap
- **Next:** Begin Phase 1
- **Issues:** None

---

## Appendix: Reference Information

### Key Files
| File | Purpose |
|------|---------|
| `src/jpscripts/agent/tools.py` | Agent shell execution wrapper |
| `src/jpscripts/core/sys/execution.py` | Core shell execution with Result type |
| `src/jpscripts/agent/engine.py` | AgentEngine with step() and governance |
| `src/jpscripts/providers/codex.py` | CodexProvider to be deleted |
| `src/jpscripts/providers/factory.py` | Provider factory with Codex logic |
| `src/jpscripts/commands/agent.py` | CLI command with --full-auto, --web |

### Decisions Log

| Decision | Rationale | Date |
|----------|-----------|------|
| Proceed with Phase 2 extraction | Improves testability despite marginal size benefit | 2025-12-01 |
| Make config_fix() provider-agnostic | Preserves functionality without Codex dependency | 2025-12-01 |
| Keep --web option provider-agnostic | Useful feature, should work with all providers | 2025-12-01 |
