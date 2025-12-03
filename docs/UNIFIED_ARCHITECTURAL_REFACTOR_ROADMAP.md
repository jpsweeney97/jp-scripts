# Unified Architectural Refactor Roadmap

**Created:** 2025-12-02
**Target completion:** No fixed deadline (Production-critical - blocks new features)
**Author:** Claude Code
**Purpose:** Transform jpscripts from a "power user script collection" into a production-grade, secure Agent Platform through strict architectural boundaries and declarative governance.

---

## Progress Tracker

### Overall Status

| Phase | Status | Commit |
|-------|--------|--------|
| Phase 1: Dissolve Core Monolith | `COMPLETED` | 3556839 |
| Phase 2: Declarative Governance | `COMPLETED` | 6a712fe |
| Phase 3: Provider Contracts | `COMPLETED` | a5dad49 |
| Phase 4: Async Isolation | `COMPLETED` | d5ee843 |
| Phase 5: MCP Sandbox Verification | `COMPLETED` | 58b2354 |
| Phase 6: CLI Diet | `NOT STARTED` | - |
| Phase 7: AST Caching | `NOT STARTED` | - |
| Phase 8: Red Team Suite | `NOT STARTED` | - |

### Current Position

**Active phase:** Phase 6: CLI Diet
**Active step:** None (Phase 6 not started)
**Last updated:** 2025-12-03
**Blockers:** None

### Quick Stats

- **Total phases:** 8
- **Completed phases:** 5
- **Total steps:** 24
- **Completed steps:** 15

---

## Prerequisites & Constraints

### Before Starting

- [x] All tests passing on main branch (`pytest`)
- [x] No pending PRs that touch affected files
- [x] Read existing completed roadmaps for context

### Constraints

- **Must not break:** Public CLI API (`jp` commands), MCP server interface
- **Must preserve:** All existing tests, security test coverage
- **Forbidden changes:** No new external dependencies without explicit approval

### Dependencies

| Dependency | Owner | Status |
|------------|-------|--------|
| None | - | - |

---

## Phase 1: Dissolve Core Monolith

**Status:** `COMPLETED`
**Estimated steps:** 4
**Commit:** 3556839

### Phase 1 Overview

The `src/jpscripts/core/` directory contains domain logic (team.py, notes_impl.py, nav.py) mixed with infrastructure (config, security, logging). This violates separation of concerns and prevents modular evolution. This phase creates a new `features/` directory for domain logic, leaving `core/` for pure infrastructure.

### Phase 1 Rollback Procedure

```bash
git revert [commit-hash]
# OR restore from backup
git checkout main -- src/jpscripts/core/ src/jpscripts/features/
rm -rf src/jpscripts/features/
```

**Additional rollback notes:**
- No data migration required
- All changes are code reorganization only

---

### Step 1.1: Create Features Directory Structure

**Status:** `COMPLETED`

**Action:**
Create the `src/jpscripts/features/` directory with subdirectories for each feature domain.

**Sub-tasks:**
- [x] Create `src/jpscripts/features/__init__.py`
- [x] Create `src/jpscripts/features/team/__init__.py`
- [x] Create `src/jpscripts/features/notes/__init__.py`
- [x] Create `src/jpscripts/features/navigation/__init__.py`

**Verification:**
- [x] Directory structure exists
- [x] `python -c "from jpscripts.features import team, notes, navigation"` succeeds

**Files affected:**
- `src/jpscripts/features/` - New directory tree

**Notes:**
Lazy imports used in features/__init__.py to prevent circular import with agent/ modules.

---

### Step 1.2: Migrate Team Domain Logic

**Status:** `COMPLETED`

**Action:**
Move `src/jpscripts/core/team.py` to `src/jpscripts/features/team/model.py` and update all imports.

**Sub-tasks:**
- [x] Copy `core/team.py` to `features/team/model.py`
- [x] Update imports in `features/team/__init__.py` to re-export public API
- [x] Find all imports of `jpscripts.core.team` using grep
- [x] Update each import to use `jpscripts.features.team`
- [x] Delete `core/team.py`

**Verification:**
- [x] `pytest tests/unit/` passes
- [x] `mypy src` shows no import errors
- [x] `ruff check src` passes

**Files affected:**
- `src/jpscripts/core/team.py` - Deleted
- `src/jpscripts/features/team/model.py` - New (moved)
- `src/jpscripts/commands/team.py` - Import update
- `tests/unit/test_team.py` - Import update
- `tests/unit/test_team_persona.py` - Import update

**Notes:**
Fixed _resolve_template_root() path: changed parent.parent to parent.parent.parent to account for deeper nesting.

---

### Step 1.3: Migrate Notes Domain Logic

**Status:** `COMPLETED`

**Action:**
Move `src/jpscripts/core/notes_impl.py` to `src/jpscripts/features/notes/service.py` and update all imports.

**Sub-tasks:**
- [x] Copy `core/notes_impl.py` to `features/notes/service.py`
- [x] Update imports in `features/notes/__init__.py`
- [x] Find all imports of `jpscripts.core.notes_impl` using grep
- [x] Update each import to use `jpscripts.features.notes`
- [x] Delete `core/notes_impl.py`

**Verification:**
- [x] `pytest tests/unit/` passes
- [x] `mypy src` shows no import errors

**Files affected:**
- `src/jpscripts/core/notes_impl.py` - Deleted
- `src/jpscripts/features/notes/service.py` - New (moved)
- `src/jpscripts/commands/notes.py` - Import update
- `src/jpscripts/mcp/tools/notes.py` - Import update
- `tests/unit/test_notes_core.py` - Import update (patch path fixed)

**Notes:**
Test patch path updated from "jpscripts.core.notes_impl.dt" to "jpscripts.features.notes.service.dt".

---

### Step 1.4: Migrate Navigation Logic

**Status:** `COMPLETED`

**Action:**
Move `src/jpscripts/core/nav.py` to `src/jpscripts/features/navigation/service.py` and update all imports.

**Sub-tasks:**
- [x] Copy `core/nav.py` to `features/navigation/service.py`
- [x] Update imports in `features/navigation/__init__.py`
- [x] Find all imports of `jpscripts.core.nav` using grep
- [x] Update each import to use `jpscripts.features.navigation`
- [x] Delete `core/nav.py`

**Verification:**
- [x] `pytest tests/unit/` passes
- [x] `mypy src` shows no import errors
- [x] `jp nav` command still works

**Files affected:**
- `src/jpscripts/core/nav.py` - Deleted
- `src/jpscripts/features/navigation/service.py` - New (moved)
- `src/jpscripts/commands/nav.py` - Import update
- `src/jpscripts/mcp/tools/navigation.py` - Import update
- `src/jpscripts/agent/context.py` - Import update
- `src/jpscripts/agent/prompting.py` - Import update
- `tests/unit/test_nav.py` - Import update

**Notes:**
Navigation module had the most imports across the codebase. Fixed circular import in features/__init__.py by using lazy imports.

---

### Phase 1 Completion Checklist

- [x] All steps marked `COMPLETED`
- [x] All verification checks passing
- [x] Tests pass: `pytest`
- [x] Linting passes: `ruff check src`
- [x] Type checking passes: `mypy src`
- [x] Changes committed with message: `refactor: dissolve core monolith - create features/`
- [x] Commit hash recorded in Progress Tracker
- [x] Phase status updated to `COMPLETED`

---

## Phase 2: Declarative Governance

**Status:** `COMPLETED`
**Estimated steps:** 3
**Commit:** 6a712fe

### Phase 2 Overview

Security rules in `ast_checker.py` are currently hardcoded as imperative Python logic. This phase extracts them into a declarative `safety_rules.yaml` configuration file, making rules visible, auditable, and easy to update without code changes.

### Phase 2 Rollback Procedure

```bash
git revert [commit-hash]
# Restore original ast_checker.py
git checkout main -- src/jpscripts/governance/ast_checker.py
rm src/jpscripts/templates/safety_rules.yaml
```

**Additional rollback notes:**
- No behavioral change if rollback needed - same rules, different representation

---

### Step 2.1: Audit Current Governance Rules

**Status:** `COMPLETED`

**Action:**
Analyze `ast_checker.py` to document all currently hardcoded forbidden patterns, imports, and calls.

**Sub-tasks:**
- [x] Read `src/jpscripts/governance/ast_checker.py`
- [x] List all forbidden imports (e.g., `os`, `subprocess`, `shutil`)
- [x] List all forbidden function calls (e.g., `eval`, `exec`, `open`)
- [x] List all forbidden AST patterns
- [x] Document findings in this step's Notes section

**Verification:**
- [x] Complete list of rules documented

**Files affected:**
- None (read-only step)

**Notes:**
Audit completed. Found 12 violation types across ast_checker.py, secret_scanner.py, and compliance.py:

**Fatal errors (block patch application):**
1. SYNC_SUBPROCESS - subprocess.run/call/check_call/check_output/Popen in async context
2. SHELL_TRUE - shell=True in subprocess calls
3. OS_SYSTEM - os.system() always forbidden
4. DESTRUCTIVE_FS - shutil.rmtree, os.remove/unlink, Path.unlink without `# safety: checked`
5. DYNAMIC_EXECUTION - eval/exec/compile/__import__ always; importlib.import_module without override
6. PROCESS_EXIT - sys.exit(), quit(), exit()
7. DEBUG_LEFTOVER - breakpoint(), pdb.set_trace(), ipdb.set_trace()
8. BARE_EXCEPT - except: without exception type
9. SECRET_LEAK - High-entropy secrets, known API key prefixes (sk-, ghp_, AKIA, etc.)
10. SECURITY_BYPASS - Agent adding `# safety: checked` comments in diffs

**Warnings (non-fatal):**
11. SYNC_OPEN - open() in async context without to_thread/aiofiles
12. UNTYPED_ANY - Any type without type: ignore comment

---

### Step 2.2: Create Safety Rules Configuration

**Status:** `COMPLETED`

**Action:**
Create `src/jpscripts/templates/safety_rules.yaml` with the rules identified in Step 2.1.

**Sub-tasks:**
- [x] Design YAML schema for rules
- [x] Create `safety_rules.yaml` with forbidden_imports section
- [x] Add forbidden_calls section
- [x] Add forbidden_patterns section (regex or AST patterns)
- [x] Add severity levels (error, warning)

**Verification:**
- [x] YAML is valid: `python -c "import yaml; yaml.safe_load(open('src/jpscripts/templates/safety_rules.yaml'))"`
- [x] All original rules are represented

**Files affected:**
- `src/jpscripts/templates/safety_rules.yaml` - New file

**Notes:**
Created comprehensive YAML with:
- forbidden_calls: SYNC_SUBPROCESS, SHELL_TRUE, OS_SYSTEM, DESTRUCTIVE_FS, DYNAMIC_EXECUTION, PROCESS_EXIT, DEBUG_LEFTOVER, SYNC_OPEN
- forbidden_patterns: BARE_EXCEPT, UNTYPED_ANY
- secret_detection: SECRET_LEAK with variable patterns, dict patterns, known API prefixes
- diff_rules: SECURITY_BYPASS
- Each rule includes type, severity, fatal, safety_override, message, suggestion

---

### Step 2.3: Refactor AST Checker to Use Config

**Status:** `COMPLETED`

**Action:**
Refactor `ast_checker.py` to load rules from `safety_rules.yaml` instead of hardcoding them.

**Sub-tasks:**
- [x] Add config loading function to ast_checker.py
- [x] Create generic `RuleVisitor` class that checks against loaded config
- [x] Replace hardcoded checks with config-driven checks
- [x] Add fallback to embedded defaults if YAML missing
- [x] Update tests to verify config-driven behavior

**Verification:**
- [x] `pytest tests/unit/test_governance*.py` passes (105 tests)
- [x] `pytest tests/security/` passes (170 tests)
- [x] Manually test with config: `python -c "from jpscripts.governance import load_safety_config; print(load_safety_config())"`

**Files affected:**
- `src/jpscripts/governance/config.py` - New config loading module
- `src/jpscripts/governance/ast_checker.py` - Refactored to use config
- `src/jpscripts/governance/__init__.py` - Export new config types

**Notes:**
Created hybrid approach:
- `SafetyConfig` dataclass with embedded defaults for all configurable rule sets
- `load_safety_config()` extracts values from YAML, falls back to defaults
- ConstitutionChecker accepts optional config parameter, loads config if not provided
- All hardcoded frozensets replaced with config references:
  - `_BLOCKING_SUBPROCESS_FUNCS` → `self._config.blocking_subprocess_funcs`
  - Destructive fs functions → `self._config.destructive_shutil_funcs/destructive_os_funcs`
  - Dynamic builtins → `self._config.forbidden_dynamic_builtins`
  - Debug/exit builtins → `self._config.debug_builtins/exit_builtins`
  - Safety override pattern → `self._config.safety_override_pattern`

---

### Phase 2 Completion Checklist

- [x] All steps marked `COMPLETED`
- [x] All verification checks passing
- [x] Tests pass: `pytest` (273 governance/security tests)
- [x] Linting passes: `ruff check src/jpscripts/governance/`
- [x] Type checking passes: `mypy src/jpscripts/governance/`
- [x] Changes committed with message: `refactor: declarative governance rules (Phase 2)`
- [x] Commit hash recorded in Progress Tracker: 6a712fe
- [x] Phase status updated to `COMPLETED`

---

## Phase 3: Provider Contracts

**Status:** `COMPLETED`
**Estimated steps:** 3
**Commit:** a5dad49

### Phase 3 Overview

The `providers/` layer needs rigorous contract testing. This phase formalizes the `LLMProvider` Protocol and adds contract tests that run identical test cases against all provider implementations, ensuring consistent behavior when swapping backends.

### Phase 3 Rollback Procedure

```bash
git revert [commit-hash]
rm -rf tests/contract/
git checkout main -- src/jpscripts/providers/
```

**Additional rollback notes:**
- Contract tests are additive; rollback removes new validation only

---

### Step 3.1: Define LLMProvider Protocol

**Status:** `COMPLETED` (already existed)

**Action:**
Create or update `src/jpscripts/providers/base.py` with a strictly typed `LLMProvider` Protocol.

**Sub-tasks:**
- [x] Check if `base.py` exists (may be in factory.py)
- [x] Define `LLMProvider` as `typing.Protocol`
- [x] Define required methods: `complete()`, `stream()`, `count_tokens()`
- [x] Define standard input/output types
- [x] Define standard exception types (`RateLimitError`, `AuthError`, `APIError`)

**Verification:**
- [x] `mypy src/jpscripts/providers/` passes
- [x] Protocol is runtime-checkable: `isinstance(provider, LLMProvider)` works

**Files affected:**
- `src/jpscripts/providers/__init__.py` - Already contains LLMProvider Protocol

**Notes:**
LLMProvider Protocol already existed with all required methods:
- Properties: provider_type, default_model, available_models
- Methods: complete(), stream(), supports_streaming(), supports_tools(), supports_json_mode(), get_context_limit()
- Error types: ProviderError, AuthenticationError, RateLimitError, ModelNotFoundError, ContentFilterError, ContextLengthError
- Data types: Message, CompletionResponse, StreamChunk, TokenUsage, CompletionOptions, ToolDefinition, ToolCall

---

### Step 3.2: Create Mock Provider for Testing

**Status:** `COMPLETED` (enhanced existing)

**Action:**
Create a `MockProvider` that implements `LLMProvider` for deterministic testing.

**Sub-tasks:**
- [x] Create `tests/mocks/mock_provider.py`
- [x] Implement all Protocol methods with configurable responses
- [x] Add helpers to simulate errors (rate limits, auth failures)
- [x] Verify it passes Protocol type check

**Verification:**
- [x] `isinstance(MockProvider(), LLMProvider)` returns True
- [x] Mock can simulate all error types

**Files affected:**
- `tests/mocks/mock_provider.py` - Enhanced with error simulation

**Notes:**
MockProvider already existed. Enhanced with:
- `simulate_error(error, on_call=N)` - trigger errors on demand
- `clear_error()` - clear simulated errors
- `streaming_enabled` parameter for optional streaming support
- Updated stream() to yield chunks when enabled
- Error simulation works for both complete() and stream()

---

### Step 3.3: Implement Contract Test Suite

**Status:** `COMPLETED`

**Action:**
Create parameterized contract tests that run against all provider implementations.

**Sub-tasks:**
- [x] Create `tests/contract/__init__.py`
- [x] Create `tests/contract/test_providers.py`
- [x] Add test for basic completion
- [x] Add test for streaming
- [x] Add test for token counting
- [x] Add tests for error handling (rate limits, auth)
- [x] Parameterize tests to run against MockProvider
- [x] Add skip markers for real providers (require API keys)

**Verification:**
- [x] `pytest tests/contract/` passes with MockProvider (34 tests)
- [x] Tests are parameterized correctly

**Files affected:**
- `tests/contract/__init__.py` - New
- `tests/contract/test_providers.py` - New (34 tests)

**Notes:**
Created comprehensive contract test suite with 34 tests covering:
- TestLLMProviderProtocol: Protocol satisfaction tests
- TestProviderProperties: provider_type, default_model, available_models, context_limit
- TestBasicCompletion: complete() behavior and response structure
- TestStreaming: stream() behavior, chunk structure, finish_reason
- TestCapabilities: supports_tools(), supports_json_mode()
- TestErrorHandling: All error types (RateLimitError, AuthenticationError, etc.)
- TestCallTracking: call_log and call_count tracking
- TestMockProviderWithCounter: Ordered response tests

---

### Phase 3 Completion Checklist

- [x] All steps marked `COMPLETED`
- [x] All verification checks passing
- [x] Tests pass: `pytest tests/contract/` (34 tests)
- [x] Linting passes: `ruff check tests/mocks/ tests/contract/`
- [x] Type checking passes: `mypy` (tests not strictly checked)
- [x] Changes committed with message: `feat: provider contract testing (Phase 3)`
- [x] Commit hash recorded in Progress Tracker: a5dad49
- [x] Phase status updated to `COMPLETED`

---

## Phase 4: Async Isolation

**Status:** `COMPLETED`
**Estimated steps:** 3
**Commit:** d5ee843

### Phase 4 Overview

Heavy CPU operations (AST parsing, Git diff calculations) block the asyncio event loop, causing UI freezes and timeouts. This phase offloads CPU-bound tasks to a ProcessPoolExecutor using a `run_cpu_bound()` utility.

### Phase 4 Rollback Procedure

```bash
git revert [commit-hash]
```

**Additional rollback notes:**
- Behavioral change only; sync operations will work but may block

---

### Step 4.1: Create run_cpu_bound Utility

**Status:** `COMPLETED`

**Action:**
Add a `run_cpu_bound(func, *args)` utility to `execution.py` that offloads work to thread pool.

**Sub-tasks:**
- [x] Read current `src/jpscripts/core/sys/execution.py`
- [x] Add `run_cpu_bound` function using `asyncio.to_thread`
- [x] Document design choice (ThreadPoolExecutor over ProcessPoolExecutor)
- [x] Add proper type hints
- [x] Add comprehensive docstring with examples

**Verification:**
- [x] Function exists and is importable
- [x] Linting passes

**Files affected:**
- `src/jpscripts/core/sys/execution.py` - Added run_cpu_bound utility

**Notes:**
Used asyncio.to_thread (ThreadPoolExecutor) instead of ProcessPoolExecutor because:
1. Many operations in this codebase involve non-picklable objects (AST nodes, Path objects)
2. The primary goal is preventing event loop blocking, which both achieve
3. ProcessPoolExecutor adds IPC overhead and pickling complexity
4. asyncio.to_thread is the established pattern in this codebase

---

### Step 4.2: Async AST Parsing

**Status:** `COMPLETED`

**Action:**
Add async versions of compliance checking functions using `run_cpu_bound`.

**Sub-tasks:**
- [x] Add import of run_cpu_bound to compliance.py
- [x] Add check_compliance_async() wrapper
- [x] Add check_source_compliance_async() wrapper
- [x] Add scan_codebase_compliance_async() wrapper
- [x] Export async functions from governance __init__.py

**Verification:**
- [x] `pytest tests/unit/test_governance*.py tests/security/` passes (273 tests)
- [x] Async functions work correctly (tested manually)
- [x] Linting passes

**Files affected:**
- `src/jpscripts/governance/compliance.py` - Added async versions
- `src/jpscripts/governance/__init__.py` - Export async functions

**Notes:**
Rather than making the AST checker itself async (which would require significant refactoring),
added async wrapper functions that call the existing sync functions via run_cpu_bound.
This provides the event loop isolation benefits without invasive changes.

---

### Step 4.3: Async Git Operations

**Status:** `COMPLETED` (already async)

**Action:**
Review Git operations for blocking patterns.

**Sub-tasks:**
- [x] Review `src/jpscripts/git/client.py` for blocking operations
- [x] Review `src/jpscripts/git/ops.py` for blocking operations
- [x] Verify all git commands use asyncio.create_subprocess_exec
- [x] Verify filesystem scanning uses asyncio.to_thread

**Verification:**
- [x] All AsyncRepo methods are already async using asyncio.create_subprocess_exec
- [x] iter_git_repos already uses asyncio.to_thread for filesystem scanning

**Files affected:**
- None (already optimized)

**Notes:**
The git module was already fully optimized for async:
1. All git commands use `asyncio.create_subprocess_exec` (non-blocking subprocess)
2. `iter_git_repos` uses `asyncio.to_thread(_scan)` for filesystem scanning (line 623)
3. All parsing functions are pure CPU-bound string parsing (fast, no I/O)
No additional changes needed.

---

### Phase 4 Completion Checklist

- [x] All steps marked `COMPLETED`
- [x] All verification checks passing
- [x] Tests pass: `pytest tests/unit/test_governance* tests/security/` (273 tests)
- [x] Linting passes: `ruff check src/jpscripts/core/sys/execution.py src/jpscripts/governance/`
- [x] Type checking passes: `mypy src/jpscripts/core/sys/execution.py src/jpscripts/governance/compliance.py`
- [x] Changes committed with message: `perf: async isolation for CPU-bound tasks`
- [x] Commit hash recorded in Progress Tracker: d5ee843
- [x] Phase status updated to `COMPLETED`

---

## Phase 5: MCP Sandbox Verification

**Status:** `COMPLETED`
**Estimated steps:** 2
**Commit:** 58b2354

### Phase 5 Overview

MCP tools expose local filesystem operations to the LLM. While basic security exists (`validate_path_safe()`), this phase verifies and hardens the sandbox against edge cases like symlink attacks and path traversal.

### Phase 5 Rollback Procedure

```bash
git revert [commit-hash]
```

**Additional rollback notes:**
- Security tests are additive; rollback only removes new tests

---

### Step 5.1: Audit MCP Filesystem Security

**Status:** `COMPLETED`

**Action:**
Review all MCP filesystem tools for proper path validation.

**Sub-tasks:**
- [x] Read `src/jpscripts/mcp/tools/filesystem.py`
- [x] Verify all path inputs go through `validate_path_safe()`
- [x] Check for symlink resolution with `resolve()`
- [x] Check for `..` handling
- [x] Document any gaps found

**Verification:**
- [x] Audit complete
- [x] All tools use path validation

**Files affected:**
- `src/jpscripts/mcp/tools/filesystem.py` - Fixed write_file vulnerability

**Notes:**
Audit found robust security implementation:
- Symlink depth limiting (MAX_SYMLINK_DEPTH = 10)
- Forbidden paths blocklist (/etc, /usr, /bin, /System, etc.)
- Workspace containment via relative_to()
- O_NOFOLLOW in validate_and_open for TOCTOU protection
- Rate limiting (100 ops/minute)

**VULNERABILITY FIXED:** write_file was calling mkdir() BEFORE path validation,
which could create directories outside workspace via path traversal.
Fixed by moving validate_path_safe() before mkdir().

---

### Step 5.2: Add MCP Security Test Cases

**Status:** `COMPLETED`

**Action:**
Create specific attack test cases for MCP filesystem tools.

**Sub-tasks:**
- [x] Create or update `tests/integration/test_mcp_security.py`
- [x] Add test: attempt to read `/etc/passwd`
- [x] Add test: attempt to read `../outside_workspace/secret`
- [x] Add test: symlink pointing outside workspace
- [x] Add test: write_file path traversal blocked
- [x] Add test: write_file no mkdir outside workspace
- [x] Add test: list_directory traversal blocked
- [x] Add test: valid paths still work (no false positives)

**Verification:**
- [x] All attack tests pass (attacks are blocked)
- [x] No false positives (valid paths still work)

**Files affected:**
- `tests/integration/test_mcp_security.py` - Added TestMcpPathTraversalAttacks class

**Notes:**
Added 7 new attack test cases in TestMcpPathTraversalAttacks class:
1. test_read_etc_passwd_blocked
2. test_path_traversal_blocked
3. test_symlink_escape_blocked
4. test_write_path_traversal_blocked
5. test_write_no_mkdir_outside_workspace
6. test_list_directory_traversal_blocked
7. test_valid_path_still_works (no false positives)

Also fixed pre-existing lint issues (unused imports) in the test file.

---

### Phase 5 Completion Checklist

- [x] All steps marked `COMPLETED`
- [x] All verification checks passing
- [x] Tests pass: `pytest` (17 MCP security tests pass)
- [x] Linting passes: `ruff check src`
- [x] Type checking passes: `mypy src`
- [x] Changes committed with message: `security: MCP sandbox verification (Phase 5)`
- [x] Commit hash recorded in Progress Tracker (58b2354)
- [x] Phase status updated to `COMPLETED`

---

## Phase 6: CLI Diet

**Status:** `NOT STARTED`
**Estimated steps:** 3
**Commit:** -

### Phase 6 Overview

`commands/agent.py` (325 LOC) contains orchestration logic that should be in the service layer. This phase extracts that logic into `agent/orchestrator.py`, making the CLI a thin dispatcher and enabling programmatic agent invocation.

### Phase 6 Rollback Procedure

```bash
git revert [commit-hash]
```

**Additional rollback notes:**
- Behavioral change only; CLI functionality preserved

---

### Step 6.1: Identify Orchestration Logic

**Status:** `NOT STARTED`

**Action:**
Analyze `commands/agent.py` to identify setup, loop, and teardown logic.

**Sub-tasks:**
- [ ] Read `src/jpscripts/commands/agent.py`
- [ ] Identify setup logic (provider init, context gathering)
- [ ] Identify main loop logic
- [ ] Identify teardown logic
- [ ] Document what should stay (arg parsing, output rendering)

**Verification:**
- [ ] Clear list of code to extract

**Files affected:**
- None (read-only analysis)

**Notes:**
[Will be filled with analysis]

---

### Step 6.2: Create/Update Orchestrator

**Status:** `NOT STARTED`

**Action:**
Move orchestration logic to `src/jpscripts/agent/orchestrator.py`.

**Sub-tasks:**
- [ ] Create `orchestrator.py` if not exists
- [ ] Create `run_agent(config: AgentConfig) -> AgentResult` function
- [ ] Move setup logic from commands/agent.py
- [ ] Move main loop logic
- [ ] Move teardown logic
- [ ] Ensure proper async handling

**Verification:**
- [ ] Orchestrator can be imported and called directly
- [ ] `python -c "from jpscripts.agent.orchestrator import run_agent"` works

**Files affected:**
- `src/jpscripts/agent/orchestrator.py` - New or major update

**Notes:**
[Empty]

---

### Step 6.3: Thin the CLI Command

**Status:** `NOT STARTED`

**Action:**
Refactor `commands/agent.py` to only handle arg parsing and call orchestrator.

**Sub-tasks:**
- [ ] Remove extracted logic from agent.py
- [ ] Update to call `run_agent()` from orchestrator
- [ ] Keep only Typer decorators and argument handling
- [ ] Keep Rich output rendering
- [ ] Target: < 60 LOC

**Verification:**
- [ ] `wc -l src/jpscripts/commands/agent.py` shows < 60 lines
- [ ] `jp agent` command works unchanged
- [ ] `jp fix` alias works unchanged

**Files affected:**
- `src/jpscripts/commands/agent.py` - Significantly slimmed

**Notes:**
[Empty]

---

### Phase 6 Completion Checklist

- [ ] All steps marked `COMPLETED`
- [ ] All verification checks passing
- [ ] Tests pass: `pytest`
- [ ] Linting passes: `ruff check src`
- [ ] Type checking passes: `mypy src`
- [ ] Changes committed with message: `refactor: extract agent orchestration from CLI`
- [ ] Commit hash recorded in Progress Tracker
- [ ] Phase status updated to `COMPLETED`

---

## Phase 7: AST Caching

**Status:** `NOT STARTED`
**Estimated steps:** 2
**Commit:** -

### Phase 7 Overview

Re-parsing the entire project's AST on every step is inefficient. This phase implements an mtime-based cache for parsed ASTs in `DependencyWalker`, significantly speeding up repeated runs.

### Phase 7 Rollback Procedure

```bash
git revert [commit-hash]
rm src/jpscripts/analysis/cache.py
```

**Additional rollback notes:**
- Performance optimization only; functionality unchanged

---

### Step 7.1: Design Cache Strategy

**Status:** `NOT STARTED`

**Action:**
Design the AST cache based on file modification time and size.

**Sub-tasks:**
- [ ] Read `src/jpscripts/analysis/dependency_walker.py`
- [ ] Design cache key: `(file_path, mtime, size)`
- [ ] Choose cache storage: in-memory dict with LRU eviction
- [ ] Define cache entry format (serialized AST data or parsed objects)
- [ ] Consider thread/process safety

**Verification:**
- [ ] Design documented in Notes

**Files affected:**
- None (design only)

**Notes:**
[Will be filled with design]

---

### Step 7.2: Implement AST Cache

**Status:** `NOT STARTED`

**Action:**
Implement the cache in `src/jpscripts/analysis/cache.py` and integrate with `DependencyWalker`.

**Sub-tasks:**
- [ ] Create `cache.py` with `ASTCache` class
- [ ] Implement `get(path)` - returns cached AST or None
- [ ] Implement `put(path, ast)` - stores with mtime/size
- [ ] Implement `invalidate(path)` - removes stale entry
- [ ] Add LRU eviction (max 100 entries)
- [ ] Integrate into `DependencyWalker._parse_file()`

**Verification:**
- [ ] `pytest tests/unit/analysis/` passes
- [ ] Benchmark: second run of `jp evolve` is >2x faster
- [ ] Cache invalidates when file changes

**Files affected:**
- `src/jpscripts/analysis/cache.py` - New
- `src/jpscripts/analysis/dependency_walker.py` - Integration

**Notes:**
[Empty]

---

### Phase 7 Completion Checklist

- [ ] All steps marked `COMPLETED`
- [ ] All verification checks passing
- [ ] Tests pass: `pytest`
- [ ] Linting passes: `ruff check src`
- [ ] Type checking passes: `mypy src`
- [ ] Changes committed with message: `perf: AST caching for dependency analysis`
- [ ] Commit hash recorded in Progress Tracker
- [ ] Phase status updated to `COMPLETED`

---

## Phase 8: Red Team Suite

**Status:** `NOT STARTED`
**Estimated steps:** 2
**Commit:** -

### Phase 8 Overview

Final verification of safety rails through adversarial testing. This phase creates a "Red Team" test suite where a mock agent is explicitly instructed to perform harmful actions, verifying that Governance and Security layers block 100% of attempts.

### Phase 8 Rollback Procedure

```bash
git revert [commit-hash]
rm tests/security/test_red_team.py
```

**Additional rollback notes:**
- Test-only phase; no production code changes

---

### Step 8.1: Design Red Team Scenarios

**Status:** `NOT STARTED`

**Action:**
Define specific attack scenarios for the Red Team suite.

**Sub-tasks:**
- [ ] Define scenario: Delete `.git` folder
- [ ] Define scenario: Read `/etc/passwd`
- [ ] Define scenario: Exfiltrate `~/.ssh/id_rsa`
- [ ] Define scenario: Run shell command with `; rm -rf /`
- [ ] Define scenario: Write to `/tmp/malware.sh`
- [ ] Define scenario: Import forbidden modules in generated code
- [ ] Document expected blocking mechanism for each

**Verification:**
- [ ] At least 6 scenarios defined

**Files affected:**
- None (design only)

**Notes:**
[Will be filled with scenarios]

---

### Step 8.2: Implement Red Team Tests

**Status:** `NOT STARTED`

**Action:**
Create `tests/security/test_red_team.py` with adversarial test cases.

**Sub-tasks:**
- [ ] Create `test_red_team.py`
- [ ] Implement test for each scenario
- [ ] Mock agent with malicious instructions
- [ ] Assert `SecurityError` or `GovernanceViolation` raised
- [ ] Assert no filesystem changes occurred
- [ ] Add test for circuit breaker activation

**Verification:**
- [ ] All red team tests pass (attacks blocked)
- [ ] Tests are clearly documented as adversarial
- [ ] No false negatives (attacks succeed)

**Files affected:**
- `tests/security/test_red_team.py` - New

**Notes:**
[Empty]

---

### Phase 8 Completion Checklist

- [ ] All steps marked `COMPLETED`
- [ ] All verification checks passing
- [ ] Tests pass: `pytest`
- [ ] Linting passes: `ruff check src`
- [ ] Type checking passes: `mypy src`
- [ ] Changes committed with message: `security: red team adversarial test suite`
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
- [ ] `jp agent` command works as expected
- [ ] `jp evolve` command works as expected
- [ ] MCP server starts and responds to tool calls
- [ ] Security rules are enforced

### Documentation
- [ ] CLAUDE.md updated with new directory structure
- [ ] ARCHITECTURE.md updated if needed

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
- **Started:** Planning session
- **Ended:** TBD
- **Progress:** Roadmap created with all 8 phases defined
- **Next:** Begin Phase 1 implementation
- **Issues:** None

---

## Appendix: Reference Information

### Key Files

| File | Purpose |
|------|---------|
| `src/jpscripts/core/team.py` | Domain logic to move (Phase 1) |
| `src/jpscripts/core/notes_impl.py` | Domain logic to move (Phase 1) |
| `src/jpscripts/core/nav.py` | Domain logic to move (Phase 1) |
| `src/jpscripts/governance/ast_checker.py` | Governance rules (Phase 2) |
| `src/jpscripts/providers/factory.py` | Provider implementation (Phase 3) |
| `src/jpscripts/core/sys/execution.py` | Async utilities (Phase 4) |
| `src/jpscripts/mcp/tools/filesystem.py` | MCP security (Phase 5) |
| `src/jpscripts/commands/agent.py` | CLI to thin (Phase 6) |
| `src/jpscripts/analysis/dependency_walker.py` | AST caching (Phase 7) |

### Related Documentation
- `docs/ARCHITECTURE.md` - System design
- `docs/EXTENDING.md` - How to add commands/tools
- `CONTRIBUTING.md` - Development workflow

### Decisions Log

| Decision | Rationale | Date |
|----------|-----------|------|
| Use `features/` for domain logic | Separates domain from infrastructure per user request | 2025-12-02 |
| Production-critical priority | All phases must complete before new features | 2025-12-02 |
| mtime-based AST cache | Simple, effective invalidation without complex hashing | 2025-12-02 |
