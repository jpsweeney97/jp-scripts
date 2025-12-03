# jpscripts Architecture Audit & Improvement Roadmap

## Executive Summary

**Repository:** `jpscripts` - A typed Python CLI for autonomous agent orchestration with MCP server, memory embeddings, and parallel swarm execution.

**Overall Assessment:** Well-architected with strong security foundations but suffering from accumulated complexity, inconsistent module boundaries, and missing infrastructure for sustainable development.

---

## Audit Findings by Area

### 1. Architecture & Boundaries

#### Critical Issues

**1.1 Governance Module Fragmentation**

- **Problem:** Governance logic is split between `governance/` (AST checker, secret scanner) and `core/safety.py`, `core/security.py`, `core/command_validation.py`, and `agent/governance.py`. This creates confusion about where security enforcement lives.
- **Evidence:** `governance/ast_checker.py` (19KB), `core/security.py` (17KB), `core/safety.py` (7KB) - all doing overlapping security work.
- **Impact:** Risk of inconsistent enforcement; cognitive load for contributors.

**1.2 Oversized Modules**

- `src/jpscripts/agent/execution.py` (32KB, ~900 lines) - contains repair loop, patch processing, archiving, tool handling - too many responsibilities.
- `src/jpscripts/memory/store.py` (25KB) - contains JsonlArchiver, LanceDBStore, HybridMemoryStore, plus tokenization, file hashing, and more.
- `src/jpscripts/git/client.py` (20KB) - full git abstraction mixing I/O, parsing, and data models.

**1.3 Provider Abstraction Inconsistencies**

- `providers/factory.py` uses `ProviderConfig` dataclass with internal `_provider_cache` dict - mutable state in a configuration object is an anti-pattern.
- Provider lazy registration via decorator is fragile: `_ensure_providers_registered()` imports modules as side-effect.

**1.4 Missing Interface Segregation**

- `MemoryStore` protocol in `memory/models.py` but implementations tightly coupled to LanceDB/JSONL specifics.
- No clear interface for tool execution - `execute_tool()` takes raw dict tools map.

#### Recommendations

1. Consolidate all security into `core/security/` package with clear sub-modules: `path.py`, `command.py`, `rate_limit.py`, `circuit_breaker.py`.
2. Split `agent/execution.py` into: `loop.py`, `patching.py`, `archiving.py`.
3. Create `VectorStoreProtocol` and `KeywordStoreProtocol` for memory backends.

---

### 2. Code Quality & Testing

#### Critical Issues

**2.1 Test Coverage Below Threshold for Security-Critical Code**

- Current: 62% minimum enforced.
- Security modules need 90%+: `core/security.py`, `governance/ast_checker.py`, `core/command_validation.py`.
- **Missing critical path tests:**
  - No property tests for path validation edge cases beyond symlinks.
  - Circuit breaker cost calculations lack boundary testing.
  - Token budget manager truncation lacks fuzz testing.

**2.2 No CI/CD Pipeline Visible**

- No `.github/workflows/` in manifest.
- **Impact:** No automated quality gates, no enforced code review, no reproducible builds.

**2.3 Test Fixture Sprawl**

- `conftest.py` has `isolate_config`, `capture_console`, `ensure_commands_registered` as autouse.
- Integration tests mock at multiple layers inconsistently.
- `tests/mocks/mock_provider.py` is ad-hoc; should align with `LLMProvider` protocol strictly.

**2.4 Incomplete Error Path Testing**

- `test_agent_real.py` only tests happy paths.
- No tests for circuit breaker tripping mid-operation.
- No tests for concurrent access to runtime context.

#### Recommendations

1. Add GitHub Actions workflow with lint/test/coverage gates.
2. Create property-based test suite using Hypothesis for:
   - Path validation (`security.py`)
   - Token budget allocation
   - JSON extraction (`parsing.py`)
3. Split test fixtures into focused conftest files per test package.

---

### 3. Performance, Reliability & Security

#### Critical Issues

**3.1 Memory Store Fallback Complexity**

- `HybridMemoryStore.__init__` has complex try/except chains with different degradation modes.
- Lazy LanceDB import with fallback to JSONL creates unpredictable behavior.
- **Evidence:** `_lancedb_available()` checks are scattered.

**3.2 Async/Sync Boundary Violations**

- `_validate_workspace_root_cached` is sync with `@lru_cache`, but caller may be async.
- `_is_git_repo()` runs synchronous subprocess inside what could be async context.
- **Evidence:** Both sync and async variants exist (`validate_path` vs `validate_path_async`), but callers inconsistently use them.

**3.3 Circuit Breaker State Mutation**

- `CircuitBreaker` in `runtime.py` mutates `_last_check_timestamp` and `last_*` fields during checks.
- Not thread-safe despite `RuntimeContext` using context vars.
- **Impact:** Potential race conditions in parallel swarm execution.

**3.4 Rate Limiter per-Context Isolation**

- Each `RuntimeContext` creates its own `RateLimiter` instance.
- In swarm mode with multiple worktrees, rate limits aren't shared.
- Could exceed global API rate limits.

**3.5 Token Estimation Rough**

- `estimate_tokens_from_args()` uses `len(str) // 4` approximation.
- For cost-critical circuit breaker decisions, this could undercount by 2-3x for code.

#### Recommendations

1. Create explicit `StorageBackend` enum with clear degradation policy.
2. Audit all sync I/O in async functions; enforce with custom AST rule.
3. Add `threading.Lock` to `CircuitBreaker` for thread safety.
4. Consider global rate limiter shared via process-level singleton.

---

### 4. Developer Experience & Conventions

#### Critical Issues

**4.1 No CI Pipeline**

- No automated testing, no branch protection indicators in codebase.
- `make test` runs linting then tests, but not integrated into Git workflow.

**4.2 Inconsistent Import Structure**

- Some modules use relative imports (`from .models import`), others use absolute (`from jpscripts.core.result import`).
- `mcp/tools/__init__.py` re-exports from `capabilities.registry` - circular-looking import graph.

**4.3 Documentation Drift**

- `ARCHITECTURE.md` references `engine/` subdirectory structure that doesn't exist (shows `engine/__init__.py`, `response_handler.py`, etc.).
- Mermaid diagrams reference modules like `core/shell.py` that don't exist.

**4.4 Incomplete Type Stubs**

- `[[tool.mypy.overrides]]` ignores 11 external packages.
- `providers/` modules have complex generics that could benefit from `typing.ParamSpec`.

**4.5 Missing Development Scripts**

- No pre-commit hooks configured.
- No integration test script for running against real LLM.
- No benchmark suite for performance regression.

#### Recommendations

1. Add `.github/workflows/ci.yml` with lint, type-check, test stages.
2. Create `.pre-commit-config.yaml` with ruff, mypy hooks.
3. Audit and update `ARCHITECTURE.md` to match actual structure.
4. Add `scripts/benchmark.py` for performance tracking.

---

## Prioritized Roadmap for Claude Code

### Phase 1: Infrastructure & CI (Foundation)

**Duration:** 1-2 hours
**Risk Reduction:** High - prevents regression during subsequent refactors

```
# Task 1.1: Create GitHub Actions CI Pipeline
Create .github/workflows/ci.yml with:
- Python 3.11 matrix
- Install with [dev,ai] extras
- Run: ruff format --check, ruff check, mypy src
- Run: pytest --cov --cov-fail-under=62
- Upload coverage to artifact

# Task 1.2: Add Pre-Commit Hooks
Create .pre-commit-config.yaml:
- ruff (format + check)
- mypy (src only)
- check-yaml, end-of-file-fixer

# Task 1.3: Fix Documentation Drift
Update docs/ARCHITECTURE.md:
- Remove references to non-existent engine/ subdirectory
- Update module diagram to match actual structure
- Add missing modules (capabilities/, features/)
```

**Precondition:** None
**Tradeoff:** Adds CI overhead but prevents quality regression.

---

### Phase 2: Security Consolidation

**Duration:** 2-3 hours
**Risk Reduction:** Critical - centralizes security enforcement

```
# Task 2.1: Create core/security/ Package
Restructure core/security.py into:
- src/jpscripts/core/security/__init__.py (re-exports)
- src/jpscripts/core/security/path.py (validate_path, validate_workspace_root, TOCTOU helpers)
- src/jpscripts/core/security/command.py (command validation, allowlist/blocklist)
- src/jpscripts/core/security/rate_limit.py (move from core/rate_limit.py)
- src/jpscripts/core/security/circuit.py (move CircuitBreaker from runtime.py)

# Task 2.2: Add Thread Safety to CircuitBreaker
In core/security/circuit.py:
- Add threading.Lock for _last_check_timestamp mutations
- Add atomic check_and_record() method

# Task 2.3: Increase Security Test Coverage
Add to tests/security/:
- test_path_properties.py (Hypothesis-based)
- test_circuit_breaker_concurrency.py (threading tests)
- test_command_validation_exhaustive.py

Target: 90%+ coverage on core/security/ package
```

**Precondition:** Phase 1 complete (CI to catch regressions)
**Tradeoff:** Breaking import changes require updating all callers.

---

### Phase 3: Agent Module Decomposition

**Duration:** 2-3 hours
**Risk Reduction:** Medium - improves maintainability

```
# Task 3.1: Split agent/execution.py
Create:
- agent/loop.py: run_repair_loop(), _process_turn()
- agent/archive.py: _archive_session_summary(), memory helpers
- agent/patch_processor.py: _process_patch(), _process_tool_call()

Keep agent/execution.py as facade that re-exports for backward compatibility.

# Task 3.2: Extract Agent Events
Move AgentEvent, EventKind, RepairLoopConfig to agent/events.py
agent/models.py becomes purely LLM-focused (Message, ToolCall, AgentResponse)

# Task 3.3: Create ToolExecutor Protocol
In agent/tools.py, define:
class ToolExecutor(Protocol):
    async def execute(self, call: ToolCall) -> str: ...
    def available_tools(self) -> list[str]: ...

Update AgentEngine to accept ToolExecutor instead of raw dict.
```

**Precondition:** Phase 2 complete (security is stable)
**Tradeoff:** Temporary backward-compat facades add complexity.

---

### Phase 4: Memory Store Simplification

**Duration:** 2 hours
**Risk Reduction:** Medium - reduces fallback complexity

```
# Task 4.1: Create Explicit Storage Backend Enum
In memory/backends.py:
class StorageMode(Enum):
    LANCE_HYBRID = "lance_hybrid"  # LanceDB + JSONL fallback
    JSONL_ONLY = "jsonl_only"      # Pure keyword search
    LANCE_ONLY = "lance_only"      # Vector only (no archival)

# Task 4.2: Create Backend Factory
def create_memory_store(mode: StorageMode, config: AppConfig) -> MemoryStore:
    # Explicit instantiation, no lazy fallbacks

# Task 4.3: Split store.py
- memory/store.py -> memory/store/__init__.py (facade)
- memory/store/jsonl.py (JsonlArchiver)
- memory/store/lance.py (LanceDBStore)
- memory/store/hybrid.py (HybridMemoryStore)

# Task 4.4: Add Store Integration Tests
tests/integration/test_memory_stores.py:
- Test each backend in isolation
- Test fallback behavior explicitly
- Test concurrent access
```

**Precondition:** Phase 1 (CI)
**Tradeoff:** User config may need migration if storage mode becomes explicit.

---

### Phase 5: Provider Layer Cleanup

**Duration:** 1-2 hours
**Risk Reduction:** Low - cleanup, not structural

```
# Task 5.1: Remove Mutable Cache from ProviderConfig
Move provider caching to module-level or RuntimeContext:
- Remove _provider_cache from ProviderConfig dataclass
- Add provider_cache to RuntimeContext

# Task 5.2: Explicit Provider Registration
Replace decorator-based registration:
- Create providers/registry.py with explicit registration
- Remove _ensure_providers_registered() side-effect imports

# Task 5.3: Improve Token Estimation
In core/safety.py or ai/tokens.py:
- Use tiktoken for accurate token counting when model is known
- Fall back to len/4 only for unknown models
```

**Precondition:** Phase 3 (agent stable)
**Tradeoff:** Minor API change to ProviderConfig construction.

---

### Phase 6: Test Infrastructure Hardening

**Duration:** 2-3 hours
**Risk Reduction:** High - sustainable quality

```
# Task 6.1: Split conftest.py
- tests/conftest.py (minimal: runner, paths)
- tests/fixtures/config.py (isolate_config)
- tests/fixtures/console.py (capture_console)
- tests/fixtures/mcp.py (MCP-specific)

# Task 6.2: Property Test Suite
Create tests/properties/:
- test_path_validation_props.py
- test_token_budget_props.py
- test_json_extraction_props.py
- test_patch_parsing_props.py

# Task 6.3: Fuzz Testing for Parsing
Add tests/fuzz/:
- test_parse_agent_response_fuzz.py
- test_apply_patch_fuzz.py

# Task 6.4: Coverage Increase
Target 70% overall, 90% for:
- core/security/*
- governance/ast_checker.py
- agent/parsing.py
```

**Precondition:** All prior phases
**Tradeoff:** Adds test runtime but catches edge cases.

---

### Phase 7: Documentation & Onboarding

**Duration:** 1 hour
**Risk Reduction:** Low - developer velocity

```
# Task 7.1: Architecture Diagram Update
Regenerate docs/ARCHITECTURE.md diagrams to match actual:
- core/security/ package
- agent/ decomposed modules
- memory/store/ package

# Task 7.2: Add ADR Directory
Create docs/adr/:
- 001-result-type-pattern.md
- 002-context-variable-runtime.md
- 003-security-consolidation.md

# Task 7.3: Contributing Guide Update
Update CONTRIBUTING.md:
- Add pre-commit setup instructions
- Document new package structure
- Add troubleshooting section
```

**Precondition:** Phases 2-4 complete
**Tradeoff:** Documentation maintenance burden.

---

## Claude Code Execution Prompt

```markdown
# jpscripts Improvement Project

## Context

You are working on jpscripts, a Python CLI for autonomous agent orchestration.
The codebase uses:

- Python 3.11+ with strict mypy
- Result[T, E] pattern for error handling
- Context variables for thread-safe runtime state
- Typer for CLI, FastMCP for MCP server

## Current Phase: [INSERT PHASE NUMBER]

## Phase Objectives

[INSERT OBJECTIVES FROM ROADMAP]

## Files to Modify

[INSERT SPECIFIC FILE LIST]

## Success Criteria

1. All existing tests pass
2. mypy --strict passes
3. ruff check passes
4. New tests added for changed code
5. Coverage does not decrease

## Constraints

- Do not modify unrelated modules
- Maintain backward compatibility where noted
- Add deprecation warnings before removing APIs
- Update imports in all callers when moving code

## Execution Steps

1. Read current implementation of target files
2. Create new structure following the task description
3. Update all import statements across the codebase
4. Add/update tests for changed functionality
5. Run full test suite and fix any failures
6. Update documentation if applicable
```

---

## Risk Assessment

| Phase       | Risk Level | Rollback Difficulty   | Estimated Breakage |
| ----------- | ---------- | --------------------- | ------------------ |
| 1. CI/Infra | Low        | Easy (delete files)   | None               |
| 2. Security | Medium     | Hard (many callers)   | Import errors      |
| 3. Agent    | Medium     | Medium (facade helps) | Test failures      |
| 4. Memory   | Medium     | Medium                | Config changes     |
| 5. Provider | Low        | Easy                  | Minor API          |
| 6. Testing  | Low        | Easy                  | None               |
| 7. Docs     | Low        | N/A                   | None               |

---

## Assumptions Made

1. **No external CI service configured** - based on absence of `.github/` in manifest. If CI exists elsewhere, Phase 1 adjusts.
2. **LanceDB is optional dependency** - based on `[ai]` extras. Memory fallback behavior is intentional.
3. **Single-user CLI** - thread safety concerns are primarily for swarm mode, not multi-tenant.
4. **Test coverage 62%** - from pyproject.toml `fail_under`. Actual may differ.
5. **Architecture.md is stale** - references non-existent modules like `engine/`.

---

## Missing Information

1. **Actual test coverage report** - would change Phase 6 priorities.
2. **Production usage patterns** - would affect performance optimization priority.
3. **LLM provider usage split** - Anthropic vs OpenAI affects token estimation priority.
4. **Memory store size in practice** - affects need for pagination/streaming.
