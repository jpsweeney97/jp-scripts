# JPScripts God-Mode Handbook

Concise rules for operating `jp` with precision. No fluff, no drift.

---

## The Zero-State (≤3 minutes to operational)

1) **Prereqs**: Python 3.11+, `git`, `ruff`, `rg`, `fzf`. Optional: `uv` or `pipx` for isolated install.  
2) **Bootstrap**:  
   - `git clone https://github.com/.../jp-scripts.git` (or sync your fork)  
   - `cd jp-scripts`  
   - `uv tool install .` _or_ `pipx install .` (fallback: `pip install -e .[dev]`)  
3) **Config**: `jp init` to generate `~/.jpconfig`; set `workspace_root` and `memory_store`.  
4) **Verify**: `jp doctor` then `jp status-all`. You are live.

---

## The OODA Loop (Observe → Orient → Decide → Act)

- **Observe**: `jp map --depth 3` for structure; `jp recent` for hot files.  
- **Orient**: `jp status-all` for repo health; `jp doctor` if tools misbehave.  
- **Decide**: `jp fix --recent "State objective"` chooses the target; on failure it escalates context (see Dynamic Context Expansion).  
- **Act**: `jp watch` runs God-Mode maintenance (syntax gate + live embedding refresh); keep it running in a dedicated pane.

---

## Dynamic Context Expansion (jp fix / jp agent)

Retries get smarter, not just louder:
- **Attempt 1 – Fast**: Run command, capture diagnostics, include directly referenced files.  
- **Attempt 2 – Deep**: Parse new errors, add imported dependencies and referenced modules. System notice injected to analyze cross-module interactions.  
- **Attempt 3 – Step Back**: Tool use disabled for the turn, reasoning_effort=high, demand a Root Cause Analysis plan before patching.  
Result: slower per retry, higher precision, fewer insanity loops.

---

## God-Mode Watch (jp watch)

- Monitors `workspace_root` via `watchdog`, honoring `ignore_dirs`.  
- On `.py` save: `ruff check --select E9,F821` (syntax gate); red alert on failure.  
- On text save: debounced (5s) LanceDB embedding refresh so `jp memory search` is always current.  
- Live dashboard (`rich.live`) displays recent events and task status. Leave it running.

---

## Emergency Protocols

- **Agent stuck/unsafe**: Re-run with `--read-only` (no patch apply). Inspect diffs manually.  
- **Repeated failure**: Drop to manual mode: `jp map` + `jp status-all`, open files, patch by hand, then re-run `jp fix` once.  
- **Tool failure**: `jp doctor` to validate binaries; verify `.jpconfig` paths.  
- **Sandbox breach risk**: Manually vet commands; never bypass `validate_command`.  
- **Rollback**: If auto patches misfire, revert affected files (`git checkout -- <paths>` or `jp gundo-last` if ahead-only).  
- **Operational pause**: Stop `jp watch` before destructive refactors; restart after structure stabilizes.

---

## Prompt Engineering Guide (Precision Comms)

- **Architect Persona** (strategy/constraints):  
  - Use when you need plans, guardrails, or multi-step execution.  
  - Speak in objectives and constraints: “Design a 3-step migration for X; forbid data loss; prefer async I/O.”  
  - Ask for risks and rollback steps.

- **Engineer Persona** (surgical edits):  
  - Use for concrete patches and tests.  
  - Provide the failing command, error excerpt, and desired end state.  
  - Specify boundaries: files to touch, patterns to avoid, acceptable refactors.

- **General Rules**:  
  - Supply minimal but exact context (paths, commands, error tail).  
  - Declare priorities (perf vs. correctness vs. speed).  
  - Request diffs, not prose. Demand tests when touching critical paths.

---

## Field Notes (Daily Discipline)

- Keep `jp watch` running; let it block on syntax before you git-add.  
- Run `jp status-all` before commits; refuse to ship dirty/behind states without intent.  
- Capture lessons with `jp memory add ... --tag <area>`; this feeds immediate recall.  
- When handing off, run `jp serialize snapshot` to produce a lossless manifest for teammates or agents.
