<constitution version="1.0" system="jpscripts">
  <metadata>
    <owner>jp-scripts maintainers</owner>
    <purpose>Codify non-negotiable engineering standards for agents and humans.</purpose>
    <scope>All code, tests, and tools within the jpscripts workspace.</scope>
  </metadata>

  <invariant id="typing">
    <rule>All code must pass mypy --strict; every function, method, and attribute carries explicit type annotations, including -> None for procedures.</rule>
    <rule>Any use of Any is prohibited unless wrapping third-party APIs without stubs and is justified with a type: ignore comment plus runtime validation at the boundary.</rule>
    <rule>All MCP tool parameters and configuration objects use pydantic.BaseModel or dataclass; no raw dicts for structured data.</rule>
  </invariant>

  <invariant id="async-io">
    <rule>All I/O (git, filesystem, subprocess, HTTP) executes in async def; blocking calls must be wrapped in asyncio.to_thread.</rule>
    <rule>Shell execution never uses shell=True; commands are tokenized with shlex.split and executed via asyncio.create_subprocess_exec or core.system.run_safe_shell.</rule>
    <rule>All shell-capable code must invoke core.command_validation.validate_command before execution.</rule>
  </invariant>

  <invariant id="error-handling">
    <rule>No bare except clauses; catch specific exceptions and map to Result[T, JPScriptsError] variants.</rule>
    <rule>Core layers return Result without printing; command layers pattern-match Ok/Err and render via console; MCP layers use tool_error_handler.</rule>
    <rule>Raw stack traces are never shown to users; errors are wrapped in typed domain errors.</rule>
  </invariant>

  <invariant id="context-and-memory">
    <rule>Structured content (.py/.json/.yaml) is never naively sliced; use smart_read_context and TokenBudgetManager for truncation.</rule>
    <rule>File memories include content hashes; prune memory entries on drift or deletion; retrieval uses reciprocal rank fusion (k=60).</rule>
  </invariant>

  <invariant id="testing">
    <rule>Every CLI command has a smoke test in tests/test_smoke.py covering --help and basic invocation.</rule>
    <rule>Async tests use pytest.mark.asyncio; security-sensitive code ships explicit security tests; bug fixes include regression coverage.</rule>
  </invariant>

  <protocol id="reflexion">
    <step>On first failure, retry with a minimal adjustment.</step>
    <step>On second failure, halt and perform structured reflection: identify root causes, question assumptions, and choose an alternate approach.</step>
    <step>After reflection, proceed with the revised plan before further action.</step>
  </protocol>

  <protocol id="commits-and-edits">
    <step>Never amend or revert user changes unless explicitly requested.</step>
    <step>Validate file existence before reading or writing; prefer idempotent operations.</step>
    <step>Avoid destructive commands (reset, checkout, rm) without explicit approval.</step>
  </protocol>

  <protocol id="command-execution">
    <step>Validate every command via core.command_validation.validate_command.</step>
    <step>Execute with asyncio.create_subprocess_exec or core.system.run_safe_shell; never use shell=True or os.system.</step>
    <step>Warn when commands reference paths outside the workspace.</step>
  </protocol>

  <security_clearance>
    <allowed>Read operations on paths validated by security.validate_path within the workspace root.</allowed>
    <allowed>Write operations limited to workspace and configured writable roots; external writes require explicit approval.</allowed>
    <allowed>Async subprocess execution after command validation; use shlex.split for tokenization.</allowed>
    <forbidden>Following symlinks outside the workspace, naive truncation of structured files, exposing raw stack traces, or executing shell=True/unsafe commands.</forbidden>
  </security_clearance>
</constitution>
