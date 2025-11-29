{
  "constitution": {
    "version": "1.0",
    "system": "jpscripts",
    "metadata": {
      "owner": "jp-scripts maintainers",
      "purpose": "Codify non-negotiable engineering standards for agents and humans.",
      "scope": "All code, tests, and tools within the jpscripts workspace."
    },
    "invariants": [
      {
        "id": "typing",
        "rules": [
          "All code must pass mypy --strict; every function, method, and attribute carries explicit type annotations, including -> None for procedures.",
          "Any use of Any is prohibited unless wrapping third-party APIs without stubs and is justified with a type: ignore comment plus runtime validation at the boundary.",
          "All MCP tool parameters and configuration objects use pydantic.BaseModel or dataclass; no raw dicts for structured data."
        ]
      },
      {
        "id": "async-io",
        "rules": [
          "All I/O (git, filesystem, subprocess, HTTP) executes in async def; blocking calls must be wrapped in asyncio.to_thread.",
          "Shell execution never uses shell=True; commands are tokenized with shlex.split and executed via asyncio.create_subprocess_exec or core.system.run_safe_shell.",
          "All shell-capable code must invoke core.command_validation.validate_command before execution."
        ]
      },
      {
        "id": "error-handling",
        "rules": [
          "No bare except clauses; catch specific exceptions and map to Result[T, JPScriptsError] variants.",
          "Core layers return Result without printing; command layers pattern-match Ok/Err and render via console; MCP layers use tool_error_handler.",
          "Raw stack traces are never shown to users; errors are wrapped in typed domain errors."
        ]
      },
      {
        "id": "context-and-memory",
        "rules": [
          "Structured content (.py/.json/.yaml) is never naively sliced; use smart_read_context and TokenBudgetManager for truncation.",
          "File memories include content hashes; prune memory entries on drift or deletion; retrieval uses reciprocal rank fusion (k=60)."
        ]
      },
      {
        "id": "testing",
        "rules": [
          "Every CLI command has a smoke test in tests/test_smoke.py covering --help and basic invocation.",
          "Async tests use pytest.mark.asyncio; security-sensitive code ships explicit security tests; bug fixes include regression coverage."
        ]
      },
      {
        "id": "destructive-fs",
        "rules": [
          "Destructive Python file operations (shutil.rmtree, os.remove, etc.) are forbidden unless the line contains `# safety: checked`. Enforced by AST."
        ]
      },
      {
        "id": "dynamic-execution",
        "rules": [
          "Dynamic Execution: Usage of `eval`, `exec`, or dynamic imports is strictly forbidden to prevent obfuscated shell injection."
        ]
      }
    ],
    "cognitive_standards": [
      {
        "id": "safety-scan",
        "rules": [
          "Before executing any tool, performing any write, or running any command, you must explicitly list potential side effects (e.g., file modification, process termination) and verify reversibility."
        ]
      },
      {
        "id": "invariant-citation",
        "rules": [
          "Every step in a proposed plan must explicitly cite the Constitution Invariant ID it satisfies (e.g., [invariant:async-io])."
        ]
      },
      {
        "id": "anti-pattern-check",
        "rules": [
          "Verify the proposed approach against known anti-patterns (Sync I/O, Shell Injection, Bare Except) before finalizing the response."
        ]
      }
    ],
    "protocols": [
      {
        "id": "reflexion",
        "steps": [
          "On first failure, retry with a minimal adjustment.",
          "On second failure, halt and perform structured reflection: identify root causes, question assumptions, and choose an alternate approach.",
          "After reflection, proceed with the revised plan before further action."
        ]
      },
      {
        "id": "commits-and-edits",
        "steps": [
          "Never amend or revert user changes unless explicitly requested.",
          "Validate file existence before reading or writing; prefer idempotent operations.",
          "Avoid destructive commands (reset, checkout, rm) without explicit approval."
        ]
      },
      {
        "id": "command-execution",
        "steps": [
          "Validate every command via core.command_validation.validate_command.",
          "Execute with asyncio.create_subprocess_exec or core.system.run_safe_shell; never use shell=True or os.system.",
          "Warn when commands reference paths outside the workspace."
        ]
      }
    ],
    "security_clearance": {
      "allowed": [
        "Read operations on paths validated by security.validate_path within the workspace root.",
        "Write operations limited to workspace and configured writable roots; external writes require explicit approval.",
        "Async subprocess execution after command validation; use shlex.split for tokenization."
      ],
      "forbidden": [
        "Following symlinks outside the workspace, naive truncation of structured files, exposing raw stack traces, or executing shell=True/unsafe commands."
      ]
    }
  }
}
