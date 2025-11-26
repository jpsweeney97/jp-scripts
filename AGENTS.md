# JPScripts Agent Constitution

## I. Operational Invariants (Violations = Immediate Failure)

1. **The Context Invariant**: You must NEVER hallucinate the existence of a file. If you have not seen it in `<file_context>` or `ls` output, it does not exist.
2. **The Verification Invariant**: Every code change MUST be followed by a verification step (`pytest` or reproduction script). You cannot mark a task "Complete" without a passing test log.
3. **The Sandbox Invariant**: You operate in `workspace_root`. Any attempt to access `/etc`, `/var`, or `~/.ssh` is a violation of your core programming.

## II. Tooling Protocols

- **Reading**: Use `read_file` for small files. For files >50KB, you MUST use `read_file_paged`.
- **Search**: Prefer `search_codebase` (ripgrep) over `find` for code queries. It is faster and respects `.gitignore`.
- **Memory**: If you solve a novel error, you MUST call `remember(fact="...")`.

## III. Interaction Mode

- **Brevity**: Do not apologize. Do not explain standard Python concepts. Output code and shell commands.
- **Diffs**: When applying changes, prefer writing the full file if it is under 200 lines. Use `sed`/`patch` only for massive files where context window is tight.
