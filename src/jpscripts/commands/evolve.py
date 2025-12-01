"""
Proactive evolution command for autonomous codebase improvement.

The `jp evolve` command identifies files with high technical debt
(complexity x fix frequency) and autonomously refactors them,
creating a PR for review.

Usage:
    jp evolve run --dry-run          # Analyze without changes
    jp evolve run --threshold 15     # Only act on debt score > 15
    jp evolve report                 # Show complexity report
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Awaitable
from pathlib import Path

import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

from jpscripts.agent import PreparedPrompt, run_repair_loop
from jpscripts.analysis.complexity import (
    FileComplexity,
    FunctionComplexity,
    TechnicalDebtScore,
    analyze_directory_complexity,
    calculate_debt_scores,
)
from jpscripts.analysis.structure import get_import_dependencies
from jpscripts.core.config import AppConfig
from jpscripts.core.console import console, get_logger
from jpscripts.core.result import Err, Ok
from jpscripts.core.system import run_safe_shell
from jpscripts.git import client as git_core
from jpscripts.main import AppState
from jpscripts.memory import save_memory
from jpscripts.providers import CompletionOptions, Message, ProviderType, infer_provider_type
from jpscripts.providers.factory import get_provider

logger = get_logger(__name__)

app = typer.Typer(help="Autonomous code evolution and optimization.")


async def _cleanup_branch(repo: git_core.AsyncRepo, branch_name: str) -> None:
    """Clean up a failed evolution branch by returning to main."""
    await repo.run_git("checkout", "main")
    await repo.run_git("branch", "-D", branch_name)


async def _abort_evolution(
    repo: git_core.AsyncRepo,
    branch_name: str,
    message: str,
    memory_tags: list[str],
    config: AppConfig,
    reset_hard: bool = False,
) -> None:
    """Abort evolution with cleanup and optional memory logging."""
    console.print(f"[red]{message}[/red]")
    if reset_hard:
        await repo.run_git("reset", "--hard", "main")
    await _cleanup_branch(repo, branch_name)
    if memory_tags:
        await asyncio.to_thread(save_memory, message, memory_tags, config=config)


async def _collect_dependent_tests(
    python_changes: list[Path],
    tests_root: Path,
    root: Path,
) -> list[Path]:
    """Collect test files that depend on the changed Python files."""
    if not tests_root.exists():
        return []
    try:
        test_files = await asyncio.to_thread(lambda: list(tests_root.rglob("test_*.py")))
    except OSError:
        return []

    dependents: list[Path] = []
    for test_file in test_files:
        try:
            deps: set[Path] = await asyncio.to_thread(get_import_dependencies, test_file, root)
        except Exception as exc:
            logger.debug("Failed to get dependencies for %s: %s", test_file, exc)
            deps = set()
        for changed in python_changes:
            if changed in deps:
                dependents.append(test_file)
                break
    return dependents


def _build_optimizer_prompt(target: TechnicalDebtScore) -> str:
    """Build the prompt for the Optimizer persona."""
    reasons_text = (
        "\n".join(f"- {r}" for r in target.reasons) if target.reasons else "High complexity"
    )

    return f"""You are an Optimizer persona. Your task is to reduce technical debt in a specific file.

Target file: {target.path}
Current complexity score: {target.complexity_score:.1f}
Historical fix frequency: {target.fix_frequency}
Git churn (commit count): {target.churn}
Identified issues:
{reasons_text}

Your objectives:
1. **Reduce cyclomatic complexity** by extracting helper functions or simplifying logic
2. **Improve code clarity** with better naming and structure
3. **Add or improve type annotations** where missing
4. **Preserve all public interfaces** - do not change function signatures for public API
5. Ensure all changes pass `mypy --strict`

Constraints:
- Preserve all existing behavior (pure refactoring)
- All I/O must remain async where it currently is
- Follow existing patterns in the codebase
- Keep changes minimal and focused on complexity reduction

Emit a unified diff patch that addresses the technical debt. Focus on the most complex
functions first. If the file is large, prioritize the top 1-2 functions by complexity."""


async def _create_evolution_pr(
    repo: git_core.AsyncRepo,
    target: TechnicalDebtScore,
    branch_name: str,
    root: Path,
    config: AppConfig,
    verification_cmd: str,
    verification_exit: int,
) -> None:
    """Create a PR for the evolution changes."""
    # Stage and commit
    await repo.run_git("add", "-A")
    commit_message = (
        f"refactor({target.path.stem}): reduce technical debt\n\n"
        f"Complexity score reduced from {target.complexity_score:.1f}\n"
        f"Autonomous optimization via jp evolve."
    )
    await repo.run_git("commit", "-m", commit_message)

    # Push and create PR
    console.print("[cyan]Pushing branch and creating PR...[/cyan]")
    await repo.run_git("push", "-u", "origin", branch_name)

    # Create PR using gh CLI
    pr_body = f"""## Autonomous Optimization

**Target:** `{target.path}`
**Debt Score:** {target.debt_score:.1f}
**Complexity Score:** {target.complexity_score:.1f}
**Fix Frequency:** {target.fix_frequency}
**Churn:** {target.churn}

### Reasons for Selection
{chr(10).join("- " + r for r in target.reasons) if target.reasons else "- High complexity"}

### Verification
- `{verification_cmd}` -> exit {verification_exit}

### Changes
- Reduced cyclomatic complexity
- Improved code clarity
- Enhanced type annotations

### Test Plan
- [ ] `pytest tests/`
- [ ] `mypy --strict src/`

---
Generated by `jp evolve`
"""

    try:
        proc = await asyncio.create_subprocess_exec(
            "gh",
            "pr",
            "create",
            "--title",
            f"refactor({target.path.stem}): reduce technical debt",
            "--body",
            pr_body,
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            pr_url = stdout.decode().strip()
            console.print(f"[green]PR created:[/green] {pr_url}")
            # Persist evolution success to memory
            try:
                await asyncio.to_thread(
                    save_memory,
                    f"Evolution Success: Refactored `{target.path}`. "
                    f"Complexity delta: {target.complexity_score:.1f} -> reduced. "
                    f"PR: {pr_url}",
                    ["evolution", "refactor", "success", target.path.name],
                    config=config,
                )
                logger.debug("Evolution success persisted to memory")
            except Exception as exc:
                logger.debug("Failed to persist evolution to memory: %s", exc)
        else:
            error = stderr.decode().strip()
            console.print(f"[yellow]PR creation failed:[/yellow] {error}")
            console.print("[dim]You may need to create the PR manually.[/dim]")
    except FileNotFoundError:
        console.print("[yellow]gh CLI not found. Please create the PR manually.[/yellow]")
        console.print(f"[dim]Branch: {branch_name}[/dim]")


async def _run_evolve(
    config: AppConfig,
    dry_run: bool,
    model: str | None,
    threshold: float,
) -> None:
    """Core evolution logic."""
    root = Path(config.user.workspace_root).expanduser().resolve()

    # Step 1: Check clean git state
    console.print("[cyan]Checking git state...[/cyan]")
    match await git_core.AsyncRepo.open(root):
        case Err(git_err):
            console.print(f"[red]Git error: {git_err}[/red]")
            return
        case Ok(repo):
            pass

    match await repo.status():
        case Err(status_err):
            console.print(f"[red]Status error: {status_err}[/red]")
            return
        case Ok(status):
            if status.dirty:
                console.print("[red]Workspace is dirty. Commit or stash changes first.[/red]")
                return

    # Step 2: Calculate debt scores
    console.print("[cyan]Analyzing complexity and fix history...[/cyan]")
    match await calculate_debt_scores(root, config):
        case Err(debt_err):
            console.print(f"[red]Analysis failed: {debt_err}[/red]")
            return
        case Ok(scores):
            if not scores:
                console.print("[green]No files require optimization.[/green]")
                return

    # Show top candidates
    table = Table(title="Technical Debt Analysis", box=box.ROUNDED)
    table.add_column("File", style="cyan")
    table.add_column("Complexity", justify="right")
    table.add_column("Fix Freq", justify="right")
    table.add_column("Debt Score", justify="right", style="yellow")

    for score in scores[:10]:
        table.add_row(
            score.path.name,
            f"{score.complexity_score:.0f}",
            str(score.fix_frequency),
            f"{score.debt_score:.1f}",
        )

    console.print(table)

    # Step 3: Select target
    target = scores[0]
    if target.debt_score < threshold:
        console.print(
            f"[green]Top debt score {target.debt_score:.1f} is below "
            f"threshold {threshold}. No optimization needed.[/green]"
        )
        return

    console.print()
    console.print(
        Panel(
            f"[bold]Target:[/bold] {target.path}\n"
            f"[bold]Debt Score:[/bold] {target.debt_score:.1f}\n"
            f"[bold]Complexity:[/bold] {target.complexity_score:.1f}\n"
            f"[bold]Fix Frequency:[/bold] {target.fix_frequency}\n\n"
            f"[bold]Reasons:[/bold]\n" + "\n".join(f"  - {r}" for r in target.reasons),
            title="Selected for Optimization",
            box=box.ROUNDED,
        )
    )

    if dry_run:
        console.print("\n[yellow]Dry run mode - no changes will be made.[/yellow]")
        return

    # Step 4: Create branch
    branch_name = f"evolve/{target.path.stem}-optimization"
    console.print(f"\n[cyan]Creating branch: {branch_name}[/cyan]")

    try:
        await repo.run_git("checkout", "-b", branch_name)
    except Exception as exc:
        console.print(f"[red]Failed to create branch: {exc}[/red]")
        return

    # Step 5: Launch optimizer agent
    console.print("[cyan]Launching optimizer agent...[/cyan]")
    optimizer_prompt = _build_optimizer_prompt(target)
    target_model = model or config.ai.default_model

    # Determine provider (force native, not Codex)
    ptype = infer_provider_type(target_model)
    if ptype == ProviderType.CODEX:
        ptype = ProviderType.OPENAI
    provider = get_provider(config, model_id=target_model, provider_type=ptype)

    async def fetch_response(prepared: PreparedPrompt) -> str:
        messages = [Message(role="user", content=prepared.prompt)]
        options = CompletionOptions(
            temperature=prepared.temperature,
            reasoning_effort=prepared.reasoning_effort,
            max_tokens=8192,
        )
        response = await provider.complete(messages, model=target_model, options=options)
        return response.content

    def fetcher(prepared: PreparedPrompt) -> Awaitable[str]:
        return fetch_response(prepared)

    # Use py_compile as the validation command
    validation_cmd = f"python -m py_compile {target.path}"

    success = await run_repair_loop(
        base_prompt=optimizer_prompt,
        command=validation_cmd,
        model=target_model,
        attach_recent=False,
        include_diff=True,
        fetch_response=fetcher,
        app_config=config,
        workspace_root=root,
        auto_archive=True,
        max_retries=3,
        keep_failed=False,
    )

    if not success:
        console.print("[red]Optimization failed. Returning to main branch.[/red]")
        await _cleanup_branch(repo, branch_name)
        return

    console.print("[green]Optimization successful![/green]")

    # Step 6: Determine changed files for targeted testing
    match await repo.run_git("diff", "--name-only", "main..HEAD"):
        case Err(err):
            console.print(f"[red]Failed to detect changed files: {err}[/red]")
            await _cleanup_branch(repo, branch_name)
            return
        case Ok(diff_output):
            changed_paths = [line.strip() for line in diff_output.splitlines() if line.strip()]

    python_changes = [Path(root / p).resolve() for p in changed_paths if p.endswith(".py")]
    tests_root = root / "tests"

    test_targets: list[Path] = []
    test_targets.extend([p for p in python_changes if tests_root in p.parents])
    dependent_tests = await _collect_dependent_tests(python_changes, tests_root, root)
    test_targets.extend(dependent_tests)
    if not test_targets and tests_root.exists():
        test_targets.append(tests_root)
    # Preserve order while deduplicating
    test_targets = list(dict.fromkeys(test_targets))

    # Fail fast if pytest is unavailable
    pytest_cmd = "pytest"
    if await asyncio.to_thread(shutil.which, "pytest") is None:
        await _abort_evolution(
            repo,
            branch_name,
            "pytest is not available; aborting evolution.",
            [],
            config,
            reset_hard=True,
        )
        return

    test_args = (
        " ".join(str(path.relative_to(root)) for path in test_targets) if test_targets else ""
    )
    test_command = f"{pytest_cmd} -q {test_args}".strip()
    console.print(f"[cyan]Running verification tests: {test_command}[/cyan]")
    test_result = await run_safe_shell(test_command, root, "evolve.verify", config=config)
    if isinstance(test_result, Err):
        await _abort_evolution(
            repo,
            branch_name,
            f"Test execution failed: {test_result.error}",
            ["evolve", "failure", "tests"],
            config,
            reset_hard=True,
        )
        return

    result_payload = test_result.value
    if result_payload.returncode != 0:
        console.print(result_payload.stdout or result_payload.stderr)
        await _abort_evolution(
            repo,
            branch_name,
            f"Verification failed (exit {result_payload.returncode}).",
            ["evolve", "failure", "tests"],
            config,
            reset_hard=True,
        )
        return

    console.print("[green]Verification tests passed.[/green]")

    # Step 7: Create PR
    await _create_evolution_pr(
        repo,
        target,
        branch_name,
        root,
        config,
        test_command,
        result_payload.returncode,
    )


@app.command("run")
def evolve_run(
    ctx: typer.Context,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Analyze and show target without making changes."
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model to use for optimization."),
    threshold: float = typer.Option(
        10.0, "--threshold", "-t", help="Minimum debt score to trigger optimization."
    ),
) -> None:
    """
    Identify and optimize the highest technical debt file.

    Process:
    1. Analyze all Python files for cyclomatic complexity
    2. Query memory for fix frequency (files with frequent fixes)
    3. Calculate debt score = complexity x (1 + fix_frequency) x log(1 + churn)
    4. Select highest-scoring file above threshold
    5. Create branch, optimize via LLM, create PR

    Use --dry-run to see the analysis without making changes.
    """
    state: AppState = ctx.obj
    asyncio.run(_run_evolve(state.config, dry_run, model, threshold))


@app.command("report")
def evolve_report(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum files to show."),
) -> None:
    """
    Show complexity report for the codebase.

    Displays the most complex files and functions without making changes.
    """
    state: AppState = ctx.obj
    root = Path(state.config.user.workspace_root).expanduser().resolve()

    async def _report() -> None:
        match await analyze_directory_complexity(root, state.config.user.ignore_dirs):
            case Err(complexity_err):
                console.print(f"[red]Analysis failed: {complexity_err}[/red]")
                return
            case Ok(complexities):
                pass

        if not complexities:
            console.print("[yellow]No Python files found.[/yellow]")
            return

        # Sort by max complexity
        sorted_files: list[FileComplexity] = sorted(complexities, key=lambda c: -c.max_cyclomatic)[
            :limit
        ]

        churn_by_path: dict[Path, int] = {fc.path: 0 for fc in sorted_files}
        match await git_core.AsyncRepo.open(root):
            case Ok(repo):
                churn_results = await asyncio.gather(
                    *(repo.get_file_churn(fc.path) for fc in sorted_files)
                )
                for fc, result in zip(sorted_files, churn_results, strict=False):
                    match result:
                        case Ok(value):
                            churn_by_path[fc.path] = value
                        case Err(churn_err):
                            logger.debug("Churn lookup failed for %s: %s", fc.path, churn_err)
                        case _:
                            logger.debug("Unexpected churn result for %s: %s", fc.path, result)
            case Err(repo_err):
                logger.debug("Skipping churn lookup; git repo unavailable: %s", repo_err)

        table = Table(title="Complexity Report", box=box.ROUNDED)
        table.add_column("File", style="cyan")
        table.add_column("Max CC", justify="right", style="yellow")
        table.add_column("Avg CC", justify="right")
        table.add_column("Total CC", justify="right")
        table.add_column("Functions", justify="right")
        table.add_column("Churn", justify="right")

        for fc in sorted_files:
            table.add_row(
                fc.path.name,
                str(fc.max_cyclomatic),
                f"{fc.average_cyclomatic:.1f}",
                str(fc.total_cyclomatic),
                str(len(fc.functions)),
                str(churn_by_path.get(fc.path, 0)),
            )

        console.print(table)

        # Show top functions
        console.print()
        console.print("[bold]Top 10 Most Complex Functions:[/bold]")

        all_functions: list[tuple[Path, FunctionComplexity]] = []
        for fc in complexities:
            for func in fc.functions:
                all_functions.append((fc.path, func))

        all_functions.sort(key=lambda x: -x[1].cyclomatic)

        func_table = Table(box=box.SIMPLE)
        func_table.add_column("Function", style="cyan")
        func_table.add_column("File", style="dim")
        func_table.add_column("CC", justify="right", style="yellow")
        func_table.add_column("Line", justify="right")

        for path, func in all_functions[:10]:
            func_table.add_row(
                func.name,
                path.name,
                str(func.cyclomatic),
                str(func.lineno),
            )

        console.print(func_table)

    asyncio.run(_report())


@app.command("debt")
def evolve_debt(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum files to show."),
) -> None:
    """
    Show technical debt scores combining complexity, fix frequency, and git churn.

    Higher scores indicate files that are both complex AND frequently
    need fixes, making them prime candidates for refactoring.
    """
    state: AppState = ctx.obj
    root = Path(state.config.user.workspace_root).expanduser().resolve()

    async def _debt() -> None:
        console.print("[cyan]Calculating technical debt scores...[/cyan]")
        match await calculate_debt_scores(root, state.config):
            case Err(debt_err):
                console.print(f"[red]Analysis failed: {debt_err}[/red]")
                return
            case Ok(scores):
                pass

        if not scores:
            console.print("[yellow]No files analyzed.[/yellow]")
            return

        table = Table(title="Technical Debt Scores", box=box.ROUNDED)
        table.add_column("Rank", justify="right", style="dim")
        table.add_column("File", style="cyan")
        table.add_column("Complexity", justify="right")
        table.add_column("Fix Freq", justify="right")
        table.add_column("Churn", justify="right")
        table.add_column("Debt Score", justify="right", style="yellow")

        for idx, score in enumerate(scores[:limit], 1):
            table.add_row(
                str(idx),
                score.path.name,
                f"{score.complexity_score:.0f}",
                str(score.fix_frequency),
                str(score.churn),
                f"{score.debt_score:.1f}",
            )

        console.print(table)

        if scores:
            top = scores[0]
            console.print()
            console.print(
                f"[bold]Recommendation:[/bold] Run `jp evolve run` to optimize "
                f"[cyan]{top.path.name}[/cyan] (debt score: {top.debt_score:.1f})"
            )

    asyncio.run(_debt())
