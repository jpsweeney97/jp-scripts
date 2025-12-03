"""
Proactive evolution command for autonomous codebase improvement.

The `jp evolve` command identifies files with high technical debt
(complexity x fix frequency) and autonomously refactors them,
creating a PR for review.

Usage:
    jp evolve run --dry-run          # Analyze and show target without changes
    jp evolve run --threshold 15     # Only act on debt score > 15
    jp evolve report                 # Show complexity report
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

from jpscripts.analysis.complexity import (
    FileComplexity,
    FunctionComplexity,
    TechnicalDebtScore,
    analyze_directory_complexity,
    calculate_debt_scores,
)
from jpscripts.core.console import console, get_logger
from jpscripts.core.evolution import run_evolution
from jpscripts.core.result import Err, Ok
from jpscripts.git import client as git_core
from jpscripts.main import AppState

logger = get_logger(__name__)

app = typer.Typer(help="Autonomous code evolution and optimization.")


def _display_debt_table(scores: list[TechnicalDebtScore], limit: int = 10) -> None:
    """Display technical debt analysis table."""
    table = Table(title="Technical Debt Analysis", box=box.ROUNDED)
    table.add_column("File", style="cyan")
    table.add_column("Complexity", justify="right")
    table.add_column("Fix Freq", justify="right")
    table.add_column("Debt Score", justify="right", style="yellow")

    for score in scores[:limit]:
        table.add_row(
            score.path.name,
            f"{score.complexity_score:.0f}",
            str(score.fix_frequency),
            f"{score.debt_score:.1f}",
        )

    console.print(table)


def _display_target_panel(target: TechnicalDebtScore) -> None:
    """Display selected target panel."""
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
    root = Path(state.config.user.workspace_root).expanduser().resolve()

    async def _run() -> None:
        # Show analysis first for user feedback
        console.print("[cyan]Checking git state...[/cyan]")

        # For dry run, show debt analysis before calling run_evolution
        if dry_run:
            console.print("[cyan]Analyzing complexity and fix history...[/cyan]")
            match await calculate_debt_scores(root, state.config):
                case Err(debt_err):
                    console.print(f"[red]Analysis failed: {debt_err}[/red]")
                    return
                case Ok(scores):
                    if not scores:
                        console.print("[green]No files require optimization.[/green]")
                        return

            _display_debt_table(scores)

            target = scores[0]
            if target.debt_score < threshold:
                console.print(
                    f"[green]Top debt score {target.debt_score:.1f} is below "
                    f"threshold {threshold}. No optimization needed.[/green]"
                )
                return

            console.print()
            _display_target_panel(target)
            console.print("\n[yellow]Dry run mode - no changes will be made.[/yellow]")
            return

        # Non-dry-run: show progress and run evolution
        console.print("[cyan]Analyzing complexity and fix history...[/cyan]")
        match await calculate_debt_scores(root, state.config):
            case Err(debt_err):
                console.print(f"[red]Analysis failed: {debt_err}[/red]")
                return
            case Ok(scores):
                if not scores:
                    console.print("[green]No files require optimization.[/green]")
                    return

        _display_debt_table(scores)

        target = scores[0]
        if target.debt_score < threshold:
            console.print(
                f"[green]Top debt score {target.debt_score:.1f} is below "
                f"threshold {threshold}. No optimization needed.[/green]"
            )
            return

        console.print()
        _display_target_panel(target)

        console.print(f"\n[cyan]Creating branch: evolve/{target.path.stem}-optimization[/cyan]")
        console.print("[cyan]Launching optimizer agent...[/cyan]")

        match await run_evolution(state.config, dry_run, model, threshold):
            case Err(error):
                console.print(f"[red]{error}[/red]")
            case Ok(None):
                pass  # Already displayed appropriate message above
            case Ok(result) if result is not None:
                console.print("[green]Optimization successful![/green]")
                console.print("[green]Verification tests passed.[/green]")
                console.print("[cyan]Pushing branch and creating PR...[/cyan]")
                if result.pr_url:
                    console.print(f"[green]PR created:[/green] {result.pr_url}")
                else:
                    console.print("[yellow]PR creation failed. Create manually.[/yellow]")
                    console.print(f"[dim]Branch: {result.branch_name}[/dim]")

    asyncio.run(_run())


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
