from __future__ import annotations

import asyncio
import json
import shutil
import webbrowser
from pathlib import Path
from typing import TypeVar

import typer
from pydantic import BaseModel
from rich import box
from rich.table import Table

from jpscripts.commands.ui import fzf_select_async
from jpscripts.core import git as git_core
from jpscripts.core import git_ops as git_ops_core
from jpscripts.core import security
from jpscripts.core.console import console
from jpscripts.core.decorators import handle_exceptions
from jpscripts.core.result import Err, GitError, Ok, Result

app = typer.Typer()
T = TypeVar("T")


def _pick_with_fzf(
    lines: list[str], prompt: str, extra_args: list[str] | None = None
) -> str | list[str] | None:
    """Wrapper to run fzf selection without blocking the main thread."""
    return asyncio.run(fzf_select_async(lines, prompt=prompt, extra_args=extra_args))


class PullRequest(BaseModel):
    number: int
    title: str
    headRefName: str
    url: str
    author: dict[str, str]

    @property
    def label(self) -> str:
        return f"#{self.number} {self.title} ([cyan]{self.headRefName}[/])"


@app.callback()
def _git_extra_callback(ctx: typer.Context) -> None:
    """Entry point for git extra commands."""


def _unwrap_result(result: Result[T, GitError]) -> T:
    match result:
        case Ok(value):
            return value
        case Err(err):
            console.print(f"[red]{err.message}[/red]")
            raise typer.Exit(code=1)


def _ensure_repo(path: Path) -> git_core.AsyncRepo:
    repo_path = path.expanduser()
    return _unwrap_result(asyncio.run(git_core.AsyncRepo.open(repo_path)))


async def _run_passthrough_command(*args: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        error_text = stderr.decode("utf-8", errors="replace") or stdout.decode(
            "utf-8", errors="replace"
        )
        raise RuntimeError(error_text or f"{args[0]} command failed")


@handle_exceptions
def gundo_last(
    ctx: typer.Context,
    repo_path: Path = typer.Option(Path("."), "--repo", "-r", help="Repository path."),
    hard: bool = typer.Option(False, "--hard", help="Use hard reset instead of soft."),
) -> None:
    """Safely undo the last commit. Works on local branches too."""
    _ = ctx
    repo_path = repo_path.expanduser()

    repo = _ensure_repo(repo_path)
    message = _unwrap_result(asyncio.run(git_ops_core.undo_last_commit(repo, hard=hard)))

    console.print(f"[green]{message}[/green]")


app.command("gundo-last")(gundo_last)


@handle_exceptions
def gstage(
    ctx: typer.Context,
    repo_path: Path = typer.Option(Path("."), "--repo", "-r", help="Repository path."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Interactively stage files."""
    _ = ctx
    repo = _ensure_repo(repo_path.expanduser())
    status_entries = _unwrap_result(asyncio.run(repo.status_short()))

    if not status_entries:
        console.print("[green]Working tree clean.[/green]")
        return

    entries = status_entries
    use_fzf = shutil.which("fzf") and not no_fzf
    selection: str | None = None
    if use_fzf:
        lines = [f"{code}\t{path}" for code, path in entries]
        fzf_selection = _pick_with_fzf(lines, prompt="stage> ")
        selection = fzf_selection if isinstance(fzf_selection, str) else None
    else:
        table = Table(title="Changes", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("Status", style="cyan", no_wrap=True)
        table.add_column("Path", style="white")
        for code, path in entries:
            table.add_row(code, path)
        console.print(table)
        selection = entries[0][1]

    if not selection:
        return

    target_str = selection.split("\t", 1)[-1] if "\t" in selection else selection
    target_path_str = target_str.split(" -> ", 1)[-1]
    target_path = security.validate_path(repo.path / target_path_str, repo.path)

    _unwrap_result(asyncio.run(repo.add(paths=[target_path])))
    console.print(f"[green]Staged[/green] {target_path_str}")


app.command("gstage")(gstage)


@handle_exceptions
async def gpr(
    ctx: typer.Context,
    action: str = typer.Option("view", "--action", "-a", help="view, checkout, or copy"),
    limit: int = typer.Option(30, "--limit", help="Max PRs to list."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Interact with GitHub PRs via gh (Typed & Robust)."""
    _ = ctx
    try:
        prs = await _get_prs(limit)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)

    if not prs:
        console.print("[yellow]No open PRs.[/yellow]")
        return

    use_fzf = shutil.which("fzf") and not no_fzf

    if use_fzf:
        # We pass the lookup key (number) as the prefix
        lines = [f"{pr.number}\t{pr.title} ({pr.headRefName})" for pr in prs]
        selection = _pick_with_fzf(lines, prompt="pr> ")
        if not selection or not isinstance(selection, str):
            return
        number = int(selection.split("\t")[0])
    else:
        # Fallback table
        table = Table(title="Open PRs", box=box.SIMPLE_HEAVY)
        table.add_column("#", style="cyan")
        table.add_column("Title", style="white")
        table.add_column("Branch", style="dim")
        for pr in prs[:15]:
            table.add_row(str(pr.number), pr.title, pr.headRefName)
        console.print(table)
        # Simple selector logic could go here, or just exit
        return

    # Action dispatch
    if action == "checkout":
        await _run_passthrough_command("gh", "pr", "checkout", str(number))
    elif action == "view":
        await _run_passthrough_command("gh", "pr", "view", str(number), "--web")
    elif action == "copy":
        # Find the PR object to get the URL directly without another shell call
        target_pr = next((p for p in prs if p.number == number), None)
        if target_pr:
            import pyperclip

            pyperclip.copy(target_pr.url)
            console.print(f"[green]Copied[/green] {target_pr.url}")
    else:
        console.print(f"[red]Unknown action: {action}[/red]")


app.command("gpr")(gpr)


async def _get_prs(limit: int) -> list[PullRequest]:
    if not shutil.which("gh"):
        raise RuntimeError("GitHub CLI (gh) is required.")

    proc = await asyncio.create_subprocess_exec(
        "gh",
        "pr",
        "list",
        f"--limit={limit}",
        "--json",
        "number,title,headRefName,author,url",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        error_text = stderr.decode("utf-8", errors="replace") or stdout.decode(
            "utf-8", errors="replace"
        )
        raise RuntimeError(f"gh failed: {error_text}")

    data = json.loads(stdout.decode("utf-8"))
    return [PullRequest(**item) for item in data]


def _repo_web_url(remote_url: str) -> str | None:
    if remote_url.startswith("git@"):
        # git@github.com:user/repo.git -> https://github.com/user/repo
        _, rest = remote_url.split(":", 1)
        rest = rest.replace(".git", "")
        return f"https://github.com/{rest}"
    if remote_url.startswith("https://"):
        return remote_url.replace(".git", "")
    return None


@handle_exceptions
def gbrowse(
    ctx: typer.Context,
    repo_path: Path = typer.Option(Path("."), "--repo", "-r", help="Repository path."),
    target: str = typer.Option("branch", "--target", help="branch (default), commit, or repo"),
) -> None:
    """Open the current repo/branch/commit on GitHub."""
    repo = _ensure_repo(repo_path.expanduser())
    remote_url = _unwrap_result(asyncio.run(repo.get_remote_url()))

    base_url = _repo_web_url(remote_url)
    if not base_url:
        console.print("[red]Could not determine remote URL for browsing.[/red]")
        raise typer.Exit(code=1)

    if target == "repo":
        url = base_url
    elif target == "commit":
        commit_sha = _unwrap_result(asyncio.run(repo.head(short=False)))
        url = f"{base_url}/commit/{commit_sha}"
    else:
        status = _unwrap_result(asyncio.run(repo.status()))
        branch = status.branch
        if branch in {"(detached)", "(unknown)"}:
            branch = _unwrap_result(asyncio.run(repo.head()))
        url = f"{base_url}/tree/{branch}"

    webbrowser.open(url)
    console.print(f"[green]Opened[/green] {url}")


@handle_exceptions
def git_branchcheck(
    ctx: typer.Context,
    repo_path: Path = typer.Option(Path("."), "--repo", "-r", help="Repository path."),
) -> None:
    """List branches with upstream and ahead/behind counts."""
    repo_path = repo_path.expanduser()

    async def _collect() -> Result[list[git_ops_core.BranchSummary], GitError]:
        match await git_core.AsyncRepo.open(repo_path):
            case Err(err):
                return Err(err)
            case Ok(repo):
                return await git_ops_core.branch_statuses(repo)

    match asyncio.run(_collect()):
        case Err(err):
            console.print(f"[red]{err.message}[/red]")
            raise typer.Exit(code=1)
        case Ok(summaries):
            pass
    table = Table(title="Branches", box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Branch", style="cyan", no_wrap=True)
    table.add_column("Upstream", style="white", no_wrap=True)
    table.add_column("Ahead/Behind", style="white", no_wrap=True)

    for summary in summaries:
        upstream = summary.upstream or "none"
        ahead_behind = f"{summary.ahead}/{summary.behind}"
        if summary.error:
            table.add_row(summary.name, "error", summary.error, style="red")
        else:
            table.add_row(summary.name, upstream, ahead_behind)

    console.print(table)


def stashview(
    ctx: typer.Context,
    repo_path: Path = typer.Option(Path("."), "--repo", "-r", help="Repository path."),
    action: str = typer.Option("apply", "--action", "-a", help="apply (default), pop, or drop."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Browse stash entries and apply/pop/drop one."""
    repo = _ensure_repo(repo_path.expanduser())
    stash_list = _unwrap_result(asyncio.run(repo.stash_list()))

    if not stash_list:
        console.print("[yellow]No stash entries.[/yellow]")
        return

    use_fzf = shutil.which("fzf") and not no_fzf
    selection = _pick_with_fzf(stash_list, prompt="stash> ") if use_fzf else stash_list[0]
    selection_str = selection if isinstance(selection, str) else None
    if not selection_str:
        return

    ref = selection_str.split(":", 1)[0]
    if action == "apply":
        op = repo.stash_apply
    elif action == "pop":
        op = repo.stash_pop
    elif action == "drop":
        op = repo.stash_drop
    else:
        console.print("[red]Unknown action. Use apply, pop, or drop.[/red]")
        return

    _unwrap_result(asyncio.run(op(ref)))
    console.print(f"[green]{action}[/green] {ref}")
