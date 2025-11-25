from __future__ import annotations

import json
import shutil
import subprocess
import webbrowser
from pathlib import Path

import typer
from pydantic import BaseModel
from rich import box
from rich.table import Table

from jpscripts.core import git as git_core
from jpscripts.core import git_ops as git_ops_core
from jpscripts.core.console import console
from jpscripts.commands.ui import fzf_select

app = typer.Typer()

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


def _ensure_repo(path: Path) -> git_core.Repo:
    try:
        return git_core.open_repo(path)
    except Exception as exc:
        console.print(f"[red]Failed to open repo at {path}: {exc}[/red]")
        raise typer.Exit(code=1)


def gundo_last(
    ctx: typer.Context,
    repo_path: Path = typer.Option(Path("."), "--repo", "-r", help="Repository path."),
    hard: bool = typer.Option(False, "--hard", help="Use hard reset instead of soft."),
) -> None:
    """Safely undo the last commit. Works on local branches too."""
    repo_path = repo_path.expanduser()
    repo = _ensure_repo(repo_path)

    try:
        message = git_ops_core.undo_last_commit(repo, hard=hard)
    except git_ops_core.GitOperationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]{message}[/green]")


app.command("gundo-last")(gundo_last)

def gstage(
    ctx: typer.Context,
    repo_path: Path = typer.Option(Path("."), "--repo", "-r", help="Repository path."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Interactively stage files."""
    repo = _ensure_repo(repo_path.expanduser())
    status_lines = repo.git.status("--porcelain").splitlines()
    if not status_lines:
        console.print("[green]Working tree clean.[/green]")
        return

    entries = []
    for line in status_lines:
        status_code, path = line[:2].strip(), line[3:]
        entries.append((status_code, path))

    use_fzf = shutil.which("fzf") and not no_fzf
    selection: str | None = None
    if use_fzf:
        lines = [f"{code}\t{path}" for code, path in entries]
        fzf_selection = fzf_select(lines, prompt="stage> ")
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

    target_path = selection.split("\t", 1)[-1] if "\t" in selection else selection
    repo.git.add(target_path)
    console.print(f"[green]Staged[/green] {target_path}")


def gpr(
    ctx: typer.Context,
    action: str = typer.Option("view", "--action", "-a", help="view, checkout, or copy"),
    limit: int = typer.Option(30, "--limit", help="Max PRs to list."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Interact with GitHub PRs via gh (Typed & Robust)."""
    try:
        prs = _get_prs(limit)
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
        selection = fzf_select(lines, prompt="pr> ")
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
        subprocess.run(["gh", "pr", "checkout", str(number)])
    elif action == "view":
        subprocess.run(["gh", "pr", "view", str(number), "--web"])
    elif action == "copy":
        # Find the PR object to get the URL directly without another shell call
        target_pr = next((p for p in prs if p.number == number), None)
        if target_pr:
            import pyperclip
            pyperclip.copy(target_pr.url)
            console.print(f"[green]Copied[/green] {target_pr.url}")
    else:
        console.print(f"[red]Unknown action: {action}[/red]")

def _get_prs(limit: int) -> list[PullRequest]:
    if not shutil.which("gh"):
        raise RuntimeError("GitHub CLI (gh) is required.")

    proc = subprocess.run(
        ["gh", "pr", "list", f"--limit={limit}", "--json", "number,title,headRefName,author,url"],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"gh failed: {proc.stderr}")

    data = json.loads(proc.stdout)
    return [PullRequest(**item) for item in data]


def _repo_web_url(repo: git_core.Repo) -> str | None:
    try:
        origin = repo.remotes.origin.url
    except Exception:
        return None

    if origin.startswith("git@"):
        # git@github.com:user/repo.git -> https://github.com/user/repo
        _, rest = origin.split(":", 1)
        rest = rest.replace(".git", "")
        return f"https://github.com/{rest}"
    if origin.startswith("https://"):
        return origin.replace(".git", "")
    return None


def gbrowse(
    ctx: typer.Context,
    repo_path: Path = typer.Option(Path("."), "--repo", "-r", help="Repository path."),
    target: str = typer.Option("branch", "--target", help="branch (default), commit, or repo"),
) -> None:
    """Open the current repo/branch/commit on GitHub."""
    repo = _ensure_repo(repo_path.expanduser())
    base_url = _repo_web_url(repo)
    if not base_url:
        console.print("[red]Could not determine remote URL for browsing.[/red]")
        raise typer.Exit(code=1)

    if target == "repo":
        url = base_url
    elif target == "commit":
        url = f"{base_url}/commit/{repo.head.commit.hexsha}"
    else:
        branch = repo.active_branch.name if not repo.head.is_detached else repo.git.rev_parse("--short", "HEAD")
        url = f"{base_url}/tree/{branch}"

    webbrowser.open(url)
    console.print(f"[green]Opened[/green] {url}")


def git_branchcheck(
    ctx: typer.Context,
    repo_path: Path = typer.Option(Path("."), "--repo", "-r", help="Repository path."),
) -> None:
    """List branches with upstream and ahead/behind counts."""
    repo = _ensure_repo(repo_path.expanduser())
    summaries = git_ops_core.branch_statuses(repo)
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
    stash_list = repo.git.stash("list").splitlines()
    if not stash_list:
        console.print("[yellow]No stash entries.[/yellow]")
        return

    use_fzf = shutil.which("fzf") and not no_fzf
    selection = fzf_select(stash_list, prompt="stash> ") if use_fzf else stash_list[0]
    selection_str = selection if isinstance(selection, str) else None
    if not selection_str:
        return

    ref = selection_str.split(":", 1)[0]
    if action == "apply":
        repo.git.stash("apply", ref)
    elif action == "pop":
        repo.git.stash("pop", ref)
    elif action == "drop":
        repo.git.stash("drop", ref)
    else:
        console.print("[red]Unknown action. Use apply, pop, or drop.[/red]")
        return

    console.print(f"[green]{action}[/green] {ref}")
