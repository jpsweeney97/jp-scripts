from __future__ import annotations

import subprocess
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import typer
from rich import box
from rich.panel import Panel
from rich.table import Table

from jpscripts.core.console import console
from jpscripts.core import git as git_core


def _ensure_repo(path: Path) -> git_core.Repo:
    try:
        return git_core.open_repo(path)
    except Exception as exc:
        console.print(f"[red]Failed to open repo at {path}: {exc}[/red]")
        raise typer.Exit(code=1)


def _run_fzf(lines: list[str], prompt: str) -> str | None:
    proc = subprocess.run(["fzf", "--prompt", prompt], input="\n".join(lines), text=True, capture_output=True)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def gundo_last(
    ctx: typer.Context,
    repo_path: Path = typer.Option(Path("."), "--repo", "-r", help="Repository path."),
    hard: bool = typer.Option(False, "--hard", help="Use hard reset instead of soft."),
) -> None:
    """Safely undo the last commit if not behind upstream."""
    repo_path = repo_path.expanduser()
    repo = _ensure_repo(repo_path)
    status = git_core.describe_status(repo)

    if status.behind > 0:
        console.print("[red]Refusing to undo: branch is behind upstream. Pull first.[/red]")
        raise typer.Exit(code=1)
    if status.ahead == 0:
        console.print("[yellow]No commits ahead of upstream; nothing to undo.[/yellow]")
        return

    mode = "--hard" if hard else "--soft"
    repo.git.reset(mode, "HEAD~1")
    console.print(f"[green]Reset {status.branch} one commit back ({mode}).[/green]")


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

    use_fzf = subprocess.which("fzf") and not no_fzf
    selection = None
    if use_fzf:
        lines = [f"{code}\t{path}" for code, path in entries]
        selection = _run_fzf(lines, prompt="stage> ")
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
    action: str = typer.Option("view", "--action", "-a", help="view (browser), checkout, or copy"),
    limit: int = typer.Option(30, "--limit", help="Max PRs to list."),
    no_fzf: bool = typer.Option(False, "--no-fzf", help="Disable fzf even if available."),
) -> None:
    """Interact with GitHub PRs via gh."""
    if not subprocess.which("gh"):
        console.print("[red]GitHub CLI (gh) is required for gpr.[/red]")
        raise typer.Exit(code=1)

    list_proc = subprocess.run(
        ["gh", "pr", "list", f"--limit={limit}", "--json", "number,title,headRefName,author,url"],
        capture_output=True,
        text=True,
    )
    if list_proc.returncode != 0:
        console.print(f"[red]gh pr list failed:[/red] {list_proc.stderr}")
        raise typer.Exit(code=1)

    import json

    prs = json.loads(list_proc.stdout)
    if not prs:
        console.print("[yellow]No open PRs.[/yellow]")
        return

    lines = [f"{pr['number']}\t{pr['title']} ({pr['headRefName']})" for pr in prs]
    use_fzf = subprocess.which("fzf") and not no_fzf
    selection = _run_fzf(lines, prompt="pr> ") if use_fzf else lines[0]
    if not selection:
        return

    number = selection.split("\t", 1)[0]

    if action == "checkout":
        subprocess.run(["gh", "pr", "checkout", number])
        return
    if action == "view":
        subprocess.run(["gh", "pr", "view", number, "--web"])
        return
    if action == "copy":
        subprocess.run(["gh", "pr", "view", number, "--json", "url", "--jq", ".url"])
        return

    console.print("[red]Unknown action. Use view, checkout, or copy.[/red]")


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
    table = Table(title="Branches", box=box.SIMPLE_HEAVY, expand=True)
    table.add_column("Branch", style="cyan", no_wrap=True)
    table.add_column("Upstream", style="white", no_wrap=True)
    table.add_column("Ahead/Behind", style="white", no_wrap=True)

    for branch in repo.branches:
        upstream = None
        ahead = behind = 0
        try:
            tracking = branch.tracking_branch()
            upstream = str(tracking) if tracking else None
            if tracking:
                ahead = sum(1 for _ in repo.iter_commits(f"{tracking}..{branch.name}"))
                behind = sum(1 for _ in repo.iter_commits(f"{branch.name}..{tracking}"))
            table.add_row(branch.name, upstream or "none", f"{ahead}/{behind}")
        except Exception as exc:
            table.add_row(branch.name, "error", str(exc), style="red")

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

    use_fzf = subprocess.which("fzf") and not no_fzf
    selection = _run_fzf(stash_list, prompt="stash> ") if use_fzf else stash_list[0]
    if not selection:
        return

    ref = selection.split(":", 1)[0]
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
