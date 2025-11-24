from __future__ import annotations

from git import Repo

from jpscripts.commands import git_extra
from jpscripts.core import git as git_core

def test_gundo_last_local_only(runner, tmp_path, monkeypatch):
    """Verify undo works on a local branch with no upstream (the fix)."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = Repo.init(repo_dir)

    # Create commit 1
    (repo_dir / "file.txt").write_text("v1")
    repo.index.add(["file.txt"])
    repo.index.commit("commit 1")

    # Create commit 2 (the mistake)
    (repo_dir / "file.txt").write_text("v2")
    repo.index.add(["file.txt"])
    repo.index.commit("commit 2")

    assert repo.head.commit.message.strip() == "commit 2"

    # Run gundo-last
    # We must patch _ensure_repo because typer argument parsing of Path objects
    # behaves differently in tests vs CLI invocation sometimes.
    monkeypatch.setattr(git_extra, "_ensure_repo", lambda p: git_core.open_repo(repo_dir))

    result = runner.invoke(git_extra.app, ["gundo-last", "--repo", str(repo_dir)])

    # Assertions
    # If the bug was still present, this would say "No commits ahead... nothing to undo"
    assert "Reset master one commit back" in result.stdout or "Reset main one commit back" in result.stdout
    assert repo.head.commit.message.strip() == "commit 1"
