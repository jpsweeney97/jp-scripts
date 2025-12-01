"""Tests for notes commands module."""

from __future__ import annotations

import asyncio
import datetime as dt
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer

from jpscripts.commands.notes import (
    CLIPHIST_DB,
    CLIPHIST_DIR,
    RepoSummary,
    _collect_repo_commits,
    _detect_user_email,
    _init_db,
    _launch_editor,
    _migrate_legacy_history,
    cliphist,
    note,
    note_search,
    standup,
    standup_note,
)
from jpscripts.core.config import AppConfig
from jpscripts.core.result import Err, Ok
from jpscripts.git.client import GitCommit


@pytest.fixture
def test_config(tmp_path: Path) -> AppConfig:
    """Create a test configuration."""
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    return AppConfig(
        workspace_root=tmp_path,
        notes_dir=notes_dir,
        editor="nano",
        ignore_dirs=[".git"],
        max_file_context_chars=50_000,
        max_command_output_chars=20_000,
        default_model="gpt-4o-mini",
        model_context_limits={"default": 50_000},
        use_semantic_search=False,
    )


@pytest.fixture
def mock_state(test_config: AppConfig) -> MagicMock:
    """Create mock state with test config."""
    state = MagicMock()
    state.config = test_config
    return state


@pytest.fixture
def mock_ctx(mock_state: MagicMock) -> MagicMock:
    """Create mock typer context."""
    ctx = MagicMock()
    ctx.obj = mock_state
    return ctx


class TestNote:
    """Tests for the note command."""

    def test_note_with_message(self, mock_ctx: MagicMock, test_config: AppConfig) -> None:
        """Note with message appends to daily note."""
        with patch(
            "jpscripts.commands.notes.notes_impl.append_to_daily_note",
            new_callable=AsyncMock,
        ) as mock_append:
            note(mock_ctx, message="Test message")
            mock_append.assert_called_once()

    def test_note_without_message_opens_editor(
        self, mock_ctx: MagicMock, test_config: AppConfig
    ) -> None:
        """Note without message opens editor."""
        with patch(
            "jpscripts.commands.notes._launch_editor",
            new_callable=AsyncMock,
            return_value=0,
        ) as mock_editor:
            note(mock_ctx, message="")
            mock_editor.assert_called_once()

    def test_note_editor_failure(self, mock_ctx: MagicMock, test_config: AppConfig) -> None:
        """Editor returning non-zero shows error."""
        with patch(
            "jpscripts.commands.notes._launch_editor",
            new_callable=AsyncMock,
            return_value=1,
        ):
            # Should not raise, just print error
            note(mock_ctx, message="")

    def test_note_editor_not_found(self, mock_ctx: MagicMock, test_config: AppConfig) -> None:
        """Missing editor raises Exit."""
        with (
            patch(
                "jpscripts.commands.notes._launch_editor",
                new_callable=AsyncMock,
                side_effect=FileNotFoundError(),
            ),
            pytest.raises(typer.Exit),
        ):
            note(mock_ctx, message="")


class TestNoteSearch:
    """Tests for the note_search command."""

    def test_notes_dir_not_exists(
        self, mock_ctx: MagicMock, test_config: AppConfig, tmp_path: Path
    ) -> None:
        """Non-existent notes dir raises Exit."""
        mock_ctx.obj.config.notes_dir = tmp_path / "nonexistent"

        with pytest.raises(typer.Exit):
            note_search(mock_ctx, query="test", no_fzf=True)

    def test_search_without_fzf(self, mock_ctx: MagicMock, test_config: AppConfig) -> None:
        """Search without fzf runs ripgrep directly."""
        with (
            patch("shutil.which", return_value=None),
            patch(
                "jpscripts.commands.notes.search_core.run_ripgrep",
                return_value="match found",
            ),
        ):
            note_search(mock_ctx, query="test", no_fzf=True)

    def test_search_no_matches(self, mock_ctx: MagicMock, test_config: AppConfig) -> None:
        """Search with no matches shows message."""
        with (
            patch("shutil.which", return_value=None),
            patch(
                "jpscripts.commands.notes.search_core.run_ripgrep",
                return_value="",
            ),
        ):
            note_search(mock_ctx, query="nomatch", no_fzf=True)

    def test_search_ripgrep_error(self, mock_ctx: MagicMock, test_config: AppConfig) -> None:
        """Ripgrep error raises Exit."""
        with (
            patch("shutil.which", return_value=None),
            patch(
                "jpscripts.commands.notes.search_core.run_ripgrep",
                side_effect=RuntimeError("rg failed"),
            ),
            pytest.raises(typer.Exit),
        ):
            note_search(mock_ctx, query="test", no_fzf=True)


class TestCollectRepoCommits:
    """Tests for _collect_repo_commits helper."""

    @pytest.mark.asyncio
    async def test_git_open_error(self, tmp_path: Path) -> None:
        """Git open error returns summary with error."""
        since = dt.datetime.now() - dt.timedelta(days=1)

        with patch(
            "jpscripts.commands.notes.git_core.AsyncRepo.open",
            return_value=Err(MagicMock(message="Not a git repo")),
        ):
            result = await _collect_repo_commits(tmp_path, since, None, 100)

        assert result.error == "Not a git repo"
        assert result.commits == []

    @pytest.mark.asyncio
    async def test_get_commits_error(self, tmp_path: Path) -> None:
        """Get commits error returns summary with error."""
        since = dt.datetime.now() - dt.timedelta(days=1)

        mock_repo = AsyncMock()
        mock_repo.get_commits.return_value = Err(MagicMock(message="Commit error"))

        with patch(
            "jpscripts.commands.notes.git_core.AsyncRepo.open",
            return_value=Ok(mock_repo),
        ):
            result = await _collect_repo_commits(tmp_path, since, None, 100)

        assert result.error == "Commit error"
        assert result.commits == []

    @pytest.mark.asyncio
    async def test_filters_by_date_and_author(self, tmp_path: Path) -> None:
        """Commits are filtered by date and author."""
        since = dt.datetime.now() - dt.timedelta(days=1)
        cutoff = int(since.timestamp())

        old_commit = GitCommit(
            hexsha="abc123",
            summary="Old commit",
            author_name="Author",
            author_email="author@example.com",
            committed_date=cutoff - 3600,  # 1 hour before cutoff
        )
        new_commit = GitCommit(
            hexsha="def456",
            summary="New commit",
            author_name="Author",
            author_email="author@example.com",
            committed_date=cutoff + 3600,  # 1 hour after cutoff
        )
        other_author_commit = GitCommit(
            hexsha="ghi789",
            summary="Other author",
            author_name="Other",
            author_email="other@example.com",
            committed_date=cutoff + 3600,
        )

        mock_repo = AsyncMock()
        mock_repo.get_commits.return_value = Ok([old_commit, new_commit, other_author_commit])

        with patch(
            "jpscripts.commands.notes.git_core.AsyncRepo.open",
            return_value=Ok(mock_repo),
        ):
            result = await _collect_repo_commits(tmp_path, since, "author@example.com", 100)

        # Only new_commit should be included (recent + matching author)
        assert len(result.commits) == 1
        assert result.commits[0].hexsha == "def456"


class TestDetectUserEmail:
    """Tests for _detect_user_email helper."""

    @pytest.mark.asyncio
    async def test_git_open_error_returns_none(self, tmp_path: Path) -> None:
        """Git open error returns None."""
        with patch(
            "jpscripts.commands.notes.git_core.AsyncRepo.open",
            return_value=Err(MagicMock()),
        ):
            result = await _detect_user_email(tmp_path)
        assert result is None

    @pytest.mark.asyncio
    async def test_config_error_returns_none(self, tmp_path: Path) -> None:
        """Git config error returns None."""
        mock_repo = AsyncMock()
        mock_repo.run_git.return_value = Err(MagicMock())

        with patch(
            "jpscripts.commands.notes.git_core.AsyncRepo.open",
            return_value=Ok(mock_repo),
        ):
            result = await _detect_user_email(tmp_path)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_email(self, tmp_path: Path) -> None:
        """Returns email when found."""
        mock_repo = AsyncMock()
        mock_repo.run_git.return_value = Ok("test@example.com\n")

        with patch(
            "jpscripts.commands.notes.git_core.AsyncRepo.open",
            return_value=Ok(mock_repo),
        ):
            result = await _detect_user_email(tmp_path)
        assert result == "test@example.com"

    @pytest.mark.asyncio
    async def test_empty_email_returns_none(self, tmp_path: Path) -> None:
        """Empty email string returns None."""
        mock_repo = AsyncMock()
        mock_repo.run_git.return_value = Ok("")

        with patch(
            "jpscripts.commands.notes.git_core.AsyncRepo.open",
            return_value=Ok(mock_repo),
        ):
            result = await _detect_user_email(tmp_path)
        assert result is None


class TestStandup:
    """Tests for the standup command."""

    def test_no_repos_found(self, mock_ctx: MagicMock, test_config: AppConfig) -> None:
        """No repos shows message."""
        with patch(
            "jpscripts.commands.notes.git_core.iter_git_repos",
            new_callable=AsyncMock,
            return_value=Ok([]),
        ):
            standup(mock_ctx, days=3, max_depth=2)

    def test_iter_repos_error(self, mock_ctx: MagicMock, test_config: AppConfig) -> None:
        """Repo iteration error raises Exit."""
        with (
            patch(
                "jpscripts.commands.notes.git_core.iter_git_repos",
                new_callable=AsyncMock,
                return_value=Err(MagicMock(message="Scan error")),
            ),
            pytest.raises(typer.Exit),
        ):
            standup(mock_ctx, days=3, max_depth=2)


class TestStandupNote:
    """Tests for the standup_note command."""

    def test_appends_standup_to_note(self, mock_ctx: MagicMock, test_config: AppConfig) -> None:
        """Standup output is appended to note."""
        with (
            patch(
                "jpscripts.commands.notes.git_core.iter_git_repos",
                new_callable=AsyncMock,
                return_value=Ok([]),
            ),
        ):
            # Should complete without error even with no standup content
            standup_note(mock_ctx, days=3)

    def test_no_standup_output(self, mock_ctx: MagicMock, test_config: AppConfig) -> None:
        """No standup output shows message."""
        with (
            patch(
                "jpscripts.commands.notes.git_core.iter_git_repos",
                new_callable=AsyncMock,
                return_value=Ok([]),
            ),
        ):
            standup_note(mock_ctx, days=3)


class TestInitDb:
    """Tests for _init_db function."""

    def test_creates_database(self, tmp_path: Path) -> None:
        """Creates database and table."""
        with (
            patch("jpscripts.commands.notes.CLIPHIST_DIR", tmp_path),
            patch("jpscripts.commands.notes.CLIPHIST_DB", tmp_path / "history.db"),
            patch("jpscripts.commands.notes.CLIPHIST_FILE", tmp_path / "history.txt"),
        ):
            conn = _init_db()
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='history'"
            )
            assert cursor.fetchone() is not None
            conn.close()


class TestMigrateLegacyHistory:
    """Tests for _migrate_legacy_history function."""

    def test_no_legacy_file(self, tmp_path: Path) -> None:
        """No legacy file does nothing."""
        db_path = tmp_path / "history.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """
        )

        with patch("jpscripts.commands.notes.CLIPHIST_FILE", tmp_path / "nonexistent.txt"):
            _migrate_legacy_history(conn)

        # No rows should be added
        cursor = conn.execute("SELECT COUNT(*) FROM history")
        assert cursor.fetchone()[0] == 0
        conn.close()

    def test_existing_rows_skips_migration(self, tmp_path: Path) -> None:
        """Existing rows prevents migration."""
        db_path = tmp_path / "history.db"
        legacy_file = tmp_path / "history.txt"
        legacy_file.write_text("2025-01-01T00:00:00\tOld entry\n")

        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO history (timestamp, content) VALUES (?, ?)",
            ("2025-01-02", "Existing"),
        )

        with patch("jpscripts.commands.notes.CLIPHIST_FILE", legacy_file):
            _migrate_legacy_history(conn)

        # Only the existing row should remain
        cursor = conn.execute("SELECT COUNT(*) FROM history")
        assert cursor.fetchone()[0] == 1
        conn.close()


class TestCliphist:
    """Tests for the cliphist command."""

    def test_add_empty_clipboard(self, mock_ctx: MagicMock, tmp_path: Path) -> None:
        """Add with empty clipboard shows message."""
        with (
            patch("jpscripts.commands.notes.CLIPHIST_DIR", tmp_path),
            patch("jpscripts.commands.notes.CLIPHIST_DB", tmp_path / "history.db"),
            patch("jpscripts.commands.notes.CLIPHIST_FILE", tmp_path / "history.txt"),
            patch("pyperclip.paste", return_value=""),
        ):
            cliphist(mock_ctx, action="add", limit=50, no_fzf=True)

    def test_add_saves_clipboard(self, mock_ctx: MagicMock, tmp_path: Path) -> None:
        """Add saves clipboard content."""
        with (
            patch("jpscripts.commands.notes.CLIPHIST_DIR", tmp_path),
            patch("jpscripts.commands.notes.CLIPHIST_DB", tmp_path / "history.db"),
            patch("jpscripts.commands.notes.CLIPHIST_FILE", tmp_path / "history.txt"),
            patch("pyperclip.paste", return_value="Test content"),
        ):
            cliphist(mock_ctx, action="add", limit=50, no_fzf=True)

        # Verify it was saved
        conn = sqlite3.connect(tmp_path / "history.db")
        cursor = conn.execute("SELECT content FROM history")
        assert cursor.fetchone()[0] == "Test content"
        conn.close()

    def test_show_no_history(self, mock_ctx: MagicMock, tmp_path: Path) -> None:
        """Show with no history shows message."""
        with (
            patch("jpscripts.commands.notes.CLIPHIST_DIR", tmp_path),
            patch("jpscripts.commands.notes.CLIPHIST_DB", tmp_path / "history.db"),
            patch("jpscripts.commands.notes.CLIPHIST_FILE", tmp_path / "history.txt"),
        ):
            cliphist(mock_ctx, action="show", limit=50, no_fzf=True)

    def test_show_displays_history(self, mock_ctx: MagicMock, tmp_path: Path) -> None:
        """Show displays history table."""
        # Pre-populate database
        db_path = tmp_path / "history.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO history (timestamp, content) VALUES (?, ?)",
            ("2025-01-01T00:00:00", "Test entry"),
        )
        conn.commit()
        conn.close()

        with (
            patch("jpscripts.commands.notes.CLIPHIST_DIR", tmp_path),
            patch("jpscripts.commands.notes.CLIPHIST_DB", db_path),
            patch("jpscripts.commands.notes.CLIPHIST_FILE", tmp_path / "history.txt"),
        ):
            cliphist(mock_ctx, action="show", limit=50, no_fzf=True)

    def test_pick_without_fzf(self, mock_ctx: MagicMock, tmp_path: Path) -> None:
        """Pick without fzf uses first entry."""
        # Pre-populate database
        db_path = tmp_path / "history.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO history (timestamp, content) VALUES (?, ?)",
            ("2025-01-01T00:00:00", "First entry"),
        )
        conn.commit()
        conn.close()

        with (
            patch("jpscripts.commands.notes.CLIPHIST_DIR", tmp_path),
            patch("jpscripts.commands.notes.CLIPHIST_DB", db_path),
            patch("jpscripts.commands.notes.CLIPHIST_FILE", tmp_path / "history.txt"),
            patch("shutil.which", return_value=None),
            patch("pyperclip.copy") as mock_copy,
        ):
            cliphist(mock_ctx, action="pick", limit=50, no_fzf=True)
            mock_copy.assert_called_once_with("First entry")

    def test_unknown_action(self, mock_ctx: MagicMock, tmp_path: Path) -> None:
        """Unknown action shows error."""
        with (
            patch("jpscripts.commands.notes.CLIPHIST_DIR", tmp_path),
            patch("jpscripts.commands.notes.CLIPHIST_DB", tmp_path / "history.db"),
            patch("jpscripts.commands.notes.CLIPHIST_FILE", tmp_path / "history.txt"),
        ):
            cliphist(mock_ctx, action="invalid", limit=50, no_fzf=True)


class TestLaunchEditor:
    """Tests for _launch_editor helper."""

    @pytest.mark.asyncio
    async def test_launches_editor(self, tmp_path: Path) -> None:
        """Editor is launched with correct arguments."""
        note_path = tmp_path / "test.md"
        note_path.write_text("content")

        mock_proc = AsyncMock()
        mock_proc.wait.return_value = 0

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            result = await _launch_editor(["nano"], note_path)
            assert result == 0
            mock_exec.assert_called_once_with("nano", str(note_path))


class TestRepoSummary:
    """Tests for RepoSummary dataclass."""

    def test_repo_summary_fields(self, tmp_path: Path) -> None:
        """RepoSummary stores path, commits and error."""
        commits = [
            GitCommit(
                hexsha="abc123",
                summary="Test commit",
                author_name="Test",
                author_email="test@example.com",
                committed_date=1000000,
            )
        ]
        summary = RepoSummary(path=tmp_path, commits=commits, error=None)

        assert summary.path == tmp_path
        assert len(summary.commits) == 1
        assert summary.error is None

    def test_repo_summary_with_error(self, tmp_path: Path) -> None:
        """RepoSummary can store error."""
        summary = RepoSummary(path=tmp_path, commits=[], error="Git error")

        assert summary.error == "Git error"
        assert summary.commits == []
