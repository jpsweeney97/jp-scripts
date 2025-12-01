"""Git worktree management for parallel task isolation."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from jpscripts.core.dag import TaskStatus, WorktreeContext
from jpscripts.core.result import Err, GitError, Ok, Result
from jpscripts.git import AsyncRepo

# Pre-compiled pattern for worktree directory detection
_WORKTREE_DIR_PATTERN = re.compile(r"^worktree-[\w-]+-[a-f0-9]{8}$")


class WorktreeManager:
    """Manages git worktrees for parallel task isolation.

    Each parallel task runs in its own worktree to prevent:
    - Git index.lock contention
    - Filesystem race conditions
    - Merge conflicts during parallel execution

    Attributes:
        repo: The main repository
        worktree_root: Directory where worktrees are created
        preserve_on_failure: Keep failed worktrees for debugging

    [invariant:async-io] All operations use async subprocess
    """

    def __init__(
        self,
        repo: AsyncRepo,
        worktree_root: Path | None = None,
        preserve_on_failure: bool = False,
    ) -> None:
        """Initialize the worktree manager.

        Args:
            repo: The main git repository
            worktree_root: Directory for worktrees (default: temp dir)
            preserve_on_failure: Keep worktrees on failure for debugging
        """
        self._repo = repo
        self._worktree_root = worktree_root or Path(tempfile.gettempdir()) / "jp-worktrees"
        self._preserve_on_failure = preserve_on_failure
        self._active_worktrees: dict[str, WorktreeContext] = {}
        self._initialized = False

    @property
    def worktree_root(self) -> Path:
        """Get the worktree root directory."""
        return self._worktree_root

    async def initialize(self) -> Result[None, GitError]:
        """Initialize the worktree manager.

        Creates the worktree root directory if it doesn't exist and prunes
        any orphaned worktrees from previous crashed sessions.

        [invariant:async-io] Uses asyncio.to_thread for mkdir
        """
        if self._initialized:
            return Ok(None)

        def _create_root() -> None:
            self._worktree_root.mkdir(parents=True, exist_ok=True)

        try:
            await asyncio.to_thread(_create_root)
        except OSError as exc:
            return Err(GitError(f"Failed to create worktree root: {exc}"))

        # Auto-detect and clean orphans from previous sessions
        removed = await self.prune_orphaned_worktrees()
        if removed > 0:
            logger = logging.getLogger(__name__)
            logger.info("Pruned %d orphaned worktrees", removed)

        self._initialized = True
        return Ok(None)

    async def _create_worktree_context(self, task_id: str) -> WorktreeContext:
        """Create a new worktree for a task.

        Args:
            task_id: Unique task identifier

        Returns:
            WorktreeContext with paths and branch info
        """
        # Generate unique branch name
        unique_suffix = uuid.uuid4().hex[:8]
        branch_name = f"swarm/{task_id}-{unique_suffix}"
        worktree_path = self._worktree_root / f"worktree-{task_id}-{unique_suffix}"

        # Create the worktree
        result = await self._repo.worktree_add(
            worktree_path,
            branch_name,
            new_branch=True,
        )

        if isinstance(result, Err):
            raise RuntimeError(f"Failed to create worktree: {result.error}")

        ctx = WorktreeContext(
            task_id=task_id,
            worktree_path=worktree_path,
            branch_name=branch_name,
            status=TaskStatus.RUNNING,
        )

        self._active_worktrees[task_id] = ctx
        return ctx

    async def cleanup_worktree(
        self,
        ctx: WorktreeContext,
        *,
        failed: bool = False,
    ) -> Result[None, GitError]:
        """Clean up a worktree after task completion.

        Args:
            ctx: The worktree context to clean up
            failed: Whether the task failed

        Returns:
            Ok(None) on success, Err on failure
        """
        # Preserve on failure if configured
        if failed and self._preserve_on_failure:
            return Ok(None)

        # Remove the worktree
        result = await self._repo.worktree_remove(ctx.worktree_path, force=True)

        # Remove from active tracking
        if ctx.task_id in self._active_worktrees:
            del self._active_worktrees[ctx.task_id]

        if isinstance(result, Err):
            # Try pruning as fallback
            await self._repo.worktree_prune()

        return result

    @asynccontextmanager
    async def create_worktree(self, task_id: str) -> AsyncIterator[WorktreeContext]:
        """Create a worktree with automatic cleanup.

        This is the primary interface for creating worktrees.
        Uses context manager pattern to ensure cleanup.

        Args:
            task_id: Unique task identifier

        Yields:
            WorktreeContext for the created worktree

        Example:
            async with manager.create_worktree("task-001") as ctx:
                # Execute task in ctx.worktree_path
                pass
            # Worktree is automatically cleaned up
        """
        ctx = await self._create_worktree_context(task_id)
        failed = False

        try:
            yield ctx
        except Exception:
            failed = True
            raise
        finally:
            await self.cleanup_worktree(ctx, failed=failed)

    async def cleanup_all(self, force: bool = False) -> None:
        """Clean up all active worktrees.

        Args:
            force: Force cleanup even if dirty

        [invariant:async-io] Uses async worktree removal
        """
        for task_id, ctx in list(self._active_worktrees.items()):
            await self._repo.worktree_remove(ctx.worktree_path, force=force)
            del self._active_worktrees[task_id]

        # Final prune to clean up any orphaned references
        await self._repo.worktree_prune()

    async def detect_orphaned_worktrees(self) -> list[Path]:
        """Detect worktree directories from previous sessions not in memory.

        Scans worktree_root for directories matching the `worktree-*-*` pattern
        that are not currently tracked in _active_worktrees.

        Returns:
            List of orphaned worktree paths

        [invariant:async-io] Uses asyncio.to_thread for directory scan
        """
        if not self._worktree_root.exists():
            return []

        def _scan() -> list[Path]:
            return [
                d
                for d in self._worktree_root.iterdir()
                if d.is_dir() and _WORKTREE_DIR_PATTERN.match(d.name)
            ]

        candidates = await asyncio.to_thread(_scan)
        active_paths = {ctx.worktree_path for ctx in self._active_worktrees.values()}

        orphans: list[Path] = []
        for path in candidates:
            if path not in active_paths:
                orphans.append(path)

        return orphans

    async def prune_orphaned_worktrees(self, force: bool = True) -> int:
        """Remove orphaned worktrees from previous crashed sessions.

        Args:
            force: If True, use --force to remove even if dirty

        Returns:
            Number of worktrees successfully removed

        [invariant:async-io] Uses async worktree removal with fallback
        """
        orphans = await self.detect_orphaned_worktrees()
        if not orphans:
            return 0

        logger = logging.getLogger(__name__)
        logger.warning(
            "Found %d orphaned worktrees from previous session: %s",
            len(orphans),
            [p.name for p in orphans],
        )

        removed = 0
        for path in orphans:
            result = await self._repo.worktree_remove(path, force=force)
            if isinstance(result, Ok):
                removed += 1
            else:
                # Try manual cleanup if git command fails (orphan may not be registered)
                try:
                    await asyncio.to_thread(shutil.rmtree, path)
                    removed += 1
                except OSError as exc:
                    logger.error("Failed to remove orphan %s: %s", path, exc)

        # Final prune to clean git refs
        await self._repo.worktree_prune()

        return removed


__all__ = [
    "WorktreeManager",
]
