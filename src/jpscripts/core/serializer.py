from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import stat
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pathspec import PathSpec
from pydantic import BaseModel, ConfigDict
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

from jpscripts.core.console import get_logger
from jpscripts.core.result import Err, JPScriptsError, Ok, Result
from jpscripts.core.security import validate_path_safe, validate_workspace_root_safe


class FileType(str, Enum):
    TEXT = "text"
    BINARY = "binary"


class FileNode(BaseModel):
    path: str
    type: FileType
    size: int
    sha256: str
    content: str
    is_executable: bool

    model_config = ConfigDict(extra="forbid")


class RepoManifest(BaseModel):
    root: str
    timestamp: datetime
    file_count: int
    total_size_bytes: int
    files: list[FileNode]

    model_config = ConfigDict(extra="forbid")


class SerializationError(JPScriptsError):
    """Raised when repository serialization fails."""


class AsyncSerializer:
    """Serialize a repository into a manifest with async, validated I/O."""

    def __init__(self, max_concurrency: int = 50) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._logger = get_logger(__name__)

    async def serialize(self, root: Path | str) -> Result[RepoManifest, SerializationError]:
        root_result = await asyncio.to_thread(validate_workspace_root_safe, root)
        if isinstance(root_result, Err):
            workspace_err = root_result.error
            return Err(
                SerializationError(
                    f"Invalid workspace root: {workspace_err.message}",
                    context=workspace_err.context,
                )
            )

        resolved_root = root_result.value
        self._logger.debug("Starting repository serialization", extra={"root": str(resolved_root)})

        gitignore_result = await self._load_gitignore(resolved_root)
        if isinstance(gitignore_result, Err):
            return gitignore_result

        gitignore = gitignore_result.value

        paths_result = await self._collect_paths(resolved_root, gitignore)
        if isinstance(paths_result, Err):
            return paths_result

        file_paths = paths_result.value
        self._logger.debug(
            "Collected %s files for serialization", len(file_paths), extra={"root": str(resolved_root)}
        )

        try:
            file_results = await asyncio.gather(
                *(self._process_path(path, resolved_root) for path in file_paths)
            )
        except Exception as exc:
            return Err(
                SerializationError(
                    "Failed to process files during serialization",
                    context={"error": str(exc), "root": str(resolved_root)},
                )
            )

        file_nodes: list[FileNode] = []
        for result in file_results:
            if isinstance(result, Err):
                return result
            file_nodes.append(result.value)

        manifest = RepoManifest(
            root=str(resolved_root),
            timestamp=datetime.now(timezone.utc),
            file_count=len(file_nodes),
            total_size_bytes=sum(node.size for node in file_nodes),
            files=sorted(file_nodes, key=lambda node: node.path),
        )
        self._logger.debug(
            "Finished repository serialization",
            extra={
                "root": str(resolved_root),
                "file_count": manifest.file_count,
                "total_size_bytes": manifest.total_size_bytes,
            },
        )
        return Ok(manifest)

    async def _load_gitignore(self, root: Path) -> Result[PathSpec | None, SerializationError]:
        gitignore_path = root / ".gitignore"
        exists = await asyncio.to_thread(gitignore_path.exists)
        if not exists:
            return Ok(None)

        try:
            content = await asyncio.to_thread(gitignore_path.read_text, encoding="utf-8")
        except OSError as exc:
            return Err(
                SerializationError(
                    f"Failed to read .gitignore: {gitignore_path}",
                    context={"error": str(exc), "path": str(gitignore_path)},
                )
            )

        patterns = content.splitlines()
        return Ok(PathSpec.from_lines("gitwildmatch", patterns))

    async def _collect_paths(
        self,
        root: Path,
        gitignore: PathSpec | None,
    ) -> Result[list[Path], SerializationError]:
        def _walk() -> Result[list[Path], SerializationError]:
            collected: list[Path] = []
            try:
                for dirpath, dirnames, filenames in os.walk(root):
                    rel_dir = Path(dirpath).relative_to(root)
                    dirnames[:] = [
                        dirname
                        for dirname in sorted(dirnames)
                        if not self._is_ignored(rel_dir / dirname, gitignore)
                    ]
                    for filename in sorted(filenames):
                        relative = rel_dir / filename
                        if self._is_ignored(relative, gitignore):
                            continue
                        collected.append(Path(dirpath) / filename)
            except (OSError, ValueError) as exc:
                return Err(
                    SerializationError(
                        f"Failed to walk repository at {root}",
                        context={"error": str(exc), "path": str(root)},
                    )
                )

            return Ok(collected)

        return await asyncio.to_thread(_walk)

    def _is_ignored(self, relative_path: Path, gitignore: PathSpec | None) -> bool:
        if relative_path.parts and relative_path.parts[0] == ".git":
            return True
        if gitignore is None:
            return False
        return gitignore.match_file(relative_path.as_posix())

    async def _process_path(
        self,
        path: Path,
        root: Path,
    ) -> Result[FileNode, SerializationError]:
        async with self._semaphore:
            return await asyncio.to_thread(self._read_file_node, path, root)

    def _read_file_node(self, path: Path, root: Path) -> Result[FileNode, SerializationError]:
        safe_path_result = validate_path_safe(path, root)
        if isinstance(safe_path_result, Err):
            security_error = safe_path_result.error
            return Err(
                SerializationError(
                    f"Path validation failed for {path}",
                    context=security_error.context,
                )
            )

        safe_path = safe_path_result.value
        try:
            stat_result = safe_path.stat()
        except OSError as exc:
            return Err(
                SerializationError(
                    f"Failed to stat file: {safe_path}",
                    context={"error": str(exc), "path": str(safe_path)},
                )
            )

        try:
            raw_bytes = safe_path.read_bytes()
        except OSError as exc:
            return Err(
                SerializationError(
                    f"Failed to read file: {safe_path}",
                    context={"error": str(exc), "path": str(safe_path)},
                )
            )

        try:
            content = raw_bytes.decode("utf-8")
            file_type = FileType.TEXT
        except UnicodeDecodeError:
            file_type = FileType.BINARY
            content = base64.b64encode(raw_bytes).decode("ascii")

        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError as exc:
            return Err(
                SerializationError(
                    f"Path escaped workspace during serialization: {path}",
                    context={"error": str(exc), "path": str(path), "root": str(root)},
                )
            )
        is_executable = bool(stat_result.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))

        node = FileNode(
            path=relative_path,
            type=file_type,
            size=int(stat_result.st_size),
            sha256=sha256,
            content=content,
            is_executable=is_executable,
        )

        return Ok(node)


def _prepare_manifest_payload(manifest: RepoManifest) -> dict[str, object]:
    """Convert manifest to a YAML-ready payload with literal scalars for text content."""
    payload = manifest.model_dump(mode="json")
    files = payload.get("files")
    if isinstance(files, list):
        for entry in files:
            if not isinstance(entry, dict):
                continue
            file_type = entry.get("type")
            content = entry.get("content")
            if file_type == FileType.TEXT.value and isinstance(content, str):
                entry["content"] = LiteralScalarString(content)
    return payload


async def write_manifest_yaml(
    manifest: RepoManifest,
    output: Path,
    *,
    workspace_root: Path | None = None,
) -> Result[Path, SerializationError]:
    """
    Persist a manifest to YAML using literal block scalars for text content.

    I/O is dispatched to worker threads to preserve async purity.
    """
    base_root = workspace_root or Path(manifest.root)
    validated_root = await asyncio.to_thread(validate_workspace_root_safe, base_root)
    if isinstance(validated_root, Err):
        workspace_err = validated_root.error
        return Err(
            SerializationError(
                f"Invalid workspace root: {workspace_err.message}",
                context=workspace_err.context,
            )
        )

    output_result = await asyncio.to_thread(validate_path_safe, output, validated_root.value)
    if isinstance(output_result, Err):
        security_err = output_result.error
        return Err(
            SerializationError(
                f"Output path validation failed: {security_err.message}",
                context=security_err.context,
            )
        )
    safe_output = output_result.value

    def _write() -> Result[Path, SerializationError]:
        yaml = YAML(typ="safe")
        yaml.preserve_quotes = True
        yaml.default_flow_style = False
        yaml.width = 4096

        payload = _prepare_manifest_payload(manifest)
        destination = safe_output

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w", encoding="utf-8") as handle:
                yaml.dump(payload, handle)
        except OSError as exc:
            return Err(
                SerializationError(
                    f"Failed to write manifest: {destination}",
                    context={"error": str(exc), "path": str(destination)},
                )
            )
        return Ok(destination)

    return await asyncio.to_thread(_write)
