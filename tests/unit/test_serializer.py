from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import stat
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from jpscripts.core.result import Err, Ok
from jpscripts.core.serializer import AsyncSerializer, FileType, RepoManifest, write_manifest_yaml

HashEntry = tuple[str, bool]


async def _write_text_file(path: Path, content: str, *, executable: bool = False) -> None:
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_text, content, encoding="utf-8")
    mode = 0o755 if executable else 0o644
    await asyncio.to_thread(path.chmod, mode)


async def _write_binary_file(path: Path, data: bytes) -> None:
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_bytes, data)


async def _collect_hashes(root: Path, skip: set[str] | None = None) -> dict[str, HashEntry]:
    def _walk() -> dict[str, HashEntry]:
        collected: dict[str, HashEntry] = {}
        for dirpath, _, filenames in os.walk(root):
            rel_dir = Path(dirpath).relative_to(root)
            for name in filenames:
                rel_path = (rel_dir / name).as_posix()
                if skip and rel_path in skip:
                    continue
                file_path = Path(dirpath) / name
                raw = file_path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                mode = file_path.stat().st_mode
                executable = bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
                collected[rel_path] = (digest, executable)
        return collected

    return await asyncio.to_thread(_walk)


@pytest.mark.asyncio
async def test_round_trip_serialization(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    restored = tmp_path / "restored"
    await asyncio.to_thread(workspace.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(restored.mkdir, parents=True, exist_ok=True)

    # Fixtures
    gitignore_path = workspace / ".gitignore"
    await asyncio.to_thread(gitignore_path.write_text, "ignored.txt\n", encoding="utf-8")

    text_path = workspace / "src" / "app.py"
    text_content = "#!/usr/bin/env python3\nprint('hello world')\n"
    await _write_text_file(text_path, text_content, executable=True)

    nested_text_path = workspace / "docs" / "notes" / "readme.txt"
    nested_content = "line one\nline two\nline three\n"
    await _write_text_file(nested_text_path, nested_content)

    binary_path = workspace / "assets" / "blob.bin"
    binary_payload = os.urandom(256)
    await _write_binary_file(binary_path, binary_payload)

    ignored_path = workspace / "ignored.txt"
    await _write_text_file(ignored_path, "should be ignored")

    source_hashes = await _collect_hashes(workspace, skip={"ignored.txt"})

    serializer = AsyncSerializer(max_concurrency=8)
    manifest_result = await serializer.serialize(workspace)
    match manifest_result:
        case Err(error):
            pytest.fail(f"Serialization failed: {error}")
        case Ok(manifest):
            pass

    manifest_path = workspace / "manifest.yaml"
    write_result = await write_manifest_yaml(manifest, manifest_path, workspace_root=workspace)
    match write_result:
        case Err(error):
            pytest.fail(f"Writing manifest failed: {error}")
        case Ok(_):
            pass

    manifest_text = await asyncio.to_thread(manifest_path.read_text, encoding="utf-8")
    assert "content: |" in manifest_text

    yaml = YAML(typ="safe")
    manifest_data = yaml.load(manifest_text)
    assert manifest_data is not None
    repo_manifest = RepoManifest.model_validate(manifest_data)

    manifest_paths = {node.path for node in repo_manifest.files}
    assert "ignored.txt" not in manifest_paths
    assert repo_manifest.file_count == len(source_hashes)

    binary_node = next(node for node in repo_manifest.files if node.path == "assets/blob.bin")
    assert binary_node.type == FileType.BINARY
    decoded_payload = base64.b64decode(binary_node.content.encode("ascii"))
    assert decoded_payload == binary_payload

    for node in repo_manifest.files:
        target_path = restored / node.path
        await asyncio.to_thread(target_path.parent.mkdir, parents=True, exist_ok=True)
        if node.type == FileType.TEXT:
            raw_bytes = node.content.encode("utf-8")
        else:
            raw_bytes = base64.b64decode(node.content.encode("ascii"))

        await asyncio.to_thread(target_path.write_bytes, raw_bytes)
        current_mode = await asyncio.to_thread(target_path.stat)
        if node.is_executable:
            await asyncio.to_thread(
                target_path.chmod,
                current_mode.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH,
            )
        else:
            await asyncio.to_thread(
                target_path.chmod,
                current_mode.st_mode & ~(stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH),
            )

        restored_hash = hashlib.sha256(raw_bytes).hexdigest()
        assert restored_hash == node.sha256

    restored_hashes = await _collect_hashes(restored)
    assert source_hashes == restored_hashes


@pytest.mark.asyncio
async def test_serialize_arbitrary_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    target = tmp_path / "external"
    await asyncio.to_thread(workspace.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(target.mkdir, parents=True, exist_ok=True)

    target_file = target / "data.txt"
    await _write_text_file(target_file, "external content")

    serializer = AsyncSerializer(max_concurrency=4)
    manifest_result = await serializer.serialize(target)
    match manifest_result:
        case Err(error):
            pytest.fail(f"Serialization failed: {error}")
        case Ok(manifest):
            pass

    manifest_path = workspace / "external-manifest.yaml"
    write_result = await write_manifest_yaml(manifest, manifest_path, workspace_root=target)
    match write_result:
        case Err(error):
            pytest.fail(f"Writing manifest failed: {error}")
        case Ok(path):
            assert path == manifest_path

    manifest_text = await asyncio.to_thread(manifest_path.read_text, encoding="utf-8")
    assert "data.txt" in manifest_text


@pytest.mark.asyncio
async def test_default_excludes_are_enforced(tmp_path: Path) -> None:
    """Ensure DEFAULT_EXCLUDES are filtered regardless of depth or gitignore."""
    workspace = tmp_path / "workspace"
    await asyncio.to_thread(workspace.mkdir, parents=True, exist_ok=True)

    # Create nested .git directory (simulating submodule)
    nested_git = workspace / "src" / "deep" / "nested" / ".git"
    await asyncio.to_thread(nested_git.mkdir, parents=True, exist_ok=True)
    nested_git_config = nested_git / "config"
    await asyncio.to_thread(nested_git_config.write_text, "[core]\n", encoding="utf-8")

    # Create __pycache__ directory
    pycache = workspace / "__pycache__"
    await asyncio.to_thread(pycache.mkdir, parents=True, exist_ok=True)
    cache_file = pycache / "cache.pyc"
    await asyncio.to_thread(cache_file.write_bytes, b"\x00\x00")

    # Create legitimate file
    main_py = workspace / "src" / "main.py"
    await _write_text_file(main_py, "print('hello')")

    serializer = AsyncSerializer(max_concurrency=4)
    result = await serializer.serialize(workspace)

    match result:
        case Err(error):
            pytest.fail(f"Serialization failed: {error}")
        case Ok(manifest):
            pass

    paths = {node.path for node in manifest.files}

    # Legitimate file is present
    assert "src/main.py" in paths

    # Nested .git content is excluded
    assert "src/deep/nested/.git/config" not in paths

    # __pycache__ content is excluded
    assert "__pycache__/cache.pyc" not in paths
