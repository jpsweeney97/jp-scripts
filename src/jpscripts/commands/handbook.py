from __future__ import annotations

import asyncio
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import typer
from rich.markdown import Markdown
from rich.panel import Panel

from jpscripts.core.config import AppConfig
from jpscripts.core.console import console
from jpscripts.core.memory import (
    EmbeddingClient,
    EmbeddingClientProtocol,
    MemoryEntry,
    STOPWORDS,
)
from jpscripts.core.security import validate_path


class _StubVectorStore:
    """Stub store that disables vector operations until LanceDB integration is restored."""

    def available(self) -> bool:
        return False

    def insert(self, entries: list[MemoryEntry]) -> None:
        pass

    def search(self, query_vec: list[float], limit: int) -> list[MemoryEntry]:
        return []


def get_vector_store(store_path: Path, *, embedding_dim: int) -> _StubVectorStore:
    """Placeholder for handbook vector store - returns stub until LanceDB integration is restored."""
    return _StubVectorStore()

CACHE_ROOT = Path.home() / ".cache" / "jpscripts" / "handbook_index"
HANDBOOK_NAME = "HANDBOOK.md"
MAX_RESULTS = 3


@dataclass
class HandbookSection:
    id: str
    title: str
    body: str

    def renderable(self) -> str:
        return f"{self.title}\n\n{self.body}".strip()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _tokenize_text(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]


async def _ensure_dir(path: Path) -> None:
    def _mk() -> None:
        path.mkdir(parents=True, exist_ok=True)

    await asyncio.to_thread(_mk)


def _cache_paths() -> tuple[Path, Path, Path, Path]:
    base_root = Path.home()
    cache_dir = validate_path(CACHE_ROOT, base_root)
    meta_path = validate_path(cache_dir / "meta.json", base_root)
    entries_path = validate_path(cache_dir / "entries.jsonl", base_root)
    store_path = validate_path(cache_dir / "lance", base_root)
    return cache_dir, meta_path, entries_path, store_path


async def _read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None

    def _load() -> dict[str, object] | None:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
                return None
        except (OSError, json.JSONDecodeError):
            return None

    return await asyncio.to_thread(_load)


async def _write_json(path: Path, payload: dict[str, object]) -> None:
    def _dump() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    await asyncio.to_thread(_dump)


async def _read_entries(path: Path) -> list[MemoryEntry]:
    if not path.exists():
        return []

    def _load() -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    entry = MemoryEntry(
                        id=str(raw.get("id", "")),
                        ts=str(raw.get("ts", "")),
                        content=str(raw.get("content", "")),
                        tags=[str(tag) for tag in raw.get("tags", []) if str(tag)],
                        tokens=[str(tok) for tok in raw.get("tokens", []) if str(tok)],
                        embedding=[float(val) for val in raw.get("embedding", [])] if raw.get("embedding") else None,
                    )
                    entries.append(entry)
        except OSError:
            return []
        return entries

    return await asyncio.to_thread(_load)


async def _write_entries(path: Path, entries: Iterable[MemoryEntry]) -> None:
    def _dump() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for entry in entries:
                record = {
                    "id": entry.id,
                    "ts": entry.ts,
                    "content": entry.content,
                    "tags": entry.tags,
                    "tokens": entry.tokens,
                    "embedding": entry.embedding,
                }
                fh.write(json.dumps(record, ensure_ascii=True) + "\n")

    await asyncio.to_thread(_dump)


def _resolve_handbook_path() -> Path | None:
    root = _project_root()
    try:
        path = validate_path(root / HANDBOOK_NAME, root)
    except PermissionError as exc:
        console.print(f"[red]{exc}[/red]")
        return None
    except Exception as exc:
        console.print(f"[red]Failed to resolve handbook path: {exc}[/red]")
        return None

    if not path.exists():
        console.print(f"[red]{HANDBOOK_NAME} not found at {path}[/red]")
        return None
    if not path.is_file():
        console.print(f"[red]{path} is not a file.[/red]")
        return None

    return path


async def _read_handbook(path: Path) -> str | None:
    try:
        return await asyncio.to_thread(path.read_text, encoding="utf-8")
    except OSError as exc:
        console.print(f"[red]Failed to read handbook: {exc}[/red]")
        return None


async def _mtime_ns(path: Path) -> int | None:
    try:
        stat_result = await asyncio.to_thread(path.stat)
        return stat_result.st_mtime_ns
    except OSError as exc:
        console.print(f"[red]Failed to stat {path}: {exc}[/red]")
        return None


def _parse_sections(content: str) -> list[HandbookSection]:
    heading_re = re.compile(r"^(#{1,2})\s+(.*)")
    sections: list[HandbookSection] = []
    current_title: str | None = None
    current_body: list[str] = []
    section_idx = 0

    for line in content.splitlines():
        match = heading_re.match(line.strip())
        if match:
            if current_title is not None:
                sections.append(
                    HandbookSection(
                        id=f"handbook-{section_idx}",
                        title=current_title,
                        body="\n".join(current_body).strip(),
                    )
                )
                section_idx += 1
            current_title = match.group(2).strip() or "Section"
            current_body = []
            continue
        if current_title is not None:
            current_body.append(line)

    if current_title is not None:
        sections.append(
            HandbookSection(
                id=f"handbook-{section_idx}",
                title=current_title,
                body="\n".join(current_body).strip(),
            )
        )

    if sections:
        return sections

    fallback = content.strip()
    if not fallback:
        return []
    return [HandbookSection(id="handbook-0", title="Handbook", body=fallback)]


def _build_entries(sections: list[HandbookSection], source_mtime_ns: int) -> list[MemoryEntry]:
    timestamp = datetime.fromtimestamp(source_mtime_ns / 1_000_000_000, tz=timezone.utc).isoformat(timespec="seconds")
    entries: list[MemoryEntry] = []
    for section in sections:
        text = section.renderable()
        entries.append(
            MemoryEntry(
                id=section.id,
                ts=timestamp,
                content=text,
                tags=[section.title] if section.title else [],
                tokens=_tokenize_text(text),
            )
        )
    return entries


def _build_embedding_client(config: AppConfig | None) -> EmbeddingClientProtocol:
    use_semantic = True
    model_name = "all-MiniLM-L6-v2"
    server_url: str | None = None
    if config:
        use_semantic = bool(getattr(config, "use_semantic_search", True))
        model_name = getattr(config, "memory_model", model_name)
        server_url = getattr(config, "embedding_server_url", None)
    return EmbeddingClient(model_name, enabled=use_semantic, server_url=server_url)


async def _reset_store(store_path: Path) -> None:
    if not store_path.exists():
        return

    def _remove() -> None:
        shutil.rmtree(store_path, ignore_errors=True)

    await asyncio.to_thread(_remove)


async def _insert_into_store(entries: list[MemoryEntry], store_path: Path, embedding_dim: int) -> None:
    store = get_vector_store(store_path, embedding_dim=embedding_dim)
    if not store.available():
        return

    def _insert() -> None:
        store.insert(entries)

    try:
        await asyncio.to_thread(_insert)
    except Exception as exc:
        console.print(f"[yellow]Vector store unavailable: {exc}[/yellow]")


async def _load_or_index_entries(
    sections: list[HandbookSection],
    embedding_client: EmbeddingClientProtocol,
    source_mtime_ns: int,
    meta_path: Path,
    entries_path: Path,
    store_path: Path,
) -> tuple[list[MemoryEntry], int]:
    meta = await _read_json(meta_path)
    if meta and meta.get("source_mtime_ns") == source_mtime_ns:
        cached_entries = await _read_entries(entries_path)
        if cached_entries:
            dim_value = meta.get("embedding_dim", 0)
            cached_dim = int(dim_value) if isinstance(dim_value, (int, float, str)) else 0
            if cached_dim == 0 and cached_entries[0].embedding:
                cached_dim = len(cached_entries[0].embedding or [])
            return cached_entries, cached_dim

    await _reset_store(store_path)
    entries = _build_entries(sections, source_mtime_ns)
    semantic_ready = embedding_client.available()
    vectors = embedding_client.embed([entry.content for entry in entries]) if semantic_ready else None
    if semantic_ready and vectors is None:
        console.print("[yellow]Semantic embeddings unavailable; falling back to keyword search.[/yellow]")
    embedding_dim = 0
    if vectors:
        first = vectors[0] if vectors else []
        embedding_dim = len(first) if first else 0
        for entry, vector in zip(entries, vectors):
            entry.embedding = vector

    await _write_entries(entries_path, entries)
    await _write_json(
        meta_path,
        {"source_mtime_ns": source_mtime_ns, "embedding_dim": embedding_dim},
    )

    if embedding_dim > 0:
        await _insert_into_store(entries, store_path, embedding_dim)

    return entries, embedding_dim


def _cosine_similarity(lhs: list[float], rhs: list[float]) -> float:
    if len(lhs) != len(rhs) or not lhs or not rhs:
        return 0.0
    dot = sum(a * b for a, b in zip(lhs, rhs))
    left_norm = math.sqrt(sum(a * a for a in lhs))
    right_norm = math.sqrt(sum(b * b for b in rhs))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _local_vector_search(entries: list[MemoryEntry], query_vec: list[float], limit: int) -> list[MemoryEntry]:
    scored: list[tuple[float, MemoryEntry]] = []
    for entry in entries:
        if entry.embedding is None:
            continue
        score = _cosine_similarity(query_vec, entry.embedding)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for _, entry in scored[:limit]]


def _keyword_search(entries: list[MemoryEntry], query: str, limit: int) -> list[MemoryEntry]:
    tokens = _tokenize_text(query)
    scored: list[tuple[int, MemoryEntry]] = []
    for entry in entries:
        score = sum(entry.content.lower().count(token) for token in tokens) if tokens else 0
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if scored:
        return [entry for _, entry in scored[:limit]]
    return entries[:limit]


async def _search_entries(
    query: str,
    embedding_client: EmbeddingClientProtocol,
    entries: list[MemoryEntry],
    embedding_dim: int,
    store_path: Path,
    limit: int = MAX_RESULTS,
) -> list[MemoryEntry]:
    if not embedding_client.available() or embedding_dim <= 0:
        return _keyword_search(entries, query, limit)

    query_vecs = embedding_client.embed([query])
    query_vec = query_vecs[0] if query_vecs else None

    if query_vec and len(query_vec) == embedding_dim:
        store = get_vector_store(store_path, embedding_dim=embedding_dim)
        if store.available():
            try:
                return store.search(query_vec, limit)
            except Exception:
                console.print("[yellow]Vector store search failed; falling back to local scoring.[/yellow]")
        local_ranked = _local_vector_search(entries, query_vec, limit)
        if local_ranked:
            return local_ranked

    return _keyword_search(entries, query, limit)


def _render_results(results: list[MemoryEntry]) -> None:
    if not results:
        console.print("[yellow]No matching sections found.[/yellow]")
        return

    for entry in results:
        title = entry.tags[0] if entry.tags else entry.content.splitlines()[0] if entry.content else "Handbook"
        body = entry.content or "No content available."
        console.print(Panel(Markdown(body), title=title, expand=True))


def handbook(
    ctx: typer.Context,
    query: str | None = typer.Argument(
        None,
        help="Optional semantic query. Provide text to search the handbook; omit to render the full handbook.",
    ),
) -> None:
    """Render the project handbook or run a semantic search over its sections.

    Args:
        ctx: Typer context containing application state.
        query: Optional semantic query. If provided, the command searches the handbook; otherwise renders it.
    """
    config: AppConfig | None = getattr(ctx.obj, "config", None) if ctx is not None else None

    async def _run() -> None:
        handbook_path = _resolve_handbook_path()
        if handbook_path is None:
            return

        content = await _read_handbook(handbook_path)
        if not content:
            return

        if query is None:
            console.print(Markdown(content))
            return

        sections = _parse_sections(content)
        if not sections:
            console.print("[yellow]No sections found in the handbook.[/yellow]")
            return

        source_mtime_ns = await _mtime_ns(handbook_path)
        if source_mtime_ns is None:
            return

        cache_dir, meta_path, entries_path, store_path = _cache_paths()
        await _ensure_dir(cache_dir)

        embedding_client = _build_embedding_client(config)
        entries, embedding_dim = await _load_or_index_entries(
            sections,
            embedding_client,
            source_mtime_ns,
            meta_path,
            entries_path,
            store_path,
        )

        results = await _search_entries(
            query=query,
            embedding_client=embedding_client,
            entries=entries,
            embedding_dim=embedding_dim,
            store_path=store_path,
            limit=MAX_RESULTS,
        )
        _render_results(results)

    asyncio.run(_run())
