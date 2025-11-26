from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger

logger = get_logger(__name__)


STOPWORDS = {
    "the", "and", "or", "a", "an", "to", "of", "in", "on", "for", "by", "with", "is", "it", "this",
    "that", "as", "at", "be", "from", "are", "we", "use", "uses", "using",
}
MAX_ENTRIES = 5000


@dataclass
class MemoryEntry:
    ts: str
    content: str
    tags: list[str]
    tokens: list[str]
    embedding: bytes | None = None
    row_id: int | None = None


def _resolve_store(config: AppConfig | None = None, store_path: Path | None = None) -> Path:
    if store_path:
        return store_path.expanduser()
    if config and getattr(config, "memory_store", None):
        return Path(config.memory_store).expanduser()
    return Path.home() / ".jp_memory.sqlite"


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


def _format_entry(entry: MemoryEntry) -> str:
    tags = f"[{', '.join(entry.tags)}]" if entry.tags else ""
    return f"{entry.ts} {tags} {entry.content}".strip()


def _serialize_content(content: str, tags: Sequence[str]) -> str:
    payload = {"content": content.strip(), "tags": [t.strip() for t in tags if t.strip()]}
    return json.dumps(payload, ensure_ascii=True)


def _deserialize_content(raw: str) -> tuple[str, list[str]]:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            base_content = str(data.get("content", ""))
            tag_list = [str(t) for t in data.get("tags", []) if str(t).strip()]
            return base_content, tag_list
    except json.JSONDecodeError:
        pass
    return raw, []


def _compose_embedding_text(entry: MemoryEntry) -> str:
    tags = " ".join(entry.tags)
    return f"{entry.content} {tags}".strip()


def _legacy_embedding_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".embeddings.json")


class SQLiteVectorStore:
    def __init__(self, db_path: Path) -> None:
        self.path = db_path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY, content TEXT, timestamp TEXT, embedding BLOB)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory(timestamp)")
            conn.commit()

    def insert_entry(self, entry: MemoryEntry) -> MemoryEntry:
        serialized = _serialize_content(entry.content, entry.tags)
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO memory (content, timestamp, embedding) VALUES (?, ?, ?)",
                (serialized, entry.ts, entry.embedding),
            )
            entry.row_id = int(cursor.lastrowid)
        return entry

    def update_embedding(self, row_id: int, embedding: bytes) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE memory SET embedding = ? WHERE id = ?", (embedding, row_id))
            conn.commit()

    def fetch_entries(self, max_rows: int | None = None) -> list[MemoryEntry]:
        query = "SELECT id, content, timestamp, embedding FROM memory ORDER BY id DESC"
        params: tuple[int, ...] = ()
        if max_rows:
            query += " LIMIT ?"
            params = (max_rows,)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        entries: list[MemoryEntry] = []
        for row in rows:
            content, tags = _deserialize_content(row["content"] or "")
            token_source = f"{content} {' '.join(tags)}".strip()
            entries.append(
                MemoryEntry(
                    ts=row["timestamp"] or "",
                    content=content,
                    tags=tags,
                    tokens=_tokenize(token_source),
                    embedding=row["embedding"],
                    row_id=int(row["id"]),
                )
            )
        return entries


_SEMANTIC_WARNED = False


def _warn_semantic_unavailable() -> None:
    global _SEMANTIC_WARNED
    if not _SEMANTIC_WARNED:
        logger.warning("Semantic memory search unavailable. Install with `pip install jpscripts[ai]`.")
        _SEMANTIC_WARNED = True


def _load_embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        _warn_semantic_unavailable()
        return None

    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    cache_dir = cache_root / "jpscripts" / "sentence-transformers"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to load embedding model %s: %s", model_name, exc)
        _warn_semantic_unavailable()
        return None

    return model


class VectorStore:
    def __init__(self, model_name: str, enabled: bool) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self._model = None

    def available(self) -> bool:
        if not self.enabled:
            return False
        if np is None:
            _warn_semantic_unavailable()
            return False
        if self._model is not None:
            return True
        model = _load_embedding_model(self.model_name)
        if model is None:
            return False
        self._model = model
        return True

    def embed(self, texts: list[str]) -> np.ndarray | None:
        if not self.available():
            return None
        if np is None:
            return None
        assert self._model is not None
        vectors = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)

    @staticmethod
    def to_bytes(vector: np.ndarray) -> bytes:
        if np is None:
            raise RuntimeError("NumPy is required to serialize embeddings.")
        return np.asarray(vector, dtype=np.float32).tobytes()

    @staticmethod
    def from_bytes(blob: bytes) -> np.ndarray:
        if np is None:
            raise RuntimeError("NumPy is required to deserialize embeddings.")
        return np.frombuffer(blob, dtype=np.float32)

    @staticmethod
    def cosine_batch(matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
        if np is None:
            raise RuntimeError("NumPy is required for cosine similarity.")
        matrix = np.asarray(matrix, dtype=np.float32)
        query_vec = np.asarray(query_vec, dtype=np.float32)
        norms = (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)) + 1e-8
        return np.dot(matrix, query_vec) / norms


def _score(query_tokens: list[str], entry: MemoryEntry) -> float:
    if not query_tokens or not entry.tokens:
        return 0.0

    q_counts = Counter(query_tokens)
    e_counts = Counter(entry.tokens)
    overlap = sum(min(q_counts[t], e_counts[t]) for t in set(q_counts) & set(e_counts))
    tag_overlap = len(set(entry.tags) & set(query_tokens))
    return float(overlap + 0.5 * tag_overlap)


def _load_legacy_entries(path: Path, max_entries: int = MAX_ENTRIES) -> list[MemoryEntry]:
    if not path.exists():
        return []

    sidecar_map: dict[str, bytes] = {}
    sidecar = _legacy_embedding_sidecar(path)
    if sidecar.exists():
        try:
            raw_sidecar = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(raw_sidecar, list):
                for item in raw_sidecar:
                    if not isinstance(item, dict):
                        continue
                    ts = item.get("ts")
                    embedding = item.get("embedding")
                    if np is not None and isinstance(ts, str) and isinstance(embedding, list):
                        sidecar_map[ts] = np.asarray(embedding, dtype=np.float32).tobytes()
        except (json.JSONDecodeError, OSError):
            sidecar_map = {}

    entries: list[MemoryEntry] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = raw.get("content", "")
                tags = list(raw.get("tags") or [])
                token_source = f"{content} {' '.join(tags)}"
                entry = MemoryEntry(
                    ts=raw.get("ts", ""),
                    content=content,
                    tags=tags,
                    tokens=_tokenize(token_source),
                    embedding=sidecar_map.get(raw.get("ts", "")),
                )
                entries.append(entry)
                if len(entries) >= max_entries:
                    break
    except OSError as exc:
        logger.error("Failed to read legacy memory store %s: %s", path, exc)
        return []

    return entries


def save_memory(
    content: str,
    tags: Sequence[str] | None = None,
    *,
    config: AppConfig | None = None,
    store_path: Path | None = None,
) -> MemoryEntry:
    """Persist a memory entry for later recall."""
    resolved = _resolve_store(config, store_path)
    store = SQLiteVectorStore(resolved)

    normalized_tags = [t.strip() for t in (tags or []) if t.strip()]
    content_text = content.strip()
    token_source = f"{content_text} {' '.join(normalized_tags)}".strip()

    entry = MemoryEntry(
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        content=content_text,
        tags=normalized_tags,
        tokens=_tokenize(token_source),
    )

    use_semantic = True
    model_name = "all-MiniLM-L6-v2"
    if config:
        use_semantic = bool(getattr(config, "use_semantic_search", True))
        model_name = getattr(config, "memory_model", model_name)

    vector_store = VectorStore(model_name, enabled=use_semantic)
    if vector_store.available():
        vectors = vector_store.embed([_compose_embedding_text(entry)])
        if vectors is not None and len(vectors):
            entry.embedding = VectorStore.to_bytes(vectors[0])

    store.insert_entry(entry)
    return entry


def query_memory(
    query: str,
    limit: int = 5,
    *,
    config: AppConfig | None = None,
    store_path: Path | None = None,
) -> list[str]:
    """Retrieve the most relevant memory snippets for a query."""
    resolved_store = _resolve_store(config, store_path)
    store = SQLiteVectorStore(resolved_store)
    entries = store.fetch_entries(MAX_ENTRIES)
    if not entries:
        return []

    use_semantic = True
    model_name = "all-MiniLM-L6-v2"
    if config:
        use_semantic = bool(getattr(config, "use_semantic_search", True))
        model_name = getattr(config, "memory_model", model_name)

    vector_store = VectorStore(model_name, enabled=use_semantic)

    if vector_store.available() and np is not None:
        missing = [entry for entry in entries if entry.embedding is None]
        if missing:
            vectors = vector_store.embed([_compose_embedding_text(entry) for entry in missing])
            if vectors is not None:
                for entry, vec in zip(missing, vectors):
                    entry.embedding = VectorStore.to_bytes(vec)
                    if entry.row_id is not None:
                        store.update_embedding(entry.row_id, entry.embedding)

        with_embeddings = [entry for entry in entries if entry.embedding is not None]
        if with_embeddings:
            matrix = np.vstack([VectorStore.from_bytes(entry.embedding) for entry in with_embeddings])
            query_vecs = vector_store.embed([query])
            if query_vecs is not None:
                scores = VectorStore.cosine_batch(matrix, query_vecs[0])
                paired = list(zip(with_embeddings, scores))
                paired.sort(key=lambda x: x[1], reverse=True)
                return [_format_entry(entry) for entry, _ in paired[:limit]]

    tokens = _tokenize(query)
    scored_kw = [(entry, _score(tokens, entry)) for entry in entries]
    scored_kw = [item for item in scored_kw if item[1] > 0]
    scored_kw.sort(key=lambda x: x[1], reverse=True)

    return [_format_entry(entry) for entry, _ in scored_kw[:limit]]


def reindex_memory(
    *,
    config: AppConfig | None = None,
    legacy_path: Path | None = None,
    target_path: Path | None = None,
) -> Path:
    """
    Migrate existing JSONL memory data to the SQLite vector store.
    """
    target_store = _resolve_store(config, target_path)
    source = legacy_path or target_store.with_suffix(".jsonl")
    entries = _load_legacy_entries(source)
    if not entries:
        return target_store

    store = SQLiteVectorStore(target_store)
    for entry in entries:
        store.insert_entry(entry)
        if entry.embedding and entry.row_id is not None:
            store.update_embedding(entry.row_id, entry.embedding)

    return target_store
