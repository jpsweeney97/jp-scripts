from __future__ import annotations

import json
import os
import re
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger

logger = get_logger(__name__)


STOPWORDS = {
    "the", "and", "or", "a", "an", "to", "of", "in", "on", "for", "by", "with", "is", "it", "this",
    "that", "as", "at", "be", "from", "are", "we", "use", "uses", "using",
}


@dataclass
class MemoryEntry:
    ts: str
    content: str
    tags: list[str]
    tokens: list[str]
    embedding: list[float] | None = None


def _resolve_store(config: AppConfig | None = None, store_path: Path | None = None) -> Path:
    if store_path:
        return store_path.expanduser()
    if config and getattr(config, "memory_store", None):
        return config.memory_store.expanduser()
    return Path.home() / ".jp_memory.jsonl"


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


def _format_entry(entry: MemoryEntry) -> str:
    tags = f"[{', '.join(entry.tags)}]" if entry.tags else ""
    return f"{entry.ts} {tags} {entry.content}".strip()


def _embedding_sidecar_path(store_path: Path) -> Path:
    return store_path.with_suffix(store_path.suffix + ".embeddings.json")


_SEMANTIC_WARNED = False


def _warn_semantic_unavailable() -> None:
    global _SEMANTIC_WARNED
    if not _SEMANTIC_WARNED:
        logger.warning("Semantic memory search unavailable. Install with `pip install jpscripts[ai]`.")
        _SEMANTIC_WARNED = True


def save_memory(
    content: str,
    tags: Sequence[str] | None = None,
    *,
    config: AppConfig | None = None,
    store_path: Path | None = None,
) -> MemoryEntry:
    """Persist a memory entry for later recall."""
    resolved = _resolve_store(config, store_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    entry = MemoryEntry(
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        content=content.strip(),
        tags=[t.strip() for t in (tags or []) if t.strip()],
        tokens=_tokenize(content + " " + " ".join(tags or [])),
    )

    try:
        with resolved.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.__dict__, ensure_ascii=True) + "\n")
    except OSError as exc:
        logger.error("Failed to write memory entry to %s: %s", resolved, exc)
        raise

    return entry


def _load_entries(path: Path, max_entries: int = 5000) -> list[MemoryEntry]:
    if not path.exists():
        return []

    sidecar_map: dict[str, list[float]] = {}
    sidecar = _embedding_sidecar_path(path)
    if sidecar.exists():
        try:
            raw_sidecar = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(raw_sidecar, list):
                for item in raw_sidecar:
                    if not isinstance(item, dict):
                        continue
                    ts = item.get("ts")
                    embedding = item.get("embedding")
                    if isinstance(ts, str) and isinstance(embedding, list):
                        sidecar_map[ts] = embedding
        except (json.JSONDecodeError, OSError):
            sidecar_map = {}

    entries: deque[MemoryEntry] = deque(maxlen=max_entries)
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

                tokens = raw.get("tokens") or _tokenize(raw.get("content", ""))
                entry = MemoryEntry(
                    ts=raw.get("ts", ""),
                    content=raw.get("content", ""),
                    tags=list(raw.get("tags") or []),
                    tokens=tokens,
                    embedding=sidecar_map.get(raw.get("ts", "")),
                )
                entries.append(entry)
    except OSError as exc:
        logger.error("Failed to read memory store %s: %s", path, exc)
        return []

    return list(entries)


def _persist_embeddings(store_path: Path, entries: Sequence[MemoryEntry]) -> None:
    sidecar = _embedding_sidecar_path(store_path)
    tmp_path = sidecar.with_suffix(sidecar.suffix + ".tmp")
    payload = [
        {"ts": entry.ts, "embedding": entry.embedding}
        for entry in entries
        if entry.embedding is not None
    ]
    try:
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp_path, sidecar)
    except OSError as exc:
        logger.debug("Failed to persist embeddings sidecar %s: %s", sidecar, exc)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _load_embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as _np  # type: ignore
    except ImportError:
        _warn_semantic_unavailable()
        return None, None

    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    cache_dir = cache_root / "jpscripts" / "sentence-transformers"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to load embedding model %s: %s", model_name, exc)
        _warn_semantic_unavailable()
        return None, None

    return model, _np


class VectorStore:
    def __init__(self, model_name: str, enabled: bool) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        self._np = None

    def available(self) -> bool:
        if not self.enabled:
            return False
        if self._model is not None:
            return True
        model, np_mod = _load_embedding_model(self.model_name)
        if model is None or np_mod is None:
            return False
        self._model = model
        self._np = np_mod
        return True

    def embed(self, texts: list[str]):
        if not self.available():
            return None
        assert self._model is not None
        vectors = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return self._np.asarray(vectors)

    def as_array(self, data):
        assert self._np is not None
        return self._np.asarray(data, dtype=float)

    def cosine(self, a, b) -> float:
        assert self._np is not None
        denom = (self._np.linalg.norm(a) * self._np.linalg.norm(b)) + 1e-8
        return float(self._np.dot(a, b) / denom)


def _score(query_tokens: list[str], entry: MemoryEntry) -> float:
    if not query_tokens or not entry.tokens:
        return 0.0

    q_counts = Counter(query_tokens)
    e_counts = Counter(entry.tokens)
    overlap = sum(min(q_counts[t], e_counts[t]) for t in set(q_counts) & set(e_counts))
    tag_overlap = len(set(entry.tags) & set(query_tokens))
    return float(overlap + 0.5 * tag_overlap)


def query_memory(
    query: str,
    limit: int = 5,
    *,
    config: AppConfig | None = None,
    store_path: Path | None = None,
) -> list[str]:
    """Retrieve the most relevant memory snippets for a query."""
    resolved_store = _resolve_store(config, store_path)
    entries = _load_entries(resolved_store)
    if not entries:
        return []

    use_semantic = True
    model_name = "all-MiniLM-L6-v2"
    if config:
        use_semantic = bool(getattr(config, "use_semantic_search", True))
        model_name = getattr(config, "memory_model", model_name)

    vector_store = VectorStore(model_name, enabled=use_semantic)

    if vector_store.available():
        # Ensure embeddings are present
        missing_indices = [idx for idx, entry in enumerate(entries) if entry.embedding is None]
        if missing_indices:
            texts = [entries[idx].content for idx in missing_indices]
            vectors = vector_store.embed(texts)
            if vectors is not None:
                for idx, vec in zip(missing_indices, vectors):
                    if hasattr(vec, "tolist"):
                        entries[idx].embedding = vec.tolist()
                    else:
                        entries[idx].embedding = list(vec)
            _persist_embeddings(resolved_store, entries)

        # Compute query embedding
        query_vecs = vector_store.embed([query])
        if query_vecs is not None:
            query_vec = query_vecs[0]
            scored: list[tuple[MemoryEntry, float]] = []
            for entry in entries:
                if entry.embedding is None:
                    continue
                entry_vec = vector_store.as_array(entry.embedding)
                score = vector_store.cosine(entry_vec, query_vec)
                scored.append((entry, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            return [_format_entry(entry) for entry, _ in scored[:limit]]

    # Fallback to keyword scoring
    tokens = _tokenize(query)
    scored_kw = [
        (entry, _score(tokens, entry))
        for entry in entries
    ]
    scored_kw = [item for item in scored_kw if item[1] > 0]
    scored_kw.sort(key=lambda x: x[1], reverse=True)

    return [_format_entry(entry) for entry, _ in scored_kw[:limit]]
