from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Coroutine, Protocol, Sequence, TYPE_CHECKING, TypeVar, cast
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from uuid import uuid4

from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger

if TYPE_CHECKING:
    from lancedb.pydantic import LanceModel as LanceModelBase  # type: ignore[import-not-found]
    from lancedb.table import LanceTable  # type: ignore[import-not-found]
    from sentence_transformers import SentenceTransformer
else:  # pragma: no cover - runtime fallbacks when optional deps are missing
    class LanceModelBase:  # type: ignore[misc]
        pass

    class LanceTable:  # type: ignore[misc]
        ...

logger = get_logger(__name__)
T = TypeVar("T")


STOPWORDS = {
    "the",
    "and",
    "or",
    "a",
    "an",
    "to",
    "of",
    "in",
    "on",
    "for",
    "by",
    "with",
    "is",
    "it",
    "this",
    "that",
    "as",
    "at",
    "be",
    "from",
    "are",
    "we",
    "use",
    "uses",
    "using",
}
MAX_ENTRIES = 5000
DEFAULT_STORE = Path.home() / ".jp_memory.lance"
FALLBACK_SUFFIX = ".jsonl"
_SEMANTIC_WARNED = False


def _run_coroutine(coro: Coroutine[Any, Any, T]) -> T | None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if loop.is_running():  # pragma: no cover - defensive; embed calls run in worker threads
        logger.debug("Async loop already running; skipping coroutine execution.")
        return None

    return loop.run_until_complete(coro)


def _normalize_server_url(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    if "://" not in cleaned:
        cleaned = f"http://{cleaned}"
    parsed = urllib_parse.urlparse(cleaned)
    if not parsed.netloc and parsed.path:
        parsed = urllib_parse.urlparse(f"http://{parsed.path}")
    return parsed.geturl() if parsed.netloc else None


async def _async_check_port(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
    except (asyncio.TimeoutError, OSError):
        return False
    try:
        writer.close()
        await writer.wait_closed()
    except Exception:  # pragma: no cover - best effort cleanup
        pass
    return True


def _check_server_online(host: str, port: int) -> bool:
    result = _run_coroutine(_async_check_port(host, port))
    return bool(result)


def _post_json(url: str, payload: dict[str, Any], timeout: float = 2.0) -> dict[str, Any] | None:
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = urllib_request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError, OSError, ValueError) as exc:
        logger.debug("Embedding HTTP request failed: %s", exc)
        return None

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        logger.debug("Embedding HTTP response parse failed: %s", exc)
        return None
    return parsed if isinstance(parsed, dict) else None


async def _async_post_json(url: str, payload: dict[str, Any], timeout: float = 2.0) -> dict[str, Any] | None:
    return await asyncio.to_thread(_post_json, url, payload, timeout)


def _extract_embeddings(payload: dict[str, Any]) -> list[list[float]] | None:
    candidates: list[list[float]] = []
    if "data" in payload and isinstance(payload["data"], list):
        for item in payload["data"]:
            if isinstance(item, dict) and isinstance(item.get("embedding"), list):
                candidates.append([float(val) for val in item["embedding"]])
    if not candidates and "embeddings" in payload and isinstance(payload["embeddings"], list):
        for vector in payload["embeddings"]:
            if isinstance(vector, list):
                candidates.append([float(val) for val in vector])
    if not candidates and "embedding" in payload and isinstance(payload["embedding"], list):
        candidates.append([float(val) for val in payload["embedding"]])
    return candidates or None


@dataclass
class MemoryEntry:
    id: str
    ts: str
    content: str
    tags: list[str]
    tokens: list[str]
    embedding: list[float] | None = None


class VectorStore(Protocol):
    def insert(self, entries: Sequence[MemoryEntry]) -> None:
        ...

    def search(self, vector: list[float], limit: int) -> list[MemoryEntry]:
        ...

    def available(self) -> bool:
        ...


class EmbeddingClientProtocol(Protocol):
    @property
    def dimension(self) -> int | None:
        ...

    def available(self) -> bool:
        ...

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        ...


def _resolve_store(config: AppConfig | None = None, store_path: Path | None = None) -> Path:
    if store_path:
        return Path(store_path).expanduser()
    if config and getattr(config, "memory_store", None):
        return Path(config.memory_store).expanduser()
    return DEFAULT_STORE


def _fallback_path(base_path: Path) -> Path:
    return base_path.with_suffix(FALLBACK_SUFFIX)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


def _format_entry(entry: MemoryEntry) -> str:
    tags = f"[{', '.join(entry.tags)}]" if entry.tags else ""
    return f"{entry.ts} {tags} {entry.content}".strip()


def _compose_embedding_text(entry: MemoryEntry) -> str:
    tags = " ".join(entry.tags)
    return f"{entry.content} {tags}".strip()


def _embedding_settings(config: AppConfig | None) -> tuple[bool, str, str | None]:
    use_semantic = True
    model_name = "all-MiniLM-L6-v2"
    server_url: str | None = None
    if config:
        use_semantic = bool(getattr(config, "use_semantic_search", True))
        model_name = getattr(config, "memory_model", model_name)
        server_url = getattr(config, "embedding_server_url", None)
    return use_semantic, model_name, server_url


def _warn_semantic_unavailable() -> None:
    global _SEMANTIC_WARNED
    if _SEMANTIC_WARNED:
        return
    logger.warning("Semantic memory search unavailable. Install with `pip install jpscripts[ai]`.")
    _SEMANTIC_WARNED = True


class _GlobalEmbeddingClient(EmbeddingClientProtocol):
    _instance: _GlobalEmbeddingClient | None = None

    def __init__(self, model_name: str, enabled: bool, server_url: str | None) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self._server_url = _normalize_server_url(server_url)
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None
        self._remote_available: bool | None = None

    @classmethod
    def get_instance(cls, model_name: str, enabled: bool, server_url: str | None) -> _GlobalEmbeddingClient:
        normalized_url = _normalize_server_url(server_url)
        if cls._instance and cls._instance._matches(model_name, normalized_url):
            cls._instance.enabled = enabled
            return cls._instance
        cls._instance = cls(model_name, enabled, normalized_url)
        return cls._instance

    def _matches(self, model_name: str, server_url: str | None) -> bool:
        return self.model_name == model_name and self._server_url == server_url

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def _server_host_port(self) -> tuple[str, int] | None:
        if not self._server_url:
            return None
        parsed = urllib_parse.urlparse(self._server_url)
        host = parsed.hostname
        port = parsed.port
        if host is None:
            return None
        if port is None:
            if parsed.scheme == "https":
                port = 443
            elif parsed.scheme == "http":
                port = 80
            else:
                return None
        return host, port

    def _check_remote_available(self) -> bool:
        if self._remote_available is not None:
            return self._remote_available
        host_port = self._server_host_port()
        if host_port is None:
            self._remote_available = False
            return False
        host, port = host_port
        self._remote_available = _check_server_online(host, port)
        return self._remote_available

    def available(self) -> bool:
        if not self.enabled:
            return False
        if self._check_remote_available():
            return True
        return self._model is not None

    def _embed_remote(self, texts: list[str]) -> list[list[float]] | None:
        if not self._server_url or not self._check_remote_available():
            return None
        payload = {"input": texts, "model": self.model_name}
        response = _run_coroutine(_async_post_json(self._server_url, payload))
        if response is None:
            self._remote_available = False
            return None
        vectors = _extract_embeddings(response)
        if vectors is None:
            logger.debug("Embedding server returned no embeddings from %s", self._server_url)
            self._remote_available = False
            return None
        if vectors and self._dimension is None and vectors[0]:
            self._dimension = len(vectors[0])
        self._remote_available = True
        return vectors

    def _get_model(self) -> SentenceTransformer | None:
        if self._model is not None:
            return self._model
        model, dimension = _load_sentence_transformer(self.model_name)
        self._model = model
        self._dimension = dimension
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not self.enabled:
            return None
        if not texts:
            return []

        remote_vectors = self._embed_remote(texts)
        if remote_vectors is not None:
            return remote_vectors

        model = self._get_model()
        if model is None:
            return None
        vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [vector.tolist() for vector in vectors]


def EmbeddingClient(model_name: str, *, enabled: bool, server_url: str | None = None) -> EmbeddingClientProtocol:
    return _GlobalEmbeddingClient.get_instance(model_name, enabled, server_url)


def _load_sentence_transformer(model_name: str) -> tuple[SentenceTransformer | None, int | None]:
    try:
        from sentence_transformers import SentenceTransformer as STModel  # type: ignore[import-not-found]
    except ImportError:
        _warn_semantic_unavailable()
        return None, None

    cache_root = Path.home() / ".cache" / "jpscripts" / "sentence-transformers"
    cache_root.mkdir(parents=True, exist_ok=True)

    try:
        model = STModel(model_name, cache_folder=str(cache_root))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to load embedding model %s: %s", model_name, exc)
        _warn_semantic_unavailable()
        return None, None

    dimension: int | None = None
    if hasattr(model, "get_sentence_embedding_dimension"):
        dim_val = model.get_sentence_embedding_dimension()
        if dim_val is not None:
            dimension = int(dim_val)

    return model, dimension


def _load_entries(path: Path, max_entries: int = MAX_ENTRIES) -> list[MemoryEntry]:
    if not path.exists():
        return []

    entries: list[MemoryEntry] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if len(entries) >= max_entries:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = str(raw.get("content", "")).strip()
                tags = [str(tag).strip() for tag in raw.get("tags", []) if str(tag).strip()]
                token_source = f"{content} {' '.join(tags)}".strip()
                raw_tokens = raw.get("tokens")
                if isinstance(raw_tokens, list) and raw_tokens:
                    tokens = [str(tok) for tok in raw_tokens if str(tok)]
                else:
                    tokens = _tokenize(token_source)
                embedding = raw.get("embedding")
                embedding_list = [float(val) for val in embedding] if isinstance(embedding, list) else None
                entries.append(
                    MemoryEntry(
                        id=str(raw.get("id", uuid4().hex)),
                        ts=str(raw.get("ts", raw.get("timestamp", ""))),
                        content=content,
                        tags=tags,
                        tokens=tokens,
                        embedding=embedding_list,
                    )
                )
    except OSError as exc:
        logger.debug("Failed to read memory entries from %s: %s", path, exc)
        return []

    return entries


def _append_entry(path: Path, entry: MemoryEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "id": entry.id,
        "ts": entry.ts,
        "content": entry.content,
        "tags": entry.tags,
        "tokens": entry.tokens,
        "embedding": entry.embedding,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")


def _write_entries(path: Path, entries: Sequence[MemoryEntry]) -> None:
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


def _load_lancedb_dependencies() -> tuple[Any, type[LanceModelBase]] | None:
    try:
        lancedb = import_module("lancedb")
        pydantic_module = import_module("lancedb.pydantic")
        lance_model = cast(type[LanceModelBase], getattr(pydantic_module, "LanceModel"))
    except Exception as exc:
        logger.debug("LanceDB unavailable: %s", exc)
        return None
    return lancedb, lance_model


def _build_memory_record_model(base: type[LanceModelBase]) -> type[LanceModelBase]:
    class MemoryRecord(base):  # type: ignore[misc, valid-type]
        id: str
        timestamp: str
        content: str
        tags: list[str]
        embedding: list[float] | None

    return MemoryRecord


class LanceDBStore:
    def __init__(self, db_path: Path, embedding_dim: int) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        lance_imports = _load_lancedb_dependencies()
        if lance_imports is None:
            raise ImportError("lancedb is not installed")
        lancedb_module, lance_model_base = lance_imports

        self._db_path = db_path.expanduser()
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._lancedb = lancedb_module
        self._embedding_dim = embedding_dim
        self._model_cls: type[LanceModelBase] = _build_memory_record_model(lance_model_base)
        self._table: LanceTable = self._ensure_table()
        self._available = True

    def _ensure_table(self) -> LanceTable:
        db = self._lancedb.connect(str(self._db_path))
        model = self._model_cls
        if "memory" not in db.table_names():
            return db.create_table("memory", schema=model, exist_ok=True)
        return db.open_table("memory")

    def _validate_embedding(self, embedding: list[float] | None) -> list[float] | None:
        if embedding is None:
            return None
        if len(embedding) != self._embedding_dim:
            raise ValueError(f"Expected embedding dim {self._embedding_dim}, got {len(embedding)}")
        return embedding

    def insert(self, entries: Sequence[MemoryEntry]) -> None:
        model = self._model_cls
        payload = [
            model(
                id=entry.id,
                timestamp=entry.ts,
                content=entry.content,
                tags=entry.tags,
                embedding=self._validate_embedding(entry.embedding),
            )
            for entry in entries
        ]
        self._table.add(payload)

    def search(self, vector: list[float], limit: int) -> list[MemoryEntry]:
        results = self._table.search(vector).limit(limit).to_pydantic(self._model_cls)
        matches: list[MemoryEntry] = []
        for row in results:
            tags = list(row.tags or [])
            token_source = f"{row.content} {' '.join(tags)}".strip()
            matches.append(
                MemoryEntry(
                    id=row.id,
                    ts=row.timestamp,
                    content=row.content,
                    tags=tags,
                    tokens=_tokenize(token_source),
                    embedding=list(row.embedding) if row.embedding is not None else None,
                )
            )
        return matches

    def available(self) -> bool:
        return self._available


class NoOpStore:
    def insert(self, entries: Sequence[MemoryEntry]) -> None:  # pragma: no cover - trivial
        return

    def search(self, vector: list[float], limit: int) -> list[MemoryEntry]:  # pragma: no cover - trivial
        return []

    def available(self) -> bool:
        return False


def get_vector_store(db_path: Path, embedding_dim: int) -> VectorStore:
    try:
        return LanceDBStore(db_path, embedding_dim=embedding_dim)
    except ImportError:
        _warn_semantic_unavailable()
    except Exception as exc:
        logger.debug("Vector store unavailable at %s: %s", db_path, exc)
    return NoOpStore()


def _score(query_tokens: list[str], entry: MemoryEntry) -> float:
    if not query_tokens or not entry.tokens:
        return 0.0

    q_counts = Counter(query_tokens)
    e_counts = Counter(entry.tokens)
    overlap = sum(min(q_counts[t], e_counts[t]) for t in set(q_counts) & set(e_counts))
    tag_overlap = len(set(entry.tags) & set(query_tokens))
    return float(overlap + 0.5 * tag_overlap)


def save_memory(
    content: str,
    tags: Sequence[str] | None = None,
    *,
    config: AppConfig | None = None,
    store_path: Path | None = None,
) -> MemoryEntry:
    """Persist a memory entry for later recall."""
    resolved_store = _resolve_store(config, store_path)
    fallback_path = _fallback_path(resolved_store)

    normalized_tags = [t.strip() for t in (tags or []) if t.strip()]
    content_text = content.strip()
    token_source = f"{content_text} {' '.join(normalized_tags)}".strip()

    entry = MemoryEntry(
        id=uuid4().hex,
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        content=content_text,
        tags=normalized_tags,
        tokens=_tokenize(token_source),
    )

    use_semantic, model_name, server_url = _embedding_settings(config)
    embedding_client = EmbeddingClient(model_name, enabled=use_semantic, server_url=server_url)
    vectors = embedding_client.embed([_compose_embedding_text(entry)])
    if vectors:
        entry.embedding = vectors[0]

    _append_entry(fallback_path, entry)

    if entry.embedding and len(entry.embedding) > 0:
        store = get_vector_store(resolved_store, embedding_dim=len(entry.embedding))
        if store.available():
            try:
                store.insert([entry])
            except Exception as exc:
                logger.debug("Failed to persist memory to LanceDB at %s: %s", resolved_store, exc)

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
    fallback_path = _fallback_path(resolved_store)
    entries = _load_entries(fallback_path, MAX_ENTRIES)
    if not entries:
        return []

    use_semantic, model_name, server_url = _embedding_settings(config)
    embedding_client = EmbeddingClient(model_name, enabled=use_semantic, server_url=server_url)
    query_vecs = embedding_client.embed([query])
    if query_vecs:
        vector = query_vecs[0]
        if len(vector) == 0:
            return []
        store = get_vector_store(resolved_store, embedding_dim=len(vector))
        if store.available():
            try:
                vector_results = store.search(vector, limit)
                if vector_results:
                    return [_format_entry(entry) for entry in vector_results]
            except Exception as exc:
                logger.debug("Vector search failed, falling back to keywords: %s", exc)

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
    Migrate existing JSONL memory data to the LanceDB store and refresh embeddings.
    """
    target_store = _resolve_store(config, target_path)
    fallback_target = _fallback_path(target_store)
    source = legacy_path or fallback_target
    entries = _load_entries(source)
    if not entries:
        return target_store

    use_semantic, model_name, server_url = _embedding_settings(config)
    embedding_client = EmbeddingClient(model_name, enabled=use_semantic, server_url=server_url)
    for entry in entries:
        if entry.embedding is None:
            vectors = embedding_client.embed([_compose_embedding_text(entry)])
            if vectors:
                entry.embedding = vectors[0]
    populated = [entry for entry in entries if entry.embedding and len(entry.embedding) > 0]
    if populated:
        embedding_dim = len(populated[0].embedding or [])
        if embedding_dim > 0:
            store = get_vector_store(target_store, embedding_dim=embedding_dim)
            if store.available():
                try:
                    store.insert(populated)
                except Exception as exc:
                    logger.debug("Failed to populate LanceDB during reindex: %s", exc)

    _write_entries(fallback_target, entries)
    return target_store
