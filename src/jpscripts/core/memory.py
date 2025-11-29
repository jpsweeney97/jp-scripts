from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import import_module
from math import sqrt
import os
from pathlib import Path
from typing import Any, Coroutine, Protocol, Sequence, TYPE_CHECKING, TypeVar, cast
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from uuid import uuid4

from jpscripts.core import structure
from jpscripts.core.config import AppConfig, ConfigError
from jpscripts.core.console import get_logger
from jpscripts.core.result import CapabilityMissingError, ConfigurationError, Err, JPScriptsError, Ok, Result
from jpscripts.providers import CompletionOptions, Message as ProviderMessage, ProviderError
from jpscripts.providers.factory import get_provider

if TYPE_CHECKING:
    from lancedb.pydantic import LanceModel as LanceModelBase  # type: ignore[import-untyped]
    from lancedb.table import LanceTable  # type: ignore[import-untyped]
    from sentence_transformers import SentenceTransformer
else:  # pragma: no cover - runtime fallbacks when optional deps are missing
    class LanceModelBase:  # type: ignore[misc]
        pass

    class LanceTable:  # type: ignore[misc]
        ...

logger = get_logger(__name__)
T = TypeVar("T")


# TODO: Load from a resource file to enable updates without code changes.
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
    "self",
    "cls",
    "def",
    "class",
    "import",
    "return",
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
    source_path: str | None = None
    content_hash: str | None = None
    related_files: list[str] = field(default_factory=list)


@dataclass
class Pattern:
    """A generalized pattern extracted from successful execution traces.

    Patterns are synthesized from clusters of similar successful fixes
    and stored in a dedicated LanceDB collection for RAG-based injection
    into agent prompts.
    """

    id: str
    created_at: str
    pattern_type: str  # "fix_pattern", "refactor_pattern", "test_pattern"
    description: str
    trigger: str  # When to apply this pattern
    solution: str  # What to do
    source_traces: list[str]  # Trace IDs that contributed
    confidence: float  # 0.0-1.0
    embedding: list[float] | None = None


class MemoryStore(Protocol):
    def add(self, entry: MemoryEntry) -> Result[MemoryEntry, JPScriptsError]:
        ...

    def search(
        self,
        query_vec: list[float] | None,
        limit: int,
        *,
        query_tokens: list[str] | None = None,
    ) -> Result[list[MemoryEntry], JPScriptsError]:
        ...

    def prune(self, root: Path) -> Result[int, JPScriptsError]:
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


def _compute_file_hash(path: Path) -> str | None:
    """Compute MD5 hash of file content. Returns None if file cannot be read."""
    try:
        resolved = path.resolve()
        digest = hashlib.md5()
        with resolved.open("rb") as fh:
            while True:
                chunk = fh.read(4096)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, IOError):
        return None


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


def _normalize_related_path(path: Path, root: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(resolved_path)


def _is_ignored_path(path: Path, root: Path, ignore_dirs: Sequence[str]) -> bool:
    try:
        rel_parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        return True
    ignore_set = {ignore.strip("/").strip() for ignore in ignore_dirs if ignore.strip()}
    return any(part in ignore_set for part in rel_parts)


def _collect_related_files(source_path: Path, root: Path, ignore_dirs: Sequence[str]) -> list[str]:
    if not source_path.exists():
        return []

    resolved_root = root.resolve()
    resolved_source = source_path.resolve()
    related: set[str] = set()

    for dep in structure.get_import_dependencies(resolved_source, resolved_root):
        if _is_ignored_path(dep, resolved_root, ignore_dirs):
            continue
        related.add(_normalize_related_path(dep, resolved_root))

    try:
        candidates = list(resolved_root.rglob("*.py"))
    except OSError:
        candidates = []

    for candidate in candidates:
        if candidate.resolve() == resolved_source:
            continue
        if _is_ignored_path(candidate, resolved_root, ignore_dirs):
            continue
        dependencies = structure.get_import_dependencies(candidate, resolved_root)
        if any(dep.resolve() == resolved_source for dep in dependencies):
            related.add(_normalize_related_path(candidate, resolved_root))

    return sorted(related)


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
        from sentence_transformers import SentenceTransformer as STModel
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
                raw_source_path = raw.get("source_path")
                source_path = str(raw_source_path) if raw_source_path else None
                raw_content_hash = raw.get("content_hash")
                content_hash = str(raw_content_hash) if raw_content_hash else None
                raw_related = raw.get("related_files") or []
                related_files = [str(path) for path in raw_related if str(path).strip()]
                entries.append(
                    MemoryEntry(
                        id=str(raw.get("id", uuid4().hex)),
                        ts=str(raw.get("ts", raw.get("timestamp", ""))),
                        content=content,
                        tags=tags,
                        tokens=tokens,
                        embedding=embedding_list,
                        source_path=source_path,
                        content_hash=content_hash,
                        related_files=related_files,
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
        "source_path": entry.source_path,
        "content_hash": entry.content_hash,
        "related_files": entry.related_files,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=True) + "\n")


def _write_entries(path: Path, entries: Sequence[MemoryEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            record = {
                "id": entry.id,
                "ts": entry.ts,
                "content": entry.content,
                "tags": entry.tags,
                "tokens": entry.tokens,
                "embedding": entry.embedding,
                "source_path": entry.source_path,
                "content_hash": entry.content_hash,
                "related_files": entry.related_files,
            }
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temp_path, path)


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
    class MemoryRecord(base):  # type: ignore[misc]
        id: str
        timestamp: str
        content: str
        tags: list[str]
        embedding: list[float] | None
        source_path: str | None
        related_files: list[str] | None

    return MemoryRecord


class LanceDBStore(MemoryStore):
    def __init__(self, db_path: Path, lancedb_module: Any, lance_model_base: type[LanceModelBase]) -> None:
        self._db_path = db_path.expanduser()
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._lancedb = lancedb_module
        self._model_cls: type[LanceModelBase] = _build_memory_record_model(lance_model_base)
        self._table: LanceTable | None = None
        self._embedding_dim: int | None = None

    def _ensure_table(self, embedding_dim: int) -> LanceTable:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self._embedding_dim is not None and self._embedding_dim != embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: {self._embedding_dim} != {embedding_dim}")

        if self._table is None or self._embedding_dim is None:
            db = self._lancedb.connect(str(self._db_path))
            model = self._model_cls
            if "memory" not in db.table_names():
                self._table = db.create_table("memory", schema=model, exist_ok=True)
            else:
                self._table = db.open_table("memory")
                try:
                    schema = getattr(self._table, "schema", None)
                    names_attr = getattr(schema, "names", None)
                    existing_names = {str(name) for name in names_attr} if isinstance(names_attr, (list, tuple, set)) else set()
                    if "related_files" not in existing_names:
                        try:
                            self._table.add_column("related_files", list[str])
                        except Exception:
                            logger.debug("Unable to add related_files column to LanceDB table; proceeding without schema update.")
                except Exception:
                    logger.debug("Skipping related_files schema check; proceeding with existing LanceDB schema.")
            self._embedding_dim = embedding_dim
        return self._table

    def add(self, entry: MemoryEntry) -> Result[MemoryEntry, JPScriptsError]:
        if entry.embedding is None:
            return Err(ConfigurationError("Embedding required for LanceDB insert", context={"id": entry.id}))

        try:
            table = self._ensure_table(len(entry.embedding))
            model = self._model_cls(
                id=entry.id,
                timestamp=entry.ts,
                content=entry.content,
                tags=entry.tags,
                embedding=entry.embedding,
                source_path=entry.source_path,
                related_files=entry.related_files or None,
            )
            table.add([model])
        except Exception as exc:  # pragma: no cover - defensive
            return Err(ConfigurationError("Failed to persist memory to LanceDB", context={"error": str(exc)}))

        return Ok(entry)

    def search(
        self,
        query_vec: list[float] | None,
        limit: int,
        *,
        query_tokens: list[str] | None = None,
    ) -> Result[list[MemoryEntry], JPScriptsError]:
        if query_vec is None:
            return Ok([])

        try:
            table = self._ensure_table(len(query_vec))
        except Exception as exc:
            return Err(ConfigurationError("Failed to prepare LanceDB table", context={"error": str(exc)}))

        try:
            results = table.search(query_vec).limit(limit).to_pydantic(self._model_cls)
        except Exception as exc:  # pragma: no cover - defensive
            return Err(ConfigurationError("LanceDB search failed", context={"error": str(exc)}))

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
                    source_path=row.source_path,
                    related_files=list(row.related_files or []),
                )
            )
        return Ok(matches)

    def prune(self, root: Path) -> Result[int, JPScriptsError]:  # pragma: no cover - not required for LanceDB
        _ = root
        return Ok(0)


class JsonlArchiver(MemoryStore):
    def __init__(self, path: Path, max_entries: int = MAX_ENTRIES) -> None:
        self._path = path
        self._max_entries = max_entries

    @property
    def path(self) -> Path:
        return self._path

    def load_entries(self) -> list[MemoryEntry]:
        return _load_entries(self._path, self._max_entries)

    def add(self, entry: MemoryEntry) -> Result[MemoryEntry, JPScriptsError]:
        try:
            _append_entry(self._path, entry)
        except OSError as exc:
            return Err(
                ConfigurationError(
                    "Failed to append memory entry",
                    context={"path": str(self._path), "error": str(exc)},
                )
            )
        return Ok(entry)

    def search(
        self,
        query_vec: list[float] | None,
        limit: int,
        *,
        query_tokens: list[str] | None = None,
    ) -> Result[list[MemoryEntry], JPScriptsError]:
        _ = query_vec
        entries = self.load_entries()
        if not query_tokens:
            return Ok([])
        scored = [(entry, _score(query_tokens, entry)) for entry in entries]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return Ok([entry for entry, _score_val in scored[:limit]])

    def prune(self, root: Path) -> Result[int, JPScriptsError]:
        entries = self.load_entries()
        if not entries:
            return Ok(0)

        workspace_root = Path(root).expanduser().resolve()
        kept: list[MemoryEntry] = []
        pruned_count = 0

        for entry in entries:
            if entry.source_path is None:
                kept.append(entry)
                continue

            source = Path(entry.source_path)
            if not source.is_absolute():
                source = workspace_root / source

            try:
                if not source.exists():
                    pruned_count += 1
                    logger.debug("Pruning stale memory entry: %s (file missing: %s)", entry.id, entry.source_path)
                    continue

                # Check for content drift via hash mismatch
                if entry.content_hash is not None:
                    current_hash = _compute_file_hash(source)
                    if current_hash is not None and current_hash != entry.content_hash:
                        pruned_count += 1
                        logger.debug("Pruning drifted memory entry: %s (hash mismatch)", entry.id)
                        continue

                kept.append(entry)
            except OSError:
                kept.append(entry)

        try:
            _write_entries(self._path, kept)
        except OSError as exc:
            return Err(ConfigurationError("Failed to rewrite pruned memory archive", context={"error": str(exc)}))

        return Ok(pruned_count)


class HybridMemoryStore(MemoryStore):
    def __init__(self, archiver: JsonlArchiver, vector_store: LanceDBStore | None) -> None:
        self._archiver = archiver
        self._vector_store = vector_store

    @property
    def archiver(self) -> JsonlArchiver:
        return self._archiver

    @property
    def vector_store(self) -> LanceDBStore | None:
        return self._vector_store

    def add(self, entry: MemoryEntry) -> Result[MemoryEntry, JPScriptsError]:
        match self._archiver.add(entry):
            case Err(err):
                return Err(err)
            case Ok(_):
                pass

        if self._vector_store and entry.embedding is not None:
            vector_result = self._vector_store.add(entry)
            if isinstance(vector_result, Err):
                return vector_result

        return Ok(entry)

    def search(
        self,
        query_vec: list[float] | None,
        limit: int,
        *,
        query_tokens: list[str] | None = None,
    ) -> Result[list[MemoryEntry], JPScriptsError]:
        entries = self._archiver.load_entries()
        vector_results: list[MemoryEntry] = []
        vector_ranks: dict[str, int] = {}

        if self._vector_store and query_vec is not None:
            match self._vector_store.search(query_vec, limit, query_tokens=query_tokens):
                case Err(err):
                    return Err(err)
                case Ok(results):
                    vector_results = results
                    vector_ranks = {entry.id: idx + 1 for idx, entry in enumerate(results)}

        keyword_ranks: dict[str, int] = {}
        if query_tokens:
            kw_scored = [(entry, _score(query_tokens, entry)) for entry in entries]
            kw_scored = [item for item in kw_scored if item[1] > 0]
            kw_scored.sort(key=lambda item: item[1], reverse=True)
            keyword_ranks = {entry.id: idx + 1 for idx, (entry, _score_val) in enumerate(kw_scored)}

        if not vector_ranks and not keyword_ranks:
            return Ok([])

        k_const = 60.0
        entry_lookup: dict[str, MemoryEntry] = {entry.id: entry for entry in entries}
        for entry in vector_results:
            entry_lookup[entry.id] = entry

        fused: list[tuple[float, MemoryEntry]] = []
        seen_ids = set(vector_ranks) | set(keyword_ranks)
        for entry_id in seen_ids:
            v_rank = vector_ranks.get(entry_id)
            k_rank = keyword_ranks.get(entry_id)
            score = 0.0
            if v_rank is not None:
                score += 1.0 / (k_const + v_rank)
            if k_rank is not None:
                score += 1.0 / (k_const + k_rank)
            found_entry = entry_lookup.get(entry_id)
            if found_entry is not None:
                fused.append((score, found_entry))

        fused.sort(key=lambda item: item[0], reverse=True)
        return Ok([entry for _, entry in fused[:limit]])

    def prune(self, root: Path) -> Result[int, JPScriptsError]:
        return self._archiver.prune(root)


def _score(query_tokens: list[str], entry: MemoryEntry) -> float:
    if not query_tokens or not entry.tokens:
        return 0.0

    q_counts = Counter(query_tokens)
    e_counts = Counter(entry.tokens)
    overlap = sum(min(q_counts[t], e_counts[t]) for t in set(q_counts) & set(e_counts))
    tag_overlap = len(set(entry.tags) & set(query_tokens))
    base_score = float(overlap + 0.5 * tag_overlap)
    if base_score == 0.0:
        return 0.0

    decay = 1.0
    try:
        timestamp = datetime.fromisoformat(entry.ts)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        days_since = max((datetime.now(timezone.utc) - timestamp).days, 0)
        decay = 1 / (1 + 0.1 * float(days_since))
    except Exception:
        decay = 1.0

    return base_score * decay


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sqrt(sum(a * a for a in vec_a))
    norm_b = sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_memory_store(
    config: AppConfig,
    store_path: Path | None = None,
) -> Result[MemoryStore, ConfigError | CapabilityMissingError]:
    resolved_store = _resolve_store(config, store_path)
    archiver = JsonlArchiver(_fallback_path(resolved_store))

    use_semantic, _model_name, _server_url = _embedding_settings(config)
    vector_store: LanceDBStore | None = None
    if use_semantic:
        deps = _load_lancedb_dependencies()
        if deps is None:
            return Err(
                CapabilityMissingError(
                    "LanceDB is required for semantic memory. Install with `pip install \"jpscripts[ai]\"`.",
                    context={"path": str(resolved_store)},
                )
            )
        lancedb_module, lance_model_base = deps
        try:
            vector_store = LanceDBStore(resolved_store, lancedb_module, lance_model_base)
        except Exception as exc:  # pragma: no cover - defensive
            return Err(ConfigError(f"Failed to initialize LanceDB store at {resolved_store}: {exc}"))

    return Ok(HybridMemoryStore(archiver, vector_store))


def _graph_expand(entries: Sequence[MemoryEntry]) -> list[MemoryEntry]:
    if not entries:
        return []

    top = entries[0]
    top_related = set(top.related_files)
    related_union: set[str] = set()
    for entry in entries:
        related_union.update(entry.related_files)

    def _rank_score(index: int, entry: MemoryEntry) -> float:
        base = 1.0 / float(index + 1)
        entry_related = set(entry.related_files)
        score = base
        if top_related and entry_related & top_related:
            score += 0.75
        if top.source_path and entry.source_path and top.source_path == entry.source_path:
            score += 0.5
        if top.source_path and top.source_path in entry_related:
            score += 0.5
        if entry.source_path and entry.source_path in related_union:
            score += 0.25
        return score

    ranked = sorted(
        enumerate(entries),
        key=lambda item: (_rank_score(item[0], item[1]), -item[0]),
        reverse=True,
    )
    return [entries[idx] for idx, _ in ranked]


async def cluster_memories(
    config: AppConfig,
    similarity_threshold: float = 0.85,
) -> Result[list[list[MemoryEntry]], JPScriptsError]:
    """Group memories into clusters based on cosine similarity."""
    match get_memory_store(config):
        case Err(err):
            if isinstance(err, JPScriptsError):
                return Err(err)
            return Err(ConfigurationError(str(err)))
        case Ok(store):
            pass

    if not isinstance(store, HybridMemoryStore) or store.vector_store is None:
        return Err(CapabilityMissingError("LanceDB is required for clustering memories."))

    entries = await asyncio.to_thread(store.archiver.load_entries)
    candidates = [entry for entry in entries if entry.embedding is not None]
    if not candidates:
        return Ok([])

    candidates.sort(key=lambda e: e.ts)
    clusters: list[list[MemoryEntry]] = []

    for entry in candidates:
        embedding = entry.embedding
        if embedding is None:
            continue
        placed = False
        for cluster in clusters:
            representative = cluster[0]
            rep_embedding = representative.embedding or []
            if _cosine_similarity(embedding, rep_embedding) >= similarity_threshold:
                cluster.append(entry)
                placed = True
                break
        if not placed:
            clusters.append([entry])

    dense_clusters = [cluster for cluster in clusters if len(cluster) > 1]
    return Ok(dense_clusters)


async def synthesize_cluster(
    entries: Sequence[MemoryEntry],
    config: AppConfig,
    *,
    model: str | None = None,
) -> Result[MemoryEntry, JPScriptsError]:
    """Synthesize a canonical memory from a cluster."""
    if not entries:
        return Err(ConfigurationError("Cannot synthesize from an empty cluster."))

    model_id = model or config.default_model
    try:
        provider = get_provider(config, model_id=model_id)
    except Exception as exc:
        return Err(ConfigurationError("Failed to initialize provider", context={"error": str(exc)}))

    ordered = sorted(entries, key=lambda e: e.ts)
    lines = []
    for entry in ordered:
        line_tags = ",".join(entry.tags) if entry.tags else "none"
        source = entry.source_path or "unknown"
        lines.append(f"- ts={entry.ts}; tags={line_tags}; source={source}; content={entry.content}")

    system_prompt = (
        "These memories describe the same topic. Merge them into a single, canonical 'Truth' entry. "
        "Resolve contradictions by favoring the most recent timestamp."
    )
    messages = [
        ProviderMessage(role="system", content=system_prompt),
        ProviderMessage(role="user", content="\n".join(lines)),
    ]
    options = CompletionOptions(temperature=0.2, reasoning_effort="high")

    try:
        response = await provider.complete(messages=messages, model=model_id, options=options)
    except ProviderError as exc:
        return Err(ConfigurationError("LLM synthesis failed", context={"error": str(exc)}))
    except Exception as exc:  # pragma: no cover - defensive
        return Err(ConfigurationError("LLM synthesis failed", context={"error": str(exc)}))

    synthesized_content = response.content.strip()
    aggregated_tags = sorted({tag for entry in entries for tag in entry.tags} | {"truth"})
    token_source = f"{synthesized_content} {' '.join(aggregated_tags)}".strip()
    source_paths = sorted({entry.source_path for entry in entries if entry.source_path})
    source_metadata = ";".join(source_paths) if source_paths else None

    new_entry = MemoryEntry(
        id=uuid4().hex,
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        content=synthesized_content,
        tags=aggregated_tags,
        tokens=_tokenize(token_source),
        source_path=source_metadata,
        content_hash=None,
    )

    use_semantic, model_name, server_url = _embedding_settings(config)
    embedding_client = EmbeddingClient(model_name, enabled=use_semantic, server_url=server_url)
    vectors = embedding_client.embed([_compose_embedding_text(new_entry)]) if use_semantic else None
    if vectors:
        new_entry.embedding = vectors[0]

    return Ok(new_entry)


def save_memory(
    content: str,
    tags: Sequence[str] | None = None,
    *,
    config: AppConfig | None = None,
    store_path: Path | None = None,
    source_path: str | None = None,
) -> MemoryEntry:
    """Persist a memory entry for later recall.

    Args:
        content: Memory content or ADR/lesson learned.
        tags: Tags to associate with the entry.
        config: Application configuration.
        store_path: Override store location.
        source_path: Optional file path this memory is related to.
    """
    if config is None:
        raise ConfigError("AppConfig is required to save memory.")

    resolved_store = _resolve_store(config, store_path)
    root = Path(getattr(config, "workspace_root", Path.cwd())).expanduser().resolve()

    normalized_tags = [t.strip() for t in (tags or []) if t.strip()]
    content_text = content.strip()
    token_source = f"{content_text} {' '.join(normalized_tags)}".strip()

    # Compute content hash if source_path provided
    computed_hash: str | None = None
    related_files: list[str] = []
    normalized_source: str | None = None
    if source_path:
        source = Path(source_path).expanduser()
        if not source.is_absolute():
            source = root / source
        computed_hash = _compute_file_hash(source)
        normalized_source = _normalize_related_path(source, root)
        related_files = _collect_related_files(source, root, getattr(config, "ignore_dirs", []))

    entry = MemoryEntry(
        id=uuid4().hex,
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        content=content_text,
        tags=normalized_tags,
        tokens=_tokenize(token_source),
        source_path=normalized_source,
        content_hash=computed_hash,
        related_files=related_files,
    )

    use_semantic, model_name, server_url = _embedding_settings(config)
    embedding_client = EmbeddingClient(model_name, enabled=use_semantic, server_url=server_url)
    vectors = embedding_client.embed([_compose_embedding_text(entry)]) if use_semantic else None
    if vectors:
        entry.embedding = vectors[0]

    match get_memory_store(config, store_path=resolved_store):
        case Err(err):
            raise err
        case Ok(store):
            add_result = store.add(entry)
            if isinstance(add_result, Err):
                raise add_result.error

    return entry


def query_memory(
    query: str,
    limit: int = 5,
    *,
    config: AppConfig | None = None,
    store_path: Path | None = None,
) -> list[str]:
    """Retrieve the most relevant memory snippets for a query using reciprocal rank fusion."""
    if config is None:
        raise ConfigError("AppConfig is required to query memory.")

    match get_memory_store(config, store_path=store_path):
        case Err(err):
            raise err
        case Ok(store):
            use_semantic, model_name, server_url = _embedding_settings(config)
            embedding_client = EmbeddingClient(model_name, enabled=use_semantic, server_url=server_url)
            query_vecs = embedding_client.embed([query]) if use_semantic else None
            query_vec = query_vecs[0] if query_vecs else None
            tokens = _tokenize(query)

            search_result = store.search(query_vec, limit, query_tokens=tokens)
            if isinstance(search_result, Err):
                raise search_result.error
            entries = search_result.value
            if not entries:
                return []
            ranked_entries = _graph_expand(entries)
            return [_format_entry(entry) for entry in ranked_entries[:limit]]


def reindex_memory(
    *,
    config: AppConfig | None = None,
    legacy_path: Path | None = None,
    target_path: Path | None = None,
) -> Path:
    """
    Migrate existing JSONL memory data to the LanceDB store and refresh embeddings.
    """
    if config is None:
        raise ConfigError("AppConfig is required to reindex memory.")

    target_store = _resolve_store(config, target_path)
    fallback_target = _fallback_path(target_store)
    source = legacy_path or fallback_target
    entries = _load_entries(source)
    if not entries:
        return target_store

    use_semantic, model_name, server_url = _embedding_settings(config)
    embedding_client = EmbeddingClient(model_name, enabled=use_semantic, server_url=server_url)
    for entry in entries:
        if entry.embedding is None and use_semantic:
            vectors = embedding_client.embed([_compose_embedding_text(entry)])
            if vectors:
                entry.embedding = vectors[0]

    _write_entries(fallback_target, entries)

    match get_memory_store(config, store_path=target_store):
        case Err(err):
            raise err
        case Ok(store):
            if isinstance(store, HybridMemoryStore) and store.vector_store:
                for entry in entries:
                    if entry.embedding:
                        _ = store.vector_store.add(entry)

    return target_store


def prune_memory(config: AppConfig) -> int:
    """Remove memory entries related to deleted files to maintain vector store hygiene.

    Loads all entries, checks if source_path exists (relative to workspace_root or absolute),
    and removes stale entries from JSONL. Triggers reindex afterward.

    Args:
        config: Application configuration with workspace_root.

    Returns:
        Count of pruned entries.
    """
    match get_memory_store(config):
        case Err(err):
            raise err
        case Ok(store):
            result = store.prune(Path(config.workspace_root))
            if isinstance(result, Err):
                raise result.error
            return result.value


# ---------------------------------------------------------------------------
# Pattern Storage and Synthesis
# ---------------------------------------------------------------------------


def _build_pattern_record_model(base: type[LanceModelBase]) -> type[LanceModelBase]:
    """Build LanceDB model for patterns collection."""

    class PatternRecord(base):  # type: ignore[misc]
        id: str
        created_at: str
        pattern_type: str
        description: str
        trigger: str
        solution: str
        source_traces: list[str]
        confidence: float
        embedding: list[float] | None

    return PatternRecord


class PatternStore:
    """Dedicated LanceDB collection for extracted patterns.

    Patterns are stored separately from memories to enable specialized
    retrieval for prompt injection without polluting the general memory space.
    """

    def __init__(
        self, db_path: Path, lancedb_module: Any, lance_model_base: type[LanceModelBase]
    ) -> None:
        self._db_path = db_path.expanduser()
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._lancedb = lancedb_module
        self._model_cls = _build_pattern_record_model(lance_model_base)
        self._table: LanceTable | None = None
        self._embedding_dim: int | None = None

    def _ensure_table(self, embedding_dim: int | None = None) -> LanceTable:
        """Ensure the patterns table exists."""
        if self._table is None:
            db = self._lancedb.connect(str(self._db_path))
            if "patterns" not in db.table_names():
                self._table = db.create_table("patterns", schema=self._model_cls, exist_ok=True)
            else:
                self._table = db.open_table("patterns")
            if embedding_dim is not None:
                self._embedding_dim = embedding_dim
        return self._table

    def add(self, pattern: Pattern) -> Result[Pattern, JPScriptsError]:
        """Add a pattern to the store."""
        try:
            embedding_dim = len(pattern.embedding) if pattern.embedding else None
            table = self._ensure_table(embedding_dim)
            record = self._model_cls(
                id=pattern.id,
                created_at=pattern.created_at,
                pattern_type=pattern.pattern_type,
                description=pattern.description,
                trigger=pattern.trigger,
                solution=pattern.solution,
                source_traces=pattern.source_traces,
                confidence=pattern.confidence,
                embedding=pattern.embedding,
            )
            table.add([record])
        except Exception as exc:
            return Err(
                ConfigurationError(
                    "Failed to persist pattern to LanceDB",
                    context={"error": str(exc)},
                )
            )
        return Ok(pattern)

    def search(
        self, query_vec: list[float] | None, limit: int
    ) -> Result[list[Pattern], JPScriptsError]:
        """Search patterns by embedding similarity."""
        if query_vec is None:
            return Ok([])

        try:
            table = self._ensure_table(len(query_vec))
        except Exception as exc:
            return Err(
                ConfigurationError(
                    "Failed to prepare patterns table", context={"error": str(exc)}
                )
            )

        try:
            results = table.search(query_vec).limit(limit).to_pydantic(self._model_cls)
        except Exception as exc:
            return Err(
                ConfigurationError("Pattern search failed", context={"error": str(exc)})
            )

        patterns: list[Pattern] = []
        for row in results:
            patterns.append(
                Pattern(
                    id=row.id,
                    created_at=row.created_at,
                    pattern_type=row.pattern_type,
                    description=row.description,
                    trigger=row.trigger,
                    solution=row.solution,
                    source_traces=list(row.source_traces or []),
                    confidence=float(row.confidence),
                    embedding=list(row.embedding) if row.embedding else None,
                )
            )
        return Ok(patterns)

    def get_all(self, limit: int = 100) -> Result[list[Pattern], JPScriptsError]:
        """Get all patterns (up to limit)."""
        try:
            table = self._ensure_table()
            # LanceDB doesn't have a simple get_all, so we do a dummy search if we have any patterns
            # For now, use a simple approach
            df = table.to_pandas()
            patterns: list[Pattern] = []
            for _, row in df.head(limit).iterrows():
                patterns.append(
                    Pattern(
                        id=str(row.get("id", "")),
                        created_at=str(row.get("created_at", "")),
                        pattern_type=str(row.get("pattern_type", "")),
                        description=str(row.get("description", "")),
                        trigger=str(row.get("trigger", "")),
                        solution=str(row.get("solution", "")),
                        source_traces=list(row.get("source_traces", [])),
                        confidence=float(row.get("confidence", 0.0)),
                        embedding=list(row.get("embedding", [])) if row.get("embedding") is not None else None,
                    )
                )
            return Ok(patterns)
        except Exception as exc:
            return Err(
                ConfigurationError(
                    "Failed to retrieve patterns", context={"error": str(exc)}
                )
            )


def get_pattern_store(config: AppConfig) -> Result[PatternStore, JPScriptsError]:
    """Get the pattern store for the given configuration."""
    resolved_store = _resolve_store(config)

    deps = _load_lancedb_dependencies()
    if deps is None:
        return Err(
            CapabilityMissingError(
                "LanceDB is required for pattern storage. Install with `pip install \"jpscripts[ai]\"`.",
                context={"path": str(resolved_store)},
            )
        )

    lancedb_module, lance_model_base = deps
    try:
        store = PatternStore(resolved_store, lancedb_module, lance_model_base)
        return Ok(store)
    except Exception as exc:
        return Err(
            ConfigurationError(
                f"Failed to initialize pattern store: {exc}",
                context={"path": str(resolved_store)},
            )
        )


async def consolidate_patterns(
    config: AppConfig,
    *,
    trace_limit: int = 50,
    model: str | None = None,
) -> Result[list[Pattern], JPScriptsError]:
    """
    Extract generalized patterns from successful execution traces.

    This function:
    1. Loads the last N successful trace steps from trace_dir
    2. Groups similar traces by error type and solution approach
    3. Uses the LLM to extract generalized patterns
    4. Stores patterns in the dedicated patterns LanceDB collection

    Args:
        config: Application configuration
        trace_limit: Maximum number of traces to analyze
        model: Model to use for pattern extraction

    Returns:
        List of extracted patterns
    """
    trace_dir = Path(config.trace_dir).expanduser()
    if not trace_dir.exists():
        return Err(ConfigurationError("Trace directory not found", context={"path": str(trace_dir)}))

    # Load successful traces
    traces = await _load_successful_traces(trace_dir, trace_limit)
    if not traces:
        return Ok([])

    # Get pattern store
    match get_pattern_store(config):
        case Err(err):
            return Err(err)
        case Ok(store):
            pass

    # Get provider for synthesis
    model_id = model or config.default_model
    try:
        provider = get_provider(config, model_id=model_id)
    except Exception as exc:
        return Err(ConfigurationError("Failed to initialize provider", context={"error": str(exc)}))

    # Group similar traces
    clusters = _cluster_traces_by_similarity(traces)

    # Extract patterns from clusters
    patterns: list[Pattern] = []
    use_semantic, model_name, server_url = _embedding_settings(config)
    embedding_client = EmbeddingClient(model_name, enabled=use_semantic, server_url=server_url)

    for cluster in clusters:
        if len(cluster) < 2:  # Need multiple examples to generalize
            continue

        pattern = await _synthesize_pattern_from_cluster(cluster, provider, model_id)
        if pattern is not None:
            # Add embedding for retrieval
            if use_semantic:
                embed_text = f"{pattern.description} {pattern.trigger} {pattern.solution}"
                vectors = embedding_client.embed([embed_text])
                if vectors:
                    pattern = Pattern(
                        id=pattern.id,
                        created_at=pattern.created_at,
                        pattern_type=pattern.pattern_type,
                        description=pattern.description,
                        trigger=pattern.trigger,
                        solution=pattern.solution,
                        source_traces=pattern.source_traces,
                        confidence=pattern.confidence,
                        embedding=vectors[0],
                    )

            # Store pattern
            match store.add(pattern):
                case Err(err):
                    logger.warning("Failed to store pattern: %s", err)
                case Ok(_):
                    patterns.append(pattern)

    return Ok(patterns)


async def _load_successful_traces(trace_dir: Path, limit: int) -> list[dict[str, Any]]:
    """Load trace steps that resulted in successful outcomes."""
    traces: list[dict[str, Any]] = []

    trace_files = sorted(trace_dir.glob("*.jsonl"), reverse=True)
    for trace_file in trace_files[: limit * 2]:  # Over-fetch to filter
        try:
            raw = await asyncio.to_thread(trace_file.read_text, encoding="utf-8")
        except OSError:
            continue

        for line in raw.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                response = data.get("response", {})
                # Check for success indicators: has patch, no error
                if response.get("file_patch") and not response.get("error"):
                    traces.append(data)
            except json.JSONDecodeError:
                continue

        if len(traces) >= limit:
            break

    return traces[:limit]


def _cluster_traces_by_similarity(traces: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group traces by similarity of error type and solution approach.

    Uses simple heuristics: group by first word of thought_process
    (usually indicates the type of issue being addressed).
    """
    clusters: dict[str, list[dict[str, Any]]] = {}

    for trace in traces:
        response = trace.get("response", {})
        thought = response.get("thought_process", "")

        # Extract key from first significant word
        words = thought.split()[:3]
        key = " ".join(words).lower() if words else "other"

        if key not in clusters:
            clusters[key] = []
        clusters[key].append(trace)

    return list(clusters.values())


async def _synthesize_pattern_from_cluster(
    cluster: list[dict[str, Any]],
    provider: Any,
    model: str,
) -> Pattern | None:
    """Use LLM to extract a generalized pattern from a cluster of similar traces."""
    examples = []
    for trace in cluster[:5]:  # Limit to 5 examples
        response = trace.get("response", {})
        input_history = trace.get("input_history", [])
        context = input_history[-1].get("content", "") if input_history else ""
        examples.append(
            {
                "thought": response.get("thought_process", ""),
                "patch": response.get("file_patch", ""),
                "context_snippet": context[:500] if context else "",
            }
        )

    prompt = f"""Analyze these successful code fixes and extract a generalized pattern.

Examples of successful fixes:
{json.dumps(examples, indent=2)}

Extract a reusable pattern with:
1. pattern_type: Category (fix_pattern, refactor_pattern, test_pattern)
2. description: One sentence summary of what this pattern addresses
3. trigger: When to apply this pattern (what error/situation triggers it)
4. solution: Generic solution approach (not specific to one codebase)
5. confidence: 0.0-1.0 based on how consistent the examples are

Respond in JSON format only:
{{"pattern_type": "...", "description": "...", "trigger": "...", "solution": "...", "confidence": 0.8}}"""

    messages = [ProviderMessage(role="user", content=prompt)]
    options = CompletionOptions(temperature=0.3)

    try:
        response = await provider.complete(messages, model=model, options=options)
        content = response.content.strip()

        # Extract JSON from response
        if "```" in content:
            import re as regex

            json_match = regex.search(r"```(?:json)?\s*(.*?)```", content, regex.DOTALL)
            if json_match:
                content = json_match.group(1).strip()

        # Find JSON object
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            content = content[start:end]

        data = json.loads(content)
        return Pattern(
            id=uuid4().hex,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            pattern_type=str(data.get("pattern_type", "fix_pattern")),
            description=str(data.get("description", "")),
            trigger=str(data.get("trigger", "")),
            solution=str(data.get("solution", "")),
            source_traces=[trace.get("timestamp", "") for trace in cluster],
            confidence=float(data.get("confidence", 0.5)),
        )
    except Exception as exc:
        logger.debug("Pattern synthesis failed: %s", exc)
        return None


async def fetch_relevant_patterns(
    query: str,
    config: AppConfig,
    limit: int = 3,
    min_confidence: float = 0.6,
) -> list[Pattern]:
    """Fetch patterns relevant to the current task for prompt injection."""
    match get_pattern_store(config):
        case Err(_):
            return []
        case Ok(store):
            pass

    use_semantic, model_name, server_url = _embedding_settings(config)
    embedding_client = EmbeddingClient(model_name, enabled=use_semantic, server_url=server_url)

    query_vecs = embedding_client.embed([query]) if use_semantic else None
    query_vec = query_vecs[0] if query_vecs else None

    match store.search(query_vec, limit * 2):  # Over-fetch to filter by confidence
        case Err(_):
            return []
        case Ok(patterns):
            return [p for p in patterns if p.confidence >= min_confidence][:limit]


def format_patterns_for_prompt(patterns: list[Pattern]) -> str:
    """Format patterns as a section for injection into agent prompts."""
    if not patterns:
        return ""

    lines = ["## Learned Patterns", ""]
    for p in patterns:
        lines.append(f"### {p.pattern_type}: {p.description}")
        lines.append(f"**When:** {p.trigger}")
        lines.append(f"**Solution:** {p.solution}")
        lines.append(f"**Confidence:** {p.confidence:.0%}")
        lines.append("")

    return "\n".join(lines)
