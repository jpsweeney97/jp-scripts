"""Embedding client for memory operations.

This module provides the EmbeddingClient for generating text embeddings
using either a remote embedding server or local SentenceTransformer models.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from typing import TYPE_CHECKING, TypeVar
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger

from .models import EmbeddingClientProtocol, MemoryEntry

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)
T = TypeVar("T")

_semantic_warned = False


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def _run_coroutine(coro: Coroutine[object, object, T]) -> T | None:
    """Run a coroutine, handling the case where an event loop is already running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if loop.is_running():  # pragma: no cover - defensive; embed calls run in worker threads
        logger.debug("Async loop already running; skipping coroutine execution.")
        return None

    return loop.run_until_complete(coro)


def _normalize_server_url(raw: str | None) -> str | None:
    """Normalize a server URL, adding http:// if needed."""
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
    """Check if a port is reachable."""
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
    except (TimeoutError, OSError):
        return False
    try:
        writer.close()
        await writer.wait_closed()
    except Exception:  # pragma: no cover - best effort cleanup
        pass
    return True


def _check_server_online(host: str, port: int) -> bool:
    """Check if the embedding server is reachable."""
    result = _run_coroutine(_async_check_port(host, port))
    return bool(result)


def _post_json(
    url: str, payload: dict[str, object], timeout: float = 2.0
) -> dict[str, object] | None:
    """POST JSON to a URL and return the response."""
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = urllib_request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (
        urllib_error.HTTPError,
        urllib_error.URLError,
        TimeoutError,
        OSError,
        ValueError,
    ) as exc:
        logger.debug("Embedding HTTP request failed: %s", exc)
        return None

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        logger.debug("Embedding HTTP response parse failed: %s", exc)
        return None
    return parsed if isinstance(parsed, dict) else None


async def _async_post_json(
    url: str, payload: dict[str, object], timeout: float = 2.0
) -> dict[str, object] | None:
    """Async wrapper for JSON POST."""
    return await asyncio.to_thread(_post_json, url, payload, timeout)


def _extract_embeddings(payload: dict[str, object]) -> list[list[float]] | None:
    """Extract embedding vectors from API response."""
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


def _warn_semantic_unavailable() -> None:
    """Warn user about missing semantic search dependencies (once)."""
    global _semantic_warned
    if _semantic_warned:
        return
    logger.warning("Semantic memory search unavailable. Install with `pip install jpscripts[ai]`.")
    _semantic_warned = True


def _embedding_settings(config: AppConfig | None) -> tuple[bool, str, str | None]:
    """Extract embedding settings from configuration."""
    use_semantic = True
    model_name = "all-MiniLM-L6-v2"
    server_url: str | None = None
    if config:
        use_semantic = bool(getattr(config, "use_semantic_search", True))
        model_name = getattr(config, "memory_model", model_name)
        server_url = getattr(config, "embedding_server_url", None)
    return use_semantic, model_name, server_url


def _compose_embedding_text(entry: MemoryEntry) -> str:
    """Compose text for embedding from a memory entry."""
    tags = " ".join(entry.tags)
    return f"{entry.content} {tags}".strip()


def _load_sentence_transformer(model_name: str) -> tuple[SentenceTransformer | None, int | None]:
    """Load a SentenceTransformer model with caching."""
    try:
        from sentence_transformers import SentenceTransformer as STModel
    except ImportError:
        _warn_semantic_unavailable()
        return None, None

    from pathlib import Path

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


# -----------------------------------------------------------------------------
# Embedding Client
# -----------------------------------------------------------------------------


class _GlobalEmbeddingClient(EmbeddingClientProtocol):
    """Global singleton embedding client.

    Supports both remote embedding servers and local SentenceTransformer models.
    """

    _instance: _GlobalEmbeddingClient | None = None

    def __init__(self, model_name: str, enabled: bool, server_url: str | None) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self._server_url = _normalize_server_url(server_url)
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None
        self._remote_available: bool | None = None

    @classmethod
    def get_instance(
        cls, model_name: str, enabled: bool, server_url: str | None
    ) -> _GlobalEmbeddingClient:
        """Get or create the singleton instance."""
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
        payload: dict[str, object] = {"input": texts, "model": self.model_name}
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


def EmbeddingClient(
    model_name: str, *, enabled: bool, server_url: str | None = None
) -> EmbeddingClientProtocol:
    """Factory function for creating an embedding client.

    Returns the global singleton instance, configured with the given parameters.
    """
    return _GlobalEmbeddingClient.get_instance(model_name, enabled, server_url)


__all__ = [
    "EmbeddingClient",
    "EmbeddingClientProtocol",
    "_GlobalEmbeddingClient",
    "_compose_embedding_text",
    "_embedding_settings",
    "_warn_semantic_unavailable",
]
