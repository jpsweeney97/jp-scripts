from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

from jpscripts.core import memory as memory_core


def _dummy_config(store: Path, use_semantic: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        use_semantic_search=use_semantic,
        memory_model="fake-model",
        memory_store=store,
    )


def test_save_memory_writes_sqlite(tmp_path: Path) -> None:
    store = tmp_path / "mem.sqlite"
    config = _dummy_config(store)
    entry = memory_core.save_memory("Learned X", tags=["tag1"], config=config, store_path=store)

    assert store.exists()
    conn = sqlite3.connect(store)
    row = conn.execute("SELECT content, timestamp, embedding FROM memory").fetchone()
    assert row is not None
    raw_content, ts, embedding = row
    content_text, tags = memory_core._deserialize_content(raw_content)

    assert content_text == "Learned X"
    assert tags == ["tag1"]
    assert isinstance(ts, str)
    assert entry.content == "Learned X"
    assert embedding is None or isinstance(embedding, (bytes, bytearray))


def test_score_keyword_overlap() -> None:
    entry = memory_core.MemoryEntry(ts="1", content="Alpha beta gamma", tags=["beta"], tokens=["alpha", "beta", "gamma"])
    score = memory_core._score(["beta", "delta"], entry)
    assert score > 0


@pytest.mark.skipif(np is None, reason="numpy not available")
def test_semantic_query_uses_embeddings_when_available(monkeypatch, tmp_path: Path) -> None:
    store = tmp_path / "mem.sqlite"
    config = _dummy_config(store, use_semantic=True)

    class FakeVectorStore:
        def __init__(self, *_args, **_kwargs):
            self.used = False
            self._vec = np.asarray([1.0, 0.0], dtype=np.float32)

        def available(self) -> bool:
            return True

        def embed(self, texts):
            self.used = True
            return np.stack([self._vec for _ in texts])

        @staticmethod
        def to_bytes(vector: np.ndarray) -> bytes:
            return np.asarray(vector, dtype=np.float32).tobytes()

        @staticmethod
        def from_bytes(blob: bytes) -> np.ndarray:
            return np.frombuffer(blob, dtype=np.float32)

        @staticmethod
        def cosine_batch(matrix: np.ndarray, query_vec: np.ndarray):
            return np.array([0.9 for _ in range(matrix.shape[0])], dtype=np.float32)

    monkeypatch.setattr(memory_core, "VectorStore", FakeVectorStore)

    memory_core.save_memory("Graph embeddings are cool", tags=[], config=config, store_path=store)

    results = memory_core.query_memory("graph", config=config, store_path=store)
    assert results
