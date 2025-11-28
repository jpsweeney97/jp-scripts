from __future__ import annotations

import json
from pathlib import Path

from jpscripts.core import memory as memory_core
from jpscripts.core.config import AppConfig
from jpscripts.core.result import Ok


def _dummy_config(store: Path, use_semantic: bool = False) -> AppConfig:
    return AppConfig(memory_store=store, use_semantic_search=use_semantic, memory_model="fake-model")


def test_save_memory_writes_fallback(tmp_path: Path) -> None:
    store = tmp_path / "mem.lance"
    fallback = store.with_suffix(".jsonl")
    entry = memory_core.save_memory("Learned X", tags=["tag1"], config=_dummy_config(store), store_path=store)

    assert fallback.exists()
    content = fallback.read_text(encoding="utf-8").strip().splitlines()
    assert content
    record = json.loads(content[-1])
    assert record["content"] == "Learned X"
    assert record["tags"] == ["tag1"]
    assert entry.content == "Learned X"


def test_score_keyword_overlap() -> None:
    entry = memory_core.MemoryEntry(
        id="1",
        ts="1",
        content="Alpha beta gamma",
        tags=["beta"],
        tokens=["alpha", "beta", "gamma"],
    )
    score = memory_core._score(["beta", "delta"], entry)
    assert score > 0


def test_query_memory_prefers_vector_results(monkeypatch, tmp_path: Path) -> None:
    store = tmp_path / "mem.lance"
    fallback = store.with_suffix(".jsonl")
    base_entry = memory_core.MemoryEntry(
        id="base",
        ts="now",
        content="placeholder",
        tags=[],
        tokens=["placeholder"],
    )
    memory_core._write_entries(fallback, [base_entry])

    class FakeEmbeddingClient:
        def __init__(self, *_args, **_kwargs) -> None:
            self.called = False

        @property
        def dimension(self) -> int | None:
            return 2

        def available(self) -> bool:
            return True

        def embed(self, texts):  # type: ignore[override]
            self.called = True
            return [[0.1, 0.2] for _ in texts]

    class FakeStore:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def add(self, entry: memory_core.MemoryEntry) -> Ok[memory_core.MemoryEntry]:
            return Ok(entry)

        def search(self, _vector, _limit: int, *, query_tokens=None):  # type: ignore[override]
            return Ok(
                [
                    memory_core.MemoryEntry(
                        id="hit",
                        ts="later",
                        content="vector match",
                        tags=["hit"],
                        tokens=["vector", "match"],
                    )
                ]
            )

        def prune(self, _root):
            return Ok(0)

    monkeypatch.setattr(memory_core, "_load_lancedb_dependencies", lambda: ("db", object))
    monkeypatch.setattr(memory_core, "EmbeddingClient", FakeEmbeddingClient)
    monkeypatch.setattr(memory_core, "LanceDBStore", FakeStore)

    results = memory_core.query_memory("vector", config=_dummy_config(store, use_semantic=True), store_path=store)
    assert results
    assert "vector match" in results[0]


def test_query_memory_rrf_combines_vector_and_keyword(monkeypatch, tmp_path: Path) -> None:
    store = tmp_path / "mem.lance"
    fallback = store.with_suffix(".jsonl")

    vector_entry = memory_core.MemoryEntry(
        id="vec",
        ts="1",
        content="vector only",
        tags=["vec"],
        tokens=["alpha"],
        embedding=[0.1, 0.2],
    )
    keyword_entry = memory_core.MemoryEntry(
        id="kw",
        ts="2",
        content="keyword only",
        tags=["kw"],
        tokens=["banana", "split"],
        embedding=None,
    )

    memory_core._write_entries(fallback, [vector_entry, keyword_entry])

    class FakeEmbeddingClient:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        @property
        def dimension(self) -> int | None:
            return 2

        def available(self) -> bool:
            return True

        def embed(self, texts):  # type: ignore[override]
            return [[0.5, 0.5] for _ in texts]

    class FakeStore:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def add(self, entry: memory_core.MemoryEntry) -> Ok[memory_core.MemoryEntry]:
            return Ok(entry)

        def search(self, _vector, _limit: int, *, query_tokens=None):  # type: ignore[override]
            return Ok([vector_entry])

        def prune(self, _root):
            return Ok(0)

    monkeypatch.setattr(memory_core, "_load_lancedb_dependencies", lambda: ("db", object))
    monkeypatch.setattr(memory_core, "EmbeddingClient", FakeEmbeddingClient)
    monkeypatch.setattr(memory_core, "LanceDBStore", FakeStore)

    results = memory_core.query_memory("banana", config=_dummy_config(store, use_semantic=True), store_path=store, limit=5)
    assert results
    assert any("vector only" in item for item in results)
    assert any("keyword only" in item for item in results)
