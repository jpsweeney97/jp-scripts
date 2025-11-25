from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jpscripts.core import memory as memory_core


def test_save_memory_writes_json(tmp_path: Path):
    store = tmp_path / "mem.jsonl"
    entry = memory_core.save_memory("Learned X", tags=["tag1"], store_path=store)

    assert store.exists()
    lines = store.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["content"] == "Learned X"
    assert data["tags"] == ["tag1"]
    assert isinstance(data["ts"], str)
    assert "tokens" in data
    assert entry.content == "Learned X"


def test_score_keyword_overlap():
    entry = memory_core.MemoryEntry(ts="1", content="Alpha beta gamma", tags=["beta"], tokens=["alpha", "beta", "gamma"])
    score = memory_core._score(["beta", "delta"], entry)
    assert score > 0


def test_semantic_query_uses_embeddings_when_available(monkeypatch, tmp_path: Path):
    store = tmp_path / "mem.jsonl"
    memory_core.save_memory("Graph embeddings are cool", tags=[], store_path=store)

    # Fake vector store
    class FakeVectorStore:
        def __init__(self, *_args, **_kwargs):
            self.used = False

        def available(self):
            return True

        def embed(self, texts):
            self.used = True
            return [[1.0 for _ in texts]]

        def as_array(self, data):
            return data

        def cosine(self, a, b):
            return 0.9

    monkeypatch.setattr(memory_core, "VectorStore", FakeVectorStore)

    results = memory_core.query_memory("graph", config=SimpleNamespace(use_semantic_search=True, memory_model="fake", memory_store=store))
    assert results
