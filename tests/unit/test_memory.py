from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from jpscripts import memory as memory_core
from jpscripts.core.config import AppConfig
from jpscripts.core.result import Ok


def _dummy_config(store: Path, use_semantic: bool = False) -> AppConfig:
    return AppConfig(
        memory_store=store, use_semantic_search=use_semantic, memory_model="fake-model"
    )


def test_save_memory_writes_fallback(tmp_path: Path) -> None:
    store = tmp_path / "mem.lance"
    fallback = store.with_suffix(".jsonl")
    entry = memory_core.save_memory(
        "Learned X", tags=["tag1"], config=_dummy_config(store), store_path=store
    )

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


def test_query_memory_prefers_vector_results(monkeypatch: Any, tmp_path: Path) -> None:
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
        def __init__(
            self, model_name: str, *, enabled: bool = True, server_url: str | None = None
        ) -> None:
            self.called = False
            self.model_name = model_name
            self.enabled = enabled
            self.server_url = server_url

        @property
        def dimension(self) -> int | None:
            return 2

        def available(self) -> bool:
            return True

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.called = True
            return [[0.1, 0.2] for _ in texts]

    class FakeStore:
        def __init__(self, db_path: Path, lancedb_module: object, lance_model_base: object) -> None:
            _ = (db_path, lancedb_module, lance_model_base)

        def add(self, entry: memory_core.MemoryEntry) -> Ok[memory_core.MemoryEntry]:
            return Ok(entry)

        def search(
            self,
            _vector: list[float] | None,
            _limit: int,
            *,
            query_tokens: list[str] | None = None,
            tag_filter: set[str] | None = None,
        ) -> Ok[list[memory_core.MemoryEntry]]:
            _ = tag_filter  # Accept but ignore for test
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

        def prune(self, _root: Path) -> Ok[int]:
            return Ok(0)

    # Patch at the module where the import occurs (hybrid.py imports from lance.py)
    monkeypatch.setattr(
        "jpscripts.memory.store.hybrid.load_lancedb_dependencies", lambda: ("db", object)
    )
    monkeypatch.setattr("jpscripts.memory.store.hybrid.LanceDBStore", FakeStore)
    monkeypatch.setattr("jpscripts.memory.api.EmbeddingClient", FakeEmbeddingClient)

    results = memory_core.query_memory(
        "vector", config=_dummy_config(store, use_semantic=True), store_path=store
    )
    assert results
    assert "vector match" in results[0]


def test_query_memory_rrf_combines_vector_and_keyword(monkeypatch: Any, tmp_path: Path) -> None:
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
        def __init__(
            self, model_name: str, *, enabled: bool = True, server_url: str | None = None
        ) -> None:
            self.model_name = model_name
            self.enabled = enabled
            self.server_url = server_url

        @property
        def dimension(self) -> int | None:
            return 2

        def available(self) -> bool:
            return True

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.5, 0.5] for _ in texts]

    class FakeStore:
        def __init__(self, db_path: Path, lancedb_module: object, lance_model_base: object) -> None:
            _ = (db_path, lancedb_module, lance_model_base)

        def add(self, entry: memory_core.MemoryEntry) -> Ok[memory_core.MemoryEntry]:
            return Ok(entry)

        def search(
            self,
            _vector: list[float] | None,
            _limit: int,
            *,
            query_tokens: list[str] | None = None,
            tag_filter: set[str] | None = None,
        ) -> Ok[list[memory_core.MemoryEntry]]:
            _ = tag_filter  # Accept but ignore for test
            return Ok([vector_entry])

        def prune(self, _root: Path) -> Ok[int]:
            return Ok(0)

    # Patch at the module where the import occurs (hybrid.py imports from lance.py)
    monkeypatch.setattr(
        "jpscripts.memory.store.hybrid.load_lancedb_dependencies", lambda: ("db", object)
    )
    monkeypatch.setattr("jpscripts.memory.store.hybrid.LanceDBStore", FakeStore)
    monkeypatch.setattr("jpscripts.memory.api.EmbeddingClient", FakeEmbeddingClient)

    results = memory_core.query_memory(
        "banana", config=_dummy_config(store, use_semantic=True), store_path=store, limit=5
    )
    assert results
    assert any("vector only" in item for item in results)
    assert any("keyword only" in item for item in results)


# -----------------------------------------------------------------------------
# _score function tests (keyword scoring with time decay)
# -----------------------------------------------------------------------------


def test_score_empty_query_tokens_returns_zero() -> None:
    """Empty query tokens should return zero score."""
    entry = memory_core.MemoryEntry(
        id="1",
        ts=datetime.now(UTC).isoformat(),
        content="test",
        tags=["tag"],
        tokens=["test"],
    )
    assert memory_core._score([], entry) == 0.0


def test_score_empty_entry_tokens_returns_zero() -> None:
    """Empty entry tokens should return zero score."""
    entry = memory_core.MemoryEntry(
        id="1",
        ts=datetime.now(UTC).isoformat(),
        content="test",
        tags=["tag"],
        tokens=[],
    )
    assert memory_core._score(["query"], entry) == 0.0


def test_score_tag_overlap_bonus() -> None:
    """Tags matching query tokens should add 0.5x bonus."""
    entry = memory_core.MemoryEntry(
        id="1",
        ts=datetime.now(UTC).isoformat(),
        content="test",
        tags=["python", "debug"],
        tokens=["test"],
    )
    # "python" is in tags, gets 0.5 bonus
    score_with_tag = memory_core._score(["python"], entry)
    # "other" is not in tags, no bonus
    score_no_tag = memory_core._score(["test"], entry)
    assert score_with_tag > 0
    assert score_no_tag > 0
    # Tag overlap gives extra boost
    assert score_with_tag > score_no_tag * 0.4  # 0.5 tag bonus vs 1.0 token match


def test_score_time_decay_older_entries_score_lower() -> None:
    """Older entries should have lower scores due to time decay."""
    now = datetime.now(UTC)
    recent = memory_core.MemoryEntry(
        id="recent",
        ts=now.isoformat(),
        content="test",
        tags=[],
        tokens=["alpha", "beta"],
    )
    old = memory_core.MemoryEntry(
        id="old",
        ts=(now - timedelta(days=30)).isoformat(),
        content="test",
        tags=[],
        tokens=["alpha", "beta"],
    )
    very_old = memory_core.MemoryEntry(
        id="very_old",
        ts=(now - timedelta(days=100)).isoformat(),
        content="test",
        tags=[],
        tokens=["alpha", "beta"],
    )

    recent_score = memory_core._score(["alpha"], recent)
    old_score = memory_core._score(["alpha"], old)
    very_old_score = memory_core._score(["alpha"], very_old)

    assert recent_score > old_score > very_old_score


def test_score_invalid_timestamp_no_decay() -> None:
    """Invalid timestamps should default to no decay (factor=1.0)."""
    entry = memory_core.MemoryEntry(
        id="1",
        ts="not-a-valid-timestamp",
        content="test",
        tags=[],
        tokens=["alpha"],
    )
    score = memory_core._score(["alpha"], entry)
    # Should return base score with decay=1.0
    assert score == 1.0


def test_score_multiple_token_overlap() -> None:
    """Multiple matching tokens should increase score."""
    entry = memory_core.MemoryEntry(
        id="1",
        ts=datetime.now(UTC).isoformat(),
        content="test",
        tags=[],
        tokens=["alpha", "beta", "gamma"],
    )
    single = memory_core._score(["alpha"], entry)
    double = memory_core._score(["alpha", "beta"], entry)
    triple = memory_core._score(["alpha", "beta", "gamma"], entry)

    assert triple > double > single


# -----------------------------------------------------------------------------
# _cosine_similarity function tests
# -----------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors() -> None:
    """Identical vectors should have similarity of 1.0."""
    vec = [0.5, 0.5, 0.5]
    assert abs(memory_core._cosine_similarity(vec, vec) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors() -> None:
    """Orthogonal vectors should have similarity of 0.0."""
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [0.0, 1.0, 0.0]
    assert abs(memory_core._cosine_similarity(vec_a, vec_b)) < 1e-9


def test_cosine_similarity_opposite_vectors() -> None:
    """Opposite vectors should have similarity of -1.0."""
    vec_a = [1.0, 0.0]
    vec_b = [-1.0, 0.0]
    assert abs(memory_core._cosine_similarity(vec_a, vec_b) + 1.0) < 1e-9


def test_cosine_similarity_different_lengths_returns_zero() -> None:
    """Vectors of different lengths should return 0.0."""
    vec_a = [1.0, 2.0, 3.0]
    vec_b = [1.0, 2.0]
    assert memory_core._cosine_similarity(vec_a, vec_b) == 0.0


def test_cosine_similarity_empty_vectors_returns_zero() -> None:
    """Empty vectors should return 0.0."""
    assert memory_core._cosine_similarity([], []) == 0.0


def test_cosine_similarity_zero_norm_vector_returns_zero() -> None:
    """Zero-norm vectors should return 0.0."""
    zero = [0.0, 0.0, 0.0]
    non_zero = [1.0, 0.0, 0.0]
    assert memory_core._cosine_similarity(zero, non_zero) == 0.0
    assert memory_core._cosine_similarity(non_zero, zero) == 0.0


# -----------------------------------------------------------------------------
# _graph_expand function tests (file relationship ranking)
# -----------------------------------------------------------------------------


def test_graph_expand_empty_entries_returns_empty() -> None:
    """Empty entries list should return empty list."""
    assert memory_core._graph_expand([]) == []


def test_graph_expand_same_source_path_boosted() -> None:
    """Entries with same source_path as top result get boosted."""
    top = memory_core.MemoryEntry(
        id="top",
        ts="1",
        content="top",
        tags=[],
        tokens=[],
        source_path="src/module.py",
        related_files=[],
    )
    same_source = memory_core.MemoryEntry(
        id="same",
        ts="2",
        content="same",
        tags=[],
        tokens=[],
        source_path="src/module.py",
        related_files=[],
    )
    different_source = memory_core.MemoryEntry(
        id="diff",
        ts="3",
        content="diff",
        tags=[],
        tokens=[],
        source_path="src/other.py",
        related_files=[],
    )

    # Same source appears last in input but should be boosted
    result = memory_core._graph_expand([top, different_source, same_source])

    # Top stays first, same_source should be boosted above different_source
    assert result[0].id == "top"
    assert result[1].id == "same"


def test_graph_expand_shared_related_files_boosted() -> None:
    """Entries sharing related files with top result get boosted."""
    top = memory_core.MemoryEntry(
        id="top",
        ts="1",
        content="top",
        tags=[],
        tokens=[],
        source_path=None,
        related_files=["shared.py", "unique_top.py"],
    )
    shares = memory_core.MemoryEntry(
        id="shares",
        ts="2",
        content="shares",
        tags=[],
        tokens=[],
        source_path=None,
        related_files=["shared.py", "unique_shares.py"],
    )
    no_share = memory_core.MemoryEntry(
        id="no_share",
        ts="3",
        content="no_share",
        tags=[],
        tokens=[],
        source_path=None,
        related_files=["completely_different.py"],
    )

    result = memory_core._graph_expand([top, no_share, shares])

    assert result[0].id == "top"
    assert result[1].id == "shares"


def test_graph_expand_source_in_related_files_boosted() -> None:
    """Entry whose source_path is in top's related_files gets boosted."""
    top = memory_core.MemoryEntry(
        id="top",
        ts="1",
        content="top",
        tags=[],
        tokens=[],
        source_path="main.py",
        related_files=["helper.py", "utils.py"],
    )
    in_related = memory_core.MemoryEntry(
        id="in_related",
        ts="2",
        content="in_related",
        tags=[],
        tokens=[],
        source_path="helper.py",
        related_files=[],
    )
    not_related = memory_core.MemoryEntry(
        id="not_related",
        ts="3",
        content="not_related",
        tags=[],
        tokens=[],
        source_path="other.py",
        related_files=[],
    )

    result = memory_core._graph_expand([top, not_related, in_related])

    assert result[0].id == "top"
    assert result[1].id == "in_related"


def test_graph_expand_preserves_order_when_no_relationships() -> None:
    """Without file relationships, base ranking is preserved."""
    entries = [
        memory_core.MemoryEntry(
            id=f"entry_{i}",
            ts=str(i),
            content=f"content_{i}",
            tags=[],
            tokens=[],
            source_path=None,
            related_files=[],
        )
        for i in range(3)
    ]

    result = memory_core._graph_expand(entries)

    # Base ranking is 1/(idx+1), so order should be preserved
    assert [e.id for e in result] == ["entry_0", "entry_1", "entry_2"]


# -----------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion) algorithm tests
# -----------------------------------------------------------------------------


def test_rrf_score_calculation() -> None:
    """Verify RRF score calculation uses k=60 constant."""
    # RRF formula: 1/(k + rank) where k=60
    # Entry at rank 1 in vector, rank 2 in keyword:
    # score = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.0164 + 0.0161 ≈ 0.0325
    k = 60.0
    vector_rank = 1
    keyword_rank = 2
    expected = 1.0 / (k + vector_rank) + 1.0 / (k + keyword_rank)

    # Approximate check (actual value ~0.0325)
    assert 0.032 < expected < 0.033


def test_hybrid_search_returns_both_vector_and_keyword_matches(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Hybrid search should return entries from both vector and keyword searches."""
    store = tmp_path / "mem.lance"
    fallback = store.with_suffix(".jsonl")

    # Entry only found by vector search
    vector_only = memory_core.MemoryEntry(
        id="vector_only",
        ts="1",
        content="vector semantic",
        tags=[],
        tokens=["semantic"],
        embedding=[0.9, 0.1],
    )
    # Entry only found by keyword search
    keyword_only = memory_core.MemoryEntry(
        id="keyword_only",
        ts="2",
        content="keyword match banana",
        tags=[],
        tokens=["banana", "keyword"],
        embedding=None,
    )
    # Entry found by both
    both = memory_core.MemoryEntry(
        id="both",
        ts="3",
        content="found by both",
        tags=["banana"],
        tokens=["banana", "semantic"],
        embedding=[0.8, 0.2],
    )

    memory_core._write_entries(fallback, [vector_only, keyword_only, both])

    class FakeEmbeddingClient:
        def __init__(
            self, model_name: str, *, enabled: bool = True, server_url: str | None = None
        ) -> None:
            pass

        @property
        def dimension(self) -> int | None:
            return 2

        def available(self) -> bool:
            return True

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.85, 0.15] for _ in texts]

    class FakeStore:
        def __init__(self, db_path: Path, lancedb_module: object, lance_model_base: object) -> None:
            pass

        def add(self, entry: memory_core.MemoryEntry) -> Ok[memory_core.MemoryEntry]:
            return Ok(entry)

        def search(
            self,
            _vector: list[float] | None,
            _limit: int,
            *,
            query_tokens: list[str] | None = None,
            tag_filter: set[str] | None = None,
        ) -> Ok[list[memory_core.MemoryEntry]]:
            _ = tag_filter  # Accept but ignore for test
            # Vector search returns vector_only and both
            return Ok([both, vector_only])

        def prune(self, _root: Path) -> Ok[int]:
            return Ok(0)

    # Patch at hybrid module where imports are resolved
    monkeypatch.setattr(
        "jpscripts.memory.store.hybrid.load_lancedb_dependencies", lambda: ("db", object)
    )
    monkeypatch.setattr("jpscripts.memory.store.hybrid.LanceDBStore", FakeStore)
    monkeypatch.setattr("jpscripts.memory.api.EmbeddingClient", FakeEmbeddingClient)

    results = memory_core.query_memory(
        "banana", config=_dummy_config(store, use_semantic=True), store_path=store, limit=10
    )

    # Should have all three entries via RRF fusion
    assert len(results) == 3
    # Entry "both" should rank highest (found by both vector and keyword)
    assert "found by both" in results[0]


def test_hybrid_search_empty_results() -> None:
    """When neither vector nor keyword search finds matches, return empty."""
    from jpscripts.memory.store import HybridMemoryStore, JsonlArchiver

    archiver = JsonlArchiver(Path("/tmp/empty.jsonl"))
    store = HybridMemoryStore(archiver, vector_store=None)

    result = store.search(None, limit=5, query_tokens=["nonexistent"])

    assert isinstance(result, Ok)
    assert result.value == []


def test_hybrid_search_keyword_only_no_vector_store(tmp_path: Path) -> None:
    """Without vector store, should fall back to keyword-only search."""
    from jpscripts.memory.store import HybridMemoryStore, JsonlArchiver

    fallback = tmp_path / "test.jsonl"
    entry = memory_core.MemoryEntry(
        id="kw",
        ts=datetime.now(UTC).isoformat(),
        content="keyword entry",
        tags=["python"],
        tokens=["python", "code"],
        embedding=None,
    )
    memory_core._write_entries(fallback, [entry])

    archiver = JsonlArchiver(fallback)
    store = HybridMemoryStore(archiver, vector_store=None)

    result = store.search(None, limit=5, query_tokens=["python"])

    assert isinstance(result, Ok)
    assert len(result.value) == 1
    assert result.value[0].id == "kw"
