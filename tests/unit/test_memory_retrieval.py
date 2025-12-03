"""
Unit tests for memory retrieval and clustering functions.

Tests cover cosine similarity, graph expansion, and clustering algorithms.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from jpscripts.memory.models import MemoryEntry
from jpscripts.memory.retrieval import (
    _cosine_similarity,
    _graph_expand,
    _simple_cluster,
)


def _make_entry(
    content: str = "test",
    embedding: list[float] | None = None,
    source_path: str | None = None,
    related_files: list[str] | None = None,
    tags: list[str] | None = None,
) -> MemoryEntry:
    """Helper to create test memory entries."""
    return MemoryEntry(
        id=uuid4().hex,
        ts=datetime.now(UTC).isoformat(timespec="seconds"),
        content=content,
        tags=tags or [],
        tokens=content.split(),
        embedding=embedding,
        source_path=source_path,
        related_files=related_files or [],
    )


class TestCosineSimilarity:
    """Test _cosine_similarity function."""

    def test_identical_vectors(self) -> None:
        """Identical vectors have similarity 1.0."""
        vec = [1.0, 2.0, 3.0]
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors have similarity 0.0."""
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        assert _cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        """Opposite vectors have similarity -1.0."""
        vec_a = [1.0, 1.0]
        vec_b = [-1.0, -1.0]
        assert _cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0)

    def test_empty_vectors(self) -> None:
        """Empty vectors return 0.0."""
        assert _cosine_similarity([], []) == 0.0

    def test_mismatched_length_returns_zero(self) -> None:
        """Vectors of different lengths return 0.0."""
        vec_a = [1.0, 2.0]
        vec_b = [1.0, 2.0, 3.0]
        assert _cosine_similarity(vec_a, vec_b) == 0.0

    def test_zero_vector_returns_zero(self) -> None:
        """Zero vector returns 0.0."""
        zero = [0.0, 0.0, 0.0]
        nonzero = [1.0, 2.0, 3.0]
        assert _cosine_similarity(zero, nonzero) == 0.0
        assert _cosine_similarity(nonzero, zero) == 0.0

    def test_similar_vectors(self) -> None:
        """Similar vectors have high similarity."""
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [1.1, 2.1, 3.1]
        sim = _cosine_similarity(vec_a, vec_b)
        assert sim > 0.99  # Very similar


class TestGraphExpand:
    """Test _graph_expand re-ranking function."""

    def test_empty_entries(self) -> None:
        """Empty input returns empty output."""
        assert _graph_expand([]) == []

    def test_single_entry(self) -> None:
        """Single entry is returned unchanged."""
        entry = _make_entry("test", source_path="a.py")
        result = _graph_expand([entry])
        assert result == [entry]

    def test_shared_files_boost(self) -> None:
        """Entries sharing files with top result are boosted."""
        top = _make_entry("top", source_path="main.py", related_files=["util.py"])
        related = _make_entry("related", source_path="other.py", related_files=["util.py"])
        unrelated = _make_entry("unrelated", source_path="alone.py", related_files=["random.py"])

        result = _graph_expand([top, unrelated, related])

        # Related should be boosted above unrelated
        assert result[0] == top
        # The related entry should be ranked higher than unrelated
        related_idx = result.index(related)
        unrelated_idx = result.index(unrelated)
        assert related_idx < unrelated_idx

    def test_same_source_path_boost(self) -> None:
        """Entries with same source path as top are boosted."""
        top = _make_entry("top", source_path="main.py")
        same_source = _make_entry("same", source_path="main.py")
        diff_source = _make_entry("diff", source_path="other.py")

        result = _graph_expand([top, diff_source, same_source])

        # same_source should be boosted higher
        same_idx = result.index(same_source)
        diff_idx = result.index(diff_source)
        assert same_idx < diff_idx

    def test_source_in_related_boost(self) -> None:
        """Entries where top's source appears in related_files are boosted."""
        top = _make_entry("top", source_path="main.py")
        mentions_top = _make_entry("mentions", related_files=["main.py", "util.py"])
        no_mention = _make_entry("nomention", related_files=["other.py"])

        result = _graph_expand([top, no_mention, mentions_top])

        # mentions_top should be boosted
        mentions_idx = result.index(mentions_top)
        no_mention_idx = result.index(no_mention)
        assert mentions_idx < no_mention_idx


class TestSimpleCluster:
    """Test _simple_cluster fallback clustering."""

    def test_empty_input(self) -> None:
        """Empty input returns empty clusters."""
        assert _simple_cluster([], 0.85) == []

    def test_no_embeddings(self) -> None:
        """Entries without embeddings are skipped."""
        entries = [
            _make_entry("a", embedding=None),
            _make_entry("b", embedding=None),
        ]
        assert _simple_cluster(entries, 0.85) == []

    def test_single_entry_no_cluster(self) -> None:
        """Single entry doesn't form a cluster (need >= 2)."""
        entries = [_make_entry("a", embedding=[1.0, 0.0])]
        assert _simple_cluster(entries, 0.85) == []

    def test_two_identical_embeddings_cluster(self) -> None:
        """Two entries with identical embeddings form a cluster."""
        emb = [1.0, 0.0, 0.0]
        entries = [
            _make_entry("a", embedding=emb),
            _make_entry("b", embedding=emb),
        ]
        clusters = _simple_cluster(entries, 0.85)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_dissimilar_no_cluster(self) -> None:
        """Dissimilar entries don't cluster."""
        entries = [
            _make_entry("a", embedding=[1.0, 0.0, 0.0]),
            _make_entry("b", embedding=[0.0, 1.0, 0.0]),
        ]
        # Orthogonal vectors have 0.0 similarity
        clusters = _simple_cluster(entries, 0.85)
        assert len(clusters) == 0

    def test_multiple_clusters(self) -> None:
        """Multiple distinct clusters are formed."""
        # Cluster 1: similar embeddings
        emb1 = [1.0, 0.0, 0.0]
        # Cluster 2: different direction
        emb2 = [0.0, 1.0, 0.0]

        entries = [
            _make_entry("a1", embedding=emb1),
            _make_entry("a2", embedding=[0.99, 0.01, 0.0]),  # Similar to emb1
            _make_entry("b1", embedding=emb2),
            _make_entry("b2", embedding=[0.01, 0.99, 0.0]),  # Similar to emb2
        ]
        clusters = _simple_cluster(entries, 0.9)

        # Should have 2 clusters
        assert len(clusters) == 2

    def test_threshold_boundary(self) -> None:
        """Entries at threshold boundary cluster correctly."""
        # Create embeddings with ~0.9 similarity
        emb1 = [1.0, 0.0]
        emb2 = [0.95, 0.312]  # Approximately 0.95 similarity

        entries = [
            _make_entry("a", embedding=emb1),
            _make_entry("b", embedding=emb2),
        ]

        # Should cluster at 0.9 threshold
        high_threshold = _simple_cluster(entries, 0.99)
        low_threshold = _simple_cluster(entries, 0.9)

        # High threshold: no cluster (similarity ~0.95 < 0.99)
        assert len(high_threshold) == 0
        # Low threshold: should cluster
        assert len(low_threshold) == 1


class TestVectorizedCluster:
    """Test _vectorized_cluster with numpy."""

    def test_vectorized_cluster_empty(self) -> None:
        """Empty input returns empty clusters."""
        from jpscripts.memory.retrieval import _vectorized_cluster

        assert _vectorized_cluster([], 0.85) == []

    def test_vectorized_cluster_no_embeddings(self) -> None:
        """Entries without embeddings return empty clusters."""
        from jpscripts.memory.retrieval import _vectorized_cluster

        entries = [
            _make_entry("a", embedding=None),
            _make_entry("b", embedding=None),
        ]
        assert _vectorized_cluster(entries, 0.85) == []

    def test_vectorized_cluster_identical(self) -> None:
        """Identical embeddings cluster together."""
        from jpscripts.memory.retrieval import _vectorized_cluster

        emb = [1.0, 0.0, 0.0]
        entries = [
            _make_entry("a", embedding=emb),
            _make_entry("b", embedding=emb),
        ]
        clusters = _vectorized_cluster(entries, 0.85)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_vectorized_matches_simple(self) -> None:
        """Vectorized clustering matches simple clustering results."""
        from jpscripts.memory.retrieval import _vectorized_cluster

        # Create test data that should produce same clusters
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]

        entries = [
            _make_entry("a1", embedding=emb1),
            _make_entry("a2", embedding=[0.99, 0.01, 0.0]),
            _make_entry("b1", embedding=emb2),
            _make_entry("b2", embedding=[0.01, 0.99, 0.0]),
        ]

        simple_clusters = _simple_cluster(entries, 0.9)
        vectorized_clusters = _vectorized_cluster(entries, 0.9)

        # Should have same number of clusters
        assert len(simple_clusters) == len(vectorized_clusters)

        # Check cluster sizes match
        simple_sizes = sorted(len(c) for c in simple_clusters)
        vectorized_sizes = sorted(len(c) for c in vectorized_clusters)
        assert simple_sizes == vectorized_sizes
