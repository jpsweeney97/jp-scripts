"""Unit tests for memory store package."""

from __future__ import annotations

from pathlib import Path

from jpscripts.memory.models import MemoryEntry
from jpscripts.memory.store import (
    DEFAULT_STORE,
    FALLBACK_SUFFIX,
    MAX_ENTRIES,
    STOPWORDS,
    StorageMode,
    fallback_path,
    format_entry,
    resolve_store_path,
    score_entry,
    tokenize,
)


class TestStorageMode:
    """Tests for StorageMode enum."""

    def test_storage_mode_values(self) -> None:
        """StorageMode has expected values."""
        assert StorageMode.JSONL_ONLY.value == 1
        assert StorageMode.LANCE_ONLY.value == 2
        assert StorageMode.HYBRID.value == 3

    def test_storage_mode_by_name(self) -> None:
        """Can access StorageMode by name."""
        assert StorageMode["JSONL_ONLY"] == StorageMode.JSONL_ONLY
        assert StorageMode["HYBRID"] == StorageMode.HYBRID


class TestConstants:
    """Tests for module constants."""

    def test_default_store_is_home_based(self) -> None:
        """DEFAULT_STORE is in home directory."""
        assert str(DEFAULT_STORE).startswith(str(Path.home()))
        assert ".jp_memory.lance" in str(DEFAULT_STORE)

    def test_fallback_suffix_is_jsonl(self) -> None:
        """FALLBACK_SUFFIX is .jsonl."""
        assert FALLBACK_SUFFIX == ".jsonl"

    def test_max_entries_is_reasonable(self) -> None:
        """MAX_ENTRIES is a reasonable limit."""
        assert MAX_ENTRIES > 0
        assert MAX_ENTRIES == 5000

    def test_stopwords_contains_common_words(self) -> None:
        """STOPWORDS contains common stop words."""
        assert "the" in STOPWORDS
        assert "and" in STOPWORDS
        assert "def" in STOPWORDS
        assert "class" in STOPWORDS


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_fallback_path_changes_suffix(self) -> None:
        """fallback_path changes .lance to .jsonl."""
        base = Path("/tmp/test.lance")
        result = fallback_path(base)
        assert result == Path("/tmp/test.jsonl")

    def test_resolve_store_path_default(self) -> None:
        """resolve_store_path returns DEFAULT_STORE when no args."""
        result = resolve_store_path()
        assert result == DEFAULT_STORE

    def test_resolve_store_path_explicit(self) -> None:
        """resolve_store_path honors explicit path."""
        explicit = Path("/custom/store.lance")
        result = resolve_store_path(store_path=explicit)
        assert result == explicit

    def test_tokenize_filters_stopwords(self) -> None:
        """tokenize removes stopwords and single chars."""
        text = "the quick fox and a dog"
        tokens = tokenize(text)
        assert "the" not in tokens
        assert "and" not in tokens
        assert "a" not in tokens
        assert "quick" in tokens
        assert "fox" in tokens
        assert "dog" in tokens

    def test_tokenize_lowercases(self) -> None:
        """tokenize converts to lowercase."""
        text = "QuickBrown FOX"
        tokens = tokenize(text)
        assert "quickbrown" in tokens
        assert "fox" in tokens

    def test_format_entry_includes_content(self) -> None:
        """format_entry includes content."""
        entry = MemoryEntry(
            id="1",
            ts="2024-01-01",
            content="Test content",
            tags=["tag1"],
            tokens=["test"],
        )
        result = format_entry(entry)
        assert "Test content" in result
        assert "2024-01-01" in result
        assert "[tag1]" in result

    def test_format_entry_no_tags(self) -> None:
        """format_entry works without tags."""
        entry = MemoryEntry(
            id="1",
            ts="2024-01-01",
            content="Test content",
            tags=[],
            tokens=["test"],
        )
        result = format_entry(entry)
        assert "Test content" in result
        assert "2024-01-01" in result
        assert "[" not in result

    def test_score_entry_zero_for_no_overlap(self) -> None:
        """score_entry returns 0 for no token overlap."""
        entry = MemoryEntry(
            id="1",
            ts="2024-01-01",
            content="foo bar",
            tags=[],
            tokens=["foo", "bar"],
        )
        score = score_entry(["baz", "qux"], entry)
        assert score == 0.0

    def test_score_entry_positive_for_overlap(self) -> None:
        """score_entry returns positive for token overlap."""
        entry = MemoryEntry(
            id="1",
            ts="2024-01-01",
            content="foo bar",
            tags=[],
            tokens=["foo", "bar"],
        )
        score = score_entry(["foo", "baz"], entry)
        assert score > 0.0

    def test_score_entry_tag_bonus(self) -> None:
        """score_entry gives bonus for tag matches."""
        entry_no_tag = MemoryEntry(
            id="1",
            ts="2024-01-01",
            content="foo bar",
            tags=[],
            tokens=["foo", "bar"],
        )
        entry_with_tag = MemoryEntry(
            id="2",
            ts="2024-01-01",
            content="foo bar",
            tags=["baz"],
            tokens=["foo", "bar"],
        )
        score_no_tag = score_entry(["foo", "baz"], entry_no_tag)
        score_with_tag = score_entry(["foo", "baz"], entry_with_tag)
        # Tag match should boost score
        assert score_with_tag > score_no_tag


class TestComputeFileHash:
    """Tests for compute_file_hash function."""

    def test_compute_hash_returns_md5(self, tmp_path: Path) -> None:
        """compute_file_hash returns MD5 hash of file content."""
        from jpscripts.memory.store import compute_file_hash

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world", encoding="utf-8")
        result = compute_file_hash(test_file)
        assert result is not None
        assert len(result) == 32  # MD5 hex digest is 32 chars

    def test_compute_hash_returns_none_for_missing(self, tmp_path: Path) -> None:
        """compute_file_hash returns None for missing files."""
        from jpscripts.memory.store import compute_file_hash

        missing_file = tmp_path / "does_not_exist.txt"
        result = compute_file_hash(missing_file)
        assert result is None

    def test_compute_hash_different_content(self, tmp_path: Path) -> None:
        """compute_file_hash returns different hashes for different content."""
        from jpscripts.memory.store import compute_file_hash

        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content one", encoding="utf-8")
        file2.write_text("content two", encoding="utf-8")
        hash1 = compute_file_hash(file1)
        hash2 = compute_file_hash(file2)
        assert hash1 != hash2


class TestParseEntry:
    """Tests for parse_entry function."""

    def test_parse_entry_basic(self) -> None:
        """parse_entry creates MemoryEntry from raw dict."""
        from jpscripts.memory.store import parse_entry

        raw = {
            "id": "test-id",
            "ts": "2024-01-01",
            "content": "test content",
            "tags": ["tag1", "tag2"],
        }
        entry = parse_entry(raw)
        assert entry.id == "test-id"
        assert entry.content == "test content"
        assert entry.tags == ["tag1", "tag2"]

    def test_parse_entry_with_embedding(self) -> None:
        """parse_entry preserves embedding data."""
        from jpscripts.memory.store import parse_entry

        raw = {
            "id": "test-id",
            "ts": "2024-01-01",
            "content": "test",
            "tags": [],
            "embedding": [0.1, 0.2, 0.3],
        }
        entry = parse_entry(raw)
        assert entry.embedding == [0.1, 0.2, 0.3]

    def test_parse_entry_generates_tokens(self) -> None:
        """parse_entry generates tokens from content."""
        from jpscripts.memory.store import parse_entry

        raw = {
            "id": "test-id",
            "ts": "2024-01-01",
            "content": "hello world",
            "tags": [],
        }
        entry = parse_entry(raw)
        assert "hello" in entry.tokens
        assert "world" in entry.tokens


class TestIterEntries:
    """Tests for iter_entries function."""

    def test_iter_entries_empty_file(self, tmp_path: Path) -> None:
        """iter_entries yields nothing for empty file."""
        from jpscripts.memory.store import iter_entries

        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")
        entries = list(iter_entries(empty_file))
        assert entries == []

    def test_iter_entries_missing_file(self, tmp_path: Path) -> None:
        """iter_entries yields nothing for missing file."""
        from jpscripts.memory.store import iter_entries

        missing_file = tmp_path / "missing.jsonl"
        entries = list(iter_entries(missing_file))
        assert entries == []

    def test_iter_entries_skips_invalid_json(self, tmp_path: Path) -> None:
        """iter_entries skips invalid JSON lines."""
        from jpscripts.memory.store import iter_entries

        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text(
            '{"id":"1","ts":"2024","content":"valid","tags":[]}\n'
            "not valid json\n"
            '{"id":"2","ts":"2024","content":"also valid","tags":[]}\n',
            encoding="utf-8",
        )
        entries = list(iter_entries(jsonl_file))
        assert len(entries) == 2


class TestJsonlArchiver:
    """Tests for JsonlArchiver class."""

    def test_init_and_path_property(self, tmp_path: Path) -> None:
        """JsonlArchiver initializes with path."""
        from jpscripts.memory.store import JsonlArchiver

        path = tmp_path / "test.jsonl"
        archiver = JsonlArchiver(path)
        assert archiver.path == path

    def test_add_creates_entry(self, tmp_path: Path) -> None:
        """JsonlArchiver.add creates entry in file."""
        from jpscripts.core.result import Ok
        from jpscripts.memory.store import JsonlArchiver

        path = tmp_path / "test.jsonl"
        archiver = JsonlArchiver(path)
        entry = MemoryEntry(
            id="test-1",
            ts="2024-01-01",
            content="test content",
            tags=["tag1"],
            tokens=["test", "content"],
        )
        result = archiver.add(entry)
        assert isinstance(result, Ok)
        assert result.value == entry
        assert path.exists()
        loaded = archiver.load_entries()
        assert len(loaded) == 1
        assert loaded[0].content == "test content"

    def test_search_returns_empty_without_tokens(self, tmp_path: Path) -> None:
        """JsonlArchiver.search returns empty when no query tokens."""
        from jpscripts.core.result import Ok
        from jpscripts.memory.store import JsonlArchiver

        path = tmp_path / "test.jsonl"
        archiver = JsonlArchiver(path)
        result = archiver.search(None, 10)
        assert isinstance(result, Ok)
        assert result.value == []

    def test_search_finds_matching_entries(self, tmp_path: Path) -> None:
        """JsonlArchiver.search finds entries by keyword."""
        from jpscripts.core.result import Ok
        from jpscripts.memory.store import JsonlArchiver

        path = tmp_path / "test.jsonl"
        archiver = JsonlArchiver(path)
        entry = MemoryEntry(
            id="test-1",
            ts="2024-01-01",
            content="python programming",
            tags=[],
            tokens=["python", "programming"],
        )
        archiver.add(entry)
        result = archiver.search(None, 10, query_tokens=["python"])
        assert isinstance(result, Ok)
        assert len(result.value) == 1
        assert result.value[0].content == "python programming"

    def test_prune_empty_archive(self, tmp_path: Path) -> None:
        """JsonlArchiver.prune handles empty archive."""
        from jpscripts.core.result import Ok
        from jpscripts.memory.store import JsonlArchiver

        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        archiver = JsonlArchiver(path)
        result = archiver.prune(tmp_path)
        assert isinstance(result, Ok)
        assert result.value == 0

    def test_prune_keeps_entries_without_source(self, tmp_path: Path) -> None:
        """JsonlArchiver.prune keeps entries without source_path."""
        from jpscripts.core.result import Ok
        from jpscripts.memory.store import JsonlArchiver

        path = tmp_path / "test.jsonl"
        archiver = JsonlArchiver(path)
        entry = MemoryEntry(
            id="test-1",
            ts="2024-01-01",
            content="no source",
            tags=[],
            tokens=["no", "source"],
            source_path=None,
        )
        archiver.add(entry)
        result = archiver.prune(tmp_path)
        assert isinstance(result, Ok)
        assert result.value == 0
        loaded = archiver.load_entries()
        assert len(loaded) == 1

    def test_prune_removes_missing_files(self, tmp_path: Path) -> None:
        """JsonlArchiver.prune removes entries for missing files."""
        from jpscripts.core.result import Ok
        from jpscripts.memory.store import JsonlArchiver

        path = tmp_path / "test.jsonl"
        archiver = JsonlArchiver(path)
        entry = MemoryEntry(
            id="test-1",
            ts="2024-01-01",
            content="missing source",
            tags=[],
            tokens=["missing"],
            source_path=str(tmp_path / "nonexistent.py"),
        )
        archiver.add(entry)
        result = archiver.prune(tmp_path)
        assert isinstance(result, Ok)
        assert result.value == 1
        loaded = archiver.load_entries()
        assert len(loaded) == 0

    def test_prune_removes_hash_mismatch(self, tmp_path: Path) -> None:
        """JsonlArchiver.prune removes entries with content hash mismatch."""
        from jpscripts.core.result import Ok
        from jpscripts.memory.store import JsonlArchiver

        source_file = tmp_path / "source.py"
        source_file.write_text("original content", encoding="utf-8")
        path = tmp_path / "test.jsonl"
        archiver = JsonlArchiver(path)
        entry = MemoryEntry(
            id="test-1",
            ts="2024-01-01",
            content="drifted",
            tags=[],
            tokens=["drifted"],
            source_path=str(source_file),
            content_hash="wrong_hash_value",
        )
        archiver.add(entry)
        result = archiver.prune(tmp_path)
        assert isinstance(result, Ok)
        assert result.value == 1

    def test_prune_keeps_matching_hash(self, tmp_path: Path) -> None:
        """JsonlArchiver.prune keeps entries with matching hash."""
        from jpscripts.core.result import Ok
        from jpscripts.memory.store import JsonlArchiver, compute_file_hash

        source_file = tmp_path / "source.py"
        source_file.write_text("original content", encoding="utf-8")
        correct_hash = compute_file_hash(source_file)
        path = tmp_path / "test.jsonl"
        archiver = JsonlArchiver(path)
        entry = MemoryEntry(
            id="test-1",
            ts="2024-01-01",
            content="valid",
            tags=[],
            tokens=["valid"],
            source_path=str(source_file),
            content_hash=correct_hash,
        )
        archiver.add(entry)
        result = archiver.prune(tmp_path)
        assert isinstance(result, Ok)
        assert result.value == 0
        loaded = archiver.load_entries()
        assert len(loaded) == 1
