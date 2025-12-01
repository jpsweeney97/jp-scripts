"""Pattern storage and synthesis.

This module provides:
- PatternStore: Dedicated LanceDB collection for learned patterns
- Pattern consolidation from execution traces
- Pattern retrieval for prompt injection
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import SupportsFloat, cast
from uuid import uuid4

from jpscripts.core.config import AppConfig
from jpscripts.core.console import get_logger
from jpscripts.core.result import (
    CapabilityMissingError,
    ConfigurationError,
    Err,
    JPScriptsError,
    Ok,
    Result,
)
from jpscripts.providers import CompletionOptions, LLMProvider
from jpscripts.providers import Message as ProviderMessage
from jpscripts.providers.factory import get_provider

from .embedding import EmbeddingClient, _embedding_settings
from .models import (
    LanceDBConnectionProtocol,
    LanceDBModuleProtocol,
    LanceModelBase,
    LanceTable,
    Pattern,
    PatternRecordProtocol,
)
from .store import _resolve_store

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# LanceDB Helpers
# -----------------------------------------------------------------------------


def _load_lancedb_dependencies() -> tuple[LanceDBModuleProtocol, type[LanceModelBase]] | None:
    """Load LanceDB dependencies, returning None if unavailable."""
    try:
        lancedb = import_module("lancedb")
        pydantic_module = import_module("lancedb.pydantic")
        lance_model = cast(type[LanceModelBase], pydantic_module.LanceModel)
    except Exception as exc:
        logger.debug("LanceDB unavailable: %s", exc)
        return None
    return cast(LanceDBModuleProtocol, lancedb), lance_model


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


# -----------------------------------------------------------------------------
# Pattern Store
# -----------------------------------------------------------------------------


class PatternStore:
    """Dedicated LanceDB collection for extracted patterns.

    Patterns are stored separately from memories to enable specialized
    retrieval for prompt injection without polluting the general memory space.
    """

    def __init__(
        self,
        db_path: Path,
        lancedb_module: LanceDBModuleProtocol,
        lance_model_base: type[LanceModelBase],
    ) -> None:
        self._db_path = db_path.expanduser()
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._lancedb: LanceDBModuleProtocol = lancedb_module
        self._model_cls = _build_pattern_record_model(lance_model_base)
        self._table: LanceTable | None = None
        self._embedding_dim: int | None = None

    def _ensure_table(self, embedding_dim: int | None = None) -> LanceTable:
        """Ensure the patterns table exists."""
        if self._table is None:
            db: LanceDBConnectionProtocol = self._lancedb.connect(str(self._db_path))
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
            record = self._model_cls(  # pyright: ignore[reportCallIssue]
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
                ConfigurationError("Failed to prepare patterns table", context={"error": str(exc)})
            )

        try:
            results = cast(
                Sequence[PatternRecordProtocol],
                table.search(query_vec).limit(limit).to_pydantic(self._model_cls),
            )
        except Exception as exc:
            return Err(ConfigurationError("Pattern search failed", context={"error": str(exc)}))

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
                row_mapping = row
                patterns.append(
                    Pattern(
                        id=str(row_mapping.get("id", "")),
                        created_at=str(row_mapping.get("created_at", "")),
                        pattern_type=str(row_mapping.get("pattern_type", "")),
                        description=str(row_mapping.get("description", "")),
                        trigger=str(row_mapping.get("trigger", "")),
                        solution=str(row_mapping.get("solution", "")),
                        source_traces=list(
                            cast(Iterable[str], row_mapping.get("source_traces", []))
                        ),
                        confidence=float(cast(SupportsFloat, row_mapping.get("confidence", 0.0))),
                        embedding=list(cast(Iterable[float], row_mapping.get("embedding", [])))
                        if row_mapping.get("embedding") is not None
                        else None,
                    )
                )
            return Ok(patterns)
        except Exception as exc:
            return Err(
                ConfigurationError("Failed to retrieve patterns", context={"error": str(exc)})
            )


# -----------------------------------------------------------------------------
# Factory Function
# -----------------------------------------------------------------------------


def get_pattern_store(config: AppConfig) -> Result[PatternStore, JPScriptsError]:
    """Get the pattern store for the given configuration."""
    resolved_store = _resolve_store(config)

    deps = _load_lancedb_dependencies()
    if deps is None:
        return Err(
            CapabilityMissingError(
                'LanceDB is required for pattern storage. Install with `pip install "jpscripts[ai]"`.',
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


# -----------------------------------------------------------------------------
# Pattern Consolidation
# -----------------------------------------------------------------------------


async def _load_successful_traces(trace_dir: Path, limit: int) -> list[dict[str, object]]:
    """Load trace steps that resulted in successful outcomes."""
    traces: list[dict[str, object]] = []

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


def _cluster_traces_by_similarity(traces: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    """Group traces by similarity of error type and solution approach.

    Uses simple heuristics: group by first word of thought_process
    (usually indicates the type of issue being addressed).
    """
    clusters: dict[str, list[dict[str, object]]] = {}

    for trace in traces:
        response = cast(dict[str, object], trace.get("response", {}))
        thought = str(response.get("thought_process", ""))

        # Extract key from first significant word
        words = thought.split()[:3]
        key = " ".join(words).lower() if words else "other"

        if key not in clusters:
            clusters[key] = []
        clusters[key].append(trace)

    return list(clusters.values())


async def _synthesize_pattern_from_cluster(
    cluster: list[dict[str, object]],
    provider: LLMProvider,
    model: str,
) -> Pattern | None:
    """Use LLM to extract a generalized pattern from a cluster of similar traces."""
    examples = []
    for trace in cluster[:5]:  # Limit to 5 examples
        response = cast(dict[str, object], trace.get("response", {}))
        input_history = cast(list[dict[str, object]], trace.get("input_history", []))
        context = str(input_history[-1].get("content", "")) if input_history else ""
        examples.append(
            {
                "thought": str(response.get("thought_process", "")),
                "patch": str(response.get("file_patch", "")),
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
        llm_response = await provider.complete(messages, model=model, options=options)
        content = llm_response.content.strip()

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
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            pattern_type=str(data.get("pattern_type", "fix_pattern")),
            description=str(data.get("description", "")),
            trigger=str(data.get("trigger", "")),
            solution=str(data.get("solution", "")),
            source_traces=[str(trace.get("timestamp", "")) for trace in cluster],
            confidence=float(data.get("confidence", 0.5)),
        )
    except Exception as exc:
        logger.debug("Pattern synthesis failed: %s", exc)
        return None


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
        return Err(
            ConfigurationError("Trace directory not found", context={"path": str(trace_dir)})
        )

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


# -----------------------------------------------------------------------------
# Pattern Retrieval
# -----------------------------------------------------------------------------


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


__all__ = [
    "PatternStore",
    "consolidate_patterns",
    "fetch_relevant_patterns",
    "format_patterns_for_prompt",
    "get_pattern_store",
]
