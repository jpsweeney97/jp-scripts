"""Interactive handbook for exploring jpscripts functionality.

Provides CLI commands for:
    - Browsing available commands and tools
    - Searching documentation
    - Viewing command help and usage
    - Generating CLI reference documentation
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import re
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import click
import typer
from rich.markdown import Markdown
from rich.panel import Panel
from typer.main import get_command

from jpscripts.agent.tools import AUDIT_PREFIX, run_safe_shell
from jpscripts.core.config import AppConfig
from jpscripts.core.console import console
from jpscripts.core.mcp_registry import get_tool_metadata, get_tool_registry
from jpscripts.core.result import CapabilityMissingError, Err, Ok
from jpscripts.core.security import validate_path, validate_workspace_root
from jpscripts.memory import (
    STOPWORDS,
    EmbeddingClient,
    EmbeddingClientProtocol,
    MemoryEntry,
    get_memory_store,
)

CACHE_ROOT = Path.home() / ".cache" / "jpscripts" / "handbook_index"
HANDBOOK_NAME = "HANDBOOK.md"
MAX_RESULTS = 3
PROTOCOL_PATTERN = re.compile(
    r"\[Protocol:\s*(?P<name>[^\]]+)\]\s*->\s*run\s*\"(?P<command>[^\"]+)\"", re.IGNORECASE
)
CLI_REFERENCE_HEADING = "## CLI Reference"

# Pre-compiled patterns for performance
_HEADING_PATTERN = re.compile(r"^(#{1,2})\s+(.*)")
_CLI_REFERENCE_PATTERN = re.compile(
    rf"{re.escape(CLI_REFERENCE_HEADING)}.*?(?=^## |\Z)", re.DOTALL | re.MULTILINE
)

app = typer.Typer(invoke_without_command=True, no_args_is_help=False)


@dataclass
class HandbookSection:
    id: str
    title: str
    body: str

    def renderable(self) -> str:
        return f"{self.title}\n\n{self.body}".strip()


@dataclass
class CLICommandRef:
    name: str
    args: str
    summary: str


@dataclass
class MCPToolRef:
    name: str
    params: str
    description: str


def _format_click_params(params: Sequence[click.Parameter]) -> str:
    parts: list[str] = []
    for param in params:
        if bool(getattr(param, "hidden", False)):
            continue
        if param.opts:
            parts.append("/".join(param.opts))
        else:
            parts.append(param.name or "")
    return ", ".join(part for part in parts if part)


def _collect_cli_commands() -> list[CLICommandRef]:
    from jpscripts.main import app as main_app

    click_app = get_command(main_app)
    refs: list[CLICommandRef] = []

    def _walk(command: click.Command, prefix: str) -> None:
        if getattr(command, "hidden", False):
            return
        if isinstance(command, click.Group):
            commands = getattr(command, "commands", {})
            for name, child in sorted(commands.items()):
                if name == "help":
                    continue
                _walk(child, f"{prefix} {name}".strip())
        else:
            args = _format_click_params(command.params)
            summary_raw = (command.help or command.short_help or "").strip()
            summary = " ".join(summary_raw.split())
            refs.append(
                CLICommandRef(name=prefix or command.name or "", args=args, summary=summary or "—")
            )

    if isinstance(click_app, click.Command):  # pyright: ignore[reportUnnecessaryIsInstance]
        _walk(click_app, "")
    return sorted(refs, key=lambda ref: ref.name)


def _type_name(obj: object) -> str:
    if obj is inspect.Parameter.empty:
        return "Any"
    if isinstance(obj, type):
        return obj.__name__
    return str(obj)


def _collect_mcp_tools() -> list[MCPToolRef]:
    tools = get_tool_registry()
    refs: list[MCPToolRef] = []
    for name, func in sorted(tools.items(), key=lambda item: item[0]):
        sig = inspect.signature(func)
        params: list[str] = []
        for param_name, param in sig.parameters.items():
            annotation = _type_name(param.annotation)
            default = "" if param.default is inspect.Parameter.empty else f"={param.default!r}"
            params.append(f"{param_name}: {annotation}{default}")
        doc = (inspect.getdoc(func) or "").strip()
        metadata = get_tool_metadata(func) or {}
        description = metadata.get("description", "") if isinstance(metadata, dict) else ""  # pyright: ignore[reportUnnecessaryIsInstance]
        combined_desc = " ".join((description or doc or "—").split())
        refs.append(MCPToolRef(name=name, params=", ".join(params), description=combined_desc))
    return refs


def generate_reference() -> tuple[str, str]:
    cli_refs: list[CLICommandRef] = _collect_cli_commands()
    mcp_refs: list[MCPToolRef] = _collect_mcp_tools()

    cli_lines = ["| Command | Args | Description |", "| :--- | :--- | :--- |"]
    for cli_ref in cli_refs:
        cli_lines.append(f"| `{cli_ref.name}` | {cli_ref.args or '—'} | {cli_ref.summary or '—'} |")
    cli_table = "\n".join(cli_lines)

    mcp_lines = ["| Tool | Params | Description |", "| :--- | :--- | :--- |"]
    for tool_ref in mcp_refs:
        mcp_lines.append(
            f"| `{tool_ref.name}` | {tool_ref.params or '—'} | {tool_ref.description or '—'} |"
        )
    mcp_table = "\n".join(mcp_lines)

    return cli_table, mcp_table


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _tokenize_text(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]


async def _ensure_dir(path: Path) -> None:
    def _mk() -> None:
        path.mkdir(parents=True, exist_ok=True)

    await asyncio.to_thread(_mk)


def _cache_paths() -> tuple[Path, Path, Path, Path]:
    base_root = Path.home()
    cache_dir = validate_path(CACHE_ROOT, base_root)
    meta_path = validate_path(cache_dir / "meta.json", base_root)
    entries_path = validate_path(cache_dir / "entries.jsonl", base_root)
    store_path = validate_path(cache_dir / "lance", base_root)
    return cache_dir, meta_path, entries_path, store_path


async def _read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None

    def _load() -> dict[str, object] | None:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return dict(data)  # Cast to explicit dict[str, object]
                return None
        except (OSError, json.JSONDecodeError):
            return None

    return await asyncio.to_thread(_load)


async def _write_json(path: Path, payload: dict[str, object]) -> None:
    def _dump() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    await asyncio.to_thread(_dump)


async def _read_entries(path: Path) -> list[MemoryEntry]:
    if not path.exists():
        return []

    def _load() -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    entry = MemoryEntry(
                        id=str(raw.get("id", "")),
                        ts=str(raw.get("ts", "")),
                        content=str(raw.get("content", "")),
                        tags=[str(tag) for tag in raw.get("tags", []) if str(tag)],
                        tokens=[str(tok) for tok in raw.get("tokens", []) if str(tok)],
                        embedding=[float(val) for val in raw.get("embedding", [])]
                        if raw.get("embedding")
                        else None,
                    )
                    entries.append(entry)
        except OSError:
            return []
        return entries

    return await asyncio.to_thread(_load)


async def _write_entries(path: Path, entries: Iterable[MemoryEntry]) -> None:
    def _dump() -> None:
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

    await asyncio.to_thread(_dump)


def _resolve_handbook_path() -> Path | None:
    root = _project_root()
    try:
        path = validate_path(root / HANDBOOK_NAME, root)
    except PermissionError as exc:
        console.print(f"[red]{exc}[/red]")
        return None
    except Exception as exc:
        console.print(f"[red]Failed to resolve handbook path: {exc}[/red]")
        return None

    if not path.exists():
        console.print(f"[red]{HANDBOOK_NAME} not found at {path}[/red]")
        return None
    if not path.is_file():
        console.print(f"[red]{path} is not a file.[/red]")
        return None

    return path


async def _read_handbook(path: Path) -> str | None:
    try:
        return await asyncio.to_thread(path.read_text, encoding="utf-8")
    except OSError as exc:
        console.print(f"[red]Failed to read handbook: {exc}[/red]")
        return None


async def _mtime_ns(path: Path) -> int | None:
    try:
        stat_result = await asyncio.to_thread(path.stat)
        return stat_result.st_mtime_ns
    except OSError as exc:
        console.print(f"[red]Failed to stat {path}: {exc}[/red]")
        return None


def _parse_sections(content: str) -> list[HandbookSection]:
    sections: list[HandbookSection] = []
    current_title: str | None = None
    current_body: list[str] = []
    section_idx = 0

    for line in content.splitlines():
        match = _HEADING_PATTERN.match(line.strip())
        if match:
            if current_title is not None:
                sections.append(
                    HandbookSection(
                        id=f"handbook-{section_idx}",
                        title=current_title,
                        body="\n".join(current_body).strip(),
                    )
                )
                section_idx += 1
            current_title = match.group(2).strip() or "Section"
            current_body = []
            continue
        if current_title is not None:
            current_body.append(line)

    if current_title is not None:
        sections.append(
            HandbookSection(
                id=f"handbook-{section_idx}",
                title=current_title,
                body="\n".join(current_body).strip(),
            )
        )

    if sections:
        return sections

    fallback = content.strip()
    if not fallback:
        return []
    return [HandbookSection(id="handbook-0", title="Handbook", body=fallback)]


def _build_entries(sections: list[HandbookSection], source_mtime_ns: int) -> list[MemoryEntry]:
    timestamp = datetime.fromtimestamp(source_mtime_ns / 1_000_000_000, tz=UTC).isoformat(
        timespec="seconds"
    )
    entries: list[MemoryEntry] = []
    for section in sections:
        text = section.renderable()
        entries.append(
            MemoryEntry(
                id=section.id,
                ts=timestamp,
                content=text,
                tags=[section.title] if section.title else [],
                tokens=_tokenize_text(text),
            )
        )
    return entries


def _render_cli_reference_section(cli_table: str, mcp_table: str) -> str:
    return (
        f"{CLI_REFERENCE_HEADING}\n\n### CLI Commands\n{cli_table}\n\n### MCP Tools\n{mcp_table}\n"
    )


async def _replace_cli_reference(path: Path, cli_table: str, mcp_table: str) -> bool:
    new_section = _render_cli_reference_section(cli_table, mcp_table)

    def _rewrite() -> tuple[bool, str]:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            return False, ""

        if _CLI_REFERENCE_PATTERN.search(content):
            updated = _CLI_REFERENCE_PATTERN.sub(new_section.strip() + "\n\n", content)
        else:
            updated = content.rstrip() + "\n\n" + new_section

        return updated != content, updated

    changed, updated_content = await asyncio.to_thread(_rewrite)
    if not changed or not updated_content:
        return False

    await asyncio.to_thread(path.write_text, updated_content, "utf-8")
    return True


def _build_embedding_client(config: AppConfig | None) -> EmbeddingClientProtocol:
    use_semantic = True
    model_name = "all-MiniLM-L6-v2"
    server_url: str | None = None
    if config:
        use_semantic = bool(getattr(config, "use_semantic_search", True))
        model_name = getattr(config, "memory_model", model_name)
        server_url = getattr(config, "embedding_server_url", None)
    return EmbeddingClient(model_name, enabled=use_semantic, server_url=server_url)


async def _reset_store(store_path: Path) -> None:
    if not store_path.exists():
        return

    def _remove() -> None:
        shutil.rmtree(store_path, ignore_errors=True)  # safety: checked

    await asyncio.to_thread(_remove)


async def _insert_into_store(
    entries: list[MemoryEntry],
    store_path: Path,
    config: AppConfig,
) -> None:
    """Insert entries into the memory store with graceful fallback."""
    store_result = get_memory_store(config, store_path=store_path)

    match store_result:
        case Err(CapabilityMissingError()):
            # LanceDB unavailable - entries will be searched via keyword fallback
            return
        case Err(error):
            console.print(f"[yellow]Memory store unavailable: {error}[/yellow]")
            return
        case Ok(store):
            pass

    def _insert() -> None:
        for entry in entries:
            result = store.add(entry)
            if isinstance(result, Err):
                raise result.error

    try:
        await asyncio.to_thread(_insert)
    except Exception as exc:
        console.print(f"[yellow]Memory store insertion failed: {exc}[/yellow]")


async def _load_or_index_entries(
    sections: list[HandbookSection],
    embedding_client: EmbeddingClientProtocol,
    source_mtime_ns: int,
    meta_path: Path,
    entries_path: Path,
    store_path: Path,
    config: AppConfig,
) -> tuple[list[MemoryEntry], int]:
    meta = await _read_json(meta_path)
    if meta and meta.get("source_mtime_ns") == source_mtime_ns:
        cached_entries = await _read_entries(entries_path)
        if cached_entries:
            dim_value = meta.get("embedding_dim", 0)
            cached_dim = int(dim_value) if isinstance(dim_value, (int, float, str)) else 0
            if cached_dim == 0 and cached_entries[0].embedding:
                cached_dim = len(cached_entries[0].embedding or [])
            return cached_entries, cached_dim

    await _reset_store(store_path)
    entries = _build_entries(sections, source_mtime_ns)
    semantic_ready = embedding_client.available()
    vectors = (
        embedding_client.embed([entry.content for entry in entries]) if semantic_ready else None
    )
    if semantic_ready and vectors is None:
        console.print(
            "[yellow]Semantic embeddings unavailable; falling back to keyword search.[/yellow]"
        )
    embedding_dim = 0
    if vectors:
        first = vectors[0] if vectors else []
        embedding_dim = len(first) if first else 0
        for entry, vector in zip(entries, vectors, strict=False):
            entry.embedding = vector

    await _write_entries(entries_path, entries)
    await _write_json(
        meta_path,
        {"source_mtime_ns": source_mtime_ns, "embedding_dim": embedding_dim},
    )

    if embedding_dim > 0:
        await _insert_into_store(entries, store_path, config)

    return entries, embedding_dim


def _cosine_similarity(lhs: list[float], rhs: list[float]) -> float:
    if len(lhs) != len(rhs) or not lhs or not rhs:
        return 0.0
    dot = sum(a * b for a, b in zip(lhs, rhs, strict=False))
    left_norm = math.sqrt(sum(a * a for a in lhs))
    right_norm = math.sqrt(sum(b * b for b in rhs))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _local_vector_search(
    entries: list[MemoryEntry], query_vec: list[float], limit: int
) -> list[MemoryEntry]:
    scored: list[tuple[float, MemoryEntry]] = []
    for entry in entries:
        if entry.embedding is None:
            continue
        score = _cosine_similarity(query_vec, entry.embedding)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for _, entry in scored[:limit]]


def _keyword_search(entries: list[MemoryEntry], query: str, limit: int) -> list[MemoryEntry]:
    tokens = _tokenize_text(query)
    scored: list[tuple[int, MemoryEntry]] = []
    for entry in entries:
        score = sum(entry.content.lower().count(token) for token in tokens) if tokens else 0
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if scored:
        return [entry for _, entry in scored[:limit]]
    return entries[:limit]


async def _search_entries(
    query: str,
    embedding_client: EmbeddingClientProtocol,
    entries: list[MemoryEntry],
    store_path: Path,
    config: AppConfig,
    limit: int = MAX_RESULTS,
) -> list[MemoryEntry]:
    """Search entries using core memory store with fallbacks."""
    store_result = get_memory_store(config, store_path=store_path)

    # Fallback to keyword search if store unavailable
    # Yellow warning for non-CapabilityMissing errors, silent for expected missing deps
    if isinstance(store_result, Err):
        if not isinstance(store_result.error, CapabilityMissingError):
            console.print(f"[yellow]Memory store unavailable: {store_result.error}[/yellow]")
        return _keyword_search(entries, query, limit)

    store = store_result.value

    # Get query embeddings if available
    query_vec: list[float] | None = None
    if embedding_client.available():
        query_vecs = embedding_client.embed([query])
        query_vec = query_vecs[0] if query_vecs else None

    # Tokenize for hybrid search
    query_tokens = _tokenize_text(query)

    # Use store's hybrid search (RRF fusion of vector + keyword)
    search_result = store.search(query_vec, limit, query_tokens=query_tokens)

    if isinstance(search_result, Err):
        console.print(f"[yellow]Store search failed: {search_result.error}[/yellow]")
        return _keyword_search(entries, query, limit)

    results = search_result.value

    # Fall back to local search if store returns empty
    if not results:
        if query_vec:
            return _local_vector_search(entries, query_vec, limit)
        return _keyword_search(entries, query, limit)

    return results


def _render_results(results: list[MemoryEntry]) -> None:
    if not results:
        console.print("[yellow]No matching sections found.[/yellow]")
        return

    for entry in results:
        title = (
            entry.tags[0]
            if entry.tags
            else entry.content.splitlines()[0]
            if entry.content
            else "Handbook"
        )
        body = entry.content or "No content available."
        console.print(Panel(Markdown(body), title=title, expand=True))


def parse_protocols(content: str) -> dict[str, list[str]]:
    """Extract protocol definitions from handbook content."""
    protocols: dict[str, list[str]] = {}
    for match in PROTOCOL_PATTERN.finditer(content):
        name = match.group("name").strip().lower()
        command = match.group("command").strip()
        if not name or not command:
            continue
        protocols.setdefault(name, []).append(command)
    return protocols


@app.callback(invoke_without_command=True, no_args_is_help=False)
def handbook(
    ctx: typer.Context,
    query: str | None = typer.Argument(
        None,
        help="Optional semantic query. Provide text to search the handbook; omit to render the full handbook.",
    ),
) -> None:
    """Render the project handbook or run a semantic search over its sections.

    Args:
        ctx: Typer context containing application state.
        query: Optional semantic query. If provided, the command searches the handbook; otherwise renders it.
    """
    config: AppConfig | None = getattr(ctx.obj, "config", None)

    async def _run() -> None:
        handbook_path = _resolve_handbook_path()
        if handbook_path is None:
            return

        content = await _read_handbook(handbook_path)
        if not content:
            return

        if query is None:
            console.print(Markdown(content))
            return

        sections = _parse_sections(content)
        if not sections:
            console.print("[yellow]No sections found in the handbook.[/yellow]")
            return

        source_mtime_ns = await _mtime_ns(handbook_path)
        if source_mtime_ns is None:
            return

        cache_dir, meta_path, entries_path, store_path = _cache_paths()
        await _ensure_dir(cache_dir)

        embedding_client = _build_embedding_client(config)

        # Config required for memory store integration
        if config is None:
            console.print(
                "[yellow]Configuration unavailable; falling back to keyword search.[/yellow]"
            )
            entries = _build_entries(sections, source_mtime_ns)
            results = _keyword_search(entries, query, MAX_RESULTS)
            _render_results(results)
            return

        entries, _embedding_dim = await _load_or_index_entries(
            sections,
            embedding_client,
            source_mtime_ns,
            meta_path,
            entries_path,
            store_path,
            config,
        )

        results = await _search_entries(
            query=query,
            embedding_client=embedding_client,
            entries=entries,
            store_path=store_path,
            config=config,
            limit=MAX_RESULTS,
        )
        _render_results(results)

    asyncio.run(_run())


@app.command("verify-protocol")
def verify_protocol(
    ctx: typer.Context,
    name: str = typer.Option("pre-commit", "--name", "-n", help="Protocol name to execute."),
) -> None:
    """Execute Handbook protocol commands for the given context."""
    state = ctx.obj

    async def _run() -> int:
        handbook_path = _resolve_handbook_path()
        if handbook_path is None:
            return 1

        content = await _read_handbook(handbook_path)
        if not content:
            console.print("[red]Handbook is empty or unreadable.[/red]")
            return 1

        agents_path = Path(_project_root()) / "AGENTS.md"
        if not agents_path.exists():
            console.print(
                "[red]AGENTS.md is missing; cannot satisfy governance requirements.[/red]"
            )
            return 1
        try:
            agents_text = await asyncio.to_thread(agents_path.read_text, encoding="utf-8")
        except OSError:
            console.print(
                "[red]AGENTS.md is unreadable; fix repository state before proceeding.[/red]"
            )
            return 1
        if "invariants" not in agents_text:
            console.print("[red]AGENTS.md lacks the required Invariants section.[/red]")
            return 1

        protocols = parse_protocols(content)
        commands = protocols.get(name.lower())
        if not commands:
            console.print(f"[yellow]No protocol named '{name}' found in handbook.[/yellow]")
            return 1

        config: AppConfig | None = getattr(state, "config", None)
        if config is None:
            console.print("[red]Configuration unavailable; cannot execute protocols.[/red]")
            return 1

        try:
            root = await asyncio.to_thread(validate_workspace_root, config.user.workspace_root)
        except Exception as exc:
            console.print(f"[red]Workspace validation failed: {exc}[/red]")
            return 1

        for cmd in commands:
            output = await run_safe_shell(
                cmd, root, f"{AUDIT_PREFIX}.protocol.{name}", config=config
            )
            if output and output.startswith("SecurityError"):
                console.print(f"[red]{output}[/red]")
                return 1
            if output:
                console.print(output)
        return 0

    exit_code = asyncio.run(_run())
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@app.command("internal-update-reference", hidden=True)
def internal_update_reference(ctx: typer.Context) -> None:
    """Regenerate CLI and MCP tool reference sections in README and HANDBOOK."""

    async def _run() -> int:
        cli_table, mcp_table = generate_reference()
        root = _project_root()

        targets: list[Path] = []
        for name in ("README.md", HANDBOOK_NAME):
            try:
                targets.append(validate_path(root / name, root))
            except Exception as exc:
                console.print(f"[red]Failed to resolve {name}: {exc}[/red]")
                return 1

        updates = await asyncio.gather(
            *(_replace_cli_reference(path, cli_table, mcp_table) for path in targets)
        )
        updated_any = any(updates)
        if not updated_any:
            console.print(
                "[yellow]No CLI reference updates applied (already current or files missing).[/yellow]"
            )
            return 0

        console.print("[green]CLI references updated in README and HANDBOOK.[/green]")
        return 0

    exit_code = asyncio.run(_run())
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
