from __future__ import annotations

import datetime as dt
import socket
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import trafilatura
import typer
import yaml
from rich import box
from rich.panel import Panel
from rich.table import Table

from jpscripts.core.console import console
from jpscripts.core.config import AppConfig

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

def _slugify_url(url: str, today: dt.date) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.replace(".", "-")
    path_parts = [part for part in parsed.path.split("/") if part]
    path_slug = "-".join(path_parts) if path_parts else "home"
    return f"{domain}-{path_slug}_{today.isoformat()}.yaml"


def _write_yaml(metadata: dict, content: str, dest: Path) -> None:
    docs = [
        metadata,
        {"content": content},
    ]
    yaml_str = ""
    for doc in docs:
        yaml_str += "---\n"
        yaml_str += yaml.dump(doc, allow_unicode=False, sort_keys=False, default_flow_style=False)
    dest.write_text(yaml_str, encoding="utf-8")


def web_snap(
    ctx: typer.Context,
    url: str = typer.Argument(..., help="URL to fetch and snapshot."),
) -> None:
    """Fetch a webpage, extract main content, and save as a YAML snapshot."""
    state = ctx.obj
    config: AppConfig = state.config
    target_dir = (config.snapshots_dir or Path(".")).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Fetching[/cyan] {url} ...")
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        console.print(f"[red]Failed to fetch[/red] {url}")
        raise typer.Exit(code=1)

    extracted = trafilatura.extract(
        downloaded,
        include_comments=False,
        output_format="markdown",
        url=url,
    )
    if not extracted:
        console.print(f"[red]Failed to extract main content for[/red] {url}")
        raise typer.Exit(code=1)

    parsed = urlparse(url)
    title = None
    try:
        meta_doc = trafilatura.extract_metadata(downloaded, default_url=url)
        meta = meta_doc.as_dict() if meta_doc else {}
        title = meta.get("title")
    except Exception:
        title = None

    metadata = {
        "url": url,
        "domain": parsed.netloc,
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "title": title,
    }

    filename = _slugify_url(url, dt.date.today())
    output_path = target_dir / filename
    _write_yaml(metadata, extracted, output_path)

    table = Table(title="Snapshot saved", box=box.SIMPLE, expand=True)
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("File", str(output_path))
    table.add_row("Domain", metadata["domain"])
    table.add_row("Timestamp", metadata["timestamp"])
    console.print(table)

    if socket.gethostname().endswith(".local"):  # crude macOS hint
        try:
            subprocess.run(["open", "-R", str(output_path)], check=False)
        except FileNotFoundError:
            pass
