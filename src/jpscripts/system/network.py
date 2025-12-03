"""Network utilities.

Provides:
- get_ssh_hosts to read SSH config and list available hosts
- run_temp_server to start a temporary HTTP server
"""

from __future__ import annotations

import asyncio
import functools
import http.server
from collections.abc import Callable
from pathlib import Path

from jpscripts.core.console import get_logger
from jpscripts.core.result import Err, Ok, Result, SystemResourceError

logger = get_logger(__name__)


async def get_ssh_hosts(config_path: Path | None = None) -> Result[list[str], SystemResourceError]:
    """Read SSH hosts from config file.

    Args:
        config_path: Optional path to SSH config (defaults to ~/.ssh/config)

    Returns:
        Result containing sorted list of host names
    """
    target = config_path or Path.home() / ".ssh" / "config"
    if not target.exists():
        return Ok([])

    def _read_hosts() -> list[str]:
        hosts: list[str] = []
        content = target.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("host "):
                entries = line.split()[1:]
                hosts.extend([h for h in entries if h != "*"])
        return sorted(hosts)

    try:
        hosts = await asyncio.to_thread(_read_hosts)
    except OSError as exc:
        return Err(
            SystemResourceError(
                "Failed to read SSH config", context={"path": str(target), "error": str(exc)}
            )
        )

    return Ok(hosts)


async def run_temp_server(directory: Path, port: int) -> Result[None, SystemResourceError]:
    """Start a temporary HTTP server serving a directory.

    Args:
        directory: Directory to serve
        port: Port number to bind

    Returns:
        Result containing None on success
    """
    if not directory.is_dir():
        return Err(
            SystemResourceError(
                "Serve directory is not a folder", context={"directory": str(directory)}
            )
        )

    def _serve() -> None:
        handler: Callable[..., http.server.SimpleHTTPRequestHandler] = functools.partial(
            http.server.SimpleHTTPRequestHandler, directory=str(directory)
        )
        httpd = http.server.ThreadingHTTPServer(("", port), handler)
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()

    try:
        await asyncio.to_thread(_serve)
    except OSError as exc:
        return Err(
            SystemResourceError(
                "Failed to start HTTP server",
                context={"directory": str(directory), "error": str(exc)},
            )
        )

    return Ok(None)


__all__ = [
    "get_ssh_hosts",
    "run_temp_server",
]
