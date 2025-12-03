"""Package management utilities (Homebrew).

Provides:
- search_brew to search Homebrew packages
- get_brew_info to get detailed package information
"""

from __future__ import annotations

import asyncio
import shutil

from jpscripts.core.console import get_logger
from jpscripts.core.result import Err, Ok, Result, SystemResourceError

logger = get_logger(__name__)


async def search_brew(query: str | None) -> Result[list[str], SystemResourceError]:
    """Search Homebrew packages.

    Args:
        query: Optional search query

    Returns:
        Result containing list of matching package names
    """
    brew = shutil.which("brew")
    if not brew:
        return Err(SystemResourceError("Homebrew is required."))

    args = [brew, "search"]
    if query:
        args.append(query)

    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return Err(SystemResourceError("Homebrew is required."))
    except Exception as exc:
        return Err(SystemResourceError("Failed to start brew search", context={"error": str(exc)}))

    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        return Err(
            SystemResourceError("brew search failed", context={"stderr": stderr.decode().strip()})
        )

    return Ok([line.strip() for line in stdout.decode().splitlines() if line.strip()])


async def get_brew_info(name: str) -> Result[str, SystemResourceError]:
    """Get detailed information about a Homebrew package.

    Args:
        name: Package name

    Returns:
        Result containing package info text
    """
    brew = shutil.which("brew")
    if not brew:
        return Err(SystemResourceError("Homebrew is required."))

    try:
        proc = await asyncio.create_subprocess_exec(
            brew,
            "info",
            name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return Err(SystemResourceError("Homebrew is required."))
    except Exception as exc:
        return Err(SystemResourceError("Failed to start brew info", context={"error": str(exc)}))

    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        return Err(
            SystemResourceError("brew info failed", context={"stderr": stderr.decode().strip()})
        )

    return Ok(stdout.decode().strip())


__all__ = [
    "get_brew_info",
    "search_brew",
]
