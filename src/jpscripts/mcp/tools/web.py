from __future__ import annotations

import asyncio

from jpscripts.mcp import tool


@tool()
async def fetch_url_content(url: str) -> str:
    """Fetch and parse a webpage into clean Markdown."""
    try:
        return await asyncio.to_thread(_fetch_content, url)
    except ImportError:
        return "Error: trafilatura not installed. Run `pip install jpscripts[full]`"
    except Exception as e:
        return f"Error fetching URL: {str(e)}"


def _fetch_content(url: str) -> str:
    import trafilatura

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return f"Error: Failed to download {url}"
    text = trafilatura.extract(downloaded, include_comments=False, output_format="markdown", url=url)
    return text if text else "Error: Could not extract content."
