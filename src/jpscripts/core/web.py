from __future__ import annotations

from jpscripts.core.console import get_logger

logger = get_logger(__name__)


def fetch_page_content(url: str) -> str:
    """Fetch a webpage and return extracted markdown content.

    Args:
        url: The URL to fetch.

    Returns:
        Extracted markdown content, or an error message if fetching fails.
    """
    try:
        import trafilatura
    except ImportError:
        return "trafilatura not installed. Install with `pip install jpscripts[full]`."

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return f"Failed to fetch {url}"

    try:
        extracted: str | None = trafilatura.extract(
            downloaded,
            include_comments=False,
            output_format="markdown",
            url=url,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Trafilatura extract failed for %s: %s", url, exc)
        return f"Failed to extract content for {url}: {exc}"

    if not extracted:
        return f"Failed to extract content for {url}"

    return extracted
