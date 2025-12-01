"""Tests for core/web.py - web page content fetching."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestFetchPageContent:
    """Test the fetch_page_content function."""

    def test_fetch_page_content_success(self) -> None:
        """Test successful fetch and extraction."""
        mock_trafilatura = MagicMock()
        mock_trafilatura.fetch_url.return_value = "<html><body>Test</body></html>"
        mock_trafilatura.extract.return_value = "# Extracted Markdown\n\nTest content"

        with patch.dict("sys.modules", {"trafilatura": mock_trafilatura}):
            # Import fresh each time to use the mocked module
            from jpscripts.core.web import fetch_page_content

            result = fetch_page_content("https://example.com")

            assert result == "# Extracted Markdown\n\nTest content"
            mock_trafilatura.fetch_url.assert_called_once_with("https://example.com")
            mock_trafilatura.extract.assert_called_once()

    def test_fetch_page_content_trafilatura_missing(self) -> None:
        """Test behavior when trafilatura is not installed."""
        # Make trafilatura import fail
        with patch.dict("sys.modules", {"trafilatura": None}):
            # Import the module - it will try to import trafilatura inside the function
            # Patch the import statement inside the function
            import builtins

            from jpscripts.core.web import fetch_page_content

            original_import = builtins.__import__

            def mock_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
                if name == "trafilatura":
                    raise ImportError("No module named 'trafilatura'")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", mock_import):
                result = fetch_page_content("https://example.com")

        assert "trafilatura not installed" in result
        assert "pip install" in result

    def test_fetch_page_content_fetch_fails(self) -> None:
        """Test when URL fetch returns None."""
        mock_trafilatura = MagicMock()
        mock_trafilatura.fetch_url.return_value = None

        with patch.dict("sys.modules", {"trafilatura": mock_trafilatura}):
            from jpscripts.core.web import fetch_page_content

            result = fetch_page_content("https://example.com/nonexistent")

            assert "Failed to fetch" in result
            assert "https://example.com/nonexistent" in result

    def test_fetch_page_content_extract_returns_none(self) -> None:
        """Test when extract returns None."""
        mock_trafilatura = MagicMock()
        mock_trafilatura.fetch_url.return_value = "<html></html>"
        mock_trafilatura.extract.return_value = None

        with patch.dict("sys.modules", {"trafilatura": mock_trafilatura}):
            from jpscripts.core.web import fetch_page_content

            result = fetch_page_content("https://example.com/empty")

            assert "Failed to extract content" in result
            assert "https://example.com/empty" in result

    def test_fetch_page_content_extract_exception(self) -> None:
        """Test when extract raises an exception (defensive branch)."""
        mock_trafilatura = MagicMock()
        mock_trafilatura.fetch_url.return_value = "<html><body>Bad content</body></html>"
        mock_trafilatura.extract.side_effect = Exception("Parse error")

        with patch.dict("sys.modules", {"trafilatura": mock_trafilatura}):
            from jpscripts.core.web import fetch_page_content

            result = fetch_page_content("https://example.com/bad")

            assert "Failed to extract content" in result
            assert "Parse error" in result

    def test_fetch_page_content_extract_called_with_correct_params(self) -> None:
        """Test that extract is called with expected parameters."""
        mock_trafilatura = MagicMock()
        mock_trafilatura.fetch_url.return_value = "<html><body>Test</body></html>"
        mock_trafilatura.extract.return_value = "content"

        with patch.dict("sys.modules", {"trafilatura": mock_trafilatura}):
            from jpscripts.core.web import fetch_page_content

            fetch_page_content("https://example.com/page")

            # Verify extract was called with expected kwargs
            call_kwargs = mock_trafilatura.extract.call_args.kwargs
            assert call_kwargs.get("include_comments") is False
            assert call_kwargs.get("output_format") == "markdown"
            assert call_kwargs.get("url") == "https://example.com/page"
