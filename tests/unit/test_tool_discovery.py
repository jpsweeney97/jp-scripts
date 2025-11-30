from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch


class TestToolDiscovery:
    """Tests for dynamic MCP tool discovery."""

    def test_discovers_all_existing_modules(self) -> None:
        """TOOL_MODULES should discover all existing tool modules."""
        from jpscripts.mcp.tools import TOOL_MODULES

        assert isinstance(TOOL_MODULES, list)
        # Should find at least the original 9 modules
        assert len(TOOL_MODULES) >= 9

        # All should be fully qualified module names
        assert all(m.startswith("jpscripts.mcp.tools.") for m in TOOL_MODULES)

        # Check known modules are present
        expected = {
            "jpscripts.mcp.tools.filesystem",
            "jpscripts.mcp.tools.git",
            "jpscripts.mcp.tools.memory",
            "jpscripts.mcp.tools.navigation",
            "jpscripts.mcp.tools.notes",
            "jpscripts.mcp.tools.search",
            "jpscripts.mcp.tools.system",
            "jpscripts.mcp.tools.tests",
            "jpscripts.mcp.tools.web",
        }
        assert expected.issubset(set(TOOL_MODULES))

    def test_excludes_private_modules(self) -> None:
        """Private modules (starting with _) should be excluded."""
        from jpscripts.mcp.tools import TOOL_MODULES

        for module in TOOL_MODULES:
            module_name = module.split(".")[-1]
            assert not module_name.startswith("_"), f"Private module found: {module}"

    def test_returns_sorted_list(self) -> None:
        """TOOL_MODULES should be sorted for deterministic ordering."""
        from jpscripts.mcp.tools import TOOL_MODULES

        assert sorted(TOOL_MODULES) == TOOL_MODULES

    def test_handles_missing_path_gracefully(self) -> None:
        """Should return empty list and warn when __path__ is None."""
        from jpscripts.mcp.tools import _discover_tool_module_names

        mock_package = MagicMock()
        mock_package.__path__ = None

        with patch("jpscripts.mcp.tools.import_module", return_value=mock_package):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = _discover_tool_module_names()

                assert result == []
                assert len(caught) == 1
                assert "no __path__" in str(caught[0].message)

    def test_handles_import_failure_gracefully(self) -> None:
        """Should return empty list and warn when import fails."""
        from jpscripts.mcp.tools import _discover_tool_module_names

        with (
            patch(
                "jpscripts.mcp.tools.import_module",
                side_effect=ImportError("test error"),
            ),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            result = _discover_tool_module_names()

            assert result == []
            assert len(caught) == 1
            assert "Failed to import" in str(caught[0].message)

    def test_handles_non_iterable_path(self) -> None:
        """Should return empty list and warn when __path__ is not iterable."""
        from jpscripts.mcp.tools import _discover_tool_module_names

        mock_package = MagicMock()
        # Make __path__ raise TypeError when iterated
        mock_package.__path__ = 42  # Not iterable

        with patch("jpscripts.mcp.tools.import_module", return_value=mock_package):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = _discover_tool_module_names()

                assert result == []
                assert len(caught) == 1
                assert "not iterable" in str(caught[0].message)

    def test_handles_iter_modules_exception(self) -> None:
        """Should warn and return partial results on iter_modules failure."""
        from jpscripts.mcp.tools import _discover_tool_module_names

        mock_package = MagicMock()
        mock_package.__path__ = ["/fake/path"]

        def failing_iter_modules(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise OSError("test error")

        with patch("jpscripts.mcp.tools.import_module", return_value=mock_package):
            with patch("jpscripts.mcp.tools.pkgutil.iter_modules", failing_iter_modules):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = _discover_tool_module_names()

                    # Should return empty list (partial results)
                    assert result == []
                    assert len(caught) == 1
                    assert "Error during tool discovery" in str(caught[0].message)

    def test_handles_empty_path_list(self) -> None:
        """Should return empty list when __path__ is empty."""
        from jpscripts.mcp.tools import _discover_tool_module_names

        mock_package = MagicMock()
        mock_package.__path__ = []

        with patch("jpscripts.mcp.tools.import_module", return_value=mock_package):
            result = _discover_tool_module_names()
            assert result == []
