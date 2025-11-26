from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from jpscripts.core.config import AppConfig
from jpscripts.core import structure
from jpscripts.core import system


def test_get_import_dependencies(tmp_path: Path) -> None:
    root = tmp_path
    file_a = root / "a.py"
    file_b = root / "b.py"
    file_b.write_text("VALUE = 1\n", encoding="utf-8")
    file_a.write_text("import b\n", encoding="utf-8")

    deps = structure.get_import_dependencies(file_a, root)

    assert file_b.resolve() in deps


def test_kill_process_dry_run(tmp_path: Path) -> None:
    config = AppConfig(workspace_root=tmp_path, dry_run=True)
    with patch("psutil.Process") as mock_process:
        result = system.kill_process(1234, force=True, config=config)

    mock_process.assert_not_called()
    assert "dry-run" in result
