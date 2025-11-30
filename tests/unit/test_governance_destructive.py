from __future__ import annotations

from pathlib import Path

from jpscripts.core.governance import ViolationType, check_source_compliance


def _violations_for_source(source: str) -> list[ViolationType]:
    violations = check_source_compliance(source, Path("example.py"))
    return [violation.type for violation in violations]


def test_rmtree_fails() -> None:
    source = "import shutil\n\ndef main() -> None:\n    shutil.rmtree('foo')\n"
    violations = _violations_for_source(source)
    assert ViolationType.DESTRUCTIVE_FS in violations


def test_rmtree_override() -> None:
    source = "import shutil\n\ndef main() -> None:\n    shutil.rmtree('foo') # safety: checked\n"
    violations = _violations_for_source(source)
    assert ViolationType.DESTRUCTIVE_FS not in violations
