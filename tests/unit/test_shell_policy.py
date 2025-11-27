from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _grep(pattern: str, root: Path) -> list[Path]:
    matches: list[Path] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if pattern in text:
            matches.append(path)
    return matches


def test_no_create_subprocess_shell() -> None:
    """Shell=True is forbidden; ensure we never call create_subprocess_shell."""
    offenders = _grep("create_subprocess_shell", REPO_ROOT / "src")
    assert not offenders, f"Found forbidden create_subprocess_shell in: {', '.join(str(p) for p in offenders)}"


def test_no_blocking_subprocess_run_in_commands() -> None:
    """
    Guard against introducing new blocking subprocess.run/Popen calls in command modules.
    ui.py is exempt because it encapsulates fzf interactions and already threads them off.
    """
    roots = [
        REPO_ROOT / "src" / "jpscripts" / "commands",
        REPO_ROOT / "src" / "jpscripts" / "core",
        REPO_ROOT / "src" / "jpscripts" / "mcp",
    ]
    allowlist = {
        REPO_ROOT / "src" / "jpscripts" / "commands" / "ui.py",
        REPO_ROOT / "src" / "jpscripts" / "core" / "git.py",
        REPO_ROOT / "src" / "jpscripts" / "core" / "system.py",
        REPO_ROOT / "src" / "jpscripts" / "core" / "search.py",
    }

    run_offenders: list[Path] = []
    popen_offenders: list[Path] = []
    for root in roots:
        run_offenders.extend([path for path in _grep("subprocess.run(", root) if path not in allowlist])
        popen_offenders.extend([path for path in _grep("subprocess.Popen", root) if path not in allowlist])

    assert not run_offenders, f"Blocking subprocess.run found in: {', '.join(str(p) for p in run_offenders)}"
    assert not popen_offenders, f"subprocess.Popen found in: {', '.join(str(p) for p in popen_offenders)}"
