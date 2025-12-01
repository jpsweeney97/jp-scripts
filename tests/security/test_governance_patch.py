"""Tests for governance patch blind spot vulnerability.

These tests verify that governance checks operate on PATCHED content,
not the original content on disk. This is critical for detecting
malicious code introduced by patches.
"""

from __future__ import annotations

from pathlib import Path

from jpscripts.governance import ViolationType, check_compliance


class TestGovernancePatchBlindSpot:
    """Test that governance catches violations in patches, not just disk content."""

    def test_catches_malicious_new_file(self, tmp_path: Path) -> None:
        """Governance MUST detect violations in NEW files created by patches.

        This is the critical blind spot: when a patch creates a new file,
        the old implementation would skip it because the file doesn't exist on disk.
        """
        diff = """\
--- /dev/null
+++ b/evil.py
@@ -0,0 +1,3 @@
+import os
+def attack():
+    os.system('rm -rf /')
"""
        violations = check_compliance(diff, tmp_path)
        assert any(v.type == ViolationType.OS_SYSTEM for v in violations), (
            f"Expected OS_SYSTEM violation, got: {violations}"
        )

    def test_catches_shell_true_in_new_file(self, tmp_path: Path) -> None:
        """Governance MUST detect shell=True in NEW files."""
        diff = """\
--- /dev/null
+++ b/bad_subprocess.py
@@ -0,0 +1,3 @@
+import subprocess
+def run():
+    subprocess.run("ls -la", shell=True)
"""
        violations = check_compliance(diff, tmp_path)
        assert any(v.type == ViolationType.SHELL_TRUE for v in violations), (
            f"Expected SHELL_TRUE violation, got: {violations}"
        )

    def test_catches_bare_except_in_new_file(self, tmp_path: Path) -> None:
        """Governance MUST detect bare except in NEW files."""
        diff = """\
--- /dev/null
+++ b/bad_except.py
@@ -0,0 +1,5 @@
+def unsafe():
+    try:
+        risky()
+    except:
+        pass
"""
        violations = check_compliance(diff, tmp_path)
        assert any(v.type == ViolationType.BARE_EXCEPT for v in violations), (
            f"Expected BARE_EXCEPT violation, got: {violations}"
        )

    def test_catches_violation_added_to_existing_file(self, tmp_path: Path) -> None:
        """Governance MUST detect violations ADDED to existing files.

        The old implementation would read the disk content (which is safe)
        instead of the patched content (which has violations).
        """
        # Create a safe file on disk
        safe_file = tmp_path / "safe.py"
        safe_file.write_text("def safe():\n    return 42\n")

        # Patch adds a violation
        diff = """\
--- a/safe.py
+++ b/safe.py
@@ -1,2 +1,5 @@
 def safe():
     return 42
+
+def unsafe():
+    os.system('rm -rf /')
"""
        violations = check_compliance(diff, tmp_path)
        assert any(v.type == ViolationType.OS_SYSTEM for v in violations), (
            f"Expected OS_SYSTEM violation, got: {violations}"
        )

    def test_catches_debug_leftover_in_new_file(self, tmp_path: Path) -> None:
        """Governance MUST detect debug breakpoints in NEW files."""
        diff = """\
--- /dev/null
+++ b/debug_leftover.py
@@ -0,0 +1,3 @@
+def debug_me():
+    breakpoint()
+    return 42
"""
        violations = check_compliance(diff, tmp_path)
        assert any(v.type == ViolationType.DEBUG_LEFTOVER for v in violations), (
            f"Expected DEBUG_LEFTOVER violation, got: {violations}"
        )

    def test_catches_sync_subprocess_in_async_new_file(self, tmp_path: Path) -> None:
        """Governance MUST detect sync subprocess in async context in NEW files."""
        diff = """\
--- /dev/null
+++ b/sync_in_async.py
@@ -0,0 +1,4 @@
+import subprocess
+async def run():
+    result = subprocess.run(["ls"])
+    return result
"""
        violations = check_compliance(diff, tmp_path)
        assert any(v.type == ViolationType.SYNC_SUBPROCESS for v in violations), (
            f"Expected SYNC_SUBPROCESS violation, got: {violations}"
        )


class TestGovernancePatchMultipleFiles:
    """Test governance across multiple files in a single patch."""

    def test_catches_violations_in_multiple_new_files(self, tmp_path: Path) -> None:
        """Governance MUST check ALL files in a multi-file patch."""
        diff = """\
--- /dev/null
+++ b/file_a.py
@@ -0,0 +1,3 @@
+import os
+def bad_a():
+    os.system('echo a')
--- /dev/null
+++ b/file_b.py
@@ -0,0 +1,5 @@
+def bad_b():
+    try:
+        x = 1
+    except:
+        pass
"""
        violations = check_compliance(diff, tmp_path)

        # Should catch os.system in file_a.py
        os_violations = [v for v in violations if v.type == ViolationType.OS_SYSTEM]
        assert len(os_violations) >= 1, f"Expected OS_SYSTEM violation, got: {violations}"

        # Should catch bare except in file_b.py
        except_violations = [v for v in violations if v.type == ViolationType.BARE_EXCEPT]
        assert len(except_violations) >= 1, f"Expected BARE_EXCEPT violation, got: {violations}"
