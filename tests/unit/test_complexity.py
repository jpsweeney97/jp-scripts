"""Tests for McCabe cyclomatic complexity analysis."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from jpscripts.core.complexity import (
    McCabeVisitor,
    analyze_file_complexity_sync,
)
from jpscripts.core.result import Err, Ok


class TestMcCabeVisitor:
    """Test the McCabe complexity visitor."""

    def test_base_complexity_is_one(self) -> None:
        """Empty function has complexity 1."""
        import ast

        code = "def foo(): pass"
        tree = ast.parse(code)
        func = tree.body[0]
        visitor = McCabeVisitor()
        visitor.visit(func)
        assert visitor.complexity == 1

    def test_if_adds_complexity(self) -> None:
        """Each if statement adds 1."""
        import ast

        code = """\
def foo(x):
    if x > 0:
        return 1
    return 0
"""
        tree = ast.parse(code)
        func = tree.body[0]
        visitor = McCabeVisitor()
        visitor.visit(func)
        assert visitor.complexity == 2  # 1 base + 1 if

    def test_nested_ifs(self) -> None:
        """Nested ifs each add 1."""
        import ast

        code = """\
def foo(x, y):
    if x > 0:
        if y > 0:
            return 1
    return 0
"""
        tree = ast.parse(code)
        func = tree.body[0]
        visitor = McCabeVisitor()
        visitor.visit(func)
        assert visitor.complexity == 3  # 1 base + 2 ifs

    def test_for_loop_adds_complexity(self) -> None:
        """For loops add 1."""
        import ast

        code = """\
def foo(items):
    for item in items:
        print(item)
"""
        tree = ast.parse(code)
        func = tree.body[0]
        visitor = McCabeVisitor()
        visitor.visit(func)
        assert visitor.complexity == 2  # 1 base + 1 for

    def test_while_loop_adds_complexity(self) -> None:
        """While loops add 1."""
        import ast

        code = """\
def foo(x):
    while x > 0:
        x -= 1
"""
        tree = ast.parse(code)
        func = tree.body[0]
        visitor = McCabeVisitor()
        visitor.visit(func)
        assert visitor.complexity == 2  # 1 base + 1 while

    def test_except_handler_adds_complexity(self) -> None:
        """Except handlers add 1."""
        import ast

        code = """\
def foo():
    try:
        risky()
    except ValueError:
        pass
    except TypeError:
        pass
"""
        tree = ast.parse(code)
        func = tree.body[0]
        visitor = McCabeVisitor()
        visitor.visit(func)
        assert visitor.complexity == 3  # 1 base + 2 except handlers

    def test_bool_op_adds_complexity(self) -> None:
        """Boolean operators add complexity."""
        import ast

        code = """\
def foo(a, b, c):
    if a and b and c:
        return True
    return False
"""
        tree = ast.parse(code)
        func = tree.body[0]
        visitor = McCabeVisitor()
        visitor.visit(func)
        # 1 base + 1 if + 2 (for 3 values in and: n-1 operators)
        assert visitor.complexity == 4

    def test_comprehension_adds_complexity(self) -> None:
        """List comprehensions add complexity."""
        import ast

        code = """\
def foo(items):
    return [x for x in items if x > 0]
"""
        tree = ast.parse(code)
        func = tree.body[0]
        visitor = McCabeVisitor()
        visitor.visit(func)
        # 1 base + 1 for clause + 1 if filter
        assert visitor.complexity == 3

    def test_ternary_adds_complexity(self) -> None:
        """Ternary expressions add 1."""
        import ast

        code = """\
def foo(x):
    return 1 if x > 0 else 0
"""
        tree = ast.parse(code)
        func = tree.body[0]
        visitor = McCabeVisitor()
        visitor.visit(func)
        assert visitor.complexity == 2  # 1 base + 1 ternary


class TestAnalyzeFileComplexity:
    """Test file-level complexity analysis."""

    def test_analyzes_simple_file(self, tmp_path: Path) -> None:
        """Simple file with one function."""
        test_file = tmp_path / "simple.py"
        test_file.write_text("""\
def foo():
    return 42
""")
        result = analyze_file_complexity_sync(test_file)
        assert isinstance(result, Ok)
        fc = result.value
        assert fc.path == test_file
        assert len(fc.functions) == 1
        assert fc.functions[0].name == "foo"
        assert fc.functions[0].cyclomatic == 1
        assert fc.max_cyclomatic == 1
        assert fc.average_cyclomatic == 1.0

    def test_analyzes_complex_file(self, tmp_path: Path) -> None:
        """File with multiple functions of varying complexity."""
        test_file = tmp_path / "complex.py"
        test_file.write_text("""\
def simple():
    return 1

def medium(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0

def complex_func(items):
    total = 0
    for item in items:
        if item > 0:
            total += item
        elif item < 0:
            total -= item
    return total
""")
        result = analyze_file_complexity_sync(test_file)
        assert isinstance(result, Ok)
        fc = result.value
        assert len(fc.functions) == 3
        assert fc.max_cyclomatic >= 3  # complex_func has for + 2 ifs
        assert fc.average_cyclomatic > 1

    def test_handles_async_functions(self, tmp_path: Path) -> None:
        """Async functions are properly analyzed."""
        test_file = tmp_path / "async_file.py"
        test_file.write_text("""\
async def fetch_data(url):
    if url:
        return await make_request(url)
    return None
""")
        result = analyze_file_complexity_sync(test_file)
        assert isinstance(result, Ok)
        fc = result.value
        assert len(fc.functions) == 1
        assert fc.functions[0].is_async is True
        assert fc.functions[0].cyclomatic == 2  # 1 base + 1 if

    def test_returns_error_for_nonexistent(self, tmp_path: Path) -> None:
        """Non-existent file returns error."""
        result = analyze_file_complexity_sync(tmp_path / "missing.py")
        assert isinstance(result, Err)

    def test_returns_error_for_non_python(self, tmp_path: Path) -> None:
        """Non-Python file returns error."""
        test_file = tmp_path / "readme.md"
        test_file.write_text("# Hello")
        result = analyze_file_complexity_sync(test_file)
        assert isinstance(result, Err)

    def test_returns_error_for_syntax_error(self, tmp_path: Path) -> None:
        """File with syntax error returns error."""
        test_file = tmp_path / "broken.py"
        test_file.write_text("def foo(\n")
        result = analyze_file_complexity_sync(test_file)
        assert isinstance(result, Err)

    def test_empty_file_returns_zero_metrics(self, tmp_path: Path) -> None:
        """File with no functions returns zero metrics."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("# Just a comment\nx = 42\n")
        result = analyze_file_complexity_sync(test_file)
        assert isinstance(result, Ok)
        fc = result.value
        assert len(fc.functions) == 0
        assert fc.max_cyclomatic == 0
        assert fc.total_cyclomatic == 0
        assert fc.average_cyclomatic == 0.0

    def test_functions_sorted_by_complexity(self, tmp_path: Path) -> None:
        """Functions should be sorted by complexity (descending)."""
        test_file = tmp_path / "mixed.py"
        test_file.write_text("""\
def simple():
    return 1

def complex(x, y, z):
    if x:
        if y:
            if z:
                return 1
    return 0

def medium(a):
    if a:
        return 1
    return 0
""")
        result = analyze_file_complexity_sync(test_file)
        assert isinstance(result, Ok)
        fc = result.value
        # First function should be most complex
        assert fc.functions[0].name == "complex"
        assert fc.functions[0].cyclomatic == 4  # 1 + 3 ifs
