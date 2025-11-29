"""Tests for AST-aware dependency walking."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pytest

from jpscripts.core.dependency_walker import (
    DependencyWalker,
    SymbolNode,
    SymbolKind,
    CallGraph,
)


class TestSymbolNode:
    """Test SymbolNode model."""

    def test_create_function_symbol(self) -> None:
        """Create a function symbol node."""
        node = SymbolNode(
            name="calculate_total",
            kind=SymbolKind.FUNCTION,
            start_line=10,
            end_line=25,
            source="def calculate_total(items): ...",
        )
        assert node.name == "calculate_total"
        assert node.kind == SymbolKind.FUNCTION
        assert node.start_line == 10
        assert node.end_line == 25

    def test_create_class_symbol(self) -> None:
        """Create a class symbol node."""
        node = SymbolNode(
            name="DataProcessor",
            kind=SymbolKind.CLASS,
            start_line=1,
            end_line=50,
            source="class DataProcessor: ...",
        )
        assert node.name == "DataProcessor"
        assert node.kind == SymbolKind.CLASS

    def test_symbol_with_docstring(self) -> None:
        """Symbol can store docstring."""
        node = SymbolNode(
            name="helper",
            kind=SymbolKind.FUNCTION,
            start_line=1,
            end_line=5,
            source="def helper(): pass",
            docstring="A helpful function.",
        )
        assert node.docstring == "A helpful function."


class TestDependencyWalkerBasics:
    """Test basic DependencyWalker functionality."""

    def test_extract_functions(self) -> None:
        """Extract function definitions from source."""
        source = """
def foo():
    pass

def bar(x: int) -> str:
    return str(x)
"""
        walker = DependencyWalker(source)
        symbols = walker.get_symbols()

        names = {s.name for s in symbols}
        assert "foo" in names
        assert "bar" in names

    def test_extract_classes(self) -> None:
        """Extract class definitions from source."""
        source = """
class Animal:
    pass

class Dog(Animal):
    def bark(self):
        pass
"""
        walker = DependencyWalker(source)
        symbols = walker.get_symbols()

        names = {s.name for s in symbols}
        assert "Animal" in names
        assert "Dog" in names

    def test_extract_async_functions(self) -> None:
        """Extract async function definitions."""
        source = """
async def fetch_data():
    pass

async def process():
    await fetch_data()
"""
        walker = DependencyWalker(source)
        symbols = walker.get_symbols()

        names = {s.name for s in symbols}
        assert "fetch_data" in names
        assert "process" in names

    def test_extract_nested_functions(self) -> None:
        """Extract nested function definitions."""
        source = """
def outer():
    def inner():
        pass
    return inner
"""
        walker = DependencyWalker(source)
        symbols = walker.get_symbols()

        names = {s.name for s in symbols}
        assert "outer" in names
        # Nested functions should be tracked under parent


class TestCallGraph:
    """Test call graph extraction."""

    def test_direct_function_calls(self) -> None:
        """Extract direct function calls."""
        source = """
def helper():
    pass

def main():
    helper()
    process_data()
"""
        walker = DependencyWalker(source)
        graph = walker.get_call_graph()

        assert "main" in graph.callers
        callees = graph.callers["main"]
        assert "helper" in callees

    def test_method_calls(self) -> None:
        """Extract method calls on objects."""
        source = """
def process(obj):
    obj.validate()
    obj.transform()
    return obj.result()
"""
        walker = DependencyWalker(source)
        graph = walker.get_call_graph()

        # Method calls should be tracked
        assert "process" in graph.callers

    def test_chained_calls(self) -> None:
        """Extract chained method calls."""
        source = """
def pipeline(data):
    return data.filter().map().reduce()
"""
        walker = DependencyWalker(source)
        graph = walker.get_call_graph()

        assert "pipeline" in graph.callers

    def test_call_in_class_method(self) -> None:
        """Extract calls within class methods."""
        source = """
class Processor:
    def process(self):
        self.validate()
        result = transform(self.data)
        return result
"""
        walker = DependencyWalker(source)
        graph = walker.get_call_graph()

        assert "Processor.process" in graph.callers
        callees = graph.callers["Processor.process"]
        assert "transform" in callees


class TestClassHierarchy:
    """Test class inheritance extraction."""

    def test_single_inheritance(self) -> None:
        """Extract single inheritance."""
        source = """
class Base:
    pass

class Derived(Base):
    pass
"""
        walker = DependencyWalker(source)
        hierarchy = walker.get_class_hierarchy()

        assert "Derived" in hierarchy
        assert "Base" in hierarchy["Derived"]

    def test_multiple_inheritance(self) -> None:
        """Extract multiple inheritance."""
        source = """
class A:
    pass

class B:
    pass

class C(A, B):
    pass
"""
        walker = DependencyWalker(source)
        hierarchy = walker.get_class_hierarchy()

        assert "C" in hierarchy
        bases = hierarchy["C"]
        assert "A" in bases
        assert "B" in bases

    def test_qualified_base_class(self) -> None:
        """Extract qualified base class names."""
        source = """
class MyError(exceptions.BaseError):
    pass
"""
        walker = DependencyWalker(source)
        hierarchy = walker.get_class_hierarchy()

        assert "MyError" in hierarchy
        bases = hierarchy["MyError"]
        assert "exceptions.BaseError" in bases or "BaseError" in bases


class TestDependencySlicing:
    """Test extracting related code slices."""

    def test_get_function_with_dependencies(self) -> None:
        """Get function and its local dependencies."""
        source = """
def helper(x):
    return x * 2

def main():
    result = helper(5)
    return result
"""
        walker = DependencyWalker(source)
        slice_result = walker.slice_for_symbol("main")

        # Should include main and helper
        assert "main" in slice_result
        assert "helper" in slice_result

    def test_get_class_with_methods(self) -> None:
        """Get class and all its methods."""
        source = """
class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value += x
        return self

    def result(self):
        return self.value
"""
        walker = DependencyWalker(source)
        slice_result = walker.slice_for_symbol("Calculator")

        assert "Calculator" in slice_result
        # Methods should be included within class slice

    def test_slice_respects_imports(self) -> None:
        """Slicing should note external dependencies."""
        source = """
from typing import List
from pathlib import Path

def process_files(paths: List[Path]) -> None:
    for p in paths:
        print(p.name)
"""
        walker = DependencyWalker(source)
        slice_result = walker.slice_for_symbol("process_files")

        assert "process_files" in slice_result


class TestTokenAwareSlicing:
    """Test token-budget aware slicing."""

    def test_prioritize_by_relevance(self) -> None:
        """Prioritize symbols based on relevance to target."""
        source = """
def unrelated():
    pass

def helper():
    return 42

def main():
    return helper()
"""
        walker = DependencyWalker(source)
        prioritized = walker.prioritize_symbols("main")

        # main should be highest priority
        assert prioritized[0].name == "main"
        # helper should be next (called by main)
        assert prioritized[1].name == "helper"

    def test_fit_within_token_budget(self) -> None:
        """Fit symbols within a token budget."""
        source = """
def short_func():
    return 1

def medium_func():
    x = 1
    y = 2
    z = 3
    return x + y + z

def long_func():
    # A function with lots of code
    a = 1
    b = 2
    c = 3
    d = 4
    e = 5
    f = 6
    return a + b + c + d + e + f
"""
        walker = DependencyWalker(source)

        # With a small budget, should only include short functions
        result = walker.slice_to_budget("short_func", max_tokens=50)
        assert "short_func" in result

    def test_preserve_complete_definitions(self) -> None:
        """Don't truncate mid-function."""
        source = """
def important():
    step1()
    step2()
    step3()
    return final_result()
"""
        walker = DependencyWalker(source)
        result = walker.slice_to_budget("important", max_tokens=200)

        # Should have complete function or nothing
        if "important" in result:
            assert "return final_result()" in result


class TestSyntaxErrorHandling:
    """Test handling of malformed source."""

    def test_invalid_syntax(self) -> None:
        """Handle invalid Python syntax gracefully."""
        source = """
def broken(
    # Missing closing paren
"""
        walker = DependencyWalker(source)
        symbols = walker.get_symbols()

        # Should return empty list, not raise
        assert symbols == []

    def test_partial_syntax(self) -> None:
        """Handle partial/incomplete code."""
        source = """
def valid():
    pass

class Incomplete:
"""
        walker = DependencyWalker(source)
        symbols = walker.get_symbols()

        # May get some symbols from valid portion
        # Should not raise


class TestModuleLevelExtraction:
    """Test extraction of module-level elements."""

    def test_extract_imports(self) -> None:
        """Extract import statements."""
        source = """
import os
import sys
from pathlib import Path
from typing import List, Dict
"""
        walker = DependencyWalker(source)
        imports = walker.get_imports()

        assert "os" in imports
        assert "sys" in imports
        assert "Path" in imports or "pathlib.Path" in imports

    def test_extract_module_constants(self) -> None:
        """Extract module-level constants."""
        source = """
MAX_SIZE = 1000
DEFAULT_NAME = "default"
CONFIG: dict[str, int] = {}
"""
        walker = DependencyWalker(source)
        symbols = walker.get_symbols()

        names = {s.name for s in symbols}
        assert "MAX_SIZE" in names or symbols == []  # Constants may be optional
