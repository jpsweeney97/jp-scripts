"""AST-aware dependency walking for semantic code slicing.

This module provides tools to analyze Python source code and extract:
- Symbol definitions (functions, classes, constants)
- Call graphs (what functions call what)
- Class hierarchies (inheritance relationships)
- Import dependencies

Key classes:
- SymbolNode: Represents a code symbol with metadata
- CallGraph: Maps callers to callees
- DependencyWalker: Main analyzer class

[invariant:typing] All types are explicit; mypy --strict compliant.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Final


class SymbolKind(Enum):
    """Classification of code symbols."""

    FUNCTION = auto()
    ASYNC_FUNCTION = auto()
    CLASS = auto()
    CONSTANT = auto()
    IMPORT = auto()


@dataclass(frozen=True)
class SymbolNode:
    """A code symbol extracted from the AST.

    Attributes:
        name: Symbol name (e.g., 'calculate_total', 'DataProcessor')
        kind: Type of symbol (function, class, etc.)
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed)
        source: Original source code for this symbol
        docstring: Optional docstring if present
        parent: Parent symbol name for nested definitions
    """

    name: str
    kind: SymbolKind
    start_line: int
    end_line: int
    source: str
    docstring: str | None = None
    parent: str | None = None


@dataclass
class CallGraph:
    """Maps function/method callers to their callees.

    Attributes:
        callers: Dict mapping caller names to sets of callee names
        callees: Dict mapping callee names to sets of caller names (reverse index)
    """

    callers: dict[str, set[str]] = field(default_factory=dict)
    callees: dict[str, set[str]] = field(default_factory=dict)

    def add_call(self, caller: str, callee: str) -> None:
        """Record a call relationship."""
        if caller not in self.callers:
            self.callers[caller] = set()
        self.callers[caller].add(callee)

        if callee not in self.callees:
            self.callees[callee] = set()
        self.callees[callee].add(caller)


class _CallVisitor(ast.NodeVisitor):
    """Visitor to extract function/method calls."""

    def __init__(self, current_scope: str) -> None:
        self.current_scope = current_scope
        self.calls: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        """Extract call target name."""
        target = self._extract_call_name(node.func)
        if target:
            self.calls.add(target)
        self.generic_visit(node)

    def _extract_call_name(self, node: ast.expr) -> str | None:
        """Extract the name of a called function/method."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            # For obj.method(), extract 'method'
            # For qualified calls like module.func(), extract 'func'
            return node.attr
        return None


class DependencyWalker:
    """Walks Python AST to extract symbols and their dependencies.

    Usage:
        walker = DependencyWalker(source_code)
        symbols = walker.get_symbols()
        graph = walker.get_call_graph()
        slice_code = walker.slice_for_symbol("main")

    [invariant:typing] All types explicit; mypy --strict compliant
    """

    _CHARS_PER_TOKEN: Final[int] = 4

    def __init__(self, source: str) -> None:
        """Initialize with Python source code.

        Args:
            source: Python source code to analyze
        """
        self._source = source
        self._lines = source.splitlines()
        self._tree: ast.Module | None = None
        self._symbols: list[SymbolNode] | None = None
        self._call_graph: CallGraph | None = None
        self._class_hierarchy: dict[str, list[str]] | None = None
        self._imports: set[str] | None = None
        self._parse_error: bool = False

        self._parse()

    def _parse(self) -> None:
        """Parse the source code into an AST."""
        try:
            self._tree = ast.parse(self._source)
        except SyntaxError:
            self._parse_error = True
            self._tree = None

    def get_symbols(self) -> list[SymbolNode]:
        """Get all symbol definitions from the source.

        Returns:
            List of SymbolNode objects for each definition
        """
        if self._symbols is not None:
            return self._symbols

        if self._tree is None:
            self._symbols = []
            return self._symbols

        symbols: list[SymbolNode] = []

        for node in ast.walk(self._tree):
            symbol = self._node_to_symbol(node)
            if symbol:
                symbols.append(symbol)

        # Sort by line number
        symbols.sort(key=lambda s: s.start_line)
        self._symbols = symbols
        return self._symbols

    def _node_to_symbol(
        self,
        node: ast.AST,
        parent: str | None = None,
    ) -> SymbolNode | None:
        """Convert an AST node to a SymbolNode if applicable."""
        if isinstance(node, ast.FunctionDef):
            return self._function_to_symbol(node, SymbolKind.FUNCTION, parent)
        if isinstance(node, ast.AsyncFunctionDef):
            return self._function_to_symbol(node, SymbolKind.ASYNC_FUNCTION, parent)
        if isinstance(node, ast.ClassDef):
            return self._class_to_symbol(node, parent)
        return None

    def _function_to_symbol(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        kind: SymbolKind,
        parent: str | None,
    ) -> SymbolNode:
        """Convert a function node to SymbolNode."""
        start = node.lineno
        end = node.end_lineno or start
        source = self._extract_source(start, end)
        docstring = ast.get_docstring(node, clean=True)

        name = f"{parent}.{node.name}" if parent else node.name

        return SymbolNode(
            name=name,
            kind=kind,
            start_line=start,
            end_line=end,
            source=source,
            docstring=docstring,
            parent=parent,
        )

    def _class_to_symbol(
        self,
        node: ast.ClassDef,
        parent: str | None,
    ) -> SymbolNode:
        """Convert a class node to SymbolNode."""
        start = node.lineno
        end = node.end_lineno or start
        source = self._extract_source(start, end)
        docstring = ast.get_docstring(node, clean=True)

        name = f"{parent}.{node.name}" if parent else node.name

        return SymbolNode(
            name=name,
            kind=SymbolKind.CLASS,
            start_line=start,
            end_line=end,
            source=source,
            docstring=docstring,
            parent=parent,
        )

    def _extract_source(self, start: int, end: int) -> str:
        """Extract source code lines."""
        if start < 1:
            start = 1
        if end > len(self._lines):
            end = len(self._lines)
        return "\n".join(self._lines[start - 1 : end])

    def get_call_graph(self) -> CallGraph:
        """Get the call graph for functions/methods.

        Returns:
            CallGraph mapping callers to callees
        """
        if self._call_graph is not None:
            return self._call_graph

        self._call_graph = CallGraph()

        if self._tree is None:
            return self._call_graph

        self._extract_calls_from_tree(self._tree, None)
        return self._call_graph

    def _extract_calls_from_tree(
        self,
        tree: ast.AST,
        scope: str | None,
    ) -> None:
        """Recursively extract calls from AST nodes."""
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                func_scope = f"{scope}.{node.name}" if scope else node.name
                self._extract_calls_from_function(node, func_scope)
            elif isinstance(node, ast.ClassDef):
                class_scope = f"{scope}.{node.name}" if scope else node.name
                self._extract_calls_from_tree(node, class_scope)
            else:
                self._extract_calls_from_tree(node, scope)

    def _extract_calls_from_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        scope: str,
    ) -> None:
        """Extract all calls made within a function."""
        visitor = _CallVisitor(scope)
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                visitor.visit_Call(child)

        assert self._call_graph is not None
        for callee in visitor.calls:
            self._call_graph.add_call(scope, callee)

    def get_class_hierarchy(self) -> dict[str, list[str]]:
        """Get class inheritance relationships.

        Returns:
            Dict mapping class names to lists of base class names
        """
        if self._class_hierarchy is not None:
            return self._class_hierarchy

        self._class_hierarchy = {}

        if self._tree is None:
            return self._class_hierarchy

        for node in ast.walk(self._tree):
            if isinstance(node, ast.ClassDef):
                bases: list[str] = []
                for base in node.bases:
                    base_name = self._extract_base_name(base)
                    if base_name:
                        bases.append(base_name)
                if bases:
                    self._class_hierarchy[node.name] = bases

        return self._class_hierarchy

    def _extract_base_name(self, node: ast.expr) -> str | None:
        """Extract the name of a base class."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            # Handle qualified names like module.ClassName
            parts: list[str] = [node.attr]
            current: ast.expr = node.value
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            parts.reverse()
            return ".".join(parts)
        return None

    def get_imports(self) -> set[str]:
        """Get all imported names.

        Returns:
            Set of imported module/symbol names
        """
        if self._imports is not None:
            return self._imports

        self._imports = set()

        if self._tree is None:
            return self._imports

        for node in ast.walk(self._tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    self._imports.add(name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name == "*":
                        self._imports.add(module)
                    else:
                        self._imports.add(name)

        return self._imports

    def slice_for_symbol(self, target: str) -> str:
        """Get code slice including target symbol and its dependencies.

        Args:
            target: Name of the symbol to slice for

        Returns:
            Source code slice containing target and dependencies
        """
        symbols = self.get_symbols()
        call_graph = self.get_call_graph()

        # Find the target symbol
        target_symbol: SymbolNode | None = None
        for s in symbols:
            if s.name == target:
                target_symbol = s
                break

        if target_symbol is None:
            return ""

        # Collect all dependent symbols
        needed_names = self._collect_dependencies(target, call_graph)
        needed_names.add(target)

        # Build slice from needed symbols
        included_symbols: list[SymbolNode] = []
        for s in symbols:
            base_name = s.name.split(".")[-1] if "." in s.name else s.name
            if s.name in needed_names or base_name in needed_names:
                included_symbols.append(s)

        if not included_symbols:
            return target_symbol.source

        included_symbols.sort(key=lambda s: s.start_line)
        return "\n\n".join(s.source for s in included_symbols)

    def _collect_dependencies(
        self,
        target: str,
        graph: CallGraph,
        visited: set[str] | None = None,
    ) -> set[str]:
        """Recursively collect all dependencies of a symbol."""
        if visited is None:
            visited = set()

        if target in visited:
            return set()
        visited.add(target)

        deps: set[str] = set()
        if target in graph.callers:
            for callee in graph.callers[target]:
                deps.add(callee)
                deps.update(self._collect_dependencies(callee, graph, visited))

        return deps

    def prioritize_symbols(self, target: str) -> list[SymbolNode]:
        """Get symbols prioritized by relevance to target.

        Args:
            target: Name of the primary symbol

        Returns:
            List of symbols sorted by relevance (most relevant first)
        """
        symbols = self.get_symbols()
        call_graph = self.get_call_graph()

        # Find target
        target_symbol: SymbolNode | None = None
        for s in symbols:
            if s.name == target:
                target_symbol = s
                break

        if target_symbol is None:
            return symbols

        # Calculate relevance scores
        deps = self._collect_dependencies(target, call_graph)

        def relevance_score(s: SymbolNode) -> int:
            if s.name == target:
                return 0  # Highest priority
            base_name = s.name.split(".")[-1] if "." in s.name else s.name
            if s.name in deps or base_name in deps:
                return 1  # Direct dependency
            return 2  # Unrelated

        return sorted(symbols, key=lambda s: (relevance_score(s), s.start_line))

    def slice_to_budget(
        self,
        target: str,
        max_tokens: int,
    ) -> str:
        """Get code slice fitting within token budget.

        Prioritizes:
        1. Target symbol (always included if fits)
        2. Direct dependencies
        3. Other related code

        Args:
            target: Name of the primary symbol
            max_tokens: Maximum token budget

        Returns:
            Source code slice within budget
        """
        prioritized = self.prioritize_symbols(target)
        if not prioritized:
            return ""

        max_chars = max_tokens * self._CHARS_PER_TOKEN
        result_parts: list[str] = []
        used_chars = 0

        for symbol in prioritized:
            source_len = len(symbol.source) + 2  # +2 for newlines
            if used_chars + source_len <= max_chars:
                result_parts.append(symbol.source)
                used_chars += source_len
            elif symbol.name == target:
                # Always try to include target, even if truncated
                remaining = max_chars - used_chars
                if remaining > 0:
                    result_parts.append(symbol.source[:remaining])
                    used_chars = max_chars
                break

        return "\n\n".join(result_parts)


__all__ = [
    "CallGraph",
    "DependencyWalker",
    "SymbolKind",
    "SymbolNode",
]
