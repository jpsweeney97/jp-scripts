from __future__ import annotations

import ast
import os
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from pathspec import PathSpec


def generate_map(root: Path, max_depth: int = 5) -> str:
    """Generate a high-density project map with top-level symbols."""
    root = root.expanduser().resolve()
    gitignore = _load_gitignore(root)

    lines: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        depth = len(rel_dir.parts)

        # Respect depth limit and .gitignore
        dirnames[:] = [
            name
            for name in sorted(dirnames)
            if depth < max_depth and not _is_ignored(rel_dir / name, gitignore)
        ]
        filenames = [
            name for name in sorted(filenames) if not _is_ignored(rel_dir / name, gitignore)
        ]

        for filename in filenames:
            path = Path(dirpath) / filename
            rel_path = path.relative_to(root).as_posix()
            lines.append(rel_path)

            symbols = _summarize_file(path)
            for idx, symbol in enumerate(symbols):
                connector = "├──" if idx < len(symbols) - 1 else "└──"
                lines.append(f"  {connector} {symbol}")

    return "\n".join(lines)


def _load_gitignore(root: Path) -> PathSpec | None:
    gitignore_path = root / ".gitignore"
    if not gitignore_path.exists():
        return None

    patterns = gitignore_path.read_text(encoding="utf-8").splitlines()
    return PathSpec.from_lines("gitwildmatch", patterns)


def _is_ignored(relative_path: Path, gitignore: PathSpec | None) -> bool:
    if relative_path.parts and relative_path.parts[0] == ".git":
        return True

    if gitignore is None:
        return False

    return gitignore.match_file(relative_path.as_posix())


def _summarize_file(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return _summarize_python(path)
    if suffix in {".js", ".jsx", ".ts", ".tsx"}:
        return _summarize_js_ts(path)
    return []


def _summarize_python(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.append(f"class {node.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(_format_function(node))

    return symbols


def _format_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = _format_arguments(node.args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"def {node.name}({args}){ret}"


def _format_arguments(args: ast.arguments) -> str:
    parts: list[str] = []

    def _fmt(arg: ast.arg) -> str:
        if arg.annotation:
            return f"{arg.arg}: {ast.unparse(arg.annotation)}"
        return arg.arg

    parts.extend(_fmt(arg) for arg in args.posonlyargs)
    if args.posonlyargs:
        parts.append("/")

    parts.extend(_fmt(arg) for arg in args.args)
    if args.vararg:
        parts.append(f"*{_fmt(args.vararg)}")
    elif args.kwonlyargs:
        parts.append("*")

    parts.extend(_fmt(arg) for arg in args.kwonlyargs)
    if args.kwarg:
        parts.append(f"**{_fmt(args.kwarg)}")

    return ", ".join(parts)


def _summarize_js_ts(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return []

    class_pattern = re.compile(r"^(?:export\s+)?class\s+(\w+)", re.MULTILINE)
    func_pattern = re.compile(r"^(?:export\s+)?function\s+(\w+)\s*\(([^)]*)", re.MULTILINE)
    const_func_pattern = re.compile(
        r"^(?:export\s+)?const\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>", re.MULTILINE
    )

    symbols: list[str] = []
    for match in class_pattern.finditer(source):
        symbols.append(f"class {match.group(1)}")

    for match in func_pattern.finditer(source):
        params = _normalize_params(match.group(2))
        symbols.append(f"function {match.group(1)}({params})")

    for match in const_func_pattern.finditer(source):
        params = _normalize_params(match.group(2))
        symbols.append(f"const {match.group(1)}({params})")

    return symbols


def _normalize_params(raw: str) -> str:
    params = [p.strip() for p in raw.split(",") if p.strip()]
    return ", ".join(params)


def _resolve_module_to_path(module: str, root: Path) -> Path | None:
    candidate = (root / (module.replace(".", "/"))).with_suffix(".py")
    if candidate.exists():
        return candidate.resolve()
    package_init = root / module.replace(".", "/") / "__init__.py"
    if package_init.exists():
        return package_init.resolve()
    return None


def _iter_imported_modules(tree: ast.AST, current: Path, root: Path) -> Iterable[str]:
    try:
        rel = current.resolve().relative_to(root.resolve())
    except ValueError:
        return
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    base_pkg_parts = parts[:-1] if parts else []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    yield alias.name
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            prefix = ""
            if node.level and base_pkg_parts:
                cutoff = max(0, len(base_pkg_parts) - (node.level - 1))
                prefix_parts = base_pkg_parts[:cutoff]
                prefix = ".".join(prefix_parts).rstrip(".")
                if prefix:
                    module_name = f"{prefix}.{module_name}" if module_name else prefix
            if module_name:
                yield module_name
            for alias in node.names:
                target = alias.name
                if not target:
                    continue
                if module_name:
                    yield f"{module_name}.{target}"
                elif prefix:
                    yield f"{prefix}.{target}"


@lru_cache(maxsize=2048)
def _cached_import_dependencies(path_str: str, root_str: str) -> tuple[str, ...]:
    path = Path(path_str)
    root = Path(root_str)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return ()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()

    dependencies: set[str] = set()
    for module in _iter_imported_modules(tree, path, root):
        resolved = _resolve_module_to_path(module, root)
        if resolved and resolved.exists():
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                continue
            dependencies.add(str(resolved.resolve()))
    return tuple(sorted(dependencies))


def get_import_dependencies(path: Path, root: Path) -> set[Path]:
    """
    Return resolved dependency file paths for imports within `path` under `root`.
    Only returns paths that exist. Cached for efficiency across repeated calls.
    """
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    cached = _cached_import_dependencies(str(resolved_path), str(resolved_root))
    return {Path(item) for item in cached}
