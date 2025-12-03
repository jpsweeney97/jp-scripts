"""Centralized Jinja2 template utilities.

This module provides:
- resolve_template_root: Find the templates directory
- get_template_environment: Cached template environment factory
- render_template: Render a template by name
- Template filters: cdata, tojson
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from jpscripts.core.result import Err, Ok
from jpscripts.core.security import validate_path


def _safe_cdata(content: str) -> str:
    """Escape CDATA terminators inside arbitrary content.

    This prevents XML injection when embedding user content in CDATA sections.

    Args:
        content: Raw string content.

    Returns:
        Content with CDATA terminators escaped.
    """
    return content.replace("]]>", "]]]]><![CDATA[>")


def resolve_template_root(custom_root: Path | None = None) -> Path:
    """Resolve the templates directory path.

    Searches in order:
    1. custom_root if provided and valid
    2. Package templates directory (jpscripts/templates)

    Args:
        custom_root: Optional custom templates directory.

    Returns:
        Resolved path to templates directory.

    Raises:
        FileNotFoundError: If no valid templates directory found.
    """
    if custom_root is not None and custom_root.is_dir():
        return custom_root.resolve()

    # Package templates directory
    package_templates = Path(__file__).parent.parent / "templates"
    if package_templates.is_dir():
        return package_templates.resolve()

    raise FileNotFoundError("No templates directory found")


@lru_cache(maxsize=4)
def get_template_environment(
    template_root: Path,
    *,
    with_filters: bool = True,
) -> Environment:
    """Create or retrieve a cached Jinja2 Environment.

    Args:
        template_root: Directory containing templates.
        with_filters: If True, add custom filters (cdata, tojson).

    Returns:
        Configured Jinja2 Environment.
    """
    env = Environment(loader=FileSystemLoader(str(template_root)), autoescape=False)
    if with_filters:
        env.filters["cdata"] = _safe_cdata
        env.filters["tojson"] = json.dumps
    return env


def render_template(
    name: str,
    context: dict[str, object],
    *,
    template_root: Path | None = None,
    with_filters: bool = True,
) -> str:
    """Render a template by name with the given context.

    Args:
        name: Template filename (e.g., "agent_prompt.j2").
        context: Dictionary of values to pass to the template.
        template_root: Optional custom templates directory.
        with_filters: If True, enable custom filters.

    Returns:
        Rendered template content.

    Raises:
        FileNotFoundError: If template or templates directory not found.
    """
    root = resolve_template_root(template_root)
    env = get_template_environment(root, with_filters=with_filters)
    try:
        template = env.get_template(name)
    except TemplateNotFound as exc:
        raise FileNotFoundError(f"Template {name} not found in {root}") from exc
    return template.render(**context)


def validate_template_path(path: Path | str, workspace_root: Path) -> Path:
    """Validate a template path is safe within workspace.

    Args:
        path: Path to validate.
        workspace_root: Workspace root for validation.

    Returns:
        Validated resolved path.

    Raises:
        PermissionError: If path escapes workspace.
    """
    result = validate_path(path, workspace_root)
    match result:
        case Ok(validated):
            return validated
        case Err(err):
            raise PermissionError(f"Template path validation failed: {err.message}")


__all__ = [
    "get_template_environment",
    "render_template",
    "resolve_template_root",
    "validate_template_path",
]
