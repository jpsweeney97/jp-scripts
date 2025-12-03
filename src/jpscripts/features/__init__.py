"""Feature modules for jpscripts.

This package contains domain logic separated from infrastructure.
Each feature has its own subpackage with models, services, and utilities.

Subpackages:
    team: Team management and configuration
    notes: Note-taking functionality
    navigation: Workspace navigation utilities

Note: Submodules are not eagerly imported to avoid circular imports.
Import directly from subpackages: e.g., `from jpscripts.features.team import Persona`
"""

__all__ = ["navigation", "notes", "team"]
