"""Navigation feature.

This module provides workspace navigation utilities.
"""

from jpscripts.features.navigation.service import (
    RecentEntry,
    get_zoxide_projects,
    scan_recent,
)

__all__ = [
    "RecentEntry",
    "get_zoxide_projects",
    "scan_recent",
]
