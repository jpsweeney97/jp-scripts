from __future__ import annotations

import datetime as dt

from jpscripts.commands.web import _slugify_url


def test_slugify_url_with_path() -> None:
    today = dt.date(2024, 1, 2)
    result = _slugify_url("https://example.com/foo/bar", today)
    assert result == "example-com-foo-bar_2024-01-02.yaml"


def test_slugify_url_homepage() -> None:
    today = dt.date(2024, 1, 2)
    result = _slugify_url("https://example.com", today)
    assert result == "example-com-home_2024-01-02.yaml"
