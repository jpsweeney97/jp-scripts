from __future__ import annotations

from jpscripts.core.team import Persona, get_default_swarm


def test_default_swarm_contains_expected_personas() -> None:
    personas = get_default_swarm()
    names = [p.name for p in personas]
    assert names == ["Architect", "Engineer", "QA"]
    assert all(p.style for p in personas)
    assert all(p.color for p in personas)
