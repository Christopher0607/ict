"""Executable prop-firm rulesets.

Rules live here as data, not as comments scattered through the simulator.
Every ruleset carries an ``effective_date`` because these rules change often
(Apex overhauled its whole model in March 2026).
"""

from rules.ruleset import (
    Ruleset,
    DrawdownType,
    RULESETS,
    get_ruleset,
    list_rulesets,
)

__all__ = [
    "Ruleset",
    "DrawdownType",
    "RULESETS",
    "get_ruleset",
    "list_rulesets",
]
