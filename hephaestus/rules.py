"""Deterministic labeling — declarative rules, first-match-wins by priority."""

import re
from typing import Any, Dict, List, Tuple

from .spec import RuleConfig


def _coerce(value: Any, compare: Any) -> Any:
    if isinstance(value, str) and isinstance(compare, (int, float)):
        try:
            return type(compare)(value)
        except (ValueError, TypeError):
            return value
    return value


def match_condition(when: Dict[str, Any], row: Dict[str, Any]) -> bool:
    field = when.get("field")
    op = when.get("op")
    value = when.get("value")
    if field is None or field not in row:
        return False
    actual = row[field]
    actual = _coerce(actual, value)
    if op == "eq":
        return actual == value
    if op == "neq":
        return actual != value
    if op == "gt":
        return _coerce(actual, value) > value
    if op == "gte":
        return _coerce(actual, value) >= value
    if op == "lt":
        return _coerce(actual, value) < value
    if op == "lte":
        return _coerce(actual, value) <= value
    if op == "contains":
        return str(value) in str(actual)
    if op == "startswith":
        return str(actual).startswith(str(value))
    if op == "endswith":
        return str(actual).endswith(str(value))
    if op == "in":
        return actual in value
    if op == "regex":
        return re.search(str(value), str(actual)) is not None
    return False


def evaluate_rules(
    row: Dict[str, Any],
    rules: List[RuleConfig],
    default_class: str,
    default_score: int,
) -> Tuple[str, int]:
    """First matching rule (lowest priority number wins) returns (class, score)."""
    for r in sorted(rules, key=lambda x: x.priority):
        if match_condition(r.when, row):
            return r.class_name, r.score
    return default_class, default_score
