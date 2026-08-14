from hephaestus.spec import RuleConfig
from hephaestus.rules import evaluate_rules, match_condition


def rule(cls, score, priority, when):
    return RuleConfig(class_name=cls, score=score, priority=priority, when=when)


def test_first_match_wins_by_priority():
    rules = [
        rule("High", 95, 1, {"field": "score_v", "op": "gt", "value": 80}),
        rule("Mid", 50, 2, {"field": "score_v", "op": "gt", "value": 40}),
    ]
    cls, score = evaluate_rules({"score_v": 90}, rules, "Benign", 5)
    assert (cls, score) == ("High", 95)


def test_lower_priority_checked_when_first_doesnt_match():
    rules = [
        rule("High", 95, 1, {"field": "score_v", "op": "gt", "value": 80}),
        rule("Mid", 50, 2, {"field": "score_v", "op": "gt", "value": 40}),
    ]
    cls, score = evaluate_rules({"score_v": 60}, rules, "Benign", 5)
    assert (cls, score) == ("Mid", 50)


def test_unmatched_returns_default():
    rules = [rule("High", 95, 1, {"field": "score_v", "op": "gt", "value": 80})]
    cls, score = evaluate_rules({"score_v": 10}, rules, "Benign", 5)
    assert (cls, score) == ("Benign", 5)


def test_empty_rules_returns_default():
    cls, score = evaluate_rules({"a": 1}, [], "Benign", 5)
    assert (cls, score) == ("Benign", 5)


def test_string_ops():
    assert match_condition({"field": "path", "op": "endswith", "value": ".crypt"},
                           {"path": "/x/file.crypt"})
    assert not match_condition({"field": "path", "op": "endswith", "value": ".crypt"},
                               {"path": "/x/file.txt"})
    assert match_condition({"field": "path", "op": "contains", "value": "/tmp/"},
                           {"path": "/tmp/stage"})
    assert match_condition({"field": "name", "op": "startswith", "value": "stage"},
                           {"name": "stager86"})


def test_in_op():
    assert match_condition({"field": "event", "op": "in", "value": ["create", "modify"]},
                           {"event": "create"})
    assert not match_condition({"field": "event", "op": "in", "value": ["create"]},
                               {"event": "delete"})


def test_comparison_ops_coerce_numeric_strings():
    assert match_condition({"field": "cpu", "op": "gt", "value": 85}, {"cpu": "90"})
    assert match_condition({"field": "cpu", "op": "lte", "value": 85}, {"cpu": "85"})
    assert match_condition({"field": "cpu", "op": "gte", "value": 85}, {"cpu": "85"})
    assert match_condition({"field": "cpu", "op": "eq", "value": 5}, {"cpu": "5"})


def test_regex_op():
    assert match_condition({"field": "path", "op": "regex", "value": r"\.(crypt|locked)$"},
                           {"path": "a.locked"})


def test_missing_field_no_match():
    assert not match_condition({"field": "missing", "op": "eq", "value": 1}, {"a": 1})
