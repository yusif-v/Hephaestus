import copy
import json

from hephaestus.brain import design_spec, build_brain_prompt
from hephaestus.spec import validate, from_dict


GOOD_SPEC = {
    "spec_version": 1,
    "task": {
        "name": "filesystem-anomaly", "description": "fs anomaly",
        "task_type": "score|label",
        "classes": ["Benign", "RansomwareEncrypt", "TempStaging"],
        "score_range": [0, 100],
        "system_prompt": "You are a filesystem-security analyst",
        "output_format": "score|type",
    },
    "features": {
        "inputs": ["path", "event", "recent_count"],
        "field_types": {"path": "string", "event": "string", "recent_count": "int"},
    },
    "labeling": {
        "strategy": "hybrid",
        "rules": [
            {"class": "RansomwareEncrypt", "score": 95, "priority": 1,
             "when": {"field": "path", "op": "endswith", "value": ".crypt"}},
        ],
        "llm_gap_fill": {"enabled": True, "max_fraction": 0.15},
        "default_benign_score": 5, "default_class": "Benign",
    },
    "sources": {
        "local": [{"path": "tests/fixtures/events.jsonl", "label_field": None}],
        "public": [],
        "sampling": {"per_class_cap": 4000, "benign_cap": 20000, "seed": 42},
    },
    "augmentation": {"enabled": False},
    "output": {"train_split": 0.9, "max_seq_length": 512,
               "target_size": {"min": 800, "ideal": 4000}},
}


def fake_llm(prompt):
    return json.dumps(GOOD_SPEC)


def test_build_prompt_mentions_task_type():
    p = build_brain_prompt({"task_type": "score|label", "inputs": "path, event",
                            "classes": "Benign, RansomwareEncrypt",
                            "sources": "local x.jsonl", "rules": ".crypt -> ransomware",
                            "scale": "small"})
    assert "score|label" in p
    assert "JSON" in p


def test_design_spec_returns_valid_spec_dict():
    spec = design_spec({"task_type": "score|label"}, llm_fn=fake_llm)
    assert validate(from_dict(spec)) == []
    assert spec["task"]["name"] == "filesystem-anomaly"


def test_design_spec_retries_on_invalid_json():
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return "{not json"
        return json.dumps(GOOD_SPEC)

    spec = design_spec({"task_type": "score|label"}, llm_fn=flaky)
    assert calls["n"] == 2
    assert spec["task"]["task_type"] == "score|label"


def test_design_spec_retries_on_invalid_spec():
    calls = {"n": 0}
    bad = copy.deepcopy(GOOD_SPEC)
    bad["task"]["task_type"] = "freeform"  # invalid

    def flaky(prompt):
        calls["n"] += 1
        return json.dumps(bad if calls["n"] == 1 else GOOD_SPEC)

    spec = design_spec({"task_type": "score|label"}, llm_fn=flaky)
    assert calls["n"] == 2
