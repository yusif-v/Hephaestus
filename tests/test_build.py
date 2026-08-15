import json
import os

import pytest

from hephaestus.build import build_dataset, render_example, BuildError
from hephaestus.spec import RuleConfig, from_dict


def spec_fixture():
    return from_dict({
        "spec_version": 1,
        "task": {
            "name": "filesystem-anomaly",
            "description": "fs anomaly",
            "task_type": "score|label",
            "classes": ["Benign", "RansomwareEncrypt", "TempStaging"],
            "score_range": [0, 100],
            "system_prompt": "You are a filesystem-security analyst",
            "output_format": "score|type",
        },
        "features": {
            "inputs": ["path", "event", "recent_count", "score_v"],
            "field_types": {"path": "string", "event": "string",
                            "recent_count": "int", "score_v": "int"},
        },
        "labeling": {
            "strategy": "hybrid",
            "rules": [
                {"class": "RansomwareEncrypt", "score": 95, "priority": 1,
                 "when": {"field": "path", "op": "endswith", "value": ".crypt"}},
                {"class": "TempStaging", "score": 51, "priority": 2,
                 "when": {"field": "path", "op": "startswith", "value": "/tmp/"}},
                {"class": "LogTamper", "score": 91, "priority": 3,
                 "when": {"field": "event", "op": "eq", "value": "append"}},
            ],
            "llm_gap_fill": {"enabled": False, "max_fraction": 0.15},
            "default_benign_score": 5,
            "default_class": "Benign",
        },
        "sources": {
            "local": [{"path": "tests/fixtures/events.jsonl", "label_field": None}],
            "public": [],
            "sampling": {"per_class_cap": 1000, "benign_cap": 1000, "seed": 42},
        },
        "augmentation": {"enabled": False},
        "output": {"train_split": 0.6, "max_seq_length": 512,
                   "target_size": {"min": 3, "ideal": 5}},
    })


def test_render_example_contains_prompt_and_answer():
    ex = render_example(
        {"path": "/x/a.crypt", "event": "create", "recent_count": 3, "score_v": 90},
        "RansomwareEncrypt", 95, spec_fixture())
    msgs = ex["messages"]
    assert msgs[0]["role"] == "system"
    assert "path:" in msgs[1]["content"]
    assert msgs[2]["content"] == "95|RansomwareEncrypt"


def test_build_dataset_emits_train_test_and_card(tmp_path):
    card = build_dataset(spec_fixture(), str(tmp_path))
    assert card["total_rows"] == 5
    assert card["spec_hash"]
    out_dir = os.path.join(str(tmp_path), card["spec_hash"])
    assert os.path.exists(os.path.join(out_dir, "train.jsonl"))
    assert os.path.exists(os.path.join(out_dir, "test.jsonl"))
    assert os.path.exists(os.path.join(out_dir, "dataset_card.json"))
    # no label leakage: 'score_v' is an input feature, never appears in assistant
    for fn in ("train.jsonl", "test.jsonl"):
        for line in open(os.path.join(out_dir, fn)):
            ex = json.loads(line)
            assert "score_v" not in ex["messages"][-1]["content"]


def test_build_dataset_is_reproducible(tmp_path):
    a = build_dataset(spec_fixture(), str(tmp_path))
    b = build_dataset(spec_fixture(), str(tmp_path))
    assert a["spec_hash"] == b["spec_hash"]
    pa = os.path.join(str(tmp_path), a["spec_hash"], "train.jsonl")
    pb = os.path.join(str(tmp_path), b["spec_hash"], "train.jsonl")
    assert open(pa).read() == open(pb).read()


def test_build_detects_no_signal(tmp_path):
    spec = spec_fixture()
    # single-class rule that matches everything -> no signal
    spec.labeling.rules = [
        RuleConfig(class_name="RansomwareEncrypt", score=95, priority=1,
                   when={"field": "event", "op": "neq", "value": "__never__"})
    ]
    with pytest.raises(BuildError):
        build_dataset(spec, str(tmp_path))


def test_build_rejects_missing_input_columns(tmp_path):
    spec = spec_fixture()
    spec.features.inputs.append("mystery_col")
    with pytest.raises(BuildError) as ei:
        build_dataset(spec, str(tmp_path))
    assert "mystery_col" in str(ei.value)


def test_gap_fill_applied_and_capped(tmp_path):
    spec = spec_fixture()
    spec.labeling.llm_gap_fill = {"enabled": True, "max_fraction": 0.5}
    spec.labeling.rules = [  # only .crypt matches; rest go to gap-fill
        RuleConfig(class_name="RansomwareEncrypt", score=95, priority=1,
                   when={"field": "path", "op": "endswith", "value": ".crypt"})
    ]
    calls = []

    def fake_gap_fill(row):
        calls.append(row)
        return ("TempStaging", 51)

    card = build_dataset(spec, str(tmp_path), llm_gap_fill_fn=fake_gap_fill)
    assert len(calls) >= 1
    assert card["gap_fill_count"] == len(calls)
    assert card["gap_fill_count"] <= int(5 * 0.5)  # capped at max_fraction
