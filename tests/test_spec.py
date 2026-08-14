from hephaestus.spec import DatasetSpec, from_dict, to_dict, validate, spec_hash, render_summary


def sample_spec():
    return from_dict({
        "spec_version": 1,
        "task": {
            "name": "filesystem-anomaly",
            "description": "Classify fs events",
            "task_type": "score|label",
            "classes": ["Benign", "RansomwareEncrypt", "TempStaging"],
            "score_range": [0, 100],
            "system_prompt": "You are a fs analyst",
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
            "default_benign_score": 5,
            "default_class": "Benign",
        },
        "sources": {
            "local": [{"path": "tests/fixtures/events.jsonl", "label_field": None}],
            "public": [],
            "sampling": {"per_class_cap": 4000, "benign_cap": 20000, "seed": 42},
        },
        "augmentation": {"enabled": False},
        "output": {"train_split": 0.9, "max_seq_length": 512,
                   "target_size": {"min": 800, "ideal": 4000}},
    })


def test_roundtrip_to_dict():
    s = sample_spec()
    d = to_dict(s)
    assert d["task"]["task_type"] == "score|label"
    assert d["labeling"]["rules"][0]["class"] == "RansomwareEncrypt"


def test_valid_spec_has_no_errors():
    assert validate(sample_spec()) == []


def test_unknown_task_type_rejected():
    s = sample_spec()
    s.task.task_type = "freeform"
    assert any("task_type" in e for e in validate(s))


def test_classification_requires_classes():
    s = sample_spec()
    s.task.task_type = "classification"
    s.task.classes = None
    assert any("classes" in e for e in validate(s))


def test_score_requires_score_range():
    s = sample_spec()
    s.task.task_type = "score"
    s.task.score_range = None
    assert any("score_range" in e for e in validate(s))


def test_duplicate_classes_rejected():
    s = sample_spec()
    s.task.classes = ["Benign", "Benign", "TempStaging"]
    assert any("unique" in e.lower() for e in validate(s))


def test_rule_feature_must_exist():
    s = sample_spec()
    s.labeling.rules[0].when = {"field": "nonexistent", "op": "eq", "value": 1}
    assert any("nonexistent" in e for e in validate(s))


def test_label_leakage_rejected():
    s = sample_spec()
    s.sources.local[0].label_field = "path"  # path is also an input feature
    assert any("leak" in e.lower() for e in validate(s))


def test_no_sources_rejected():
    s = sample_spec()
    s.sources.local = []
    s.sources.public = []
    assert any("source" in e.lower() for e in validate(s))


def test_local_source_missing_rejected():
    s = sample_spec()
    s.sources.local[0].path = "tests/fixtures/does-not-exist.jsonl"
    assert any("exist" in e.lower() for e in validate(s))


def test_spec_hash_is_stable():
    a = spec_hash(sample_spec())
    b = spec_hash(sample_spec())
    assert a == b
    assert len(a) == 8  # short hex


def test_render_summary_contains_task_name():
    out = render_summary(sample_spec())
    assert "filesystem-anomaly" in out
    assert "RansomwareEncrypt" in out
