# Hephaestus v0.3 — The Forge Proper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Hephaestus from a train-only framework into a forge that interviews the user, designs a reviewable dataset spec with an LLM brain, builds the dataset (rules-first, ≤15% LLM gap-fill), then trains and delivers a tailored model via Kaggle.

**Architecture:** Six isolated stages (interview → design → build → size → forge → deliver), each a pure function of its input artifact. The `dataset-spec.json` is the contract between stages. Deterministic rule-based labeling is the ground-truth backbone; the LLM only fills gaps the rules can't decide.

**Tech Stack:** Python 3.12 (venv at `.venv/`), stdlib + existing deps (yaml, json, torch/transformers/peft/trl already in venv), `pytest` for tests, `kaggle` CLI for kernels, `uv` for env management.

**Spec:** `docs/superpowers/specs/2026-08-14-forge-dataset-spec-design.md` (uncommitted — `specs/` is gitignored by project convention).

## Global Constraints

- Test runner: `.venv/bin/python -m pytest` (pytest already installed in venv). Run from repo root.
- No new runtime dependencies beyond what's in `pyproject.toml`; the brain uses stdlib `urllib` for LLM calls (no `requests`/`openai` package).
- LLM backend is **never invoked in tests** — always injected as a fake callable.
- Dataset label leakage is forbidden: an assistant answer field must never appear in the prompt fields. Enforced in `spec.validate` and `build`.
- New modules go in `hephaestus/`; tests in `tests/` with fixtures in `tests/fixtures/`.
- Existing `hephaestus/` modules (`cli.py`, `trainer.py`, `evaluator.py`, `exporter.py`, `loader.py`, `registry.py`, `config.py`, `setup.py`) are NOT rewritten except additive changes to `cli.py`.
- Kaggle kernels: match the proven pattern from `kaggle_kernel/ciciot_v31/ciciot-lora-training.ipynb` (CUDA_VISIBLE_DEVICES=0, `%pip uninstall -y torchao`, eager attention, fp16, T4).
- Commit after each task with a `feat:`/`test:`/`refactor:` prefix per repo style.
- `.env` holds `OPENROUTER_API_KEY` and `HEPHAESTUS_BRAIN_MODEL` (never committed).

---

### Task 1: Test scaffolding + spec module

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/__init__.py`
- Create: `tests/test_spec.py`
- Create: `hephaestus/spec.py`
- Modify: `pyproject.toml` (add pytest dev dep — optional, for documentation)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `hephaestus/spec.py` exposes `DatasetSpec` (dataclass tree), `from_dict(raw: dict) -> DatasetSpec`, `to_dict(spec) -> dict`, `validate(spec) -> list[str]`, `spec_hash(spec) -> str`, `render_summary(spec) -> str`. These exact names are used by every later task.

- [ ] **Step 1: Write the failing tests**

Create `tests/conftest.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
```

Create `tests/__init__.py` (empty — makes `tests.test_brain` importable from tests):

Create `tests/test_spec.py`:

```python
from hephaestus.spec import DatasetSpec, from_dict, validate, spec_hash, render_summary


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_spec.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hephaestus.spec'`

- [ ] **Step 3: Write the minimal implementation**

Create `hephaestus/spec.py`:

```python
"""Dataset spec — the contract artifact produced by the brain and executed by build."""

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

TASK_TYPES = ("classification", "score", "score|label")
RULE_OPS = ("eq", "neq", "gt", "gte", "lt", "lte",
            "contains", "startswith", "endswith", "in", "regex")


@dataclass
class TaskConfig:
    name: str = ""
    description: str = ""
    task_type: str = "classification"
    classes: Optional[List[str]] = None
    score_range: Optional[List[int]] = None
    system_prompt: str = ""
    output_format: str = ""


@dataclass
class FeaturesConfig:
    inputs: List[str] = field(default_factory=list)
    field_types: Dict[str, str] = field(default_factory=dict)


@dataclass
class RuleConfig:
    class_name: str = "Benign"
    score: int = 5
    priority: int = 100
    when: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LabelingConfig:
    strategy: str = "hybrid"
    rules: List[RuleConfig] = field(default_factory=list)
    llm_gap_fill: Dict[str, Any] = field(
        default_factory=lambda: {"enabled": False, "max_fraction": 0.15})
    default_benign_score: int = 5
    default_class: str = "Benign"


@dataclass
class LocalSourceConfig:
    path: str = ""
    label_field: Optional[str] = None


@dataclass
class PublicSourceConfig:
    dataset: str = ""
    split: str = "train"
    label_field: Optional[str] = None


@dataclass
class SamplingConfig:
    per_class_cap: int = 4000
    benign_cap: int = 20000
    seed: int = 42


@dataclass
class SourcesConfig:
    local: List[LocalSourceConfig] = field(default_factory=list)
    public: List[PublicSourceConfig] = field(default_factory=list)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


@dataclass
class AugmentationConfig:
    enabled: bool = False


@dataclass
class OutputConfig:
    train_split: float = 0.9
    max_seq_length: int = 512
    target_size: Dict[str, int] = field(
        default_factory=lambda: {"min": 800, "ideal": 4000})


@dataclass
class DatasetSpec:
    spec_version: int = 1
    task: TaskConfig = field(default_factory=TaskConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def from_dict(raw: dict) -> DatasetSpec:
    task = TaskConfig(**raw.get("task", {}))
    feats = FeaturesConfig(**raw.get("features", {}))
    lab = raw.get("labeling", {})
    rules = []
    for r in lab.get("rules", []):
        rules.append(RuleConfig(
            class_name=r.get("class", "Benign"),
            score=int(r.get("score", 5)),
            priority=int(r.get("priority", 100)),
            when=r.get("when", {}),
        ))
    src = raw.get("sources", {})
    local = [LocalSourceConfig(**s) for s in src.get("local", [])]
    public = [PublicSourceConfig(**s) for s in src.get("public", [])]
    sampling = SamplingConfig(**src.get("sampling", {}))
    out = OutputConfig(**raw.get("output", {}))
    return DatasetSpec(
        spec_version=int(raw.get("spec_version", 1)),
        task=task,
        features=feats,
        labeling=LabelingConfig(
            strategy=lab.get("strategy", "hybrid"),
            rules=rules,
            llm_gap_fill=lab.get("llm_gap_fill",
                                  {"enabled": False, "max_fraction": 0.15}),
            default_benign_score=int(lab.get("default_benign_score", 5)),
            default_class=lab.get("default_class", "Benign"),
        ),
        sources=SourcesConfig(local=local, public=public, sampling=sampling),
        augmentation=AugmentationConfig(**raw.get("augmentation", {})),
        output=out,
    )


def to_dict(spec: DatasetSpec) -> dict:
    return {
        "spec_version": spec.spec_version,
        "task": {
            "name": spec.task.name,
            "description": spec.task.description,
            "task_type": spec.task.task_type,
            "classes": spec.task.classes,
            "score_range": spec.task.score_range,
            "system_prompt": spec.task.system_prompt,
            "output_format": spec.task.output_format,
        },
        "features": {
            "inputs": spec.features.inputs,
            "field_types": spec.features.field_types,
        },
        "labeling": {
            "strategy": spec.labeling.strategy,
            "rules": [
                {"class": r.class_name, "score": r.score,
                 "priority": r.priority, "when": r.when}
                for r in spec.labeling.rules
            ],
            "llm_gap_fill": spec.labeling.llm_gap_fill,
            "default_benign_score": spec.labeling.default_benign_score,
            "default_class": spec.labeling.default_class,
        },
        "sources": {
            "local": [
                {"path": s.path, "label_field": s.label_field}
                for s in spec.sources.local
            ],
            "public": [
                {"dataset": s.dataset, "split": s.split, "label_field": s.label_field}
                for s in spec.sources.public
            ],
            "sampling": {
                "per_class_cap": spec.sources.sampling.per_class_cap,
                "benign_cap": spec.sources.sampling.benign_cap,
                "seed": spec.sources.sampling.seed,
            },
        },
        "augmentation": {"enabled": spec.augmentation.enabled},
        "output": {
            "train_split": spec.output.train_split,
            "max_seq_length": spec.output.max_seq_length,
            "target_size": spec.output.target_size,
        },
    }


def _has_label_leakage(spec: DatasetSpec) -> bool:
    inputs = set(spec.features.inputs)
    for src in list(spec.sources.local) + list(spec.sources.public):
        if src.label_field and src.label_field in inputs:
            return True
    return False


def validate(spec: DatasetSpec) -> List[str]:
    """Return list of validation error strings. Empty list means valid."""
    errors: List[str] = []
    if spec.task.task_type not in TASK_TYPES:
        errors.append(f"task_type '{spec.task.task_type}' not in {TASK_TYPES}")
    if spec.task.task_type in ("classification", "score|label"):
        if not spec.task.classes:
            errors.append("classification/score|label requires non-empty task.classes")
        elif len(set(spec.task.classes)) != len(spec.task.classes):
            errors.append("task.classes must be unique")
    if spec.task.task_type in ("score", "score|label"):
        if not spec.task.score_range or len(spec.task.score_range) != 2:
            errors.append("score/score|label requires task.score_range [min,max]")
        elif spec.task.score_range[0] >= spec.task.score_range[1]:
            errors.append("task.score_range must be [min, max] with min < max")
    if not spec.task.system_prompt:
        errors.append("task.system_prompt is required")
    known_fields = set(spec.features.inputs)
    for r in spec.labeling.rules:
        field = r.when.get("field")
        if field and field not in known_fields:
            errors.append(f"rule references unknown feature '{field}'")
        if r.when.get("op") and r.when.get("op") not in RULE_OPS:
            errors.append(f"rule uses unknown op '{r.when.get('op')}'")
    if not spec.sources.local and not spec.sources.public:
        errors.append("at least one source is required (local or public)")
    for src in spec.sources.local:
        if not os.path.exists(src.path):
            errors.append(f"local source does not exist: {src.path}")
    if _has_label_leakage(spec):
        errors.append("label leakage: a label_field cannot also be an input feature")
    frac = spec.labeling.llm_gap_fill.get("max_fraction", 0.15)
    if not (0 < frac <= 0.5):
        errors.append("llm_gap_fill.max_fraction must be in (0, 0.5]")
    if not (0 < spec.output.train_split < 1):
        errors.append("output.train_split must be in (0, 1)")
    return errors


def spec_hash(spec: DatasetSpec) -> str:
    canonical = json.dumps(to_dict(spec), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def render_summary(spec: DatasetSpec) -> str:
    t = spec.task
    lines = [
        f"Task: {t.name}",
        f"  type: {t.task_type}",
        f"  description: {t.description}",
        f"  classes: {', '.join(t.classes) if t.classes else '(score only)'}",
        f"  score_range: {t.score_range if t.score_range else 'n/a'}",
        f"  output_format: {t.output_format or '(default)'}",
        f"  system_prompt: {t.system_prompt[:100]}",
        "Inputs: " + ", ".join(spec.features.inputs),
        f"Labeling: {spec.labeling.strategy} "
        f"({len(spec.labeling.rules)} rules, "
        f"gap_fill={spec.labeling.llm_gap_fill.get('max_fraction', 0.15)})",
        f"Sources: {len(spec.sources.local)} local, {len(spec.sources.public)} public",
        f"Output: split={spec.output.train_split}, "
        f"max_seq={spec.output.max_seq_length}",
    ]
    for r in spec.labeling.rules:
        lines.append(f"  rule: [{r.priority}] {r.when} -> "
                     f"{r.class_name} ({r.score})")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_spec.py -v`
Expected: all PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add hephaestus/spec.py tests/conftest.py tests/__init__.py tests/test_spec.py
git commit -m "feat: dataset spec dataclass, validation, hash, summary"
```

---

### Task 2: Rule evaluator

**Files:**
- Create: `hephaestus/rules.py`
- Create: `tests/test_rules.py`

**Interfaces:**
- Consumes: `hephaestus/spec.py` `RuleConfig`, `LabelingConfig` (from Task 1).
- Produces: `evaluate_rules(row: dict, rules: list[RuleConfig], default_class: str, default_score: int) -> tuple[str, int]` and `match_condition(when: dict, row: dict) -> bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rules.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_rules.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hephaestus.rules'`

- [ ] **Step 3: Write the minimal implementation**

Create `hephaestus/rules.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_rules.py -v`
Expected: all PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add hephaestus/rules.py tests/test_rules.py
git commit -m "feat: deterministic rule evaluator for labeling"
```

---

### Task 3: Source ingestion

**Files:**
- Create: `hephaestus/sources.py`
- Create: `tests/fixtures/events.jsonl`
- Create: `tests/fixtures/events.csv`
- Create: `tests/test_sources.py`

**Interfaces:**
- Consumes: `hephaestus/spec.py` `SourcesConfig`, `FeaturesConfig` (Task 1).
- Produces: `ingest_local(source: LocalSourceConfig, field_types: dict) -> list[dict]`, `ingest_public(source: PublicSourceConfig, field_types: dict, loader=None) -> list[dict]` (loader is injectable for tests; defaults to `datasets.load_dataset`), `unified_table(sources: SourcesConfig, field_types: dict, public_loader=None) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/fixtures/events.jsonl` (5 rows):

```jsonl
{"path": "/tmp/stager86", "event": "create", "recent_count": 4, "score_v": 51}
{"path": "/Users/l/finance/Q2143.xlsx.crypt", "event": "modify", "recent_count": 111, "score_v": 94}
{"path": "/etc/nginx/nginx.conf", "event": "write", "recent_count": 2, "score_v": 66}
{"path": "/var/log/auth.log", "event": "append", "recent_count": 9, "score_v": 91}
{"path": "/home/u/docs/report.pdf", "event": "read", "recent_count": 1, "score_v": 5}
```

Create `tests/fixtures/events.csv`:

```csv
path,event,recent_count,score_v
/tmp/one,create,3,44
/etc/hosts,write,2,66
```

Create `tests/test_sources.py`:

```python
from hephaestus.spec import LocalSourceConfig, PublicSourceConfig, SourcesConfig
from hephaestus.sources import ingest_local, ingest_public, unified_table


def test_ingest_local_jsonl():
    src = LocalSourceConfig(path="tests/fixtures/events.jsonl", label_field=None)
    rows = ingest_local(src, {"path": "string", "recent_count": "int"})
    assert len(rows) == 5
    assert rows[0]["path"] == "/tmp/stager86"
    assert isinstance(rows[0]["recent_count"], int)


def test_ingest_local_csv():
    src = LocalSourceConfig(path="tests/fixtures/events.csv", label_field=None)
    rows = ingest_local(src, {})
    assert len(rows) == 2
    assert rows[1]["path"] == "/etc/hosts"


def test_ingest_local_missing_raises():
    src = LocalSourceConfig(path="tests/fixtures/nope.jsonl", label_field=None)
    try:
        ingest_local(src, {})
        assert False, "should raise"
    except FileNotFoundError:
        pass


def test_ingest_public_with_fake_loader():
    fake = lambda ds, split=None, trust_remote_code=False: {  # noqa: E731
        "train": [
            {"path": "/x/a.crypt", "event": "create", "recent_count": 3},
            {"path": "/y/b.txt", "event": "read", "recent_count": 1},
        ]
    }
    src = PublicSourceConfig(dataset="fake/ds", split="train", label_field=None)
    rows = ingest_public(src, {}, loader=fake)
    assert len(rows) == 2


def test_unified_table_merges_local_and_public():
    fake = lambda ds, split=None, trust_remote_code=False: {  # noqa: E731
        "train": [{"path": "/pub/x", "event": "create", "recent_count": 7}]
    }
    sources = SourcesConfig(
        local=[LocalSourceConfig(path="tests/fixtures/events.jsonl")],
        public=[PublicSourceConfig(dataset="fake/ds", split="train")],
    )
    rows = unified_table(sources, {}, public_loader=fake)
    assert len(rows) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sources.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hephaestus.sources'`

- [ ] **Step 3: Write the minimal implementation**

Create `hephaestus/sources.py`:

```python
"""Data ingestion — local files + public HF/Kaggle datasets -> unified row table."""

import csv
import json
import os
from typing import Any, Callable, Dict, List, Optional

from .spec import LocalSourceConfig, PublicSourceConfig, SourcesConfig


def _coerce_types(row: Dict[str, Any], field_types: Dict[str, str]) -> Dict[str, Any]:
    out = {}
    for k, v in row.items():
        t = field_types.get(k)
        if t == "int":
            try:
                v = int(float(v))
            except (ValueError, TypeError):
                pass
        elif t == "float":
            try:
                v = float(v)
            except (ValueError, TypeError):
                pass
        out[k] = v
    return out


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_csv(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: v for k, v in r.items()})
    return rows


def ingest_local(source: LocalSourceConfig, field_types: Dict[str, str]) -> List[Dict[str, Any]]:
    if not os.path.exists(source.path):
        raise FileNotFoundError(f"local source not found: {source.path}")
    if source.path.endswith(".csv"):
        rows = _read_csv(source.path)
    else:
        rows = _read_jsonl(source.path)
    return [_coerce_types(r, field_types) for r in rows]


def ingest_public(
    source: PublicSourceConfig,
    field_types: Dict[str, str],
    loader: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Pull a public dataset. `loader` is injectable for tests (defaults to HF datasets)."""
    if loader is None:
        from datasets import load_dataset as _default_loader
        loader = _default_loader
    ds = loader(source.dataset, split=source.split, trust_remote_code=True)
    if hasattr(ds, "to_list"):
        ds = ds.to_list()
    return [_coerce_types(dict(r), field_types) for r in ds]


def unified_table(
    sources: SourcesConfig,
    field_types: Dict[str, str],
    public_loader: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for src in sources.local:
        rows.extend(ingest_local(src, field_types))
    for src in sources.public:
        rows.extend(ingest_public(src, field_types, loader=public_loader))
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sources.py -v`
Expected: all PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add hephaestus/sources.py tests/fixtures/events.jsonl tests/fixtures/events.csv tests/test_sources.py
git commit -m "feat: local + public dataset ingestion into unified rows"
```

---

### Task 4: Build engine

**Files:**
- Create: `hephaestus/build.py`
- Create: `tests/test_build.py`

**Interfaces:**
- Consumes: `spec.py` (`DatasetSpec`, `to_dict`, `validate`), `rules.evaluate_rules`, `sources.unified_table` (Tasks 1-3).
- Produces: `render_example(row: dict, label: str, score: int, spec: DatasetSpec) -> dict` (a `{"messages": [...]}` chat example), `build_dataset(spec: DatasetSpec, out_dir: str, llm_gap_fill_fn=None) -> dict` (returns the dataset card dict; writes `train.jsonl`, `test.jsonl`, `dataset_card.json` under `out_dir/<spec_hash>/`). Raises `BuildError` when the dataset has no signal (>95% one class).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build.py`:

```python
import json
import os

import pytest

from hephaestus.build import build_dataset, render_example, BuildError
from hephaestus.spec import from_dict


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
        {"class": "RansomwareEncrypt", "score": 95, "priority": 1,
         "when": {"field": "event", "op": "neq", "value": "__never__"}}
    ]
    with pytest.raises(BuildError):
        build_dataset(spec, str(tmp_path))


def test_gap_fill_applied_and_capped(tmp_path):
    spec = spec_fixture()
    spec.labeling.llm_gap_fill = {"enabled": True, "max_fraction": 0.5}
    spec.labeling.rules = [  # only .crypt matches; rest go to gap-fill
        {"class": "RansomwareEncrypt", "score": 95, "priority": 1,
         "when": {"field": "path", "op": "endswith", "value": ".crypt"}}
    ]
    calls = []

    def fake_gap_fill(row):
        calls.append(row)
        return ("TempStaging", 51)

    card = build_dataset(spec, str(tmp_path), llm_gap_fill_fn=fake_gap_fill)
    assert len(calls) >= 1
    assert card["gap_fill_count"] == len(calls)
    assert card["gap_fill_count"] <= int(5 * 0.5)  # capped at max_fraction
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_build.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hephaestus.build'`

- [ ] **Step 3: Write the minimal implementation**

Create `hephaestus/build.py`:

```python
"""Build engine — executes a DatasetSpec into train/test JSONL + dataset card."""

import json
import os
import random
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from .rules import evaluate_rules
from .spec import DatasetSpec, from_dict, spec_hash, to_dict, validate
from .sources import unified_table


class BuildError(Exception):
    pass


def render_example(row: Dict[str, Any], label: str, score: int, spec: DatasetSpec) -> dict:
    feats = "\n".join(f"{k}: {row[k]}" for k in spec.features.inputs)
    user = f"{spec.task.description}\n\n{feats}" if spec.task.description else feats
    if spec.task.task_type == "classification":
        answer = label
    elif spec.task.task_type == "score":
        answer = str(score)
    else:  # score|label
        answer = f"{score}|{label}"
    return {"messages": [
        {"role": "system", "content": spec.task.system_prompt},
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer},
    ]}


def _stratified_sample(
    rows: List[Tuple[Dict[str, Any], str, int]],
    per_class_cap: int,
    benign_cap: int,
    benign_class: str,
    seed: int,
) -> List[Tuple[Dict[str, Any], str, int]]:
    by_class: Dict[str, List[Tuple[Dict[str, Any], str, int]]] = {}
    for item in rows:
        by_class.setdefault(item[1], []).append(item)
    rng = random.Random(seed)
    out = []
    for cls, items in by_class.items():
        cap = benign_cap if cls == benign_class else per_class_cap
        out.extend(rng.sample(items, min(cap, len(items))))
    rng.shuffle(out)
    return out


def _label_rows(
    rows: List[Dict[str, Any]],
    spec: DatasetSpec,
) -> Tuple[List[Tuple[Dict[str, Any], str, int]], List[Tuple[Dict[str, Any], str, int]]]:
    """Label all rows with rules; rows matched by a rule are returned as
    `labeled`, rows that fell to the default are returned as `unmatched`
    (candidates for LLM gap-fill)."""
    labeled, unmatched = [], []
    default_class = spec.labeling.default_class
    for row in rows:
        cls, score = evaluate_rules(
            row, spec.labeling.rules,
            spec.labeling.default_class, spec.labeling.default_benign_score,
        )
        if cls != default_class:
            labeled.append((row, cls, score))
        else:
            unmatched.append((row, cls, score))
    return labeled, unmatched


def build_dataset(
    spec: DatasetSpec,
    out_dir: str,
    llm_gap_fill_fn: Optional[Callable[[Dict[str, Any]], Tuple[str, int]]] = None,
) -> dict:
    errors = validate(spec)
    if errors:
        raise ValueError("invalid spec: " + "; ".join(errors))

    rows = unified_table(spec.sources, spec.features.field_types)
    if not rows:
        raise BuildError("no rows ingested from sources")

    labeled, unmatched = _label_rows(rows, spec)

    # LLM gap-fill: only unmatched rows, capped at max_fraction of total.
    gap_fill_count = 0
    gap_cfg = spec.labeling.llm_gap_fill
    if gap_cfg.get("enabled") and llm_gap_fill_fn:
        cap = max(0, int(len(rows) * float(gap_cfg.get("max_fraction", 0.15))))
        for row, _cls, _score in unmatched[:cap]:
            cls, score = llm_gap_fill_fn(row)
            labeled.append((row, cls, score))
            gap_fill_count += 1
        # rows beyond the cap keep their rule/default label
        labeled.extend(unmatched[cap:])
    else:
        labeled.extend(unmatched)

    # Stratified sampling.
    sampled = _stratified_sample(
        labeled, spec.sources.sampling.per_class_cap,
        spec.sources.sampling.benign_cap,
        spec.labeling.default_class, spec.sources.sampling.seed,
    )

    if not sampled:
        raise BuildError("no rows survive sampling")
    dist = Counter(cls for _, cls, _ in sampled)
    top_class_ratio = max(dist.values()) / len(sampled)
    if top_class_ratio > 0.95:
        raise BuildError(
            f"no signal: {top_class_ratio:.0%} rows are one class {dist.most_common(1)}. "
            "Edit labeling.rules or sources.")

    # Split.
    rng = random.Random(spec.sources.sampling.seed)
    rng.shuffle(sampled)
    n_test = int(len(sampled) * (1 - spec.output.train_split))
    test_items, train_items = sampled[:n_test], sampled[n_test:]

    h = spec_hash(spec)
    out = os.path.join(out_dir, h)
    os.makedirs(out, exist_ok=True)

    def _write(path, items):
        with open(path, "w") as f:
            for row, cls, score in items:
                f.write(json.dumps(render_example(row, cls, score, spec)) + "\n")

    _write(os.path.join(out, "train.jsonl"), train_items)
    _write(os.path.join(out, "test.jsonl"), test_items)

    card = {
        "spec_hash": h,
        "total_rows": len(sampled),
        "train_rows": len(train_items),
        "test_rows": len(test_items),
        "class_distribution": dict(dist),
        "gap_fill_count": gap_fill_count,
        "seed": spec.sources.sampling.seed,
        "spec": to_dict(spec),
    }
    with open(os.path.join(out, "dataset_card.json"), "w") as f:
        json.dump(card, f, indent=2)
    return card
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_build.py -v`
Expected: all PASS (5 tests). If `_label_rows` gating is wrong, adjust `matched` logic so that only rows where a rule *fired* go to the labeled list (see gap-fill test).

- [ ] **Step 5: Commit**

```bash
git add hephaestus/build.py tests/test_build.py
git commit -m "feat: spec executor — rules labeling, sampling, gap-fill, audit, JSONL emit"
```

---

### Task 5: Wizard (interactive interview)

**Files:**
- Create: `hephaestus/wizard.py`
- Create: `tests/test_wizard.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `run_wizard(ask: Optional[Callable[[str, list], str]] = None) -> dict` (draft answers dict). `ask(prompt, options)` defaults to `input()`; injected in tests.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_wizard.py`:

```python
from hephaestus.wizard import run_wizard, QUESTIONS


def test_question_keys_are_stable():
    keys = [q["key"] for q in QUESTIONS]
    assert "task_type" in keys
    assert "inputs" in keys
    assert "classes" in keys
    assert "sources" in keys
    assert "rules" in keys
    assert "scale" in keys


def test_run_wizard_with_scripted_answers():
    answers = iter([
        "score|label",
        "filesystem events: path, event, recent_count",
        "Benign, RansomwareEncrypt, TempStaging",
        "local tests/fixtures/events.jsonl",
        ".crypt -> RansomwareEncrypt",
        "small",
    ])

    def ask(prompt, options=None):
        return next(answers)

    draft = run_wizard(ask=ask)
    assert draft["task_type"] == "score|label"
    assert "RansomwareEncrypt" in draft["classes"]
    assert "events.jsonl" in draft["sources"]
    assert ".crypt" in draft["rules"]
    assert draft["scale"] == "small"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_wizard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hephaestus.wizard'`

- [ ] **Step 3: Write the minimal implementation**

Create `hephaestus/wizard.py`:

```python
"""Interactive interview — records user answers into a draft dict, no cleverness."""

from typing import Callable, List, Optional


QUESTIONS = [
    {
        "key": "task_type",
        "prompt": "What should the model output? (classification / score / score|label)",
        "options": ["classification", "score", "score|label"],
    },
    {
        "key": "inputs",
        "prompt": "Describe what the model will see. "
                  "Comma-separated field names, or path to a sample file to ingest columns.",
        "options": None,
    },
    {
        "key": "classes",
        "prompt": "What are the answer classes? (comma-separated, or 'score-only')",
        "options": None,
    },
    {
        "key": "sources",
        "prompt": "Where does the data live? Local paths and/or public dataset IDs "
                  "(comma-separated).",
        "options": None,
    },
    {
        "key": "rules",
        "prompt": "Any hard rules you already know? e.g. '.crypt -> RansomwareEncrypt'. "
                  "Leave blank for none.",
        "options": None,
    },
    {
        "key": "scale",
        "prompt": "Scale intent? (small / medium / large)",
        "options": ["small", "medium", "large"],
    },
]


def _default_ask(prompt: str, options: Optional[List[str]] = None) -> str:
    if options:
        print(f"\n{prompt}")
        for i, o in enumerate(options, 1):
            print(f"  {i}. {o}")
        return input("> ").strip()
    return input(f"\n{prompt}\n> ").strip()


def run_wizard(ask: Optional[Callable[[str, Optional[List[str]]], str]] = None) -> dict:
    if ask is None:
        ask = _default_ask
    draft = {}
    for q in QUESTIONS:
        draft[q["key"]] = ask(q["prompt"], q["options"])
    return draft
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_wizard.py -v`
Expected: all PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add hephaestus/wizard.py tests/test_wizard.py
git commit -m "feat: interactive wizard gathering draft task answers"
```

---

### Task 6: Brain (LLM spec designer)

**Files:**
- Create: `hephaestus/brain.py`
- Create: `tests/test_brain.py`

**Interfaces:**
- Consumes: `spec.py` `from_dict`, `validate`, `render_summary` (Task 1); `wizard.run_wizard` draft shape (Task 5).
- Produces: `design_spec(draft: dict, llm_fn: Optional[Callable[[str], str]] = None) -> dict` (returns a spec dict; validates, retries up to 3 on invalid JSON/spec), `_default_llm(prompt) -> str` (OpenRouter via stdlib urllib), `build_brain_prompt(draft: dict) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_brain.py`:

```python
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
    bad = dict(GOOD_SPEC)
    bad["task"]["task_type"] = "freeform"  # invalid

    def flaky(prompt):
        calls["n"] += 1
        return json.dumps(bad if calls["n"] == 1 else GOOD_SPEC)

    spec = design_spec({"task_type": "score|label"}, llm_fn=flaky)
    assert calls["n"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_brain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hephaestus.brain'`

- [ ] **Step 3: Write the minimal implementation**

Create `hephaestus/brain.py`:

```python
"""Brain — LLM client that turns draft answers into a complete DatasetSpec dict."""

import json
import os
import urllib.request
from typing import Callable, Optional


SCHEMA_HINT = (
    "Return a JSON object with these top-level keys: "
    "spec_version (int), task (name, description, task_type, classes, score_range, "
    "system_prompt, output_format), features (inputs list, field_types dict), "
    "labeling (strategy, rules list where each rule is {class, score, priority, when: "
    "{field, op, value}}, llm_gap_fill {enabled, max_fraction}, default_benign_score, "
    "default_class), sources (local [{path, label_field}], public [{dataset, split, "
    "label_field}], sampling {per_class_cap, benign_cap, seed}), augmentation {enabled}, "
    "output (train_split, max_seq_length, target_size {min, ideal}). "
    "Rules use ops: eq neq gt gte lt lte contains startswith endswith in regex. "
    "Do not invent label fields that are also input features."
)


def build_brain_prompt(draft: dict) -> str:
    return (
        "You are Hephaestus, the forge brain. Design a training dataset spec for a "
        "purpose-built LLM automaton.\n\n"
        f"Draft answers from the operator interview:\n{json.dumps(draft, indent=2)}\n\n"
        + SCHEMA_HINT
    )


def _default_llm(prompt: str) -> str:
    """Call OpenRouter (or a configured OpenAI-compatible endpoint) via stdlib urllib."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    model = os.environ.get("HEPHAESTUS_BRAIN_MODEL", "openai/gpt-4o-mini")
    url = os.environ.get(
        "HEPHAESTUS_BRAIN_URL", "https://openrouter.ai/api/v1/chat/completions")
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}" if api_key else "",
    })
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def design_spec(
    draft: dict,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """Produce a spec dict. Validates the result; retries up to 3 times."""
    if llm_fn is None:
        llm_fn = _default_llm
    prompt = build_brain_prompt(draft)
    last_error = None
    for _ in range(3):
        raw = llm_fn(prompt)
        try:
            spec = json.loads(raw)
        except json.JSONDecodeError as e:
            last_error = str(e)
            continue
        from .spec import from_dict, validate
        try:
            errors = validate(from_dict(spec))
        except Exception as e:  # malformed structure
            last_error = str(e)
            continue
        if not errors:
            return spec
        last_error = "; ".join(errors)
    raise ValueError(f"brain could not produce a valid spec after 3 tries: {last_error}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_brain.py -v`
Expected: all PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add hephaestus/brain.py tests/test_brain.py
git commit -m "feat: brain — LLM dataset-spec designer with validation retry"
```

---

### Task 7: Kaggle kernel builder

**Files:**
- Create: `hephaestus/kernel_builder.py`
- Create: `tests/test_kernel_builder.py`

**Interfaces:**
- Consumes: `spec.py` (`DatasetSpec`, `to_dict`) (Task 1).
- Produces: `build_kernel(spec: DatasetSpec, model_size: str, out_dir: str, dataset_slug: str, kernel_slug: str) -> dict` — writes `notebook.ipynb` and `kernel-metadata.json` into `out_dir`; returns `{"kernel_slug": ..., "metadata_path": ..., "notebook_path": ...}`. `MODEL_SIZES: dict[str, dict]` maps `"latency"|"balanced"|"accuracy"` to `{"model": ..., "rank": ..., "alpha": ...}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_kernel_builder.py`:

```python
import json
import os

from hephaestus.kernel_builder import build_kernel, MODEL_SIZES
from hephaestus.spec import from_dict


def spec():
    return from_dict({
        "spec_version": 1,
        "task": {
            "name": "filesystem-anomaly", "description": "fs anomaly",
            "task_type": "score|label",
            "classes": ["Benign", "RansomwareEncrypt"],
            "score_range": [0, 100],
            "system_prompt": "You are a fs analyst", "output_format": "score|type",
        },
        "features": {
            "inputs": ["path", "event"],
            "field_types": {"path": "string", "event": "string"},
        },
        "labeling": {"strategy": "rules", "rules": [], "default_benign_score": 5,
                     "default_class": "Benign"},
        "sources": {"local": [{"path": "x.jsonl"}], "public": [],
                    "sampling": {}},
        "augmentation": {"enabled": False},
        "output": {"train_split": 0.9, "max_seq_length": 512,
                   "target_size": {"min": 800, "ideal": 4000}},
    })


def test_model_sizes_cover_all_profiles():
    assert set(MODEL_SIZES) == {"latency", "balanced", "accuracy"}
    assert MODEL_SIZES["latency"]["model"].endswith("0.5B-Instruct")
    assert MODEL_SIZES["accuracy"]["rank"] > MODEL_SIZES["latency"]["rank"]


def test_build_kernel_writes_notebook_and_metadata(tmp_path):
    result = build_kernel(
        spec(), "latency", str(tmp_path),
        dataset_slug="yusifovtelman/forge-filesystem-anomaly",
        kernel_slug="yusifovtelman/forge-filesystem-anomaly-train",
    )
    nb = json.load(open(result["notebook_path"]))
    assert any("Qwen" in "".join(c.get("source", []))
               for c in nb["cells"] if c["cell_type"] == "code")
    md = json.load(open(result["metadata_path"]))
    assert md["id"] == "yusifovtelman/forge-filesystem-anomaly-train"
    assert md["machine_shape"] == "NvidiaTeslaT4"
    assert md["dataset_sources"] == ["yusifovtelman/forge-filesystem-anomaly"]
    assert "CUDA_VISIBLE_DEVICES" in open(result["notebook_path"]).read()


def test_build_kernel_uses_rank_for_size():
    from hephaestus.kernel_builder import _kernel_source_for_size
    src = _kernel_source_for_size("accuracy")
    assert src["rank"] == MODEL_SIZES["accuracy"]["rank"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_kernel_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hephaestus.kernel_builder'`

- [ ] **Step 3: Write the minimal implementation**

Create `hephaestus/kernel_builder.py`:

```python
"""Kernel builder — approved spec + model size -> Kaggle training notebook + metadata."""

import json
import os
from typing import Any, Dict

from .spec import DatasetSpec


MODEL_SIZES = {
    "latency": {"model": "Qwen/Qwen2.5-0.5B-Instruct", "rank": 64, "alpha": 128},
    "balanced": {"model": "Qwen/Qwen2.5-1.5B-Instruct", "rank": 128, "alpha": 256},
    "accuracy": {"model": "Qwen/Qwen2.5-3B-Instruct", "rank": 256, "alpha": 512},
}


def _kernel_source_for_size(model_size: str) -> Dict[str, Any]:
    if model_size not in MODEL_SIZES:
        raise ValueError(f"model_size must be one of {list(MODEL_SIZES)}")
    return MODEL_SIZES[model_size]


def _build_notebook(spec: DatasetSpec, size_cfg: Dict[str, Any]) -> dict:
    t = spec.task
    cells = []
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [
        f"# Forge — {t.name} (LoRA training)\n",
        f"Task: {t.task_type}. Model: {size_cfg['model']}.",
    ]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [
        "%pip install -q -U transformers datasets trl peft accelerate\n",
        "%pip uninstall -y torchao\n",
    ]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [
        "import os\n",
        "os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # pin single T4\n",
        "import torch\n",
        "assert torch.cuda.is_available(), 'GPU not available'\n",
    ]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [
        "import glob\n",
        "jsonl = None\n",
        "for p in glob.glob('/kaggle/input/**/train.jsonl', recursive=True):\n",
        "    jsonl = p; break\n",
        "assert jsonl, 'dataset not mounted - check dataset_sources'\n",
    ]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [
        "from datasets import load_dataset\n",
        "ds = load_dataset('json', data_files={'train': jsonl})\n",
        "train_ds = ds['train']\n",
    ]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "from peft import LoraConfig, get_peft_model\n",
        f"base = '{size_cfg['model']}'\n",
        "tok = AutoTokenizer.from_pretrained(base)\n",
        "tok.pad_token = tok.eos_token\n",
        "model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16,\n",
        "                                             device_map='cuda', attn_implementation='eager')\n",
        "lora = LoraConfig(r=%d, lora_alpha=%d, lora_dropout=0.05,\n"
        "                  target_modules=['q_proj','k_proj','v_proj','o_proj',\n"
        "                                 'gate_proj','up_proj','down_proj'],\n"
        "                  task_type='CAUSAL_LM')\n" % (size_cfg['rank'], size_cfg['alpha']),
        "model = get_peft_model(model, lora)\n",
    ]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [
        "from trl import SFTConfig, SFTTrainer\n",
        f"cfg = SFTConfig(output_dir='/kaggle/working/forge-lora',\n"
        f"                per_device_train_batch_size=8, gradient_accumulation_steps=4,\n"
        f"                num_train_epochs=1, learning_rate=2e-4, fp16=True,\n"
        f"                max_length={spec.output.max_seq_length}, report_to='none',\n"
        f"                save_strategy='epoch', seed={spec.sources.sampling.seed})\n",
        "trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds)\n",
    ]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [
        "trainer.train()\n",
        "model.save_pretrained('/kaggle/working/forge-lora')\n",
        "tok.save_pretrained('/kaggle/working/forge-lora')\n",
    ]})
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def build_kernel(
    spec: DatasetSpec,
    model_size: str,
    out_dir: str,
    dataset_slug: str,
    kernel_slug: str,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    size_cfg = _kernel_source_for_size(model_size)
    notebook = _build_notebook(spec, size_cfg)
    nb_path = os.path.join(out_dir, "notebook.ipynb")
    md_path = os.path.join(out_dir, "kernel-metadata.json")
    with open(nb_path, "w") as f:
        json.dump(notebook, f, indent=1)
    metadata = {
        "id": kernel_slug,
        "title": f"Forge {spec.task.name} LoRA Training",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [dataset_slug],
    }
    with open(md_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return {"kernel_slug": kernel_slug, "metadata_path": md_path,
            "notebook_path": nb_path}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_kernel_builder.py -v`
Expected: all PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add hephaestus/kernel_builder.py tests/test_kernel_builder.py
git commit -m "feat: Kaggle kernel builder from spec + model size"
```

---

### Task 8: Forge orchestrator + pipeline state

**Files:**
- Create: `hephaestus/forge_state.py`
- Create: `hephaestus/forge.py`
- Create: `tests/test_forge_state.py`
- Create: `tests/test_forge.py`

**Interfaces:**
- Consumes: `wizard.run_wizard`, `brain.design_spec`, `spec.from_dict/validate/render_summary/spec_hash`, `build.build_dataset`, `kernel_builder.build_kernel` (Tasks 1-7).
- Produces: `forge_state.py` exposes `load_state(out_dir) -> dict`, `save_state(out_dir, state)`, `update_stage(out_dir, stage)`. `forge.py` exposes `run_forge(interview_only=False, resume=False) -> dict` (returns final state) and `STAGES = ["designed", "built", "forged", "delivered"]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_forge_state.py`:

```python
import json
import os

from hephaestus.forge_state import load_state, save_state, update_stage


def test_save_load_roundtrip(tmp_path):
    save_state(str(tmp_path), {"stage": "designed", "spec_hash": "abc123"})
    state = load_state(str(tmp_path))
    assert state["stage"] == "designed"
    assert state["spec_hash"] == "abc123"


def test_load_missing_returns_empty(tmp_path):
    assert load_state(str(tmp_path)) == {}


def test_update_stage_keeps_other_fields(tmp_path):
    save_state(str(tmp_path), {"stage": "designed", "spec_hash": "abc"})
    update_stage(str(tmp_path), "built")
    state = load_state(str(tmp_path))
    assert state["stage"] == "built"
    assert state["spec_hash"] == "abc"
```

Create `tests/test_forge.py`:

```python
import json
import os

from hephaestus.forge import run_forge, STAGES


def test_stages_order():
    assert STAGES == ["designed", "built", "forged", "delivered"]


def test_run_forge_end_to_end_with_fakes(tmp_path, monkeypatch):
    # Wizard is fully scripted; brain returns a valid spec for the fixture.
    answers = iter([
        "score|label",
        "path, event, recent_count, score_v",
        "Benign, RansomwareEncrypt, TempStaging",
        "local tests/fixtures/events.jsonl",
        ".crypt -> RansomwareEncrypt",
        "small",
    ])
    monkeypatch.setattr("hephaestus.forge.ask_input", lambda *a: next(answers))
    # Model-size choice via input too.
    size_answers = iter(["latency"])
    monkeypatch.setattr("hephaestus.forge.ask_size", lambda *a: next(size_answers))

    from hephaestus.brain import design_spec
    from tests.test_brain import GOOD_SPEC
    monkeypatch.setattr("hephaestus.forge.design_spec",
                        lambda draft: GOOD_SPEC)
    # Skip actual Kaggle push.
    monkeypatch.setattr("hephaestus.forge._push_and_wait_kernel",
                        lambda kernel_path, slug: {"status": "COMPLETE"})

    state = run_forge(
        out_dir=str(tmp_path),
        dataset_slug="yusifovtelman/forge-test",
        kernel_slug="yusifovtelman/forge-test-train",
        push_kernel=False,
    )
    assert state["stage"] == "delivered"
    assert state["spec_hash"]
    assert os.path.exists(os.path.join(str(tmp_path), state["spec_hash"], "train.jsonl"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_forge_state.py tests/test_forge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hephaestus.forge'`

- [ ] **Step 3: Write the minimal implementation**

Create `hephaestus/forge_state.py`:

```python
"""Pipeline state — tracks which forge stage the current task is at."""

import json
import os
from typing import Any, Dict

STAGES = ["designed", "built", "forged", "delivered"]
STATE_FILE = "forge_state.json"


def _path(out_dir: str) -> str:
    return os.path.join(out_dir, STATE_FILE)


def load_state(out_dir: str) -> Dict[str, Any]:
    p = _path(out_dir)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def save_state(out_dir: str, state: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(_path(out_dir), "w") as f:
        json.dump(state, f, indent=2)


def update_stage(out_dir: str, stage: str) -> Dict[str, Any]:
    state = load_state(out_dir)
    state["stage"] = stage
    save_state(out_dir, state)
    return state
```

Create `hephaestus/forge.py`:

```python
"""Forge orchestrator — wires wizard -> brain -> build -> kernel -> deliver."""

import os
from typing import Callable, Optional

from .brain import design_spec
from .build import build_dataset
from .forge_state import STAGES, load_state, save_state, update_stage
from .kernel_builder import build_kernel
from .spec import from_dict, spec_hash
from .wizard import run_wizard

def _default_ask(prompt: str, options=None) -> str:
    if options:
        prompt += " (" + ", ".join(options) + ")"
    return input(prompt + "> ").strip()


ask_input: Callable = _default_ask
ask_size: Callable = input


def _run_interview() -> dict:
    draft = run_wizard(ask=ask_input)
    return draft


def _choose_size() -> str:
    return ask_size("Model size? (latency / balanced / accuracy) > ").strip().lower()


def _push_and_wait_kernel(kernel_path: str, slug: str) -> dict:
    """Launch kernel via kaggle CLI and poll. Stub-able in tests."""
    import subprocess
    subprocess.run(["kaggle", "kernels", "push", "-p", kernel_path], check=True)
    # Polling loop omitted for brevity — mirrors watch_v31.sh in production.
    return {"status": "COMPLETE"}


def run_forge(
    out_dir: str,
    dataset_slug: str,
    kernel_slug: str,
    push_kernel: bool = True,
    interview_only: bool = False,
    resume: bool = False,
) -> dict:
    state = load_state(out_dir)

    # Stage 1-2: interview + design (skipped when resuming after 'designed')
    if state.get("stage") in (None, "designed") or not resume:
        draft = _run_interview()
        spec_dict = design_spec(draft)
        h = spec_hash(from_dict(spec_dict))
        with open(os.path.join(out_dir, "dataset-spec.json"), "w") as f:
            json.dump(spec_dict, f, indent=2)
        state = save_state(out_dir, {"stage": "designed", "spec_hash": h})
        if interview_only:
            return load_state(out_dir)

    # Stage 3: build (only if not already built)
    if load_state(out_dir).get("stage") == "designed":
        spec = from_dict(json.load(open(os.path.join(out_dir, "dataset-spec.json"))))
        build_dataset(spec, out_dir)
        update_stage(out_dir, "built")

    # Stage 4-5: size + kernel build + (optionally) push/train
    if load_state(out_dir).get("stage") == "built":
        size = _choose_size()
        spec = from_dict(json.load(open(os.path.join(out_dir, "dataset-spec.json"))))
        kernel_dir = os.path.join(out_dir, "kernel")
        build_kernel(
            spec, size, kernel_dir,
            dataset_slug=dataset_slug,
            kernel_slug=kernel_slug,
        )
        update_stage(out_dir, "forged")
        if push_kernel and dataset_slug and kernel_slug:
            _push_and_wait_kernel(kernel_dir, kernel_slug)

    # Stage 6: deliver (marker; GGUF export wired after real kernel returns)
    if load_state(out_dir).get("stage") == "forged":
        update_stage(out_dir, "delivered")

    return load_state(out_dir)
```

**Note for the implementer:** the test above relies on `run_forge` calling `design_spec(draft)` and persisting the approved spec so later stages can reload it. If the flow is hard to fit the test exactly, adjust `run_forge` to: (1) persist `dataset-spec.json` from the approved spec dict after design, (2) reload `from_dict` at each later stage, (3) sequence build → kernel build → push/wait → update_stage through `STAGES`. The tests pin the *observable contract* (final stage `delivered`, files exist, state roundtrip) — the internal wiring can be arranged to satisfy them. The test for `test_run_forge_end_to_end_with_fakes` skips actual kernel push via `push_kernel=False`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_forge_state.py tests/test_forge.py -v`
Expected: all PASS. If `test_run_forge_end_to_end_with_fakes` fails on the final stage, extend `run_forge` so that after `built` it calls `build_kernel` and `_push_and_wait_kernel` (guarded by `push_kernel`), then sets stage `delivered`. Persist `dataset-spec.json` under `out_dir` after design and reload it as the source of truth for build/kernel stages.

- [ ] **Step 5: Commit**

```bash
git add hephaestus/forge_state.py hephaestus/forge.py tests/test_forge_state.py tests/test_forge.py
git commit -m "feat: forge orchestrator + pipeline state with resume"
```

---

### Task 9: CLI wiring

**Files:**
- Modify: `hephaestus/cli.py` (add `forge` and `dataset` subparsers + handlers)
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `forge.run_forge`, `spec.from_dict/render_summary/spec_hash`, `build.build_dataset` (Tasks 1-8).
- Produces: `hephaestus.cli.main()` handles `forge` and `dataset` subcommands.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
from hephaestus.cli import _add_forge_subparser, _add_dataset_subparser
import argparse


def test_forge_subparser_flags():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    _add_forge_subparser(sub)
    args = parser.parse_args(["forge", "--interview-only", "--resume"])
    assert args.command == "forge"
    assert args.interview_only is True
    assert args.resume is True


def test_dataset_subparser_build_command():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    _add_dataset_subparser(sub)
    args = parser.parse_args(["dataset", "build", "spec.json"])
    assert args.dataset_action == "build"
    assert args.spec == "spec.json"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ImportError: cannot import name '_add_forge_subparser'`

- [ ] **Step 3: Implement**

Modify `hephaestus/cli.py`:

Add these functions and register the subparsers in `main()`:

```python
def _add_forge_subparser(subparsers):
    p = subparsers.add_parser("forge", help="Full forge loop: interview -> design -> build -> train -> deliver")
    p.add_argument("--resume", action="store_true", help="Resume from forge_state.json")
    p.add_argument("--interview-only", action="store_true",
                   help="Stop after design and print the spec")
    p.add_argument("--out-dir", default="outputs/forge",
                   help="Working/output directory (default: outputs/forge)")
    p.add_argument("--dataset-slug", default=None, help="Kaggle dataset slug (required for train)")
    p.add_argument("--kernel-slug", default=None, help="Kaggle kernel slug (required for train)")
    return p


def _add_dataset_subparser(subparsers):
    p = subparsers.add_parser("dataset", help="Dataset operations")
    p.add_argument("dataset_action", choices=["build", "show"],
                   help="build from spec, or show readable summary")
    p.add_argument("--spec", type=str, help="Path to dataset-spec.json")
    p.add_argument("--out-dir", default="outputs/forge",
                   help="Output directory (default: outputs/forge)")
    return p


def _cmd_forge(args):
    from .forge import run_forge
    slug = args.dataset_slug
    state = run_forge(
        out_dir=args.out_dir,
        dataset_slug=slug,
        kernel_slug=args.kernel_slug,
        push_kernel=bool(slug and args.kernel_slug),
        interview_only=args.interview_only,
        resume=args.resume,
    )
    print(f"\nForge stage: {state.get('stage')} (spec_hash={state.get('spec_hash')})")


def _cmd_dataset(args):
    from .spec import from_dict, render_summary
    if args.dataset_action == "show":
        spec = from_dict(json.load(open(args.spec)))
        print(render_summary(spec))
    else:  # build
        from .build import build_dataset
        spec = from_dict(json.load(open(args.spec)))
        card = build_dataset(spec, args.out_dir)
        print(f"Built {card['total_rows']} rows -> "
              f"{args.out_dir}/{card['spec_hash']}/")
```

Register in `main()`:

```python
    forge_parser = _add_forge_subparser(subparsers)
    dataset_parser = _add_dataset_subparser(subparsers)
```

And dispatch in the command `if/elif` chain:

```python
    if args.command == "forge":
        _cmd_forge(args)
    elif args.command == "dataset":
        _cmd_dataset(args)
    elif args.command == "train":
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_cli.py -v`
Expected: all PASS (2 tests)

- [ ] **Step 5: Verify existing CLI still parses**

Run: `.venv/bin/python -m hephaestus --help`
Expected: shows `forge` and `dataset` plus existing subcommands; no error.

- [ ] **Step 6: Commit**

```bash
git add hephaestus/cli.py tests/test_cli.py
git commit -m "feat: forge + dataset CLI subcommands"
```

---

### Task 10: Full test suite + docs

**Files:**
- Modify: `README.md` (add Forge quick-start)
- Modify: `hephaestus/__init__.py` (bump `__version__` to `0.3.0`)

**Interfaces:**
- Consumes: all tasks.

- [ ] **Step 1: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all PASS (no skips, no failures)

- [ ] **Step 2: Bump version**

Modify `hephaestus/__init__.py`:

```python
__version__ = "0.3.0"
```

- [ ] **Step 3: Update README**

Append a "Forge (v0.3)" section to `README.md`:

```markdown
## Forge — design your own dataset (v0.3)

`python -m hephaestus forge` runs the full loop: an interactive interview, an LLM
brain that designs a reviewable `dataset-spec.json`, a deterministic build engine
that emits `train.jsonl`/`test.jsonl`, then a Kaggle training run and export.

`python -m hephaestus dataset show spec.json` — review a spec.
`python -m hephaestus dataset build spec.json` — build without training.
```

- [ ] **Step 4: Run full suite again**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add README.md hephaestus/__init__.py
git commit -m "docs: forge quick-start; bump v0.3.0"
```

---

## Test Inventory

| Test file | Covers |
|-----------|--------|
| `tests/test_spec.py` | spec parse/validate/hash/summary/leakage (14 tests) |
| `tests/test_rules.py` | rule priority, ops, defaults (10 tests) |
| `tests/test_sources.py` | local JSONL/CSV, public loader, unified table (5 tests) |
| `tests/test_build.py` | render, emit, reproducibility, no-signal, gap-fill cap (5 tests) |
| `tests/test_wizard.py` | question keys, scripted interview (2 tests) |
| `tests/test_brain.py` | prompt, valid design, retry on bad JSON/spec (4 tests) |
| `tests/test_kernel_builder.py` | size map, notebook+metadata, rank wiring (3 tests) |
| `tests/test_forge_state.py` | state save/load/update (3 tests) |
| `tests/test_forge.py` | stages, end-to-end with fakes (2 tests) |
| `tests/test_cli.py` | subparser flags (2 tests) |

## Out of Scope (deferred)

- Real Kaggle push/watch (production `_push_and_wait_kernel` is a stub; the watch scripts already exist in `kaggle_kernel/`).
- Local sanity-train step before Kaggle (spec Step 5 in the design doc — add once the pipeline runs end-to-end).
- Deliver/GGUF stage (merge + q4_K_M + Modelfile) — reuses existing exporter + `v31_gguf_build` pattern; wired after kernel training returns an adapter.
