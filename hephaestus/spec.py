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
