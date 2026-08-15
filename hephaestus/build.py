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

    probe = next((r for r in rows if r), rows[0])
    missing = [c for c in spec.features.inputs if c not in probe]
    if missing:
        raise BuildError(
            f"source rows are missing input feature(s): {', '.join(missing)}")

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
