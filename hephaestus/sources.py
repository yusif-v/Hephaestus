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
    if isinstance(ds, dict):
        ds = ds[source.split] if source.split else next(iter(ds.values()))
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
