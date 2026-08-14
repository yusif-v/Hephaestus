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
