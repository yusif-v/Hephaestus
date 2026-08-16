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


def test_notebook_explicitly_tokenizes_with_prompt_masking(tmp_path):
    """Regression: the generated notebook must tokenize with apply_chat_template
    and -100 prompt masking (proven v31 pattern), not pass raw messages to
    SFTTrainer — raw-messages training produces garbage score|type output."""
    result = build_kernel(
        spec(), "latency", str(tmp_path),
        dataset_slug="yusifovtelman/forge-filesystem-anomaly",
        kernel_slug="yusifovtelman/forge-filesystem-anomaly-train",
    )
    src = open(result["notebook_path"]).read()
    assert "def tok_fn(ex):" in src
    assert "apply_chat_template" in src
    assert "labels[:plen] = [-100] * plen" in src
    assert "train_ds = train_ds.map(tok_fn, batched=False)" in src


def test_notebook_uses_max_steps_when_spec_sets_it(tmp_path):
    s = spec()
    s.output.max_steps = 800
    result = build_kernel(
        s, "latency", str(tmp_path),
        dataset_slug="yusifovtelman/forge-filesystem-anomaly",
        kernel_slug="yusifovtelman/forge-filesystem-anomaly-train",
    )
    src = open(result["notebook_path"]).read()
    assert "max_steps=800" in src
    assert "num_train_epochs" not in src


def test_notebook_uses_full_epoch_when_max_steps_unset(tmp_path):
    s = spec()
    s.output.max_steps = None
    result = build_kernel(
        s, "latency", str(tmp_path),
        dataset_slug="yusifovtelman/forge-filesystem-anomaly",
        kernel_slug="yusifovtelman/forge-filesystem-anomaly-train",
    )
    src = open(result["notebook_path"]).read()
    assert "num_train_epochs=1" in src
    assert "max_steps=" not in src
