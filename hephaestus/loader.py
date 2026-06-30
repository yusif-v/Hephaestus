"""Loader — dataset and model loading, supports local files or HuggingFace datasets."""

import json
import os
from typing import List, Dict, Any, Optional


def load_dataset(config) -> List[Dict[str, Any]]:
    """Load training dataset from local file or HuggingFace."""
    path = config.dataset.train_path

    # HuggingFace dataset ID detection (contains "/" but not "/" at start and not a local path)
    if _is_hf_dataset_id(path):
        return _load_from_hf(path, "train", config.dataset.max_samples)

    # Local file
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found: {path}")

    data = _load_jsonl(path)
    if config.dataset.max_samples:
        data = data[: config.dataset.max_samples]
    print(f"Loaded {len(data)} training samples from {path}")
    return data


def load_test_dataset(config) -> List[Dict[str, Any]]:
    """Load test dataset from local file or HuggingFace."""
    path = config.dataset.test_path

    if _is_hf_dataset_id(path):
        return _load_from_hf(path, "test", config.dataset.max_val_samples or 500)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Test data not found: {path}")

    data = _load_jsonl(path)
    max_n = getattr(config.dataset, "max_val_samples", None) or len(data)
    data = data[:max_n]
    print(f"Loaded {len(data)} test samples from {path}")
    return data


def load_model(config):
    """Load base model with LoRA applied. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"Loading model: {config.model.name}")

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float16"}
    torch_dtype = __import__("torch").float16
    if config.model.dtype == "bfloat16":
        torch_dtype = __import__("torch").bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_cfg = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    return model, tokenizer


def _is_hf_dataset_id(path: str) -> bool:
    """Heuristic: HF datasets look like 'user/dataset-name' (one slash, not a path)."""
    if path.startswith("/") or path.startswith(".") or path.startswith("~"):
        return False
    parts = path.split("/")
    if len(parts) == 2 and all(p and "/" not in p for p in parts):
        return True
    return False


def _load_from_hf(dataset_id: str, split: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load a HuggingFace dataset and convert to messages format."""
    from datasets import load_dataset

    print(f"Loading HF dataset: {dataset_id} (split={split})")
    ds = load_dataset(dataset_id, trust_remote_code=True)

    # Handle split naming
    if split not in ds:
        available = list(ds.keys())
        split = available[0]
        print(f"  Split not found, using '{split}'")

    data = ds[split]
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))

    # Convert to messages format based on common CVE dataset structures
    result = []
    for item in data:
        messages = _convert_to_messages(item)
        if messages:
            result.append({"messages": messages})

    print(f"  Converted {len(result)} samples")
    return result


def _convert_to_messages(item: Dict) -> Optional[List[Dict]]:
    """Convert various HF dataset schemas to messages format.

    Supports:
    - AlicanKiraz0/All-CVE-Training-Dataset: {cve_id, description, severity, ...}
    - Generic: {instruction, input, output}
    - Generic: {question, answer}
    - Generic: {prompt, completion}
    """
    # Already in messages format
    if "messages" in item:
        return item["messages"]

    # CVE analysis format
    if "description" in item or "cve_id" in item:
        cve_id = item.get("cve_id", "Unknown")
        description = item.get("description", item.get("text", ""))
        severity = item.get("severity", item.get("label", ""))

        system = (
            "You are a cybersecurity expert specializing in CVE analysis. "
            "Given a CVE description, provide the severity level (CRITICAL, HIGH, MEDIUM, LOW) "
            "and a brief risk summary."
        )
        user = f"Analyze this CVE:\n\nCVE ID: {cve_id}\nDescription: {description}"
        assistant_output = severity if severity else "MEDIUM. Requires further analysis."

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(assistant_output)},
        ]

    # instruction/input/output format
    if "instruction" in item and "output" in item:
        user_content = item["instruction"]
        if item.get("input"):
            user_content += f"\n\n{item['input']}"
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": str(item["output"])},
        ]

    # question/answer format
    if "question" in item and "answer" in item:
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": str(item["question"])},
            {"role": "assistant", "content": str(item["answer"])},
        ]

    # prompt/completion format
    if "prompt" in item and "completion" in item:
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": str(item["prompt"])},
            {"role": "assistant", "content": str(item["completion"])},
        ]

    # Unknown schema — skip
    return None


def _load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file."""
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data
