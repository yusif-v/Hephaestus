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
        "def tok_fn(ex):\n",
        "    msgs = ex['messages']\n",
        "    prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False)\n",
        "    full = tok.apply_chat_template(msgs, tokenize=False)\n",
        "    enc = tok(full, max_length=%d, truncation=True)\n" % spec.output.max_seq_length,
        "    plen = len(tok(prompt, add_special_tokens=False)['input_ids'])\n",
        "    labels = enc['input_ids'].copy()\n",
        "    labels[:plen] = [-100] * plen\n",
        "    return {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'], 'labels': labels}\n",
        "\n",
        "train_ds = train_ds.map(tok_fn, batched=False)\n",
    ]})
    if spec.output.max_steps:
        train_cfg = (
            "                per_device_train_batch_size=8, gradient_accumulation_steps=4,\n"
            "                max_steps=%d, learning_rate=2e-4, fp16=True,\n"
            "                report_to='none', save_strategy='steps', save_steps=400,\n"
            "                logging_steps=25,\n" % spec.output.max_steps
        )
    else:
        train_cfg = (
            "                per_device_train_batch_size=8, gradient_accumulation_steps=4,\n"
            "                num_train_epochs=1, learning_rate=2e-4, fp16=True,\n"
            "                report_to='none', save_strategy='epoch',\n"
        )
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [
        "from trl import SFTConfig, SFTTrainer\n",
        f"cfg = SFTConfig(output_dir='/kaggle/working/forge-lora',\n{train_cfg}"
        f"                seed={spec.sources.sampling.seed})\n",
        "trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds,\n",
        "                     processing_class=tok)\n",
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
