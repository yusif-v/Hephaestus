"""Forge orchestrator — wires wizard -> brain -> build -> kernel -> deliver."""

import json
import os
import subprocess
import time
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


def _push_and_wait_kernel(kernel_path: str, slug: str, poll_seconds: int = 60,
                          max_checks: int = 90) -> dict:
    """Push a kernel to Kaggle, poll its status until terminal, and return.

    The adapter is NOT pulled here — the deliver stage does that so the
    orchestrator can record state between stages. This mirrors the watch_*.sh
    loop from the v31 workflow.
    """
    subprocess.run(["kaggle", "kernels", "push", "-p", kernel_path], check=True)
    for i in range(max_checks):
        status = subprocess.run(
            ["kaggle", "kernels", "status", slug],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        print(f"[forge wait {i + 1}] {status}")
        if "RUNNING" not in status:
            return {"status": status, "checks": i + 1}
        time.sleep(poll_seconds)
    return {"status": "TIMEOUT", "checks": max_checks}


def _pull_kernel_output(slug: str, out_dir: str) -> str:
    """Download kernel output (adapter) to out_dir. Returns adapter dir path."""
    subprocess.run(
        ["kaggle", "kernels", "output", slug, "-p", out_dir, "--force"],
        check=True,
    )
    adapter = os.path.join(out_dir, "forge-lora")
    if not os.path.exists(adapter):
        raise FileNotFoundError(f"no adapter pulled at {adapter}")
    return adapter


def _upload_dataset(dataset_dir: str) -> None:
    """Upload (or version) the built dataset to Kaggle as a private dataset."""
    subprocess.run(["kaggle", "datasets", "create", "-p", dataset_dir],
                   check=False)  # 409 if exists
    subprocess.run(["kaggle", "datasets", "version", "-p", dataset_dir,
                    "-m", "forge dataset build"], check=False)


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
    if not resume or state.get("stage") is None:
        draft = _run_interview()
        spec_dict = design_spec(draft)
        h = spec_hash(from_dict(spec_dict))
        with open(os.path.join(out_dir, "dataset-spec.json"), "w") as f:
            json.dump(spec_dict, f, indent=2)
        state = save_state(out_dir, {"stage": "designed", "spec_hash": h})

    # Interview-only: stop after design (the design->build approval gate).
    if interview_only:
        return load_state(out_dir)

    # Stage 3: build (only if not already built)
    if load_state(out_dir).get("stage") == "designed":
        spec = from_dict(json.load(open(os.path.join(out_dir, "dataset-spec.json"))))
        card = build_dataset(spec, out_dir)
        # Stage the built dataset for Kaggle upload (kernel expects train.jsonl).
        built_dir = os.path.join(out_dir, card["spec_hash"])
        stage_dir = os.path.join(out_dir, "dataset_upload")
        os.makedirs(stage_dir, exist_ok=True)
        import shutil
        shutil.copy(os.path.join(built_dir, "train.jsonl"),
                    os.path.join(stage_dir, "train.jsonl"))
        shutil.copy(os.path.join(built_dir, "test.jsonl"),
                    os.path.join(stage_dir, "test.jsonl"))
        update_stage(out_dir, "built")

    # Stage 4-5: size + kernel build + (optionally) upload/push/train
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
            _upload_dataset(os.path.join(out_dir, "dataset_upload"))
            result = _push_and_wait_kernel(kernel_dir, kernel_slug)
            state = save_state(out_dir, {
                **load_state(out_dir), "kernel_status": result["status"],
            })
            if result["status"] != "TIMEOUT":
                adapter = _pull_kernel_output(
                    kernel_slug, os.path.join(out_dir, "adapter"))
                state = save_state(out_dir, {
                    **load_state(out_dir), "adapter_path": adapter,
                })

    # Stage 6: deliver (marker; GGUF export wired after real kernel returns)
    if load_state(out_dir).get("stage") == "forged":
        update_stage(out_dir, "delivered")

    return load_state(out_dir)
