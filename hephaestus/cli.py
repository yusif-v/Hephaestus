"""CLI entry point."""

import argparse
import json
import os
import sys
import yaml

from .config import HephaestusConfig
from .setup import setup_environment, verify_gpu
from .loader import load_dataset, load_test_dataset, load_model
from .trainer import train_model
from .evaluator import evaluate_model
from .exporter import export_model


def main():
    parser = argparse.ArgumentParser(description="Hephaestus v0.1 — Forge tiny purpose-built LLMs")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--profile", choices=["latency", "balanced", "accuracy"],
                        help="Usage profile (auto-configures model + params)")
    parser.add_argument("--hf-token", type=str, help="HuggingFace token")
    parser.add_argument("--setup-only", action="store_true", help="Only run setup, don't train")
    args = parser.parse_args()

    # Set HF token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # Load config
    if args.config:
        config = HephaestusConfig.from_yaml(args.config)
    elif args.profile:
        config = HephaestusConfig.from_profile(args.profile)
    else:
        config = HephaestusConfig()  # defaults

    # CLI overrides
    if args.model:
        config.model.name = args.model

    # Phase 1: Environment setup
    print("=" * 60)
    print("HEPHAESTUS v0.1 — Environment Setup")
    print("=" * 60)
    setup_environment()
    gpu_info = verify_gpu()

    if args.setup_only:
        print("Setup complete, exiting.")
        return

    # Phase 2: Load data
    print("\n" + "=" * 60)
    print("HEPHAESTUS v0.1 — Data Loading")
    print("=" * 60)
    train_data = load_dataset(config)
    test_data = load_test_dataset(config)

    # Phase 3: Load model
    print("\n" + "=" * 60)
    print("HEPHAESTUS v0.1 — Model Loading")
    print("=" * 60)
    model, tokenizer = load_model(config)

    # Phase 4: Train
    print("\n" + "=" * 60)
    print("HEPHAESTUS v0.1 — Training")
    print("=" * 60)
    result, training_metrics = train_model(model, tokenizer, train_data, config)

    # Phase 5: Evaluate
    print("\n" + "=" * 60)
    print("HEPHAESTUS v0.1 — Evaluation")
    print("=" * 60)
    eval_results = evaluate_model(model, tokenizer, test_data, config)

    if not eval_results["quality_gate_passed"]:
        print("\n� Quality gate NOT passed. Consider adjusting hyperparameters.")
    else:
        print("\n✓ Quality gate passed.")

    # Phase 6: Export
    print("\n" + "=" * 60)
    print("HEPHAESTUS v0.1 — Export")
    print("=" * 60)
    model_card = export_model(model, tokenizer, eval_results, config)

    # Final summary
    print("\n" + "=" * 60)
    print("HEPHAESTUS v0.1 — COMPLETE")
    print("=" * 60)
    print(f"Model: {config.model.name}")
    print(f"Accuracy: {eval_results['accuracy']*100:.1f}%")
    print(f"F1: {eval_results['f1']*100:.1f}%")
    print(f"Train time: {training_metrics['train_time_minutes']:.1f} min")
    print(f"Output: {config.output.dir}")


if __name__ == "__main__":
    main()
