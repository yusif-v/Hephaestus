"""CLI entry point with subcommands: train, evaluate, serve, export, models."""

import argparse
import json
import os
import sys

from .config import HephaestusConfig
from .setup import setup_environment, verify_gpu
from .loader import load_dataset, load_test_dataset, load_model
from .trainer import train_model, auto_configure
from .evaluator import evaluate_model
from .exporter import export_model
from .registry import list_models, find_model, remove_model


def cmd_train(args):
    """Train a model from config."""
    config = _load_config(args)

    print("=" * 60)
    print(f"HEPHAESTUS v0.2 — Train: {config.task_name}")
    print("=" * 60)

    # Environment setup
    setup_environment()
    gpu_info = verify_gpu()

    # Auto-configure based on model size and VRAM (Phase 2.3/2.4)
    config = auto_configure(config, gpu_info.get("vram_gb"))

    # Load data
    print("\n" + "=" * 60)
    print("HEPHAESTUS v0.2 — Data Loading")
    print("=" * 60)
    train_data = load_dataset(config)
    test_data = load_test_dataset(config)

    # Load model
    print("\n" + "=" * 60)
    print("HEPHAESTUS v0.2 — Model Loading")
    print("=" * 60)
    model, tokenizer = load_model(config)

    # Train
    print("\n" + "=" * 60)
    print("HEPHAESTUS v0.2 — Training")
    print("=" * 60)
    result, training_metrics = train_model(model, tokenizer, train_data, config)

    # Evaluate
    print("\n" + "=" * 60)
    print("HEPHAESTUS v0.2 — Evaluation")
    print("=" * 60)
    eval_results = evaluate_model(model, tokenizer, test_data, config)

    # Export
    print("\n" + "=" * 60)
    print("HEPHAESTUS v0.2 — Export")
    print("=" * 60)
    model_card = export_model(model, tokenizer, eval_results, config)

    # Final summary
    _print_summary(config, eval_results, training_metrics)


def cmd_evaluate(args):
    """Evaluate an already-trained model."""
    config = _load_config(args)

    print("=" * 60)
    print(f"HEPHAESTUS v0.2 — Evaluate: {config.task_name}")
    print("=" * 60)

    setup_environment()
    verify_gpu()

    test_data = load_test_dataset(config)
    model, tokenizer = load_model(config)
    eval_results = evaluate_model(model, tokenizer, test_data, config)

    print(f"\nAccuracy: {eval_results['accuracy']*100:.1f}%")
    print(f"F1: {eval_results['f1']*100:.1f}%")


def cmd_export(args):
    """Export a merged model to different formats."""
    config = _load_config(args)

    print("=" * 60)
    print(f"HEPHAESTUS v0.2 — Export: {config.task_name}")
    print("=" * 60)

    setup_environment()
    verify_gpu()

    model, tokenizer = load_model(config)
    # For export without retraining, pass empty eval results
    eval_results = getattr(args, 'metrics', None) or {}
    model_card = export_model(model, tokenizer, eval_results, config)

    print(f"\nExport complete: {config.output.dir}")


def cmd_serve(args):
    """Serve a model via API for production (Phase 3.3 — vLLM format)."""
    model_path = args.model_path
    port = args.port or 8000

    try:
        from vllm import LLM, SamplingParams
        print(f"Serving {model_path} on port {port} (vLLM)...")
        # Note: actual vLLM serving requires separate server setup
        print("Use: python -m vllm.entrypoints.openai.api_server --model", model_path)
        print(f"Or programmatically load with: LLM(model='{model_path}')")
    except ImportError:
        print("vLLM not installed. Install with: pip install vllm")
        print(f"Fallback: serve with transformers + FastAPI on port {port}")


def cmd_models(args):
    """Model registry operations: list, find, remove."""
    if args.models_action == "list":
        models = list_models()
        if not models:
            print("No models registered yet.")
        else:
            print(f"{'Name':<45} {'Task':<20} {'Accuracy':<10} {'F1':<10} {'Size':<10}")
            print("-" * 95)
            for m in models:
                print(f"{m.get('name','?'):<45} {m.get('task','?'):<20} "
                      f"{m.get('accuracy','?'):<10} {m.get('f1','?'):<10} "
                      f"{m.get('size_mb','?')} MB")

    elif args.models_action == "find":
        entry = find_model(args.name)
        if entry:
            print(json.dumps(entry, indent=2))
        else:
            print(f"Model not found: {args.name}")

    elif args.models_action == "remove":
        remove_model(args.name)


def main():
    parser = argparse.ArgumentParser(description="Hephaestus v0.2 — Forge tiny purpose-built LLMs")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, help="Path to YAML config file")
    train_parser.add_argument("--profile", choices=["latency", "balanced", "accuracy"],
                             help="Usage profile")
    train_parser.add_argument("--model", type=str, help="Override model name")
    train_parser.add_argument("--hf-token", type=str, help="HuggingFace token")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    eval_parser.add_argument("--model-path", type=str, help="Path to trained model (overrides config)")

    # Export subcommand
    export_parser = subparsers.add_parser("export", help="Export model to different formats")
    export_parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    export_parser.add_argument("--format", choices=["safetensors", "gguf", "onnx"],
                              help="Export format")

    # Serve subcommand
    serve_parser = subparsers.add_parser("serve", help="Serve model via API")
    serve_parser.add_argument("--model-path", type=str, required=True, help="Path to model")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")

    # Models (registry) subcommand
    models_parser = subparsers.add_parser("models", help="Model registry operations")
    models_parser.add_argument("models_action", choices=["list", "find", "remove"],
                              help="Action to perform")
    models_parser.add_argument("--name", type=str, help="Model name for find/remove")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "models":
        cmd_models(args)
    else:
        parser.print_help()


def _load_config(args):
    """Load config from file or profile."""
    if hasattr(args, 'config') and args.config:
        return HephaestusConfig.from_yaml(args.config)
    elif hasattr(args, 'profile') and args.profile:
        return HephaestusConfig.from_profile(args.profile)
    else:
        return HephaestusConfig()


def _print_summary(config, eval_results, training_metrics):
    """Print final training summary."""
    print("\n" + "=" * 60)
    print("HEPHAESTUS v0.2 — COMPLETE")
    print("=" * 60)
    print(f"Task: {config.task_name}")
    print(f"Model: {config.model.name}")
    print(f"Accuracy: {eval_results['accuracy']*100:.1f}%")
    print(f"F1: {eval_results['f1']*100:.1f}%")
    print(f"Train time: {training_metrics['train_time_minutes']:.1f} min")
    print(f"Output: {config.output.dir}")

    if training_metrics.get("overfitting_warning"):
        print(f"\n⚠ {training_metrics['overfitting_warning']}")

    if not eval_results["quality_gate_passed"]:
        print("\n✗ Quality gate NOT passed.")
    else:
        print("\n✓ Quality gate passed.")


if __name__ == "__main__":
    main()
