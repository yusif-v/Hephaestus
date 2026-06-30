"""Exporter — merge LoRA, quantize, benchmark, save model card."""

import json
import os
import time
import torch


def export_model(model, tokenizer, eval_results, config):
    """Merge LoRA, optionally quantize, benchmark, save artifacts."""

    output_dir = config.output.dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. Merge LoRA into base model
    print("Merging LoRA weights into base model...")
    merged = model.merge_and_unload()

    # 2. Save merged model
    model_path = os.path.join(output_dir, "model")
    merged.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

    # 3. Benchmark
    benchmark = benchmark_model(merged, tokenizer, config)

    # 4. Model card
    model_card = {
        "project": "Hephaestus",
        "version": "0.1.0",
        "base_model": config.model.name,
        "task": "HDFS Log Anomaly Detection",
        "method": "QLoRA + SFT",
        "config": config.to_dict(),
        "metrics": eval_results,
        "benchmark": benchmark,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    card_path = os.path.join(output_dir, "model_card.json")
    with open(card_path, "w") as f:
        json.dump(model_card, f, indent=2)
    print(f"Model card saved to {card_path}")

    # 5. Clean up checkpoints
    import shutil
    ckpt_dir = output_dir + "/checkpoints"
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    return model_card


def benchmark_model(model, tokenizer, config):
    """Benchmark inference speed and model size."""
    model.eval()

    # Model size
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 2 / 1e6  # fp16 = 2 bytes per param

    # Inference speed
    dummy_input = tokenizer("Classify: test log entry", return_tensors="pt")
    dummy_input = {k: v.to(model.device) for k, v in dummy_input.items()}

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model.generate(**dummy_input, max_new_tokens=32, do_sample=False)

    # Timed runs
    times = []
    for _ in range(10):
        t0 = time.time()
        with torch.no_grad():
            _ = model.generate(**dummy_input, max_new_tokens=32, do_sample=False)
        times.append(time.time() - t0)

    avg_latency_ms = sum(times) / len(times) * 1000
    tokens_per_sec = 32 / (sum(times) / len(times))

    benchmark = {
        "total_parameters": total_params,
        "model_size_mb": round(model_size_mb, 1),
        "avg_latency_ms": round(avg_latency_ms, 1),
        "tokens_per_second": round(tokens_per_sec, 1),
    }

    print(f"Model: {total_params/1e9:.1f}B params, {model_size_mb:.0f} MB")
    print(f"Latency: {avg_latency_ms:.1f} ms, {tokens_per_sec:.0f} tokens/s")

    return benchmark
