"""Exporter — merge LoRA, quantize, benchmark, GGUF export, save model card."""

import json
import os
import time
import torch


def export_model(model, tokenizer, eval_results, config):
    """Merge LoRA, optionally quantize, benchmark, GGUF export, save artifacts."""

    output_dir = config.output.dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. Merge LoRA into base model
    print("Merging LoRA weights into base model...")
    merged = model.merge_and_unload()

    # 2. Save merged model (safetensors)
    model_path = os.path.join(output_dir, "model")
    merged.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model saved to {model_path}")

    # 3. GGUF export (Phase 3.2)
    if config.export.gguf:
        _export_gguf(merged, tokenizer, output_dir, config)

    # 4. Benchmark
    benchmark = benchmark_model(merged, tokenizer, config)

    # 5. INT4 quantization benchmark (Phase 3.1 / 3.4)
    int4_benchmark = None
    if config.export.quantize == "int4":
        int4_benchmark = _benchmark_int4(model_path, tokenizer, config)
        benchmark["comparison"] = {
            "fp16_size_mb": benchmark["model_size_mb"],
            "int4_size_mb": int4_benchmark["model_size_mb"],
            "fp16_latency_ms": benchmark["avg_latency_ms"],
            "int4_latency_ms": int4_benchmark["avg_latency_ms"],
            "size_reduction": f"{(1 - int4_benchmark['model_size_mb'] / benchmark['model_size_mb']) * 100:.0f}%",
        }

    # 6. Model card
    model_card = {
        "project": "Hephaestus",
        "version": "0.2.0",
        "base_model": config.model.name,
        "task": config.task_name,
        "task_type": config.evaluation.task_type,
        "system_prompt": config.system_prompt,
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

    # 7. Register in local model registry (Phase 4.3)
    from .registry import register_model
    register_model({
        "name": f"hephaestus-{config.task_name}-{config.model.name.split('/')[-1].lower()}",
        "task": config.task_name,
        "path": model_path,
        "accuracy": eval_results.get("accuracy"),
        "f1": eval_results.get("f1"),
        "size_mb": benchmark.get("model_size_mb"),
    })

    # 8. Clean up checkpoints
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
    dummy_input = tokenizer("Classify: test input", return_tensors="pt")
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


def _benchmark_int4(model_path, tokenizer, config):
    """Benchmark INT4 quantized model (Phase 3.1)."""
    try:
        from transformers import AutoModelForCausalLM
        import bitsandbytes as bnb

        print("\nBenchmarking INT4 quantization...")
        model_int4 = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=True,
            device_map="auto",
        )

        total_params = sum(p.numel() for p in model_int4.parameters())
        # INT4 ~ 0.5 bytes per param + 8-bit scale/zero per group
        model_size_mb = total_params * 0.6 / 1e6

        dummy_input = tokenizer("Classify: test input", return_tensors="pt")
        dummy_input = {k: v.to(model_int4.device) for k, v in dummy_input.items()}

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model_int4.generate(**dummy_input, max_new_tokens=32, do_sample=False)

        # Timed
        times = []
        for _ in range(10):
            t0 = time.time()
            with torch.no_grad():
                _ = model_int4.generate(**dummy_input, max_new_tokens=32, do_sample=False)
            times.append(time.time() - t0)

        avg_latency_ms = sum(times) / len(times) * 1000

        result = {
            "total_parameters": total_params,
            "model_size_mb": round(model_size_mb, 1),
            "avg_latency_ms": round(avg_latency_ms, 1),
        }
        print(f"INT4: {total_params/1e9:.1f}B params, {model_size_mb:.0f} MB, {avg_latency_ms:.1f} ms")
        return result

    except ImportError:
        print("bitsandbytes INT4 not available, skipping INT4 benchmark")
        return {"model_size_mb": 0, "avg_latency_ms": 0}


def _export_gguf(model, tokenizer, output_dir, config):
    """Export model to GGUF format for edge/CPU deployment (Phase 3.2)."""
    gguf_path = os.path.join(output_dir, "model.gguf")

    try:
        # Try using llama.cpp's convert script
        import subprocess
        import sys

        # Save model in a temp directory first
        tmp_dir = output_dir + "/gguf_tmp"
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)

        convert_script = config.export.get("llama_cpp_convert_path",
            "/usr/local/bin/convert-hf-to-gguf.py")

        if os.path.exists(convert_script):
            subprocess.check_call([
                sys.executable, convert_script,
                tmp_dir,
                "--outfile", gguf_path,
                "--outtype", "f16",
            ])
            print(f"GGUF exported: {gguf_path}")
        else:
            print("GGUF export: llama.cpp converter not found. Skipping.")
            print(f"Install: pip install llama-cpp-python or build llama.cpp")
            gguf_path = None

        # Cleanup
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    except Exception as e:
        print(f"GGUF export failed: {e}")
        gguf_path = None

    return gguf_path
