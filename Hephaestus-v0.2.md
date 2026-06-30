# Plan: Hephaestus v0.2 — Multi-Automaton Pipeline

## Goal
Expand from single HDFS automaton to multi-automaton system with better evaluation, production export, and professional CLI.

## Status: IN PROGRESS

---

### Phase 1: CVE Analysis Automaton [MOSTLY DONE — needs Kaggle execution]

- [x] 1.1 Create loader that supports HF datasets (AlicanKiraz0/All-CVE-Training-Dataset)
- [x] 1.2 Create configs/cve-analysis.yaml
- [x] 1.3 Add CVE-specific system prompt and task_name field to config
- [x] 1.4 Adapt evaluator for multiclass (severity: CRITICAL/HIGH/MEDIUM/LOW)
- [ ] 1.5 Run training on Kaggle P100 + upload to HF as Yusif-v/hephaestus-cve-analyzer

### Phase 2: Better Evaluation + Adaptive Config [DONE]

- [x] 2.1 Per-class metrics (per CVE severity / per class accuracy breakdown)
- [x] 2.2 Overfitting detection (eval vs train loss gap monitoring with warning)
- [x] 2.3 Auto-select LoRA rank based on model size (0.5B→64, 1.5B→128, 3B→256, 7B→512)
- [x] 2.4 Auto-select batch_size based on available VRAM

### Phase 3: Model Export Pipeline [STRUCTURE DONE — needs real run to verify]

- [x] 3.1 INT4 post-training quantization benchmark (comparison with fp16)
- [x] 3.2 GGUF export support (requires llama.cpp convert script)
- [x] 3.3 vLLM-compatible serve command
- [x] 3.4 Benchmark comparison framework (size, speed, fp16 vs int4)

### Phase 4: CLI Improvements + Multi-Task [DONE]

- [x] 4.1 Subcommands: train, evaluate, serve, export, models
- [x] 4.2 Config supports task_name + system_prompt for multi-task
- [x] 4.3 Local model registry (JSON at ~/.hephaestus/registry.json)

### Phase 5: Second Automaton (Phishing Detection) [CONFIG READY]

- [x] 5.1 Create configs/phishing.yaml with phishing-specific settings
- [ ] 5.2 Source phishing dataset (e.g., from HF or local)
- [ ] 5.3 Train + evaluate on Kaggle
- [ ] 5.4 Upload to HF as Yusif-v/hephaestus-phishing-detector

---

## Success Criteria

- [ ] CVE automaton accuracy >= 95%
- [ ] INT4 quantization with <2% accuracy loss
- [ ] GGUF export works on CPU (when llama.cpp available)
- [ ] CLI supports all subcommands train/evaluate/serve/export/models
- [ ] 2 public models on HuggingFace

## Remaining Work

1. **Run CVE training on Kaggle** — builds the actual model (Phase 1.5)
2. **Source phishing dataset** — find/download for Phase 5
3. **Run phishing training** — Phase 5.2-5.4
4. **GGGUF real export test** — needs llama.cpp installed

## Files Changed

```
hephaestus/loader.py          — NEW: dataset/model loading, HF support
hephaestus/registry.py        — NEW: local model registry
hephaestus/trainer.py         — updated: overfitting detection, auto-config
hephaestus/evaluator.py       — updated: multiclass + per-class metrics
hephaestus/exporter.py        — updated: GGUF export, INT4 benchmark
hephaestus/config.py          — updated: task_name, system_prompt, auto-rank
hephaestus/cli.py             — updated: subcommands
configs/cve-analysis.yaml     — NEW: CVE severity classification config
configs/phishing.yaml         — NEW: Phishing detection config
```

## Estimated Effort Remaining

| Task | Time |
|------|------|
| CVE training on Kaggle | ~30 min (kernel launch) |
| Phishing dataset + train | ~1h |
| Verify export + GGUF | ~30m |
| Upload models + docs | ~30m |
| **Total** | **~2.5h** |
