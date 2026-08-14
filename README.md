# Hephaestus

Forge tiny, purpose-built LLMs from any base model.

## Quick Start

```bash
pip install -e .

# Run with accuracy profile (3B, best quality)
python -m hephaestus --profile accuracy

# Run with config file
python -m hephaestus --config configs/hdfs-accuracy.yaml train
python -m hephaestus --profile balanced --training.max_steps 50 train
```

## Usage Profiles

|| Profile | Model | Accuracy | Speed ||
||---------|-------|----------|---------||
|| latency | 0.5B | ~91% | Fastest ||
|| balanced | 1.5B | ~95% | Fast ||
|| accuracy | 3B | ~96% | Slower ||

## Output

```
outputs/
├── model/                 # Merged model (safetensors)
├── model_card.json        # All metrics + config used
└── benchmarks.json        # TPS, latency, model size
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full design.

## Models

- [Yusif-v/hephaestus-hdfs-0.5b](https://huggingface.co/Yusif-v/hephaestus-hdfs-0.5b) — HDFS Log Anomaly Detector (91% accuracy)
- **Heimdall-0.5b** — File System Anomaly Detector (target: 97% accuracy) - `outputs/heimdall-0.5b`

## Results

|| Model | Accuracy | F1 | Train Time | Size ||
||-------|----------|-----|------------|------||
|| Qwen2.5-0.5B | 91.0% | 90.7% | 6 min | 988 MB ||
|| Qwen2.5-3B | 96.0% | 95.1% | 17 min | ~6 GB ||

## Forge — design your own dataset (v0.3)

`python -m hephaestus forge` runs the full loop: an interactive interview, an LLM
brain that designs a reviewable `dataset-spec.json`, a deterministic build engine
that emits `train.jsonl`/`test.jsonl`, then a Kaggle training run and export.

`python -m hephaestus dataset show spec.json` — review a spec.
`python -m hephaestus dataset build spec.json` — build without training.