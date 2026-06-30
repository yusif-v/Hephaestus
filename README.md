# Hephaestus v0.1

Forge tiny, purpose-built LLMs from any base model.

## Quick Start

```bash
pip install -e .

# Run with accuracy profile (3B, best quality)
python -m hephaestus --profile accuracy --hf-token YOUR_TOKEN

# Run with config file
python -m hephaestus --config configs/hdfs-accuracy.yaml

# Quick iteration
python -m hephaestus --profile balanced --training.max_steps 50
```

## Usage Profiles

| Profile | Model | Accuracy | Speed |
|---------|-------|----------|-------|
| latency | 0.5B | ~94% | Fastest |
| balanced | 1.5B | ~95% | Fast |
| accuracy | 3B | ~96% | Slower |

## Output

```
outputs/hdfs-3b/
├── model/                 # Merged model (safetensors)
├── model_card.json        # All metrics + config used
└── benchmarks.json        # TPS, latency, model size
```

## Architecture

See `docs/specs/2026-06-29-v0.1-workflow-design.md` for the full design.
