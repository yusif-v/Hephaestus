.PHONY: test docker test-local

# CPU-safe tests (no GPU). Installs CPU torch + test extra.
test:
	python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
	python -m pip install -e ".[test]"
	pytest -q tests/

docker:
	docker build -t hephaestus:local .

# Full training run (needs GPU + HF_TOKEN). Example:
#   docker run --gpus all -e HF_TOKEN=hf_xxx hephaestus:local \
#     python -m hephaestus train --config configs/cve-analysis.yaml
