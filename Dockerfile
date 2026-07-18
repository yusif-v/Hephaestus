# Hephaestus v0.3 — reproducible training image.
# Pinned to the PROVEN stack used by the Heimdall + CVE Kaggle kernels
# (torch 2.0.1 + cu117, unsloth, trl 0.8.6). Kaggle P100 is sm_60; cu128
# (Kaggle default) is sm_70+ and crashes. Do not bump torch without re-verifying.
FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN python3.10 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
        --index-url https://download.pytorch.org/whl/cu117 && \
    pip install --no-cache-dir unsloth transformers==4.46.3 peft==0.13.2 trl==0.8.6 \
        accelerate datasets bitsandbytes==0.46.1 pyyaml && \
    pip install --no-cache-dir -e ".[test]"

ENV PATH=/opt/venv/bin:$PATH

# Smoke entrypoint: run CPU-safe tests by default. Training is invoked explicitly:
#   docker run --gpus all hephaestus python -m hephaestus train --config configs/cve-analysis.yaml
CMD ["pytest", "-q", "tests/"]
