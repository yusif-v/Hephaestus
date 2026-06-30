"""Environment setup — install correct CUDA stack before any torch import."""

import subprocess
import os
import sys


def setup_environment():
    """Install cu121 stack and ML packages. Must run BEFORE importing torch."""
    packages = [
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
    ]
    ml_packages = [
        "transformers>=4.40",
        "peft>=0.11.0",
        "trl>=0.9.0",
        "accelerate>=0.30.0",
        "datasets>=2.19.0",
        "bitsandbytes>=0.43.0",
    ]

    # Install cu121 torch (overrides Kaggle's cu128)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages,
         "--index-url", "https://download.pytorch.org/whl/cu121"]
    )

    # Install ML packages
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *ml_packages]
    )

    # Set environment variables
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Set HF token if provided
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token and hf_token != "PASTE_YOUR_TOKEN_HERE":
        os.environ["HF_TOKEN"] = hf_token

    print("Environment setup complete")


def verify_gpu():
    """Verify GPU is available and compatible."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available!")

    gpu_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability()
    vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9

    print(f"GPU: {gpu_name}")
    print(f"CC: {cc[0]}.{cc[1]}")
    print(f"VRAM: {vram_gb:.1f} GB")
    print(f"PyTorch: {torch.__version__}")

    if cc < (7, 0) and "cu121" not in torch.__version__:
        raise RuntimeError(
            f"P100 (sm_60) requires cu121 but got {torch.__version__}. "
            "Run setup_environment() first."
        )

    return {
        "name": gpu_name,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "vram_gb": round(vram_gb, 1),
        "torch_version": torch.__version__,
    }
