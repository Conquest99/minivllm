from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
sys.path.insert(0, str(ROOT))


def model_path() -> str:
    """Prefer a locally downloaded checkpoint, fall back to the hub id."""
    if (LOCAL_MODEL / "config.json").exists():
        return str(LOCAL_MODEL)
    return os.environ.get("MINIVLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def require_local_model() -> str:
    if not (LOCAL_MODEL / "config.json").exists():
        pytest.skip(f"no local checkpoint at {LOCAL_MODEL}")
    return str(LOCAL_MODEL)


@pytest.fixture(scope="session")
def runner():
    """A ModelRunner on the device selected by MINIVLLM_TEST_DEVICE (default cpu).

    CPU + float32 keeps the numerical-comparison tests fast to trust and does not
    compete with whatever else is using the GPU.  Set MINIVLLM_TEST_DEVICE=cuda to
    validate the same code on the GPU path.
    """
    import torch

    from minivllm.config import EngineConfig
    from minivllm.model_runner import ModelRunner

    device = os.environ.get("MINIVLLM_TEST_DEVICE", "cpu")
    dtype = "float32" if device == "cpu" else "float16"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = EngineConfig(model=require_local_model(), dtype=dtype, device=device, num_gpu_blocks=64)
    return ModelRunner(cfg)
