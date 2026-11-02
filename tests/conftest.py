from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
sys.path.insert(0, str(ROOT))


def model_path() -> str:
    """优先用本地下载好的权重，没有就退回 hub id。"""
    if (LOCAL_MODEL / "config.json").exists():
        return str(LOCAL_MODEL)
    return os.environ.get("MINIVLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def require_local_model() -> str:
    if not (LOCAL_MODEL / "config.json").exists():
        pytest.skip(f"no local checkpoint at {LOCAL_MODEL}")
    return str(LOCAL_MODEL)


@pytest.fixture(scope="session")
def runner():
    """一个 ModelRunner，设备由 MINIVLLM_TEST_DEVICE 决定（默认 cpu）。

    CPU + float32 让数值对比类测试跑得快、结果可信，也不去和别人抢 GPU。设
    MINIVLLM_TEST_DEVICE=cuda 就能在 GPU 路径上验证同一份代码。
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
