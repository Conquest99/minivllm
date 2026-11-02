"""引擎级配置。

有两处值得注意，因为它们正是分页注意力存在的全部理由：

* ``block_size`` —— KV cache 按固定大小的 token *块*发放，所以一条最后只有 3 个
  token 的请求照样占一整块。块小则每请求浪费少，块大则页表查询少。
  16 是 vLLM 的默认值，也是个好折中。
* ``max_num_batched_tokens`` —— 每步的 *token* 预算（不是序列预算）。一次 1000
  token 的 prefill 和 8 个 1 token 的 decode 花的是同一份预算，这才让 prefill 和
  decode 混在一批里成为可能。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineConfig:
    # --- 模型 ---
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "float16"
    device: str = "cuda"
    seed: int = 0
    use_chat_template: bool = True
    """Wrap a plain prompt in the model's chat template before tokenizing."""

    # --- KV cache 分页 ---
    block_size: int = 16
    num_gpu_blocks: int | None = None
    """Number of physical blocks. ``None`` = size it from free VRAM at startup."""
    gpu_memory_utilization: float = 0.80
    """Fraction of *currently free* VRAM the KV cache may occupy."""

    # --- 调度 ---
    max_num_seqs: int = 8
    """How many sequences may be in the running batch at once."""
    max_num_batched_tokens: int = 1024
    """Per-step token budget shared by prefill chunks and decode steps."""
    max_model_len: int = 2048
    """Hard cap on prompt + generated tokens; also bounds the block table width."""
    enable_chunked_prefill: bool = True
    """Split a long prompt across steps so it cannot stall the decode batch."""
    long_prefill_token_threshold: int = 0
    """0 = never reject a prompt; >0 = a prompt longer than this waits for a
    step where it fits alone (how vLLM avoids one request monopolising a step)."""

    # --- 前缀缓存 ---
    enable_prefix_caching: bool = True
    """Share full KV blocks between requests that start with the same tokens."""

    # --- 仅用于压测 ---
    static_batching: bool = False
    """Emulate a pre-continuous-batching engine: a batch runs until its longest
    member finishes and no new request is admitted before the batch drains.  Used
    by the benchmark to isolate the *scheduling* effect from everything else."""

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.max_num_batched_tokens < self.block_size:
            raise ValueError("max_num_batched_tokens must be >= block_size")
        if self.long_prefill_token_threshold and (
            self.long_prefill_token_threshold < self.max_num_batched_tokens
        ):
            raise ValueError("long_prefill_token_threshold must be >= max_num_batched_tokens")

    @property
    def max_blocks_per_seq(self) -> int:
        """block 表张量的宽度：ceil(max_model_len / block_size)。"""
        return (self.max_model_len + self.block_size - 1) // self.block_size
