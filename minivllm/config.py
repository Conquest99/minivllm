"""Engine-level configuration.

Two things are worth noticing here, because they are the whole reason paged
attention exists:

* ``block_size`` -- KV cache is handed out in fixed-size *blocks* of tokens, so a
  request that ends up 3 tokens long still owns a whole block.  Small blocks
  waste less memory per request, large blocks mean fewer block-table lookups.
  16 is the vLLM default and a good compromise.
* ``max_num_batched_tokens`` -- the per-step *token* budget (not sequence
  budget).  A prefill of 1000 tokens and 8 decodes of 1 token cost the same
  budget, which is what makes mixing prefill and decode in one batch possible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineConfig:
    # --- model ---
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "float16"
    device: str = "cuda"
    seed: int = 0
    use_chat_template: bool = True
    """Wrap a plain prompt in the model's chat template before tokenizing."""

    # --- KV cache paging ---
    block_size: int = 16
    num_gpu_blocks: int | None = None
    """Number of physical blocks. ``None`` = size it from free VRAM at startup."""
    gpu_memory_utilization: float = 0.80
    """Fraction of *currently free* VRAM the KV cache may occupy."""

    # --- scheduling ---
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

    # --- prefix caching ---
    enable_prefix_caching: bool = True
    """Share full KV blocks between requests that start with the same tokens."""

    # --- benchmarking only ---
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
        """Width of the block-table tensor: ceil(max_model_len / block_size)."""
        return (self.max_model_len + self.block_size - 1) // self.block_size
