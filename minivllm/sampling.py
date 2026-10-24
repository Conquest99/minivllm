"""Sampling parameters plus the (small) sampler itself.

Deliberately independent of the model: it only sees a ``[vocab]`` logits vector
per sequence, which is also why it is trivial to unit-test on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SamplingParams:
    temperature: float = 0.0
    """0.0 = greedy, matches ``do_sample=False``."""
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 64
    ignore_eos: bool = False
    stop: list[str] = field(default_factory=list)
    """Stop *strings*. Checked on decoded text, not token ids."""
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


@torch.inference_mode()
def sample(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator | None = None) -> int:
    """Pick one token id from a 1-D ``[vocab]`` logits tensor."""
    if params.greedy:
        return int(torch.argmax(logits).item())

    logits = logits.float() / params.temperature

    if params.top_k > 0:
        k = min(params.top_k, logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if params.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        # keep the smallest set whose cumulative probability reaches top_p
        cutoff = cumulative - probs > params.top_p
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())
