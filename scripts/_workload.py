"""Shared workload generation so every benchmark measures the same requests."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# A grab-bag of prompts with very different lengths, like a real chat server sees.
PROMPTS = [
    "Hi",
    "What is paged attention?",
    "Explain the difference between continuous batching and static batching in one paragraph.",
    "Write a short story about a cache that never ran out of memory.",
    "Summarise how a copy-on-write page table works.",
    "Why does the decode phase of an LLM become memory-bandwidth bound?",
    "Give me three reasons a KV cache fragments.",
    "What happens when the scheduler has to preempt a sequence?",
    "Compare swapping and recomputation for KV cache eviction.",
    "Describe the lifecycle of a token inside a serving engine, from arrival to eviction.",
    "In two sentences: what is prefix caching good for?",
    "List five things that make the inter-token latency spike.",
]


def build_workload(
    n: int,
    seed: int = 0,
    min_output: int = 16,
    max_output: int = 256,
    distribution: str = "uniform",
) -> list[dict]:
    """``n`` requests with highly variable output lengths.

    The variance is the whole point: a static batch runs until its *longest*
    member finishes, so one 256-token request forces every 16-token request to
    keep its slot -- and its memory -- occupied.

    ``distribution="heavy_tail"`` draws ``u**3`` instead of a uniform ``u``: most
    requests are short and a few are very long.  That is what a real chat
    workload looks like, and it is where static batching loses the most.
    """
    rng = random.Random(seed)
    workload = []
    for i in range(n):
        u = rng.random()
        if distribution == "heavy_tail":
            u = u**3
        workload.append(
            {
                "prompt": PROMPTS[i % len(PROMPTS)],
                "max_tokens": max(1, min_output + int(u * (max_output - min_output))),
            }
        )
    return workload


def summarise(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values = sorted(values)
    n = len(values)

    def pct(p: float) -> float:
        return values[min(n - 1, int(p * n))]

    return {
        "mean": sum(values) / n,
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": values[-1],
    }
