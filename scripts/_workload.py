"""共用的负载生成器，保证每个 benchmark 量的是同一批请求。"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 一堆长度差异很大的 prompt，就像真实的对话服务器看到的那样。
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
    """``n`` 个输出长度差异极大的请求。

    这份方差就是全部意义：静态批要跑到它*最长*的成员结束，所以一个 256 token 的请求会
    逼着每个 16 token 的请求继续占着自己的槽位 —— 以及自己的内存。

    ``distribution="heavy_tail"`` 抽的是 ``u**3`` 而不是均匀的 ``u``：大多数请求很短，
    极少数很长。这才是真实对话负载的样子，也是静态批吃亏最狠的地方。
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
