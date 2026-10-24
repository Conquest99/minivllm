"""End-to-end: the whole loop, checked against properties rather than snapshots.

The strongest property available here is **batching independence**: with greedy
sampling, a request must generate exactly the same tokens whether it ran alone,
sharing a batch, was chunked across steps, or got preempted and recomputed.  If
any of the paging / scheduling bookkeeping is wrong, this breaks.
"""

from __future__ import annotations

import os

import pytest
from conftest import require_local_model

from minivllm.config import EngineConfig
from minivllm.llm_engine import LLMEngine
from minivllm.sampling import SamplingParams

DEVICE = os.environ.get("MINIVLLM_TEST_DEVICE", "cpu")
DTYPE = "float32" if DEVICE == "cpu" else "float16"


def make_engine(**overrides) -> LLMEngine:
    kwargs = dict(
        model=require_local_model(),
        dtype=DTYPE,
        device=DEVICE,
        block_size=16,
        max_num_seqs=8,
        max_num_batched_tokens=256,
        max_model_len=512,
        num_gpu_blocks=64,
    )
    kwargs.update(overrides)
    return LLMEngine(EngineConfig(**kwargs))


PROMPTS = [
    "Name three prime numbers.",
    "What is a page table?",
    "Why is a KV cache expensive?",
]


def greedy(max_tokens: int = 8) -> SamplingParams:
    return SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)


@pytest.fixture(scope="module")
def engine() -> LLMEngine:
    return make_engine()


def test_generate_returns_one_output_per_prompt(engine):
    outputs = engine.generate(PROMPTS, greedy(6))
    assert [o.prompt for o in outputs] == PROMPTS
    for out in outputs:
        assert out.num_output_tokens == 6
        assert out.finish_reason == "length_capped"
        assert out.ttft is not None and out.latency is not None
        assert out.ttft <= out.latency


def test_batched_output_matches_solo_output(engine):
    solo = [engine.generate([p], greedy(6))[0].token_ids for p in PROMPTS]
    batched = [o.token_ids for o in engine.generate(PROMPTS, greedy(6))]
    assert batched == solo


def test_small_batch_budget_changes_nothing(engine):
    """max_num_seqs=1 serialises everything; the tokens must still match."""
    engine.cfg.max_num_seqs = 1
    try:
        serial = [o.token_ids for o in engine.generate(PROMPTS, greedy(6))]
    finally:
        engine.cfg.max_num_seqs = 8
    assert serial == [engine.generate([p], greedy(6))[0].token_ids for p in PROMPTS]


def test_chunked_prefill_gives_the_same_tokens():
    long_prompt = "Explain, step by step and in detail, how a block table maps logical to physical pages. "
    one_shot = make_engine(enable_chunked_prefill=False, max_num_batched_tokens=256)
    chunked = make_engine(enable_chunked_prefill=True, max_num_batched_tokens=32)

    expected = one_shot.generate([long_prompt], greedy(6))[0].token_ids
    actual = chunked.generate([long_prompt], greedy(6))[0].token_ids
    assert actual == expected


def test_prefix_caching_gives_the_same_tokens():
    # max_num_seqs=1 forces the requests to be admitted one after another: prefix
    # caching only helps a request that arrives *after* the shared prefix has been
    # computed and registered, so a batch scheduled all at once cannot hit it.
    with_cache = make_engine(enable_prefix_caching=True, max_num_seqs=1)
    without = make_engine(enable_prefix_caching=False, max_num_seqs=1)
    prompts = ["You are a careful assistant. " + p for p in PROMPTS]

    cached = [o.token_ids for o in with_cache.generate(prompts, greedy(6))]
    plain = [o.token_ids for o in without.generate(prompts, greedy(6))]
    assert cached == plain
    assert with_cache.block_manager.num_prefix_hit_blocks > 0


def test_preemption_still_produces_correct_tokens():
    # 3 blocks per request (48 tokens / 16) but only 4 blocks of pool: the
    # scheduler must preempt and recompute, and the answer must not change.
    tight = make_engine(num_gpu_blocks=4, max_num_seqs=3, max_num_batched_tokens=64)
    roomy = make_engine(num_gpu_blocks=64, max_num_seqs=3, max_num_batched_tokens=64)

    prompts = ["Question number %d about caches." % i for i in range(3)]
    expected = [o.token_ids for o in roomy.generate(prompts, greedy(10))]
    actual = [o.token_ids for o in tight.generate(prompts, greedy(10))]

    assert actual == expected
    assert tight.scheduler.num_preemptions > 0, "the pool is too small not to preempt"


def test_metrics_are_consistent(engine):
    engine.metrics = type(engine.metrics)()
    outputs = engine.generate(PROMPTS, greedy(5))
    assert engine.metrics.num_generation_tokens == sum(o.num_output_tokens for o in outputs)
    assert engine.metrics.throughput > 0
    stats = engine.stats()
    assert 0.0 <= stats["utilization"] <= 1.0
