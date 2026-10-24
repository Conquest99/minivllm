"""Scheduler behaviour, with no model involved at all.

Everything the scheduler decides is expressed in tokens and pages, which is why
it can be tested on a laptop CPU in milliseconds.
"""

from __future__ import annotations

from minivllm.block_manager import BlockManager
from minivllm.config import EngineConfig
from minivllm.sampling import SamplingParams
from minivllm.scheduler import Scheduler
from minivllm.sequence import Sequence, SequenceStatus


def build(**cfg_kwargs) -> Scheduler:
    kwargs = dict(
        model="unused",
        block_size=4,
        max_num_seqs=8,
        max_num_batched_tokens=16,
        max_model_len=128,
        enable_prefix_caching=False,
    )
    kwargs.update(cfg_kwargs)
    cfg = EngineConfig(**kwargs)
    bm = BlockManager(kwargs.get("num_gpu_blocks") or 16, cfg.block_size, cfg.enable_prefix_caching)
    return Scheduler(cfg, bm)


def make_seq(seq_id: int, n_tokens: int) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=[seq_id * 1000 + i for i in range(n_tokens)],
        sampling_params=SamplingParams(max_tokens=4),
    )


def test_fcfs_and_one_decode_token_per_step():
    sched = build()
    for i in range(3):
        sched.add(make_seq(i, 4))

    out = sched.schedule()
    assert [n for _, n in out.scheduled] == [4, 4, 4]
    for seq, n in out.scheduled:
        seq.num_computed_tokens += n
        seq.append_token(0)

    out = sched.schedule()
    assert [n for _, n in out.scheduled] == [1, 1, 1]


def test_chunked_prefill_respects_the_token_budget():
    sched = build(max_num_batched_tokens=8, enable_chunked_prefill=True)
    sched.add(make_seq(0, 20))

    out = sched.schedule()
    assert out.scheduled[0][1] == 8

    seq = out.scheduled[0][0]
    seq.num_computed_tokens += 8
    out = sched.schedule()
    assert out.scheduled[0][1] == 8
    assert seq.num_computed_tokens == 8


def test_chunked_prefill_off_waits_for_a_big_enough_step():
    sched = build(max_num_batched_tokens=8, enable_chunked_prefill=False)
    sched.add(make_seq(0, 20))
    assert sched.schedule().is_empty  # cannot fit -> nothing is scheduled

    sched2 = build(max_num_batched_tokens=32, enable_chunked_prefill=False)
    sched2.add(make_seq(0, 20))
    assert sched2.schedule().scheduled[0][1] == 20


def test_decode_is_scheduled_before_new_prefill():
    sched = build(max_num_batched_tokens=6)
    sched.add(make_seq(0, 4))
    out = sched.schedule()
    running = out.scheduled[0][0]
    running.num_computed_tokens += 4
    running.append_token(0)

    sched.add(make_seq(1, 4))  # arrives while seq 0 is decoding
    out = sched.schedule()

    kinds = [(seq.seq_id, n) for seq, n in out.scheduled]
    assert kinds[0] == (0, 1), "the running decode must go first"
    assert kinds[1] == (1, 4)


def test_budget_is_shared_between_prefill_and_decode():
    sched = build(max_num_batched_tokens=8)
    sched.add(make_seq(0, 4))
    out = sched.schedule()
    running = out.scheduled[0][0]
    running.num_computed_tokens += 4
    running.append_token(0)

    sched.add(make_seq(1, 8))
    out = sched.schedule()
    assert sum(n for _, n in out.scheduled) <= 8
    # seq 1 gets what is left of the budget (7), rounded down to a page boundary (4)
    assert [(s.seq_id, n) for s, n in out.scheduled] == [(0, 1), (1, 4)]


def test_max_num_seqs_is_enforced():
    sched = build(max_num_seqs=2, max_num_batched_tokens=64)
    for i in range(4):
        sched.add(make_seq(i, 4))
    out = sched.schedule()
    assert len(out.scheduled) == 2
    assert len(sched.waiting) == 2


def test_preemption_when_the_cache_is_full():
    sched = build(num_gpu_blocks=4, max_num_seqs=8, max_num_batched_tokens=64, block_size=4)
    # 2 pages each -> only two sequences fit in a 4-page pool
    for i in range(3):
        sched.add(make_seq(i, 8))

    out = sched.schedule()
    assert len(out.scheduled) == 2
    assert len(sched.waiting) == 1

    # force a third sequence to grow into the pool while both others decode
    for seq, n in out.scheduled:
        seq.num_computed_tokens += n
        seq.append_token(0)
    sched.add(make_seq(9, 8))

    out = sched.schedule()
    assert out.preempted, "the pool is full, something must give way"
    assert sched.num_preemptions >= 1
    victim = out.preempted[0]
    assert victim.num_computed_tokens == 0  # recompute from scratch
    assert victim.num_preemptions == 1
    assert victim.status is SequenceStatus.WAITING


def test_finished_sequences_release_their_pages():
    sched = build(num_gpu_blocks=4, block_size=4, max_num_batched_tokens=64)
    sched.add(make_seq(0, 8))
    out = sched.schedule()
    seq = out.scheduled[0][0]
    seq.num_computed_tokens += 8
    seq.mark_finished(SequenceStatus.FINISHED_LENGTH_CAPPED)

    assert sched.block_manager.num_free_blocks == 2
    sched.free_finished()
    assert sched.block_manager.num_free_blocks == 4


def test_prefix_cache_skips_recomputing_a_shared_prompt():
    sched = build(enable_prefix_caching=True, max_num_batched_tokens=64)
    first = make_seq(0, 8)
    sched.add(first)
    out = sched.schedule()
    seq = out.scheduled[0][0]
    seq.num_computed_tokens += 8
    seq.mark_finished(SequenceStatus.FINISHED_LENGTH_CAPPED)
    sched.free_finished()

    # same prompt -> every block is already in the cache
    second = Sequence(
        seq_id=1, prompt_token_ids=list(first.prompt_token_ids), sampling_params=SamplingParams(max_tokens=4)
    )
    sched.add(second)
    hits = sched.block_manager.match_prefix(second.data)
    assert hits, "expected a prefix-cache hit for an identical prompt"
    out = sched.schedule()
    assert out.scheduled[0][0].block_table[: len(hits)] == hits
