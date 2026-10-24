"""Paged allocator: ref counting, copy-on-write, prefix caching, eviction."""

from __future__ import annotations

import pytest

from minivllm.block_manager import AllocStatus, BlockManager, OutOfBlocksError
from minivllm.sampling import SamplingParams
from minivllm.sequence import Sequence


def make_seq(seq_id: int, token_ids: list[int]) -> Sequence:
    return Sequence(seq_id=seq_id, prompt_token_ids=token_ids, sampling_params=SamplingParams())


def tokens(seq_id: int, n: int) -> list[int]:
    return [seq_id * 1000 + i for i in range(n)]


def test_allocate_and_free_returns_blocks():
    bm = BlockManager(num_blocks=8, block_size=4, enable_prefix_caching=False)
    seq = make_seq(1, tokens(1, 10))
    assert bm.can_allocate(seq) is AllocStatus.OK

    bm.allocate(seq, 10)
    assert len(seq.block_table) == 3
    assert bm.num_free_blocks == 5

    bm.free(seq)
    assert bm.num_free_blocks == 8
    assert seq.block_table == []


def test_allocation_is_lazy_not_reserved():
    """A 10-token request takes 3 pages of 4 -- not ceil(max_model_len / 4) pages."""
    bm = BlockManager(num_blocks=64, block_size=4, enable_prefix_caching=False)
    seq = make_seq(1, tokens(1, 10))
    bm.allocate(seq, 10)
    assert len(seq.block_table) == 3
    assert bm.naive_reservation_cost(seq, max_model_len=256) == 64
    assert bm.num_free_blocks == 61


def test_out_of_blocks():
    bm = BlockManager(num_blocks=2, block_size=4, enable_prefix_caching=False)
    seq = make_seq(1, tokens(1, 12))
    assert bm.can_allocate(seq) is AllocStatus.LATER
    with pytest.raises(OutOfBlocksError):
        bm.allocate(seq, 12)


def test_growth_allocates_only_when_crossing_a_boundary():
    bm = BlockManager(num_blocks=8, block_size=8, enable_prefix_caching=False)
    seq = make_seq(1, tokens(1, 4))
    bm.allocate(seq, 4)
    seq.num_computed_tokens = 4

    for n in range(5, 9):  # tokens 5..8 still fit in page 0
        assert bm.ensure_capacity(seq, n) == []
        assert len(seq.block_table) == 1

    assert bm.ensure_capacity(seq, 9) == []
    assert len(seq.block_table) == 2


def test_prefix_cache_shares_blocks():
    bm = BlockManager(num_blocks=16, block_size=4, enable_prefix_caching=True)
    first = make_seq(1, tokens(1, 8))  # tokens 1000..1007 -> 2 full blocks
    bm.allocate(first, 8)
    first.num_computed_tokens = 8
    bm.register_computed_blocks(first)

    second = make_seq(2, first.prompt_token_ids[:6])  # shares the first full block only
    assert bm.match_prefix(second.data) == first.block_table[:1]

    matched = bm.allocate(second, 6)
    assert matched == 1
    assert second.block_table[0] == first.block_table[0]
    assert bm.blocks[first.block_table[0]].ref_count == 2

    bm.free(first)
    assert bm.blocks[second.block_table[0]].ref_count == 1
    # still alive -> not evictable
    assert bm.num_free_blocks == 16 - 2


def test_prefix_cache_disabled_ignores_matches():
    bm = BlockManager(num_blocks=16, block_size=4, enable_prefix_caching=False)
    first = make_seq(1, tokens(1, 8))
    bm.allocate(first, 8)
    first.num_computed_tokens = 8
    bm.register_computed_blocks(first)

    assert bm.match_prefix(first.data) == []
    second = make_seq(2, first.prompt_token_ids[:6])
    assert bm.allocate(second, 6) == 0


def test_copy_on_write_after_fork():
    bm = BlockManager(num_blocks=16, block_size=4, enable_prefix_caching=False)
    parent = make_seq(1, tokens(1, 6))  # 1 full block + 1 partial
    bm.allocate(parent, 6)
    parent.num_computed_tokens = 6

    child = make_seq(2, list(parent.prompt_token_ids))
    bm.fork(parent, child)
    assert child.block_table == parent.block_table
    assert bm.blocks[parent.block_table[1]].ref_count == 2

    cow = bm.ensure_capacity(child, 7)
    assert len(cow) == 1
    src, dst = cow[0]
    assert src == parent.block_table[1]
    assert dst == child.block_table[1] != parent.block_table[1]
    assert bm.blocks[src].ref_count == 1

    # once the child has detached, the parent is the sole owner of its block and
    # writes into it without another copy -- paying one copy per divergence, not
    # one per owner.
    assert bm.ensure_capacity(parent, 7) == []


def test_eviction_drops_cache_entry():
    bm = BlockManager(num_blocks=2, block_size=4, enable_prefix_caching=True)
    first = make_seq(1, tokens(1, 8))
    bm.allocate(first, 8)
    first.num_computed_tokens = 8
    bm.register_computed_blocks(first)
    assert len(bm.hash_to_block) == 2

    bm.free(first)
    assert len(bm.hash_to_block) == 2  # cached while idle
    assert bm.num_free_blocks == 2

    other = make_seq(2, tokens(9, 4))
    bm.allocate(other, 4)
    assert len(bm.hash_to_block) == 1  # one entry got evicted to serve the allocation


def test_no_double_free():
    bm = BlockManager(num_blocks=4, block_size=4, enable_prefix_caching=False)
    seq = make_seq(1, tokens(1, 4))
    bm.allocate(seq, 4)
    bm.free(seq)
    bm.free(seq)  # idempotent: the block table is already empty
    assert bm.num_free_blocks == 4
