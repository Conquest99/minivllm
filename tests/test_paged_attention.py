"""Paged attention vs a plain dense attention reference.

The point of these tests is the property that matters: reading K/V through a page
table must be *numerically identical* to reading it from a contiguous tensor.  If
this holds, the rest of the engine is just bookkeeping.
"""

from __future__ import annotations

import pytest
import torch

from minivllm.paged_attention import (
    build_gather_index,
    copy_blocks,
    paged_attention_decode,
    paged_attention_prefill,
    write_kv_cache,
)


def dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Reference: ``q`` [Lq, H, D], ``k``/``v`` [Lk, Hkv, D], causal."""
    lq, heads, _ = q.shape
    lk = k.shape[0]
    reps = heads // k.shape[1]
    k = k.repeat_interleave(reps, dim=1).transpose(0, 1)  # [H, Lk, D]
    v = v.repeat_interleave(reps, dim=1).transpose(0, 1)
    scores = (q.transpose(0, 1) @ k.transpose(1, 2)) * scale  # [H, Lq, Lk]
    offset = lk - lq
    mask = torch.arange(lk)[None, :] <= torch.arange(lq)[:, None] + offset
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v).transpose(0, 1).reshape(lq, heads * q.shape[-1])


class PagedCache:
    """Test harness: a block pool plus the ground-truth K/V per sequence."""

    def __init__(self, num_blocks: int, block_size: int, kv_heads: int, head_dim: int, seed: int = 0):
        torch.manual_seed(seed)
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.k_cache = torch.zeros(num_blocks + 1, block_size, kv_heads, head_dim)
        self.v_cache = torch.zeros(num_blocks + 1, block_size, kv_heads, head_dim)
        self.dummy_slot = num_blocks * block_size
        self.seqs: list[dict] = []

    def add(self, block_ids: list[int], k: torch.Tensor, v: torch.Tensor) -> None:
        n = k.shape[0]
        slots = []
        for t in range(n):
            slots.append(block_ids[t // self.block_size] * self.block_size + t % self.block_size)
        write_kv_cache(
            self.k_cache,
            self.v_cache,
            torch.tensor(slots, dtype=torch.long),
            k,
            v,
        )
        self.seqs.append({"blocks": block_ids, "k": k, "v": v, "len": n})

    def block_tables(self) -> torch.Tensor:
        width = max(len(s["blocks"]) for s in self.seqs)
        rows = [s["blocks"] + [0] * (width - len(s["blocks"])) for s in self.seqs]
        return torch.tensor(rows, dtype=torch.long)

    def kv_lens(self) -> torch.Tensor:
        return torch.tensor([s["len"] for s in self.seqs], dtype=torch.long)


def random_kv(n: int, kv_heads: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(n, kv_heads, head_dim), torch.randn(n, kv_heads, head_dim)


def test_write_and_read_roundtrip():
    cache = PagedCache(num_blocks=8, block_size=4, kv_heads=2, head_dim=4)
    k, v = random_kv(6, 2, 4)
    cache.add([3, 5], k, v)  # deliberately non-monotonic physical blocks

    tables = torch.tensor([[3, 5]], dtype=torch.long)
    kv_lens = torch.tensor([6], dtype=torch.long)
    idx, valid = build_gather_index(tables, kv_lens, block_size=4, dummy_slot=cache.dummy_slot)
    gathered_k = cache.k_cache.view(-1, 2, 4).index_select(0, idx.reshape(-1))[:6]

    assert torch.allclose(gathered_k, k, atol=1e-6)
    assert valid[0, :6].all() and not valid[0, 6:].any()


def test_decode_matches_dense_for_ragged_batch():
    kv_heads, heads, head_dim = 2, 4, 8
    cache = PagedCache(num_blocks=16, block_size=4, kv_heads=kv_heads, head_dim=head_dim)

    lens = [6, 3, 11]
    blocks = [[0, 1], [2], [3, 4, 5]]
    for n, b in zip(lens, blocks, strict=False):
        k, v = random_kv(n, kv_heads, head_dim)
        cache.add(b, k, v)

    q = torch.randn(len(lens), heads, head_dim)
    out = paged_attention_decode(
        q,
        cache.k_cache,
        cache.v_cache,
        cache.block_tables(),
        cache.kv_lens(),
        scale=head_dim**-0.5,
        dummy_slot=cache.dummy_slot,
    )

    for i, seq in enumerate(cache.seqs):
        expected = dense_attention(q[i : i + 1], seq["k"], seq["v"], head_dim**-0.5)
        assert torch.allclose(out[i], expected[0], atol=1e-5), f"sequence {i} diverged"


def test_prefill_matches_dense():
    """Chunked prefill: the query block sits at the end of an already-cached prefix."""
    kv_heads, heads, head_dim = 2, 4, 8
    cache = PagedCache(num_blocks=16, block_size=4, kv_heads=kv_heads, head_dim=head_dim)

    # sequence 0 owns 11 tokens (3 pages), sequence 1 owns 5 tokens (2 pages)
    layouts = [(11, [0, 1, 2]), (5, [3, 4])]
    for n, blocks in layouts:
        k, v = random_kv(n, kv_heads, head_dim)
        cache.add(blocks, k, v)

    q_lens = [5, 2]  # the number of *new* tokens each sequence prefills this step
    q = torch.randn(sum(q_lens), heads, head_dim)
    kv_lens = cache.kv_lens()

    out = paged_attention_prefill(
        q,
        cache.k_cache,
        cache.v_cache,
        cache.block_tables(),
        q_lens,
        kv_lens,
        scale=head_dim**-0.5,
    )

    offset = 0
    for i, (seq, q_len) in enumerate(zip(cache.seqs, q_lens, strict=False)):
        expected = dense_attention(q[offset : offset + q_len], seq["k"], seq["v"], head_dim**-0.5)
        assert torch.allclose(out[offset : offset + q_len], expected, atol=1e-5), f"sequence {i} diverged"
        offset += q_len


def test_copy_blocks_duplicates_kv():
    cache = PagedCache(num_blocks=8, block_size=4, kv_heads=2, head_dim=4)
    k, v = random_kv(4, 2, 4)
    cache.add([1], k, v)

    copy_blocks(cache.k_cache, cache.v_cache, [(1, 6)], block_size=4)

    assert torch.allclose(cache.k_cache[6], cache.k_cache[1])
    assert torch.allclose(cache.v_cache[6], cache.v_cache[1])
    # and the copy is independent
    cache.k_cache[1].zero_()
    assert not torch.allclose(cache.k_cache[6], cache.k_cache[1])


@pytest.mark.parametrize("num_heads,kv_heads", [(4, 4), (4, 1), (8, 2)])
def test_gqa_head_expansion(num_heads: int, kv_heads: int):
    block_size, head_dim = 4, 8
    cache = PagedCache(num_blocks=4, block_size=block_size, kv_heads=kv_heads, head_dim=head_dim)
    k, v = random_kv(5, kv_heads, head_dim)
    cache.add([0, 1], k, v)

    q = torch.randn(1, num_heads, head_dim)
    out = paged_attention_decode(
        q,
        cache.k_cache,
        cache.v_cache,
        cache.block_tables(),
        cache.kv_lens(),
        scale=head_dim**-0.5,
        dummy_slot=cache.dummy_slot,
    )
    expected = dense_attention(q, k, v, head_dim**-0.5)
    assert torch.allclose(out, expected, atol=1e-5)
