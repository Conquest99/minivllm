"""Paged attention in plain PyTorch.

The cache is one dense tensor::

    k_cache: [layers, num_blocks, block_size, num_kv_heads, head_dim]

so a token's KV lives at flat slot ``block_id * block_size + offset`` -- *within a
layer*.  Two operations are enough to run attention over it:

* ``write_kv_cache`` -- scatter the freshly projected K/V of this step into their
  slots (``slot_mapping`` says where each token goes).
* ``gather_kv`` -- for each sequence, walk its page table to *build* a contiguous
  K/V view, which is what a fused PagedAttention kernel avoids doing.

The gather is the honest cost of doing this without a custom kernel.  Note it is
a single ``index_select`` over a padded 2-D slot index, not a Python loop, so it
is still O(batch) kernel launches rather than O(batch * blocks).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _detect_gqa_support() -> bool:
    """Feature-probe rather than guess: SDPA is a builtin, so it has no signature."""
    try:
        q = torch.zeros(1, 2, 1, 4)
        kv = torch.zeros(1, 1, 1, 4)
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
    except TypeError:
        return False
    return True


# ``enable_gqa`` landed in torch 2.5: it lets SDPA broadcast 2 KV heads over 14
# query heads internally instead of us materialising a 7x larger tensor.
_SDPA_SUPPORTS_GQA = _detect_gqa_support()


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float):
    """``q``: [.., heads, len_q, dim], ``k``/``v``: [.., kv_heads, len_kv, dim]."""
    if k.shape[-3] == q.shape[-3]:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    if _SDPA_SUPPORTS_GQA:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=True)
    reps = q.shape[-3] // k.shape[-3]
    return F.scaled_dot_product_attention(
        q,
        k.repeat_interleave(reps, dim=-3),
        v.repeat_interleave(reps, dim=-3),
        attn_mask=mask,
        scale=scale,
    )


def write_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Scatter ``key``/``value`` (``[num_tokens, num_kv_heads, head_dim]``) into the cache."""
    flat_k = k_cache.view(-1, *k_cache.shape[2:])
    flat_v = v_cache.view(-1, *v_cache.shape[2:])
    flat_k[slot_mapping] = key
    flat_v[slot_mapping] = value


def copy_blocks(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    mapping: list[tuple[int, int]],
    block_size: int,
) -> None:
    """Execute copy-on-write: copy whole blocks ``src -> dst``, for every layer.

    The block table is a CPU-side data structure, but the KV it points at is on
    the GPU, so every structural decision in the scheduler becomes a memcpy here.
    Works for ``[blocks, block_size, kv_heads, head_dim]`` and for the layered
    ``[layers, blocks, block_size, kv_heads, head_dim]`` layout, because the copy
    is identical in every layer and only the slot index changes.
    """
    if not mapping:
        return
    device = k_cache.device
    src = torch.tensor([s for s, _ in mapping], device=device)
    dst = torch.tensor([d for _, d in mapping], device=device)
    offsets = torch.arange(block_size, device=device)
    src_slots = (src[:, None] * block_size + offsets[None, :]).reshape(-1)
    dst_slots = (dst[:, None] * block_size + offsets[None, :]).reshape(-1)

    num_slots = 1
    for dim in k_cache.shape[:-2]:
        num_slots *= dim
    flat_k = k_cache.reshape(-1, num_slots, *k_cache.shape[-2:])
    flat_v = v_cache.reshape(-1, num_slots, *v_cache.shape[-2:])
    # advanced indexing on the right-hand side already materialises a copy, so the
    # in-place scatter is safe even when src and dst overlap
    flat_k[:, dst_slots] = flat_k[:, src_slots]
    flat_v[:, dst_slots] = flat_v[:, src_slots]


def build_gather_index(
    block_tables: torch.Tensor,
    kv_lens: torch.Tensor,
    block_size: int,
    dummy_slot: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn page tables into a padded gather index + validity mask.

    ``block_tables``: ``[batch, max_blocks]``, ``kv_lens``: ``[batch]``.
    Slots past a sequence's length point at ``dummy_slot`` (a permanently zero
    slot) so the batch stays rectangular without any branch in the hot path.
    """
    batch, _ = block_tables.shape
    max_len = int(kv_lens.max().item())
    offsets = torch.arange(block_size, device=block_tables.device)
    slots = (block_tables[:, :, None] * block_size + offsets[None, None, :]).reshape(batch, -1)
    slots = slots[:, :max_len]
    valid = torch.arange(max_len, device=block_tables.device)[None, :] < kv_lens[:, None]
    return torch.where(valid, slots, dummy_slot), valid


def gather_kv(k_cache: torch.Tensor, v_cache: torch.Tensor, gather_index: torch.Tensor):
    """``[batch, max_len, kv_heads, head_dim]`` K/V for a whole decode batch."""
    head_dim = k_cache.shape[-1]
    kv_heads = k_cache.shape[-2]
    flat_k = k_cache.view(-1, kv_heads, head_dim)
    flat_v = v_cache.view(-1, kv_heads, head_dim)
    idx = gather_index.reshape(-1)
    batch, max_len = gather_index.shape
    k = flat_k.index_select(0, idx).view(batch, max_len, kv_heads, head_dim)
    v = flat_v.index_select(0, idx).view(batch, max_len, kv_heads, head_dim)
    return k, v


def _gather(flat_k: torch.Tensor, flat_v: torch.Tensor, flat_idx: torch.Tensor):
    return flat_k.index_select(0, flat_idx), flat_v.index_select(0, flat_idx)


def paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    kv_lens: torch.Tensor,
    scale: float,
    dummy_slot: int,
) -> torch.Tensor:
    """Attention for a batch where every sequence contributes exactly 1 query token.

    ``query``: ``[batch, num_heads, head_dim]``.  Returns ``[batch, num_heads * head_dim]``.
    """
    batch, num_heads, head_dim = query.shape
    gather_index, valid = build_gather_index(block_tables, kv_lens, k_cache.shape[-3], dummy_slot)
    k, v = gather_kv(k_cache, v_cache, gather_index)  # [B, L, kv_heads, D]

    q = query.unsqueeze(2)  # [B, H, 1, D], one query token per sequence
    # bool mask: [B, 1, 1, L] -- True means "may attend"
    mask = valid[:, None, None, :]
    out = _sdpa(q, k.transpose(1, 2), v.transpose(1, 2), mask, scale)
    return out.transpose(1, 2).reshape(batch, num_heads * head_dim)


def paged_attention_prefill(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    query_lens: list[int],
    kv_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Attention for ragged multi-token queries (prefill / chunked prefill).

    Done per sequence: padding the query dimension would make every short decode
    in the batch pay for the longest prefill, which is exactly the waste
    continuous batching is supposed to remove.  A fused kernel gets this for free
    by keeping the query dim flat; here we eat one loop iteration per sequence.

    ``query_lens`` is a plain list on purpose: reading it back from a CUDA tensor
    with ``.item()`` inside the layer loop would stall the pipeline once per layer
    per sequence.
    """
    num_heads, head_dim = query.shape[1], query.shape[2]
    block_size = k_cache.shape[-3]
    flat_k = k_cache.reshape(-1, *k_cache.shape[-2:])
    flat_v = v_cache.reshape(-1, *v_cache.shape[-2:])
    kv_lens = kv_lens.tolist()

    outputs = []
    q_offset = 0
    for i in range(block_tables.shape[0]):
        q_len = query_lens[i]
        kv_len = kv_lens[i]
        q_i = query[q_offset : q_offset + q_len]  # [q_len, H, D]
        q_offset += q_len

        n_blocks = (kv_len + block_size - 1) // block_size
        offsets = torch.arange(block_size, device=block_tables.device)
        flat_idx = (block_tables[i, :n_blocks, None] * block_size + offsets[None, :]).reshape(-1)[:kv_len]
        k, v = _gather(flat_k, flat_v, flat_idx)  # [kv_len, kv_heads, D]

        # causal mask, with this query block sitting at the END of the KV prefix
        q_pos = torch.arange(kv_len - q_len, kv_len, device=query.device)[:, None]
        kv_pos = torch.arange(kv_len, device=query.device)[None, :]
        mask = kv_pos <= q_pos  # [q_len, kv_len], True = may attend

        out = _sdpa(
            q_i.unsqueeze(0).transpose(1, 2),  # [1, H, q_len, D]
            k.transpose(0, 1).unsqueeze(0),  # [1, kv_heads, kv_len, D]
            v.transpose(0, 1).unsqueeze(0),
            mask,
            scale,
        )
        outputs.append(out.transpose(1, 2).reshape(q_len, num_heads * head_dim))

    if not outputs:
        return query.new_zeros((0, num_heads * head_dim))
    return torch.cat(outputs, dim=0)
