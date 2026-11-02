"""用纯 PyTorch 实现的分页注意力。

cache 是一个稠密张量::

    k_cache: [layers, num_blocks, block_size, num_kv_heads, head_dim]

所以一个 token 的 KV 落在平坦槽位 ``block_id * block_size + offset`` —— *在某一层
之内*。在它上面跑注意力只需要两个操作：

* ``write_kv_cache`` —— 把这一步新投影出来的 K/V 散射到各自的槽位
  （``slot_mapping`` 说明每个 token 去哪）。
* ``gather_kv`` —— 对每条序列走它的页表，*拼出*一段连续的 K/V 视图，而这正是融合
  版 PagedAttention kernel 极力避免的动作。

这个 gather 就是不写定制 kernel 的诚实代价。注意它是对一个补齐过的二维槽位下标做
一次 ``index_select``，不是 Python 循环，所以仍然是 O(batch) 次 kernel 启动，而不是
O(batch × blocks)。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _detect_gqa_support() -> bool:
    """用特性探测代替猜测：SDPA 是内建函数，没有签名可看。"""
    try:
        q = torch.zeros(1, 2, 1, 4)
        kv = torch.zeros(1, 1, 1, 4)
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
    except TypeError:
        return False
    return True


# ``enable_gqa`` 在 torch 2.5 出现：它让 SDPA 内部把 2 个 KV head 广播到 14 个
# query head，而不是由我们物化一个 7 倍大的张量。
_SDPA_SUPPORTS_GQA = _detect_gqa_support()


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float):
    """``q``: [.., heads, len_q, dim]，``k``/``v``: [.., kv_heads, len_kv, dim]。"""
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
    """把 ``key``/``value``（``[num_tokens, num_kv_heads, head_dim]``）散射进 cache。"""
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
    """执行写时复制：把整块 ``src -> dst`` 复制，每一层都复制。

    block 表是 CPU 侧的数据结构，但它指向的 KV 在 GPU 上，所以调度器里每一个结构决策，
    到这里都变成一次 memcpy。对 ``[blocks, block_size, kv_heads, head_dim]`` 和分层的
    ``[layers, blocks, block_size, kv_heads, head_dim]`` 两种布局都成立，因为每一层的复制
    内容一样，只是槽位下标不同。
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
    # 右侧的高级索引已经物化出副本，所以就地 scatter
    # 即使 src 和 dst 重叠也是安全的
    flat_k[:, dst_slots] = flat_k[:, src_slots]
    flat_v[:, dst_slots] = flat_v[:, src_slots]


def build_gather_index(
    block_tables: torch.Tensor,
    kv_lens: torch.Tensor,
    block_size: int,
    dummy_slot: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把页表变成补齐过的 gather 下标 + 有效性掩码。

    ``block_tables``: ``[batch, max_blocks]``，``kv_lens``: ``[batch]``。超出序列长度的
    槽位指向 ``dummy_slot``（一个永远为 0 的槽），这样整批保持矩形，热路径里一个分支都
    不用有。
    """
    batch, _ = block_tables.shape
    max_len = int(kv_lens.max().item())
    offsets = torch.arange(block_size, device=block_tables.device)
    slots = (block_tables[:, :, None] * block_size + offsets[None, None, :]).reshape(batch, -1)
    slots = slots[:, :max_len]
    valid = torch.arange(max_len, device=block_tables.device)[None, :] < kv_lens[:, None]
    return torch.where(valid, slots, dummy_slot), valid


def gather_kv(k_cache: torch.Tensor, v_cache: torch.Tensor, gather_index: torch.Tensor):
    """整个 decode 批的 K/V，形状 ``[batch, max_len, kv_heads, head_dim]``。"""
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
    """每条序列恰好贡献 1 个 query token 的批的注意力。

    ``query``: ``[batch, num_heads, head_dim]``。返回 ``[batch, num_heads * head_dim]``。
    """
    batch, num_heads, head_dim = query.shape
    gather_index, valid = build_gather_index(block_tables, kv_lens, k_cache.shape[-3], dummy_slot)
    k, v = gather_kv(k_cache, v_cache, gather_index)  # [B, L, kv_heads, D]

    q = query.unsqueeze(2)  # [B, H, 1, D]，每条序列一个 query token
    # bool 掩码: [B, 1, 1, L] —— True 表示「允许注意」
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
    """长度不齐的多 token query（prefill / 分块 prefill）的注意力。

    按序列分别做：把 query 维补齐，会让批里每个短 decode 都为最长的 prefill 买单，而这
    正是连续批处理要消除的浪费。融合 kernel 靠把 query 维压平免费拿到这一点；我们这里
    每条序列吃一次循环。

    ``query_lens`` 故意是普通 list：在层循环里用 ``.item()`` 从 CUDA 张量读回它，会让
    流水线每层每序列停顿一次。
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

        # 因果掩码，这个 query 块位于 KV 前缀的末尾
        q_pos = torch.arange(kv_len - q_len, kv_len, device=query.device)[:, None]
        kv_pos = torch.arange(kv_len, device=query.device)[None, :]
        mask = kv_pos <= q_pos  # [q_len, kv_len]，True = 允许注意

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
