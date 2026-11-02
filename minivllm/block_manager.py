"""分页 KV cache 的分配器 —— PagedAttention 的心脏。

心智模型：KV cache 是一个固定的物理页池（``num_blocks`` 页），每条序列持有一张
*页表*（``seq.block_table``），把它的逻辑块号映射到物理块号。注意力 kernel 从不
假设一条序列的 KV 在内存里连续；它走页表。

三种机制共用这一个池子：

1. **不预留。**  一条序列只在实际算过的 token 上拥有页。「每个请求预留
   ``max_model_len`` 个 token」的老做法写在 ``naive_reservation_cost()`` 里，好让
   那份浪费可以被量出来。
2. **引用计数 + 写时复制。**  两条序列可以共享同一个物理块（相同 system prompt，
   或 ``n`` 路并行采样）。只有当一条序列要往一个*半满的*共享块里写时才会复制。
3. **前缀缓存。**  一个满块按它的 token 内容做哈希；引用计数降到 0 之后它仍留在
   ``hash_to_block`` 里当可逐出的缓存项，于是下一个共享同一前缀的请求可以整段跳过
   prefill。
"""

from __future__ import annotations

import hashlib
from array import array
from collections import deque
from enum import Enum
from typing import Deque

from minivllm.sequence import Sequence

_PREFIX_SALT = b"minivllm/prefix/v1"


class AllocStatus(Enum):
    OK = 1
    LATER = 2
    NEVER = 3


class OutOfBlocksError(RuntimeError):
    """Raised when the KV cache pool is exhausted (the scheduler must preempt)."""


class PhysicalBlock:
    """A page of KV cache memory: ``block_size`` token slots."""

    __slots__ = ("block_id", "ref_count", "block_hash", "tokens")

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_count = 0
        self.block_hash: bytes | None = None
        self.tokens: tuple[int, ...] = ()

    def __repr__(self) -> str:
        return f"PhysicalBlock(id={self.block_id}, ref={self.ref_count})"


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching

        self.blocks = [PhysicalBlock(i) for i in range(num_blocks)]
        # FIFO 空闲表；当某个 block 被 *共享*（而不是被逐出）时，
        # 表项可能变陈旧，所以 ``_in_free`` 才是唯一事实来源。
        self.free_blocks: Deque[PhysicalBlock] = deque(self.blocks)
        self._in_free = bytearray(b"\x01") * num_blocks
        self._num_free = num_blocks

        self.hash_to_block: dict[bytes, int] = {}

        # --- 统计 ---
        self.num_prefix_hit_blocks = 0
        self.num_cow_ops = 0
        self.peak_used_blocks = 0

    # ------------------------------------------------------------------ 统计
    @property
    def num_free_blocks(self) -> int:
        return self._num_free

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self._num_free

    @property
    def utilization(self) -> float:
        return self.num_used_blocks / self.num_blocks

    def stats(self) -> dict[str, float]:
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "used_blocks": self.num_used_blocks,
            "free_blocks": self._num_free,
            "utilization": self.utilization,
            "cached_blocks": len(self.hash_to_block),
            "prefix_hit_blocks": self.num_prefix_hit_blocks,
            "cow_ops": self.num_cow_ops,
            "peak_used_blocks": self.peak_used_blocks,
        }

    def cached_prefix_tokens(self, token_ids: list[int]) -> int:
        """Number of leading tokens whose KV is already in the pool and reusable."""
        return len(self.match_prefix(token_ids)) * self.block_size

    # ------------------------------------------------------------- 空闲表
    def _acquire_free_block(self) -> PhysicalBlock:
        while self.free_blocks:
            blk = self.free_blocks.popleft()
            if not self._in_free[blk.block_id]:
                continue  # 陈旧项：这个 block 已经通过 _share() 发出去了
            self._in_free[blk.block_id] = 0
            self._num_free -= 1
            self._evict(blk)
            blk.tokens = ()
            blk.ref_count = 1
            self.peak_used_blocks = max(self.peak_used_blocks, self.num_used_blocks)
            return blk
        raise OutOfBlocksError(f"out of KV blocks ({self.num_blocks} total)")

    def _evict(self, blk: PhysicalBlock) -> None:
        """Drop the block's cache entry -- its content is about to be overwritten."""
        if blk.block_hash is not None:
            if self.hash_to_block.get(blk.block_hash) == blk.block_id:
                del self.hash_to_block[blk.block_hash]
            blk.block_hash = None

    def _share(self, block_id: int) -> None:
        """Take a second reference on an existing block (no copy)."""
        blk = self.blocks[block_id]
        if blk.ref_count == 0:
            if not self._in_free[block_id]:
                raise AssertionError(f"block {block_id} is neither referenced nor free")
            self._in_free[block_id] = 0
            self._num_free -= 1
        blk.ref_count += 1
        self.peak_used_blocks = max(self.peak_used_blocks, self.num_used_blocks)

    # -------------------------------------------------------------- 哈希
    def _block_hash(self, prev_hash: bytes, tokens: tuple[int, ...]) -> bytes:
        h = hashlib.blake2b(digest_size=16)
        h.update(prev_hash)
        h.update(array("q", tokens).tobytes())
        return h.digest()

    def match_prefix(self, token_ids: list[int]) -> list[int]:
        """Physical blocks already holding a cached prefix of ``token_ids``.

        Only *complete* blocks participate: a block must be full before its
        content is deterministic enough to be shared.
        """
        if not self.enable_prefix_caching:
            return []
        bs = self.block_size
        matched: list[int] = []
        prev = _PREFIX_SALT
        for i in range(len(token_ids) // bs):
            chunk = tuple(token_ids[i * bs : (i + 1) * bs])
            prev = self._block_hash(prev, chunk)
            block_id = self.hash_to_block.get(prev)
            if block_id is None:
                break
            if self.blocks[block_id].tokens != chunk:
                break  # 哈希碰撞保护
            matched.append(block_id)
        return matched

    # ----------------------------------------------------------- 分配
    def can_allocate(self, seq: Sequence, num_tokens: int | None = None) -> AllocStatus:
        """Can the prefix cache + free list cover ``num_tokens`` of this sequence?"""
        num_tokens = len(seq.data) if num_tokens is None else num_tokens
        bs = self.block_size
        needed = (num_tokens + bs - 1) // bs
        if self.enable_prefix_caching:
            needed -= len(self.match_prefix(seq.data[:num_tokens]))
        if needed <= self._num_free:
            return AllocStatus.OK
        return AllocStatus.LATER

    def allocate(self, seq: Sequence, num_tokens: int | None = None) -> int:
        """Build ``seq.block_table`` covering ``num_tokens`` tokens.

        With chunked prefill ``num_tokens`` is only the first chunk, not the whole
        prompt: pages are handed out as the prompt is actually computed, never
        reserved up front.  Returns the number of blocks satisfied from the cache.
        """
        if seq.block_table:
            raise AssertionError("sequence already owns a block table")

        bs = self.block_size
        num_tokens = len(seq.data) if num_tokens is None else num_tokens
        needed = (num_tokens + bs - 1) // bs

        matched = self.match_prefix(seq.data[:num_tokens])[:needed]
        for block_id in matched:
            self._share(block_id)
            seq.block_table.append(block_id)
            seq.block_hashes.append(self.blocks[block_id].block_hash)
        self.num_prefix_hit_blocks += len(matched)

        for _ in range(needed - len(matched)):
            blk = self._acquire_free_block()
            seq.block_table.append(blk.block_id)
            seq.block_hashes.append(None)
        return len(matched)

    def ensure_capacity(self, seq: Sequence, num_tokens: int) -> list[tuple[int, int]]:
        """Grow the block table to cover ``num_tokens``, copying shared pages if needed.

        Returns ``[(src_block, dst_block), ...]`` -- copy-on-write operations the
        model runner must apply to the real KV tensors before writing new tokens.
        """
        bs = self.block_size
        needed = (num_tokens + bs - 1) // bs
        cow_ops: list[tuple[int, int]] = []

        # 第一个新 token 可能落在我们已经拥有的 block 里。如果
        # 这个 block 是共享的（fork / 前缀缓存），必须先复制，否则
        # 会把另一个持有者的 KV 写坏。
        idx = seq.num_computed_tokens // bs
        if num_tokens > seq.num_computed_tokens and idx < len(seq.block_table):
            src = seq.block_table[idx]
            if self.blocks[src].ref_count > 1:
                dst = self._acquire_free_block()
                self.blocks[src].ref_count -= 1
                seq.block_table[idx] = dst.block_id
                seq.block_hashes[idx] = None
                cow_ops.append((src, dst.block_id))
                self.num_cow_ops += 1

        while len(seq.block_table) < needed:
            blk = self._acquire_free_block()
            seq.block_table.append(blk.block_id)
            seq.block_hashes.append(None)
        return cow_ops

    def register_computed_blocks(self, seq: Sequence) -> None:
        """Publish every now-complete block of ``seq`` into the prefix cache."""
        if not self.enable_prefix_caching:
            return
        bs = self.block_size
        prev = _PREFIX_SALT
        for i in range(seq.num_computed_tokens // bs):
            blk = self.blocks[seq.block_table[i]]
            if blk.block_hash is not None:
                prev = blk.block_hash
                continue
            chunk = tuple(seq.data[i * bs : (i + 1) * bs])
            if blk.tokens and blk.tokens != chunk:
                # 这个物理 block 里装的是别人的内容：绝不能写进缓存。
                return
            prev = self._block_hash(prev, chunk)
            blk.block_hash = prev
            blk.tokens = chunk
            self.hash_to_block.setdefault(prev, blk.block_id)
            seq.block_hashes[i] = prev

    def free(self, seq: Sequence) -> None:
        for block_id in seq.block_table:
            blk = self.blocks[block_id]
            blk.ref_count -= 1
            if blk.ref_count < 0:
                raise AssertionError(f"double free of block {block_id}")
            if blk.ref_count == 0:
                self._in_free[block_id] = 1
                self.free_blocks.append(blk)
                self._num_free += 1
        seq.block_table.clear()
        seq.block_hashes.clear()

    def fork(self, parent: Sequence, child: Sequence) -> None:
        """Make ``child`` share every block of ``parent`` (ref-counted, no copy).

        Used for ``n > 1`` parallel sampling.  The pages stay shared until one of
        the two writes past the fork point, at which point ``ensure_capacity``
        copies exactly the one block that diverges.
        """
        child.block_table = list(parent.block_table)
        child.block_hashes = list(parent.block_hashes)
        child.num_computed_tokens = parent.num_computed_tokens
        for block_id in child.block_table:
            self._share(block_id)

    # -------------------------------------------------------------- 分析
    def naive_reservation_cost(self, seq: Sequence, max_model_len: int) -> int:
        """Blocks a *non-paged* engine would reserve up front for this sequence."""
        return (max_model_len + self.block_size - 1) // self.block_size
