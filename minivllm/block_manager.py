"""The paged KV cache allocator -- the heart of PagedAttention.

Mental model: the KV cache is a fixed pool of ``num_blocks`` physical pages, and
every sequence holds a *page table* (``seq.block_table``) mapping its logical
block index to a physical block id.  The attention kernel never assumes that a
sequence's KV is contiguous in memory; it walks the page table instead.

Three mechanisms share this one pool:

1. **No reservation.**  A sequence only owns pages for tokens it has actually
   computed.  The classic "reserve ``max_model_len`` tokens per request" approach
   is implemented in ``naive_reservation_cost()`` so the waste can be measured.
2. **Reference counting + copy-on-write.**  Two sequences may share the same
   physical block (identical system prompt, or ``n`` parallel samples).  A block
   is only copied when a sequence writes into a *partially filled* shared block.
3. **Prefix caching.**  A full block is hashed over its token content; once it
   falls to ref-count 0 it stays in ``hash_to_block`` as an evictable cache entry,
   so the next request with the same prefix can skip its prefill entirely.
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
        # FIFO free list; entries can go stale when a block is *shared* (not
        # evicted) out of the free list, so ``_in_free`` is the source of truth.
        self.free_blocks: Deque[PhysicalBlock] = deque(self.blocks)
        self._in_free = bytearray(b"\x01") * num_blocks
        self._num_free = num_blocks

        self.hash_to_block: dict[bytes, int] = {}

        # --- stats ---
        self.num_prefix_hit_blocks = 0
        self.num_cow_ops = 0
        self.peak_used_blocks = 0

    # ------------------------------------------------------------------ stats
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

    # ------------------------------------------------------------ free list
    def _acquire_free_block(self) -> PhysicalBlock:
        while self.free_blocks:
            blk = self.free_blocks.popleft()
            if not self._in_free[blk.block_id]:
                continue  # stale: the block was handed out through _share()
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

    # -------------------------------------------------------------- hashing
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
                break  # hash collision guard
            matched.append(block_id)
        return matched

    # ----------------------------------------------------------- allocation
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

        # The first new token may land inside a block we already own.  If that
        # block is shared (fork / prefix cache) it has to be copied first, or we
        # would corrupt the other owner's KV.
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
                # The physical block holds someone else's content: never cache it.
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

    # -------------------------------------------------------------- analysis
    def naive_reservation_cost(self, seq: Sequence, max_model_len: int) -> int:
        """Blocks a *non-paged* engine would reserve up front for this sequence."""
        return (max_model_len + self.block_size - 1) // self.block_size
