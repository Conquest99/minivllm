"""Continuous batching scheduler.

The scheduling unit is one *step* of the model, i.e. one forward pass over a set
of tokens.  Two decisions are made every step:

1. **Who runs?**  Every running sequence gets its next token (decode) or its next
   prefill chunk; leftover budget is filled with brand-new requests.  A finished
   sequence frees its slot *immediately*, so a new request can join mid-flight --
   that is the "continuous" in continuous batching.  Static batching instead waits
   for the whole batch to hit ``max_tokens``.
2. **What if the cache is full?**  Preempt the youngest running sequence and drop
   its KV completely (``recompute``), rather than swapping it to CPU.  For a small
   model the recompute is a single fast prefill, and it frees memory instantly.

Order matters: decodes go first.  A decode is 1 token and its latency is what a
user perceives as smoothness; a prefill is a burst of compute whose result nobody
sees until it finishes.  Draining decodes first is why vLLM's v1 scheduler beats
the old "prefill-first" order on inter-token latency.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

from minivllm.block_manager import AllocStatus, BlockManager, OutOfBlocksError
from minivllm.config import EngineConfig
from minivllm.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutputs:
    scheduled: list[tuple[Sequence, int]] = field(default_factory=list)
    """``(sequence, num_new_tokens)`` in batch order."""
    preempted: list[Sequence] = field(default_factory=list)
    cow_ops: list[tuple[int, int]] = field(default_factory=list)
    num_batched_tokens: int = 0

    @property
    def num_prefill_seqs(self) -> int:
        return sum(1 for seq, n in self.scheduled if n > 1)

    @property
    def num_decode_seqs(self) -> int:
        return sum(1 for _, n in self.scheduled if n == 1)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class Scheduler:
    def __init__(self, cfg: EngineConfig, block_manager: BlockManager) -> None:
        self.cfg = cfg
        self.block_manager = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

        # --- stats ---
        self.num_preemptions = 0
        self.peak_running = 0

    # ------------------------------------------------------------- interface
    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def free_finished(self) -> list[Sequence]:
        finished = [seq for seq in self.running if seq.is_finished]
        if finished:
            self.running = [seq for seq in self.running if not seq.is_finished]
            for seq in finished:
                # publish the blocks first: free() clears the block table
                self.block_manager.register_computed_blocks(seq)
                self.block_manager.free(seq)
        return finished

    def reset_cache(self) -> None:
        """Drop all prefix-cache entries (only legal when nothing is running)."""
        if self.running:
            raise RuntimeError("cannot reset the prefix cache while sequences are running")
        self.block_manager.hash_to_block.clear()
        for blk in self.block_manager.blocks:
            blk.block_hash = None
            blk.tokens = ()

    # ------------------------------------------------------------ scheduling
    def schedule(self) -> SchedulerOutputs:
        out = SchedulerOutputs()
        budget = self.cfg.max_num_batched_tokens

        self._schedule_running(out, budget)
        budget -= out.num_batched_tokens
        self._schedule_waiting(out, budget)

        out.num_batched_tokens = sum(n for _, n in out.scheduled)
        self.peak_running = max(self.peak_running, len(self.running))
        return out

    def _chunk_size(self, seq: Sequence, budget: int) -> int:
        """Tokens to run for this sequence, block-aligned when it is a partial chunk."""
        remaining = seq.num_new_tokens
        n = min(remaining, budget)
        if n == remaining:
            return n
        if not self.cfg.enable_chunked_prefill:
            return 0
        bs = self.cfg.block_size
        end = seq.num_computed_tokens + n
        aligned = end - end % bs
        return max(aligned - seq.num_computed_tokens, 0)

    def _schedule_running(self, out: SchedulerOutputs, budget: int) -> None:
        running = self.running
        i = 0
        while i < len(running) and budget > 0:
            seq = running[i]
            n = self._chunk_size(seq, budget)
            if n <= 0:
                i += 1
                continue

            try:
                cow_ops = self.block_manager.ensure_capacity(seq, seq.num_computed_tokens + n)
            except OutOfBlocksError:
                # Cache is full: sacrifice the youngest sequence.
                victim = running.pop()
                self.block_manager.free(victim)
                victim.num_computed_tokens = 0
                victim.num_preemptions += 1
                victim.status = SequenceStatus.WAITING
                self.waiting.appendleft(victim)
                out.preempted.append(victim)
                self.num_preemptions += 1
                if victim is seq:
                    continue
                continue

            out.cow_ops.extend(cow_ops)
            out.scheduled.append((seq, n))
            out.num_batched_tokens += n
            budget -= n
            i += 1

    def _schedule_waiting(self, out: SchedulerOutputs, budget: int) -> None:
        if self.cfg.static_batching and self.running:
            return  # the current batch must drain completely before refilling
        first_admitted = len(out.scheduled)

        while self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs:
            seq = self.waiting[0]

            if len(seq.data) > self.cfg.max_model_len:
                self.waiting.popleft()
                seq.mark_finished(SequenceStatus.FINISHED_LENGTH_CAPPED)
                continue

            # A prompt longer than the step budget would monopolise the step; let
            # it wait until it can be the only thing running (vLLM's behaviour).
            if (
                self.cfg.long_prefill_token_threshold
                and seq.num_new_tokens > self.cfg.long_prefill_token_threshold
                and len(self.running) > 0
            ):
                break

            if not seq.block_table and self.cfg.enable_prefix_caching:
                cached = self.block_manager.cached_prefix_tokens(seq.data)
                if cached:
                    # The KV for this prefix is already in the pool and correct, so
                    # the prefill can start after it.  Keep at least one token: the
                    # logits for the last token are the whole point of the prefill.
                    seq.num_computed_tokens = min(cached, len(seq.data) - 1)

            n = self._chunk_size(seq, budget)
            if n <= 0:
                break

            if self.block_manager.can_allocate(seq, seq.num_computed_tokens + n) != AllocStatus.OK:
                break

            self.waiting.popleft()
            if not seq.block_table:
                self.block_manager.allocate(seq, seq.num_computed_tokens + n)
            out.cow_ops.extend(self.block_manager.ensure_capacity(seq, seq.num_computed_tokens + n))
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

            out.scheduled.append((seq, n))
            out.num_batched_tokens += n
            budget -= n

        if self.cfg.static_batching:
            batch = [seq for seq, _ in out.scheduled[first_admitted:]]
            if batch:
                # a static batch shares one deadline: everybody runs as long as the
                # longest member, which is exactly the waste being measured
                longest = max(s.sampling_params.max_tokens for s in batch)
                for seq in batch:
                    seq.sampling_params = replace(seq.sampling_params, max_tokens=longest, ignore_eos=True)
