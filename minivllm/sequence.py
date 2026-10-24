"""One in-flight request.

A sequence is a *token stream that grows*.  Everything the scheduler and the
model runner need to know about it reduces to three numbers:

    len(seq.data)            -- how many tokens exist (prompt + generated)
    seq.num_computed_tokens  -- how many of them already have KV in the cache
    len(seq.data) - seq.num_computed_tokens  -- how many tokens this step must run
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto

from minivllm.sampling import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED_STOPPED = auto()
    FINISHED_LENGTH_CAPPED = auto()
    FINISHED_ABORTED = auto()

    @property
    def is_finished(self) -> bool:
        return self in (
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH_CAPPED,
            SequenceStatus.FINISHED_ABORTED,
        )


@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    prompt: str = ""
    data: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING

    # --- paged KV cache bookkeeping ---
    block_table: list[int] = field(default_factory=list)
    """Logical block index -> physical block id.  This list *is* the page table."""
    block_hashes: list[bytes | None] = field(default_factory=list)
    """Parallel to block_table; the content hash of a block once it is full."""
    num_computed_tokens: int = 0
    """Prefix length whose KV is guaranteed to be in the cache."""

    # --- observability ---
    arrival_time: float = field(default_factory=time.perf_counter)
    first_token_time: float | None = None
    finish_time: float | None = None
    num_preemptions: int = 0

    def __post_init__(self) -> None:
        if not self.data:
            self.data = list(self.prompt_token_ids)

    # ------------------------------------------------------------------ sizes
    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.data) - self.num_prompt_tokens

    @property
    def num_new_tokens(self) -> int:
        """Uncomputed suffix -- what the next forward pass has to run."""
        return len(self.data) - self.num_computed_tokens

    @property
    def is_finished(self) -> bool:
        return self.status.is_finished

    def num_blocks_required(self, block_size: int) -> int:
        return (len(self.data) + block_size - 1) // block_size

    # --------------------------------------------------------------- mutation
    def append_token(self, token_id: int) -> None:
        self.data.append(token_id)

    def mark_finished(self, status: SequenceStatus) -> None:
        self.status = status
        self.finish_time = time.perf_counter()

    # ---------------------------------------------------------------- metrics
    @property
    def ttft(self) -> float | None:
        """Time to first token (prefill + one decode step)."""
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def latency(self) -> float | None:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    @property
    def num_blocks(self) -> int:
        return len(self.block_table)

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, status={self.status.name}, "
            f"len={len(self.data)}, computed={self.num_computed_tokens}, "
            f"blocks={len(self.block_table)})"
        )
