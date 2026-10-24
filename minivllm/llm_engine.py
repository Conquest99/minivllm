"""The engine: one ``step()`` == one forward pass == one scheduling decision.

This is the loop everything else exists to serve::

    while requests are alive:
        batch = scheduler.schedule()          # who runs, how many tokens each
        model_runner.apply_cow(batch.cow_ops) # structural decisions -> memcpy
        logits = model_runner.execute_model() # paged attention over the block pool
        sample + bookkeeping                  # finished seqs leave, freed seqs return

Wall-clock is measured per step, so the engine reports the two numbers that
actually matter for an inference server: **tokens/s** (throughput) and **TTFT /
ITL** (how smooth a single stream feels).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from minivllm.block_manager import BlockManager
from minivllm.config import EngineConfig
from minivllm.model_runner import ModelRunner
from minivllm.sampling import SamplingParams, sample
from minivllm.scheduler import Scheduler
from minivllm.sequence import Sequence, SequenceStatus


@dataclass
class RequestOutput:
    seq_id: int
    prompt: str
    text: str
    token_ids: list[int]
    finish_reason: str
    num_prompt_tokens: int
    num_output_tokens: int
    ttft: float | None = None
    latency: float | None = None
    num_preemptions: int = 0


@dataclass
class EngineMetrics:
    num_steps: int = 0
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    wall_time: float = 0.0
    step_times: list[float] = field(default_factory=list)

    @property
    def throughput(self) -> float:
        """Generated tokens per second of wall clock."""
        return self.num_generation_tokens / self.wall_time if self.wall_time else 0.0

    @property
    def tokens_per_step(self) -> float:
        return self.num_generation_tokens / self.num_steps if self.num_steps else 0.0


class LLMEngine:
    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self.model_runner = ModelRunner(cfg)
        self.tokenizer = self.model_runner.tokenizer
        self.block_manager = BlockManager(
            num_blocks=self.model_runner.num_gpu_blocks,
            block_size=cfg.block_size,
            enable_prefix_caching=cfg.enable_prefix_caching,
        )
        self.scheduler = Scheduler(cfg, self.block_manager)

        self.generator = torch.Generator(device=self.model_runner.device)
        self.generator.manual_seed(cfg.seed)

        self._next_seq_id = 0
        self._outputs: dict[int, RequestOutput] = {}
        self.metrics = EngineMetrics()

    # ------------------------------------------------------------- requests
    def _encode_prompt(self, prompt: str) -> list[int]:
        if self.cfg.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
            )
        else:
            text = prompt
        return self.tokenizer(text).input_ids

    def add_request(self, prompt: str, params: SamplingParams | None = None) -> int:
        params = params or SamplingParams()
        token_ids = self._encode_prompt(prompt)
        if len(token_ids) + params.max_tokens > self.cfg.max_model_len:
            raise ValueError(
                f"prompt ({len(token_ids)} tokens) + max_tokens ({params.max_tokens}) "
                f"exceeds max_model_len ({self.cfg.max_model_len})"
            )
        seq = Sequence(
            seq_id=self._next_seq_id,
            prompt_token_ids=token_ids,
            sampling_params=params,
            prompt=prompt,
        )
        self._next_seq_id += 1
        self.metrics.num_prompt_tokens += len(token_ids)
        self.scheduler.add(seq)
        return seq.seq_id

    # ----------------------------------------------------------------- step
    def step(self) -> list[RequestOutput]:
        t0 = time.perf_counter()
        sched_out = self.scheduler.schedule()
        if sched_out.is_empty:
            return []

        if sched_out.cow_ops:
            self.model_runner.apply_cow(sched_out.cow_ops)

        seqs = [seq for seq, _ in sched_out.scheduled]
        new_tokens = [n for _, n in sched_out.scheduled]
        logits = self.model_runner.execute_model(seqs, new_tokens)

        now = time.perf_counter()
        self._postprocess(seqs, new_tokens, logits, now)

        self.metrics.num_steps += 1
        self.metrics.step_times.append(time.perf_counter() - t0)

        return self._drain_outputs()

    def _postprocess(self, seqs, new_tokens, logits, now) -> None:
        for row, (seq, n) in enumerate(zip(seqs, new_tokens, strict=False)):
            seq.num_computed_tokens += n
            if seq.num_computed_tokens < len(seq.data):
                continue  # this was a prefill chunk, no logits were materialised yet

            params = seq.sampling_params
            token_id = sample(logits[row], params, self.generator)
            seq.append_token(token_id)
            if seq.first_token_time is None:
                seq.first_token_time = now
            self.metrics.num_generation_tokens += 1

            if token_id in self.model_runner.eos_token_ids and not params.ignore_eos:
                seq.mark_finished(SequenceStatus.FINISHED_STOPPED)
            elif seq.num_output_tokens >= params.max_tokens:
                seq.mark_finished(SequenceStatus.FINISHED_LENGTH_CAPPED)
            elif params.stop and any(s in self._decoded_output(seq) for s in params.stop):
                seq.mark_finished(SequenceStatus.FINISHED_STOPPED)

        for seq in self.scheduler.free_finished():
            self._outputs[seq.seq_id] = self._make_output(seq)

    def _decoded_output(self, seq: Sequence) -> str:
        return self.model_runner.decode(seq.data[seq.num_prompt_tokens :])

    def _make_output(self, seq: Sequence) -> RequestOutput:
        return RequestOutput(
            seq_id=seq.seq_id,
            prompt=seq.prompt,
            text=self._decoded_output(seq),
            token_ids=list(seq.data[seq.num_prompt_tokens :]),
            finish_reason=seq.status.name.removeprefix("FINISHED_").lower(),
            num_prompt_tokens=seq.num_prompt_tokens,
            num_output_tokens=seq.num_output_tokens,
            ttft=seq.ttft,
            latency=seq.latency,
            num_preemptions=seq.num_preemptions,
        )

    def _drain_outputs(self) -> list[RequestOutput]:
        if not self._outputs:
            return []
        out = [self._outputs[k] for k in sorted(self._outputs)]
        self._outputs.clear()
        return out

    # ------------------------------------------------------------- generate
    def generate(self, prompts: list[str], params: SamplingParams | None = None) -> list[RequestOutput]:
        """Run to completion and return one output per prompt, in input order."""
        t0 = time.perf_counter()
        ids = [self.add_request(p, params) for p in prompts]
        done: dict[int, RequestOutput] = {}
        while self.scheduler.has_unfinished():
            for out in self.step():
                done[out.seq_id] = out
        self.metrics.wall_time += time.perf_counter() - t0
        return [done[i] for i in ids]

    # ---------------------------------------------------------------- stats
    def stats(self) -> dict[str, float]:
        s = self.block_manager.stats()
        s.update(
            {
                "num_steps": self.metrics.num_steps,
                "throughput_tok_s": self.metrics.throughput,
                "tokens_per_step": self.metrics.tokens_per_step,
                "peak_running": self.scheduler.peak_running,
                "num_preemptions": self.scheduler.num_preemptions,
            }
        )
        return s

    def free_blocks(self) -> int:
        return self.block_manager.num_free_blocks
