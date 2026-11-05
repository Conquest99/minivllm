"""引擎本体：一次 ``step()`` == 一次前向 == 一次调度决策。

这就是其它一切都在为之服务的那个循环::

    while requests are alive:
        batch = scheduler.schedule()          # 谁跑，各跑多少 token
        model_runner.apply_cow(batch.cow_ops) # 结构决策 -> memcpy
        logits = model_runner.execute_model() # 在 block 池上做分页注意力
        sample + bookkeeping                  # 结束的序列离场，释放的页回池

每步都计时，所以引擎报的正是推理服务真正关心的两个数：**tokens/s**（吞吐）和
**TTFT / ITL**（单条流顺不顺滑）。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
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
        """按墙上时间算的每秒生成 token 数。"""
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

        # 流式输出用的两个槽位：一个回调 + 每条序列已经发出去多少字符。
        # 之所以按「已发字符数」而不是「已发 token 数」记账，是因为一个中文字
        # 常被切成多个 token，逐 token 解码会吐出半个字符（U+FFFD）。
        self._token_sink: Callable[[int, str], None] | None = None
        self._sent_chars: dict[int, int] = {}
        self.last_output: RequestOutput | None = None
        """最近一次 ``generate_stream`` 结束时的完整输出，给需要元信息的调用方。"""

    # ------------------------------------------------------------- 请求
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

    # ----------------------------------------------------------------- 单步
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
                continue  # 这是一段 prefill 分块，还没有物化任何 logits

            params = seq.sampling_params
            token_id = sample(logits[row], params, self.generator)
            seq.append_token(token_id)
            if seq.first_token_time is None:
                seq.first_token_time = now
            self.metrics.num_generation_tokens += 1
            if self._token_sink is not None:
                self._token_sink(seq.seq_id, self._new_text(seq))

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

    def _new_text(self, seq: Sequence) -> str:
        """Decode the whole output so far and return only what has not been sent yet.

        Decoding the *full* prefix every step is O(n) per token, so O(n^2) overall --
        fine for the few hundred tokens this engine produces, and it is the only way to
        stay correct when a character spans several tokens.

        一个中文常用字常被切成几个 byte-level token。如果就在字符中间解码，末尾会得到
        U+FFFD；而它和真正那个字都是「一个字符」，按长度差下发就会把真字永久吞掉
        （实测：``函数`` 变成 ``\\ufffd数``）。所以末尾的替换符先扣住不发，等下一个
        token 补完再一起吐。
        """
        text = self._decoded_output(seq)
        if text.endswith("\ufffd"):
            text = text[:-1]
        sent = self._sent_chars.get(seq.seq_id, 0)
        if len(text) <= sent:
            return ""
        self._sent_chars[seq.seq_id] = len(text)
        return text[sent:]

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

    # ------------------------------------------------------------- 生成
    def generate(self, prompts: list[str], params: SamplingParams | None = None) -> list[RequestOutput]:
        """跑到全部结束，按输入顺序每条 prompt 返回一个输出。"""
        t0 = time.perf_counter()
        ids = [self.add_request(p, params) for p in prompts]
        done: dict[int, RequestOutput] = {}
        while self.scheduler.has_unfinished():
            for out in self.step():
                done[out.seq_id] = out
        self.metrics.wall_time += time.perf_counter() - t0
        return [done[i] for i in ids]

    def generate_stream(self, prompt: str, params: SamplingParams | None = None) -> Iterator[str]:
        """边跑边吐文本增量，一次只服务一条 prompt。

        必须一次一条：``step()`` 是批处理的入口，而「这条增量属于哪个请求」在多请求
        场景下要靠 seq_id 区分；流式接口的语义是「一条流」，所以这里就把并发留给上层
        —— 真实服务里，多路流是在共享的 ``step()`` 之上fan-out 出来的，不是各开一个
        引擎。

        拼接所有增量 == ``generate()`` 返回的 ``out.text``，这一点有测试锁住。
        """
        params = params or SamplingParams()
        t0 = time.perf_counter()
        self.add_request(prompt, params)
        self.last_output = None

        pending: list[str] = []

        def sink(_seq_id: int, delta: str) -> None:
            if delta:
                pending.append(delta)

        self._token_sink = sink
        try:
            while self.scheduler.has_unfinished():
                for finished in self.step():
                    # step() 每次都会清空内部输出表，所以这里必须接住，否则 finally
                    # 里再 drain 就什么也拿不到了。
                    self.last_output = finished
                if pending:
                    yield "".join(pending)
                    pending.clear()
        finally:
            self._token_sink = None
            self._sent_chars.clear()
            self.metrics.wall_time += time.perf_counter() - t0

    # ---------------------------------------------------------------- 统计
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
