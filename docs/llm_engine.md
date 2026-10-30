# `minivllm/llm_engine.py` 详解

## 这个文件干什么

**把所有部件串成一个循环**。整个推理引擎的骨架只有这 6 行：

```python
while self.scheduler.has_unfinished():
    sched_out = self.scheduler.schedule()            # 谁跑、各跑几个 token
    model_runner.apply_cow(sched_out.cow_ops)        # 结构决策 → 显存复制
    logits = model_runner.execute_model(...)         # 分页 attention 前向
    _postprocess(...)                                # 采样 + 生命周期
```

`step()` 就是一次这样的循环；`generate()` 就是「加请求 + 反复 step」。

**一个 step == 一次前向 == 一次调度决策。** 这个等式是理解全文的关键。

---

## 1. 两个数据结构

### `RequestOutput`：对外的结果

```python
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
```

为什么**同时**提供 `text` 和 `token_ids`？

* `text` 给人看
* `token_ids` 给测试比对（文本比对会被 tokenizer 的 normalize 干扰，token id 是精确的）

`finish_reason` 从枚举转成小写字符串（`"stopped"` / `"length_capped"`）：

```python
finish_reason=seq.status.name.removeprefix("FINISHED_").lower(),
```

`str.removeprefix` 是 Python 3.9+ 的方法。比 `name.replace("FINISHED_", "")` 安全（后者会把中间出现的也替掉），比切片 `name[9:]` 可读。

### `EngineMetrics`：吞吐与时延

```python
@dataclass
class EngineMetrics:
    num_steps: int = 0
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    wall_time: float = 0.0
    step_times: list[float] = field(default_factory=list)

    @property
    def throughput(self) -> float:
        return self.num_generation_tokens / self.wall_time if self.wall_time else 0.0

    @property
    def tokens_per_step(self) -> float:
        return self.num_generation_tokens / self.num_steps if self.num_steps else 0.0
```

两个派生指标，各有明确含义：

| 指标 | 公式 | 它回答什么问题 |
|---|---|---|
| `throughput` | 生成 token 数 / 墙钟时间 | 这台机器整体能吐多少字 |
| `tokens_per_step` | 生成 token 数 / step 数 | **平均每个 step 有几个 token 是"有效"的** |

`tokens_per_step` 是判断「批处理效率」的关键数字：

* 连续批处理：约等于当前 batch 里还在 decode 的序列数（本项目实测 6.0，上限 8）
* 静态批处理：因为短请求陪跑，这个数字按定义就是 8.0，但其中有大量重复/无用的生成

> 有意思的是：**静态批处理的 `tokens_per_step` 更高（8.0 > 6.0），但吞吐更低**。因为它把算力花在了 56% 的无效 token 上。这组数字是「为什么不能只看 GPU 利用率」的绝佳例子。

`if self.wall_time else 0.0` 是防除零。写成 `if` 表达式而不是 `try/except ZeroDivisionError`，因为这是预期内的状态（还没跑过）。

---

## 2. `__init__`：依赖装配

```python
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
```

### 依赖方向（单向，没有回头箭头）

```
LLMEngine
├── ModelRunner   ← 只知道 config 和显存
├── BlockManager  ← 纯 Python，块数从 ModelRunner 拿
└── Scheduler     ← 持有 BlockManager 的引用
```

`BlockManager` 的 `num_blocks` 来自 `model_runner.num_gpu_blocks`——因为**池子大小是显存决定的**，必须等模型加载完、知道还剩多少显存才能定。

### `torch.Generator`

```python
self.generator = torch.Generator(device=self.model_runner.device)
self.generator.manual_seed(cfg.seed)
```

`Generator` 是**独立的随机数状态**，传进 `torch.multinomial(..., generator=gen)` 就只受它影响。

为什么不用全局 `torch.manual_seed`？因为全局种子会被别处的代码（数据加载、初始化）干扰，无法保证「同 seed 同结果」。**引擎级的可复现必须用自己的 generator。**

### `_outputs`：一个「待取件」缓冲区

```python
self._outputs: dict[int, RequestOutput] = {}
```

`step()` 内部会产生已完成的请求，但 `step()` 的返回值是给调用者的。中间通过这个 dict 中转，最后 `_drain_outputs()` 一次性取走并清空。

为什么用 dict 而不是 list？因为一个 step 里可能有多条请求结束，用 `seq_id → output` 保证去重，且 `sorted(...)` 能给出稳定顺序。

---

## 3. `add_request`：准入与校验

```python
def add_request(self, prompt: str, params: SamplingParams | None = None) -> int:
    params = params or SamplingParams()
    token_ids = self._encode_prompt(prompt)
    if len(token_ids) + params.max_tokens > self.cfg.max_model_len:
        raise ValueError(
            f"prompt ({len(token_ids)} tokens) + max_tokens ({params.max_tokens}) "
            f"exceeds max_model_len ({self.cfg.max_model_len})"
        )
    seq = Sequence(seq_id=self._next_seq_id, prompt_token_ids=token_ids,
                   sampling_params=params, prompt=prompt)
    self._next_seq_id += 1
    self.metrics.num_prompt_tokens += len(token_ids)
    self.scheduler.add(seq)
    return seq.seq_id
```

### `params = params or SamplingParams()`

用 `or` 做默认值有个坑：如果传进来的是「假值」会被替换。`SamplingParams` 是 dataclass 实例，永远为真，所以这里安全。

（如果参数是 `int`，`or` 会把 0 误判为"没传"，那是 bug。这也是为什么很多代码写 `if params is None`。）

### 立即拒绝超长请求

```python
if len(token_ids) + params.max_tokens > self.cfg.max_model_len:
    raise ValueError(...)
```

**在入口处失败，而不是在调度器里静默丢弃。** 因为这个组合注定装不下（`max_model_len` 是硬上限），先报错能让调用者立刻知道参数配错了，而不是拿到一个"生成到一半被截断"的结果。

`Scheduler._schedule_waiting` 里还有一个 `len(seq.data) > max_model_len` 的检查——那是兜底（防止运行时 `data` 增长越界），两者职责不同。

### chat template

```python
def _encode_prompt(self, prompt: str) -> list[int]:
    if self.cfg.use_chat_template and getattr(self.tokenizer, "chat_template", None):
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
    else:
        text = prompt
    return self.tokenizer(text).input_ids
```

`apply_chat_template(..., tokenize=False)` 返回**字符串**，再自己 `tokenizer(text)`。

为什么不直接 `tokenize=True`？为了留一个「拿到最终字符串」的中间态，方便调试（打印出来能直接看出 `<|im_start|>user\n...<|im_end|>`）。

`add_generation_prompt=True` 会在末尾加上 `<|im_start|>assistant\n`——这是告诉模型「该你说话了」。**不加的话模型会接着用户的话往下编，输出质量崩坏。**

`getattr(self.tokenizer, "chat_template", None)` 是防御：base 模型没有 chat template，这时退回裸 prompt。

---

## 4. `step()`：一次前向

```python
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
```

### `if sched_out.is_empty: return []`

如果没有任何能跑的（比如所有 prompt 都太长、预算不够），返回空列表而不是死循环。`generate()` 里的 `while has_unfinished()` 会再调一次 `step()`——如果一直是空，就会**忙等**。

实际上这个组合不会死锁：`can_allocate` 失败时 `waiting` 会保留，但只要 `running` 有空位和预算就一定能推进。真需要防护的话，`generate` 里应该加「连续 N 次空 step 就报错」——这是生产代码该做的，本项目为简洁省略了。

### `apply_cow` 在 `execute_model` **之前**

顺序不能反。COW 是「复制旧页到新页」，必须在写入新 token 之前完成。否则新 token 会写进还在共享的页里，把别人的数据污染掉。

### `now = time.perf_counter()` 在采样**之前**取

这个时刻被用作 `first_token_time`。取在前向之后、采样之前，量的是「模型算完的那一刻」，不含采样开销。因为采样只是选一个 id，几十微秒，且不属于"模型延迟"。

### `step_times` 全量保存

```python
self.metrics.step_times.append(time.perf_counter() - t0)
```

保留每一步的耗时（而不是只存总和），是为了之后能算 **p99 / 抖动**。ITL 的抖动比平均值更能反映用户体验。

---

## 5. `_postprocess`：生命周期管理

```python
def _postprocess(self, seqs, new_tokens, logits, now) -> None:
    for row, (seq, n) in enumerate(zip(seqs, new_tokens, strict=False)):
        seq.num_computed_tokens += n
        if seq.num_computed_tokens < len(seq.data):
            continue  # 这是 prefill 的一块，还没有 logits

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
```

### 第一行 `seq.num_computed_tokens += n` 必须最先

因为后面的判断依赖它：

```python
if seq.num_computed_tokens < len(seq.data):
    continue
```

含义：**还有 token 没算完 → 这是 prefill 的中间块 → 没有 logits 可用。**

用具体数字看：

```
len(data) = 100
这一 step n = 32
num_computed: 0 → 32    32 < 100  → continue（不采样）
下一 step n = 32
num_computed: 32 → 64   64 < 100  → continue
下一 step n = 36
num_computed: 64 → 100  100 == 100 → 采样！
```

**注意 `logits[row]` 永远是「这一序列最后一个 token」的 logits**（因为 `_build_inputs` 只取了 `last_indices`）。所以只有在「刚好算完」时，这个 logits 才对应真正的最后一个 token。

> 这也是为什么 chunked prefill 的中间块可以不产出 logits 而不浪费——`execute_model` 里 `last_indices` 对这些序列指向的是「这一块的最后一个 token」，算了但不用。严格说有一点浪费，可以优化成「只对刚好算完的序列取 logits」。这是本项目明确保留的简化。

### 结束判定的优先级

```python
if token_id in eos and not ignore_eos:        → STOPPED
elif num_output_tokens >= max_tokens:          → LENGTH_CAPPED
elif stop 串命中:                              → STOPPED
```

* **EOS 优先**：模型自己决定结束是最自然的。
* **`ignore_eos`**：压测时忽略 EOS，保证每条请求长度可控。
* **长度上限**：防止无限生成。
* **stop 字符串**：按**解码后的文本**匹配，因为一个词可能被切成多个 token。

`params.stop and any(s in self._decoded_output(seq) for s in params.stop)` 每次都要解码整个输出——O(n²) 的开销。生产实现会增量解码 + 前缀树匹配。本项目为了可读性直接全量解。

### 最后统一回收

```python
for seq in self.scheduler.free_finished():
    self._outputs[seq.seq_id] = self._make_output(seq)
```

**在循环外**做回收，而不是在循环内立即回收。原因：`free_finished` 会修改 `self.running` 列表，而我们在循环里正遍历 `zip(seqs, ...)`——虽然 `seqs` 是快照不影响，但把「状态变更」集中在循环之后更不容易出错。

---

## 6. `generate()`：跑到底

```python
def generate(self, prompts: list[str], params: SamplingParams | None = None) -> list[RequestOutput]:
    t0 = time.perf_counter()
    ids = [self.add_request(p, params) for p in prompts]
    done: dict[int, RequestOutput] = {}
    while self.scheduler.has_unfinished():
        for out in self.step():
            done[out.seq_id] = out
    self.metrics.wall_time += time.perf_counter() - t0
    return [done[i] for i in ids]
```

三个细节：

1. **先全加进队列，再循环**。这样 3 条 prompt 会被连续批处理混在一起跑，而不是一条一条串行。
2. **`return [done[i] for i in ids]`**：按**输入顺序**返回。`done` 是 dict，完成顺序是无序的；用 `ids` 重排保证调用者拿到的顺序和传入一致。
3. **`+=` 而不是 `=`**：多次调用 `generate` 时累计墙钟时间。所以测吞吐要先把 `metrics` 重置成新的 `EngineMetrics()`（benchmark 脚本就是这么做的）。

---

## 7. `stats()`：把内部状态摊平成一个 dict

```python
def stats(self) -> dict[str, float]:
    s = self.block_manager.stats()
    s.update({
        "num_steps": self.metrics.num_steps,
        "throughput_tok_s": self.metrics.throughput,
        "tokens_per_step": self.metrics.tokens_per_step,
        "peak_running": self.scheduler.peak_running,
        "num_preemptions": self.scheduler.num_preemptions,
    })
    return s
```

设计上的好处：**所有诊断信息通过一个函数暴露**，benchmark 脚本不需要知道内部结构，直接 `json.dumps(engine.stats())` 就能存档。

里面的字段值得一个个看（它们就是「引擎健康度」）：

| 字段 | 健康表现 | 异常意味着 |
|---|---|---|
| `utilization` | 高 | 低说明池子开太大或并发不足 |
| `cached_blocks` | 稳定 | 持续增长可能是泄漏（块没被复用） |
| `prefix_hit_blocks` | > 0（有共享前缀时） | 0 说明前缀缓存没生效 |
| `cow_ops` | 小 | 大说明共享太多、分叉频繁 |
| `peak_running` | 接近 `max_num_seqs` | 远小说明请求来不够快 |
| `num_preemptions` | 0 或很小 | 大说明显存不够、池子该调大 |

---

## 用到的 PyTorch / Python 特性

| API | 作用 |
|---|---|
| `torch.Generator(device=...)` + `manual_seed` | 独立可复现的随机源 |
| `enumerate(zip(a, b, strict=False))` | 同时遍历两个列表并取序号 |
| `dict` 作「待取件」缓冲 | 按键去重、按序取出 |
| `@dataclass` | 结果对象 / 指标容器 |
| `str.removeprefix` | 枚举名 → 小写原因 |
| `time.perf_counter()` | 单调时钟 |
| `x or default` | 简单默认值（仅对非假值安全） |
| `dict.update` | 合并统计 |
| `sorted(dict)` | 稳定顺序取出 |

---

## 设计取舍

**为什么 `step()` 要暴露出来，而不是只给 `generate()`？**

因为**流式输出**（server-sent events）需要每步拿到新 token，而不是等整条生成完。真实推理服务器的核心接口就是 `step()`；`generate()` 只是它的一个便利封装。

**为什么不在 `step()` 里自动回收内存？**

已经在回收了（`free_finished`）。但**不驱逐前缀缓存**——那是刻意的：缓存留着才能让后续请求命中。

**为什么 `_outputs` 是引擎级字段而不是 `step()` 的局部变量？**

因为 `_postprocess` 是分开的方法，需要和 `step()` 共享。也可以合并成一个方法、用局部变量，但那样 `step()` 会有 60 行，可读性下降。

---

## 常见误解

* 「`step()` 处理一个 token」——不一定。prefill 的 step 可能一次处理几百个 token。
* 「`logits` 行数等于 batch 里的 token 数」——等于 `len(seqs)`（每个序列一行）。
* 「`num_computed_tokens += n` 应该放在采样之后」——必须在之前，它是"是否需要采样"的判据。
* 「`generate` 的返回顺序是完成顺序」——是按 `ids`（输入顺序）重排的。
* 「`throughput` 可以直接跨多次 `generate` 比较」——不会自动清零，需要手动 `engine.metrics = EngineMetrics()`。

---

## 动手练习

1. 在 `step()` 里把 `apply_cow` 移到 `execute_model` 之后，跑 `tests/test_engine_e2e.py`，看 `test_prefix_caching_gives_the_same_tokens` 是否失败。
2. 把 `_postprocess` 里 `seq.num_computed_tokens += n` 挪到 `continue` 之后，跑 `tests/test_model_runner.py::test_chunked_prefill_matches_single_shot`。
3. 打个日志，在 32 请求的压测里打印每个 step 的 `len(sched_out.scheduled)` 和 token 数，你会看到 batch 从满到空再到满的锯齿——这就是连续批处理的呼吸。
4. 把 `metrics.wall_time +=` 改成 `=`，连续调两次 `generate`，观察 `throughput_tok_s` 的变化。
