# `tests/test_scheduler.py` 详解

## 这个文件干什么

给调度器写 **9 条策略断言**，**完全不涉及模型和 torch**。

这是整个测试套件里性价比最高的文件：毫秒级跑完，却把「连续批处理」的每一条规则都钉死了。

---

## 1. `build()`：一行建一个调度器

```python
def build(**cfg_kwargs) -> Scheduler:
    kwargs = dict(
        model="unused",
        block_size=4,
        max_num_seqs=8,
        max_num_batched_tokens=16,
        max_model_len=128,
        enable_prefix_caching=False,
    )
    kwargs.update(cfg_kwargs)
    cfg = EngineConfig(**kwargs)
    bm = BlockManager(kwargs.get("num_gpu_blocks") or 16, cfg.block_size, cfg.enable_prefix_caching)
    return Scheduler(cfg, bm)
```

三个设计点：

* **`model="unused"`**：`EngineConfig` 要求这个字段，但调度器根本不看它。刻意填一个假值，**证明调度器与模型无关**。
* **`block_size=4`**：极小的块让「跨页」发生在第 5 个 token，测试里的数字容易手算。
* **`**cfg_kwargs` 覆盖默认值**：每个测试只写自己要改的那一两个参数，其余保持不变（减少视觉噪音）。

`num_gpu_blocks` 从 `kwargs` 里取——因为它是给 `BlockManager` 的，不是 `EngineConfig` 的字段（`EngineConfig` 里叫 `num_gpu_blocks` 也有，但这里故意用独立的池子大小）。

```python
def make_seq(seq_id: int, n_tokens: int) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=[seq_id * 1000 + i for i in range(n_tokens)],
        sampling_params=SamplingParams(max_tokens=4),
    )
```

`seq_id * 1000` 保证不同序列的 token 不重叠（防止前缀缓存误命中）。

---

## 2. `test_fcfs_and_one_decode_token_per_step`

```python
sched = build()
for i in range(3):
    sched.add(make_seq(i, 4))

out = sched.schedule()
assert [n for _, n in out.scheduled] == [4, 4, 4]

for seq, n in out.scheduled:
    seq.num_computed_tokens += n
    seq.append_token(0)

out = sched.schedule()
assert [n for _, n in out.scheduled] == [1, 1, 1]
```

守两件事：

1. **FCFS**：三条按加入顺序排入（`[4, 4, 4]`）
2. **decode 每步只算 1 个 token**：prefill 完成后 `num_new_tokens` 变成 1

注意 `[n for _, n in out.scheduled]` 这个惯用法：**取出所有 `n`，忽略序列**。断言只关心「这一步排了多少 token」，不关心具体是哪条（顺序由前一行保证）。

> 手动模拟引擎的动作（`num_computed_tokens += n; append_token(0)`）是这类测试的关键：调度器的输入依赖序列状态，必须自己推进状态才能测下一步。

---

## 3. `test_chunked_prefill_respects_the_token_budget`

```python
sched = build(max_num_batched_tokens=8, enable_chunked_prefill=True)
sched.add(make_seq(0, 20))

out = sched.schedule()
assert out.scheduled[0][1] == 8

seq = out.scheduled[0][0]
seq.num_computed_tokens += 8
out = sched.schedule()
assert out.scheduled[0][1] == 8
assert seq.num_computed_tokens == 8
```

20 个 token 的 prompt，预算 8 → 先切 8。

**第二段断言为什么还是 8？** 因为剩下的 12 个 token，预算 8，一块（4 个）对齐后 `end = 8+8 = 16`，`aligned = 16`，`n = 8`。正好是 8。

所以这条测试实际上验证了：**分块在预算内尽量装满，且按块对齐**。

---

## 4. `test_chunked_prefill_off_waits_for_a_big_enough_step`

```python
sched = build(max_num_batched_tokens=8, enable_chunked_prefill=False)
sched.add(make_seq(0, 20))
assert sched.schedule().is_empty       # 装不下 → 整个 step 空转

sched2 = build(max_num_batched_tokens=32, enable_chunked_prefill=False)
sched2.add(make_seq(0, 20))
assert sched2.schedule().scheduled[0][1] == 20
```

关闭分块时：20 > 8 → 返回 0 → `scheduled` 空 → `is_empty` 为真。

预算调到 32 之后，一次算完 20。

**这条测试的价值**：它明确了「关闭分块会浪费 step」这个代价——空转的 step 意味着 GPU 什么都没干。这就是为什么默认 `enable_chunked_prefill=True`。

---

## 5. `test_decode_is_scheduled_before_new_prefill`

```python
sched = build(max_num_batched_tokens=6)
sched.add(make_seq(0, 4))
out = sched.schedule()
running = out.scheduled[0][0]
running.num_computed_tokens += 4
running.append_token(0)

sched.add(make_seq(1, 4))        # 序列 0 正在 decode 时来了新请求
out = sched.schedule()

kinds = [(seq.seq_id, n) for seq, n in out.scheduled]
assert kinds[0] == (0, 1), "running 的 decode 必须排在最前面"
assert kinds[1] == (1, 4)
```

**这条测试守的是全文最重要的设计决策**（decode 优先）。

如果实现顺序反了（先 `_schedule_waiting` 再 `_schedule_running`），结果会变成 `[(1, 4), (0, 1)]`——新请求的 prefill 抢在 decode 前面，所有正在输出的用户都会卡一下。

断言里的中文消息就是给未来的自己看的：**失败时能立刻知道违反了哪条设计**。

---

## 6. `test_budget_is_shared_between_prefill_and_decode`

```python
sched = build(max_num_batched_tokens=8)
sched.add(make_seq(0, 4))
out = sched.schedule()
running = out.scheduled[0][0]
running.num_computed_tokens += 4
running.append_token(0)

sched.add(make_seq(1, 8))
out = sched.schedule()
assert sum(n for _, n in out.scheduled) <= 8
# seq 1 拿到剩余预算（7），向下对齐到页边界（4）
assert [(s.seq_id, n) for s, n in out.scheduled] == [(0, 1), (1, 4)]
```

预算 8：decode 先花 1，剩 7 给 prefill。7 按 `block_size=4` 对齐 → 只能算 4 个。

**这条把「预算共享」和「块对齐」两个机制一起钉住了。**

第一版我写成 `[(0, 1), (1, 7)]`——忘了对齐那一步。测试失败反而帮我确认了对齐逻辑真的生效。

> 断言写成精确的 `== [...]` 而不是 `<=`，是有意的：这个行为（7 → 4）是设计的一部分，不是实现细节。

---

## 7. `test_max_num_seqs_is_enforced`

```python
sched = build(max_num_seqs=2, max_num_batched_tokens=64)
for i in range(4):
    sched.add(make_seq(i, 4))
out = sched.schedule()
assert len(out.scheduled) == 2
assert len(sched.waiting) == 2
```

4 条请求、只允许 2 条并发 → 2 条进 running，2 条留队列。

**注意预算是 64，足够 4 条**（4×4=16），所以这里限制来自 `max_num_seqs` 而不是预算。**控制变量**：一次只测一个约束。

---

## 8. `test_preemption_when_the_cache_is_full`

```python
sched = build(num_gpu_blocks=4, max_num_seqs=8, max_num_batched_tokens=64, block_size=4)
for i in range(3):
    sched.add(make_seq(i, 8))      # 每条 2 块，池子只有 4 块

out = sched.schedule()
assert len(out.scheduled) == 2     # 只能进 2 条
assert len(sched.waiting) == 1

for seq, n in out.scheduled:
    seq.num_computed_tokens += n
    seq.append_token(0)
sched.add(make_seq(9, 8))

out = sched.schedule()
assert out.preempted, "池子满了，必须有东西让位"
victim = out.preempted[0]
assert victim.num_computed_tokens == 0    # 抢占 = 丢 KV + 重算
assert victim.num_preemptions == 1
assert victim.status is SequenceStatus.WAITING
```

这是最复杂的一条。要点：

* 池子 4 块、每条要 2 块 → 最多 2 条
* 第 3 步时加一条新请求，但池子已满 → 触发抢占
* 断言 `num_computed_tokens == 0`：**这是「recompute 策略」的签名**。如果实现改成 swap，这个值应该保持（KV 在 CPU 上），测试就必须改。

**为什么这条测试能跑通而不死循环**：抢占后 `victim` 被 `appendleft` 回 waiting，当前序列重试成功，step 推进。如果抢占逻辑写错（比如抢了又不重试），就会一直空转。

---

## 9. `test_finished_sequences_release_their_pages`

```python
sched = build(num_gpu_blocks=4, block_size=4, max_num_batched_tokens=64)
sched.add(make_seq(0, 8))
out = sched.schedule()
seq = out.scheduled[0][0]
seq.num_computed_tokens += 8
seq.mark_finished(SequenceStatus.FINISHED_LENGTH_CAPPED)

assert sched.block_manager.num_free_blocks == 2     # 还没回收
sched.free_finished()
assert sched.block_manager.num_free_blocks == 4     # 回收了
```

守的是「**显存回收发生在显式调用时**，不是标记完成时」。

这个分离很重要：`mark_finished` 只是改状态，真正的回收在 `free_finished()`——因为回收前还要先登记前缀缓存（顺序在 [scheduler.md](scheduler.md) 里讲）。

---

## 10. `test_prefix_cache_skips_recomputing_a_shared_prompt`

```python
sched = build(enable_prefix_caching=True, max_num_batched_tokens=64)
first = make_seq(0, 8)
sched.add(first)
out = sched.schedule()
seq = out.scheduled[0][0]
seq.num_computed_tokens += 8
seq.mark_finished(SequenceStatus.FINISHED_LENGTH_CAPPED)
sched.free_finished()

second = Sequence(seq_id=1, prompt_token_ids=list(first.prompt_token_ids),
                  sampling_params=SamplingParams(max_tokens=4))
sched.add(second)
hits = sched.block_manager.match_prefix(second.data)
assert hits, "相同 prompt 应该命中前缀缓存"
out = sched.schedule()
assert out.scheduled[0][0].block_table[: len(hits)] == hits
```

这是唯一一条同时涉及调度器和缓存交互的测试。

它验证的是**修 bug 之后**的行为：`schedule()` 里会把 `second.num_computed_tokens` 推到缓存长度（`cached_prefix_tokens`），所以块表前几项直接复用。

> 严格说这条断言略弱（只检查块表对不对齐，没检查「少算了多少 token」）。更严的版本是：
> ```python
> assert out.scheduled[0][1] == 8 - len(hits) * sched.cfg.block_size
> ```
> 但那样会把「至少留 1 个 token」的边界也耦合进来。当前版本更稳。

`list(first.prompt_token_ids)` 复制一份——如果直接共享 list，后续 `append_token` 会互相影响。

---

## 覆盖矩阵

| 规则 | 测试 |
|---|---|
| FCFS 顺序 | 2 |
| decode 每步 1 token | 2 |
| 分块 prefill 按预算切 | 3、6 |
| 分块按块对齐 | 3、6 |
| 关闭分块时拒绝 | 4 |
| **decode 优先于 prefill** | 5 |
| 预算共享 | 6 |
| 并发上限 | 7 |
| **抢占 = 丢 KV 重算** | 8 |
| 显式回收 | 9 |
| 前缀缓存与调度交互 | 10 |

**没覆盖**：`long_prefill_token_threshold`、`static_batching`、`AllocStatus.NEVER`、饥饿场景。

---

## 用到的 pytest / Python 特性

| 特性 | 说明 |
|---|---|
| `**cfg_kwargs` + `dict.update` | 覆盖默认参数 |
| `assert ..., "消息"` | 失败时打印说明 |
| `is` 比较枚举 | `status is SequenceStatus.WAITING` |
| 列表推导取子集 | `[n for _, n in out.scheduled]` |
| `list(...)` 复制 | 防止共享可变对象 |

---

## 设计取舍

**为什么完全不加载模型？**

因为调度决策**只依赖 token 数和块数**，与语义无关。这个分层让 9 条规则在 10 毫秒内验证完——而如果每条都跑真模型，要 2 分钟。

**为什么用 `block_size=4` 而不是默认 16？**

让「跨页」发生在很小的数字上，断言里可以直接写 `3`、`4`、`[8, 4]` 这样的值，不用心算大数。

**为什么断言写成精确值而不是范围？**

调度器是**确定性**的（没有随机、没有并发）。这种情况下精确断言能抓出更多回归。如果哪天实现变成随机选择 victim，这些测试会提醒你「行为变了」。

---

## 常见误解

* 「`schedule()` 会推进序列状态」——不会。它只读状态、产出计划；状态由 `_postprocess` 推进。测试里必须手动模拟。
* 「抢占会丢已生成的 token」——`data` 保留，只清 `num_computed_tokens`。
* 「`is_empty` 只在实际没请求时出现」——长 prompt + 关分块时也会出现（空转 step）。
* 「`max_num_seqs` 是批内 token 上限」——不是，是序列条数上限。

---

## 动手练习

1. 把 `_schedule_running` 调到 `_schedule_waiting` 之后，跑测试 5，确认报错信息是那句中文。
2. 把 `_chunk_size` 的块对齐去掉（`return n`），跑测试 6，看它如何从 `4` 变成 `7`。
3. 把抢占从 `running.pop()` 改成 `running.pop(0)`，跑测试 8，观察 `victim.seq_id` 变化。
4. 新增测试：设置 `long_prefill_token_threshold=32`，验证长 prompt 在 running 非空时被推迟。
