# `minivllm/scheduler.py` 详解 ★核心

## 这个文件干什么

每个 step 回答两个问题：

1. **这一批跑谁、每个跑几个 token？**
2. **显存不够时怎么办？**

它是「连续批处理（continuous batching）」这个概念的全部实现，一共不到 200 行。

---

## 1. 核心观念：调度单位是「token」，不是「请求」

这是理解全文的前提。

| | 静态批处理 | 连续批处理 |
|---|---|---|
| 调度单位 | 一个 batch | 一个 **step** |
| 谁在跑 | 批次固定，跑完才换 | 每 step 重新决定 |
| 短请求 | 陪跑到最长那条结束 | 一完成立刻离场 |
| 新请求 | 等整批结束 | 下一步就能进来 |
| 显存 | 整批占着 | 完成的立刻还 |

代码里对应 `SchedulerOutputs.scheduled = [(seq, n), ...]`：**每条序列配一个数字 n = 本 step 要算几个 token**。

* `n = 1` → decode
* `n > 1` → prefill（或 chunked prefill 的一块）

---

## 2. `SchedulerOutputs`：一次调度的产出

```python
@dataclass
class SchedulerOutputs:
    scheduled: list[tuple[Sequence, int]] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    cow_ops: list[tuple[int, int]] = field(default_factory=list)
    num_batched_tokens: int = 0
```

只有 4 个字段，但信息很密：

* `scheduled`：**要跑什么**（顺序有意义，见第 4 节）
* `preempted`：这次牺牲了谁（只用于统计/调试）
* `cow_ops`：调度过程中产生的 COW 决策，交给 `ModelRunner.apply_cow` 执行
* `num_batched_tokens`：本 step 总共处理多少 token

三个 `@property` 是纯派生量（`num_prefill_seqs`、`num_decode_seqs`、`is_empty`），**刻意不存字段**——存了就可能和 `scheduled` 不一致。

### 为什么 `cow_ops` 要「带出来」而不是就地执行

因为 `BlockManager` 是纯 Python 的，而复制页要操作显存。**调度阶段只产出决策，执行阶段才碰张量**。这个分层让 `tests/test_scheduler.py` 完全不需要 GPU。

---

## 3. 两个队列

```python
self.waiting: deque[Sequence] = deque()   # 排队等准入
self.running: list[Sequence] = []          # 已在跑
```

* `waiting` 用 `deque`：需要从两端 O(1) 操作（`append` 队尾；抢占时 `appendleft` 插队首）。
* `running` 用 `list`：需要按下标遍历、`pop()` 拿最后一个。

### 为什么抢占要 `appendleft`（插回队首）

被抢占的序列**已经算过一部分 prefill 了**（那些工作被丢弃了）。如果插到队尾，它要等所有新请求跑完才能重来——那前面的工作等于白扔，用户感知到的延迟会非常糟。

插回队首 = 「尽快让它重试」。这是**公平性**的体现，也是为什么 `deque` 而不是 `list`。

---

## 4. `schedule()`：两步走

```python
def schedule(self) -> SchedulerOutputs:
    out = SchedulerOutputs()
    budget = self.cfg.max_num_batched_tokens

    self._schedule_running(out, budget)
    budget -= out.num_batched_tokens
    self._schedule_waiting(out, budget)

    out.num_batched_tokens = sum(n for _, n in out.scheduled)
    self.peak_running = max(self.peak_running, len(self.running))
    return out
```

**先跑 running，再收 waiting，预算共享。**

### 为什么 decode 优先（这是全文最重要的设计决策）

假设一个 step 的预算是 1024 token，队列里有一个 1000 token 的 prefill 和 8 个正在 decode 的请求。

* **prefill 优先**：8 个 decode 全部推迟一步 → 8 个用户同时感到卡顿；那个 prefill 的用户也要等 1000 token 算完才有第一个字。
* **decode 优先**：8 个 decode 先花 8 token（几乎瞬间），剩下的 1016 token 给 prefill → 没有任何人感到卡顿。

所以顺序是：**先把所有 running 的下一步（各 1 个 token）排进去，剩下的预算才给新请求。**

vLLM v1 把这叫 "decode-first"，之前 v0 是 "prefill-first"，两者在 ITL（inter-token latency）上的差距很可观。

### 预算为什么是 token 而不是请求数

因为一次 prefill 可能占 1000 个 token 的算力，而一次 decode 只占 1 个。用「请求数」限额，会让「一个超长 prompt + 7 个 decode」和「8 个 decode」看起来一样重——但前者慢 100 倍。

**用 token 计费，才能让每 step 的实际工作量可控。**

---

## 5. `_chunk_size`：切多少 token 给它

```python
def _chunk_size(self, seq: Sequence, budget: int) -> int:
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
```

四种情况：

| 情况 | 返回 | 含义 |
|---|---|---|
| `remaining ≤ budget` | `remaining` | 一次算完（decode 或短 prompt） |
| 算不完 + 关掉分块 | `0` | **拒绝**，等一个更大的 step |
| 算不完 + 开着分块 | 对齐后的大小 | 切一块 |
| 对齐后为 0 | `0` | 一块都切不出来 |

### 为什么分块要按 block 边界对齐

假设 `bs = 16`，已经算了 32 个 token，预算只剩 10。**不对齐**就切 10 个 → 算完 42 个 token，落在第 3 块（逻辑块 2）的中间。

后果有两个：

1. **前缀缓存无法登记这一块**。`register_computed_blocks` 只登记完整的块（`num_computed_tokens // bs`）。第 3 块只填了 10/16，它的内容还会变，不能哈希。
2. 下一次写第 3 块时要重新判断 COW、重新算 slot，边界情况变多。

对齐后：`end = 42`，`aligned = 42 - 42%16 = 32`，`aligned - 32 = 0` → 这一 step 先不算它，把预算留给别人。**下一 step 有更多预算时再切整块。**

> `max(..., 0)` 是防御：如果 `num_computed_tokens` 本身就是块边界（`32`），而 `n` 很小（比如 1，这是 decode），`end=33`，`aligned=32`，差 0 —— 但 decode 不会被切（`n == remaining` 先返回了）。所以 0 的情况实际只出现在「预算不足一整块」时。

### 为什么「关掉分块就返回 0」

返回 0 意味着「这个 step 不排它」，但序列仍留在 `waiting`/`running` 里。等某个 step 预算够大（其它序列都结束了），它就能一次算完。

这是 vLLM 早期的策略：**长 prompt 不允许被打断**。缺点是它可能饿很久。所以工程上常用 `long_prefill_token_threshold` 来「让它等到能被独占时再跑」。

---

## 6. `_schedule_running`：抢占逻辑

```python
def _schedule_running(self, out, budget):
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
            victim = running.pop()
            self.block_manager.free(victim)
            victim.num_computed_tokens = 0
            victim.num_preemptions += 1
            victim.status = SequenceStatus.WAITING
            self.waiting.appendleft(victim)
            out.preempted.append(victim)
            self.num_preemptions += 1
            continue

        out.cow_ops.extend(cow_ops)
        out.scheduled.append((seq, n))
        out.num_batched_tokens += n
        budget -= n
        i += 1
```

### 抢占的五步

```python
victim = running.pop()                 # 1. 挑最后进来的那条
self.block_manager.free(victim)        # 2. 归还它所有页（ref_count 减一）
victim.num_computed_tokens = 0         # 3. 忘掉"算过多少"——KV 已经没了
victim.num_preemptions += 1            # 4. 记账
victim.status = WAITING
self.waiting.appendleft(victim)        # 5. 插队首，尽快重试
```

**关键点 `num_computed_tokens = 0`**：这一行让序列「假装从没算过」。下次它被调度时，`num_new_tokens = len(data)`，会从头 prefill 一遍。**这就是 "recompute" 策略**。

### 为什么用 `pop()` 而不是 `remove(seq)`

`pop()` 拿**最后一个**（最新进来的）。两个好处：

1. O(1)。`remove` 是 O(n)。
2. 策略上「牺牲最新的」比较合理——老请求已经投入了更多计算，丢掉更可惜（类似 LRU/MRU 的权衡，这里相当于牺牲 MRU）。

### 为什么抢占后要 `continue` 重试

因为腾出了一批页之后，**当前这条序列可能就够了**。所以循环回到 `while` 再试一次。如果不重试，这一 step 就白浪费了。

> 代码里原本还有 `if victim is seq: continue` 这样一行（和后面的 `continue` 等价，属于冗余分支）。逻辑上：如果被抢的正是当前序列，`running` 变短后 `i < len(running)` 自然为假，循环退出，行为一致。

### 为什么选 recompute 而不是 swap 到 CPU

| 方案 | 代价 | 何时更优 |
|---|---|---|
| **recompute**（本项目） | 重跑一次 prefill | prompt 短、模型小、PCIe 慢 |
| swap | 一次 D2H（写回 CPU 内存）+ 一次 H2D | prompt 极长、CPU 内存充足、有 NVLink |

0.5B 模型、几百 token 的 prompt，重算只要几十毫秒；而一次 100 MB 的 D2H/H2D 在 PCIe 4.0 x16 上就要 ~20 ms 往返，还要额外占用内存带宽。**小模型场景 recompute 明显更划算**——这也是 vLLM 默认的策略。

---

## 7. `_schedule_waiting`：准入新请求

```python
def _schedule_waiting(self, out, budget):
    if self.cfg.static_batching and self.running:
        return  # 静态批处理：批次没排空前不收新请求
    first_admitted = len(out.scheduled)

    while self.waiting and budget > 0 and len(self.running) < self.cfg.max_num_seqs:
        seq = self.waiting[0]

        if len(seq.data) > self.cfg.max_model_len:      # ① 太长，永远装不下
            self.waiting.popleft()
            seq.mark_finished(SequenceStatus.FINISHED_LENGTH_CAPPED)
            continue

        if (self.cfg.long_prefill_token_threshold        # ② 长 prompt 等独占
                and seq.num_new_tokens > self.cfg.long_prefill_token_threshold
                and len(self.running) > 0):
            break

        if not seq.block_table and self.cfg.enable_prefix_caching:   # ③ 前缀缓存
            cached = self.block_manager.cached_prefix_tokens(seq.data)
            if cached:
                seq.num_computed_tokens = min(cached, len(seq.data) - 1)

        n = self._chunk_size(seq, budget)
        if n <= 0:
            break

        if self.block_manager.can_allocate(seq, seq.num_computed_tokens + n) != AllocStatus.OK:
            break                                        # ④ 显存不够，等

        self.waiting.popleft()
        if not seq.block_table:
            self.block_manager.allocate(seq, seq.num_computed_tokens + n)
        out.cow_ops.extend(self.block_manager.ensure_capacity(seq, seq.num_computed_tokens + n))
        seq.status = SequenceStatus.RUNNING
        self.running.append(seq)

        out.scheduled.append((seq, n))
        out.num_batched_tokens += n
        budget -= n
```

### 三个 `break` 和两个 `continue` 的区别

**这是本函数唯一需要仔细分辨的地方**：

| 关键字 | 用在 | 含义 |
|---|---|---|
| `continue` | ① 超长请求 | 这条**废了**，从队列移除并标记结束，继续看下一条 |
| `break` | ②③④ | 队列是 **FIFO**，队首放不进去，后面的也不该越过去 → 整个准入阶段结束 |

如果把 ②③④ 写成 `continue`，就会**跳过队首去调度后面的请求**，FIFO 语义被破坏（长请求会被无限期饿死）。

### ③ 前缀缓存：光共享页不够，还要推进计算进度

```python
cached = self.block_manager.cached_prefix_tokens(seq.data)
if cached:
    seq.num_computed_tokens = min(cached, len(seq.data) - 1)
```

这是**修 bug 修出来的一段**。最初只有 `BlockManager.match_prefix` 共享了物理页，但 `num_computed_tokens` 仍然是 0，于是：

```
页表已经指向那些 KV → 数据是对的
但调度器以为"什么都没算" → 让 prefill 从头算一遍
→ 算了也白算（写进同样的页），TTFT 一点没降
```

修完之后，warm TTFT 从 0.779s → 0.077s（**10.2 倍**）。

### `min(cached, len(seq.data) - 1)` 为什么要减 1

如果整条 prompt 都被缓存命中了（`cached == len(data)`），`num_computed_tokens` 就等于 `len(data)`，`num_new_tokens = 0`，`_chunk_size` 返回 0，这条序列**永远排不进去**，死锁。

**必须至少留 1 个 token 让模型跑一次**，因为我们需要它的 logits 才能出第一个生成 token。这就是 `- 1` 的全部理由。

> 这是一个典型的「缓存命中的边界比想象中麻烦」：命中 100% 反而需要特殊处理。

### `can_allocate` 传的是 `num_computed_tokens + n`

不是 `len(seq.data)`。因为 chunked prefill 只需要为**这一块**分配页，不需要为整个 prompt 分配。这是「按需分配」的直接体现。

### 静态批处理模式

```python
if self.cfg.static_batching and self.running:
    return
...
if self.cfg.static_batching:
    batch = [seq for seq, _ in out.scheduled[first_admitted:]]
    if batch:
        longest = max(s.sampling_params.max_tokens for s in batch)
        for seq in batch:
            seq.sampling_params = replace(seq.sampling_params, max_tokens=longest, ignore_eos=True)
```

两处：

1. 开头 return：批次没排空前不接新请求。
2. 结尾把整批的 `max_tokens` 拉齐到最长那条——**这就是静态批处理「陪跑」的精确形式**：短请求被迫生成长度 = 批内最长。

`replace()` 来自 `dataclasses`，生成一个新对象（不改原对象），这样「用户传进来的 params」不会被悄悄修改。这个模式在生产代码里很常见。

---

## 8. `free_finished`：回收顺序很重要

```python
def free_finished(self) -> list[Sequence]:
    finished = [seq for seq in self.running if seq.is_finished]
    if finished:
        self.running = [seq for seq in self.running if not seq.is_finished]
        for seq in finished:
            # 先登记再释放：free() 会清空 block_table
            self.block_manager.register_computed_blocks(seq)
            self.block_manager.free(seq)
    return finished
```

**顺序反了前缀缓存就永远失效**（而且不报错）。因为 `free()` 里最后会 `seq.block_table.clear()`，之后再登记就没了页表可读。

> 这就是我在 [block_manager.md](block_manager.md) 里说的「我踩过的坑」之一。

---

## 9. `reset_cache`：只允许在空闲时调用

```python
def reset_cache(self) -> None:
    if self.running:
        raise RuntimeError("cannot reset the prefix cache while sequences are running")
    self.block_manager.hash_to_block.clear()
    for blk in self.block_manager.blocks:
        blk.block_hash = None
        blk.tokens = ()
```

benchmark 在两次测量之间需要清空前缀缓存（否则第二次测会因为缓存而偏快）。`running` 非空时清空会破坏引用语义，所以直接拒绝。

---

## 10. 一次真实调度长什么样

配置：`max_num_batched_tokens = 16`，`block_size = 4`，`max_num_seqs = 8`。

假设 `running = [A(已算4), B(已算4)]`，`waiting = [C(20 token), D(4 token)]`。

**Step 1**
```
budget = 16
_schedule_running:
  A: num_new = 1 (decode) → 排入，budget = 15
  B: num_new = 1         → 排入，budget = 14
budget = 14
_schedule_waiting:
  C: num_new = 20 > 14，分块开启 → end = 0+14 = 14, aligned = 12, n = 12
     can_allocate(12) → 需要 3 块 → 排入，budget = 2
  D: num_new = 4 > 2 → end = 2, aligned = 0, n = 0 → break
结果: [(A,1), (B,1), (C,12)]，共 14 token
```

**Step 2**
```
budget = 16
_schedule_running:
  A: 1 → 15
  B: 1 → 14
  C: num_new = 8, 8 ≤ 14 → n = 8 → 排入，budget = 6
budget = 6
_schedule_waiting:
  D: 4 ≤ 6 → 排入，budget = 2
结果: [(A,1),(B,1),(C,8),(D,4)] —— 注意 C 在这一步把自己的 prefill 做完了
```

**Step 3**：A、B、C、D 全部 decode，各 1 token。

可以看到 C 的 20 token prompt 被拆成 12 + 8，**中间还夹着两次 A/B 的 decode**——这就是「chunked prefill 避免卡住 decode」的直观样子。

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `deque` | 双端 O(1)，`appendleft` 用于插队 |
| `@dataclass` + `field(default_factory=list)` | 每次新建独立的空 list |
| `dataclasses.replace` | 复制并修改不可变 dataclass |
| `@property` | 派生量，不存字段 |
| `try/except` 自定义异常 | 用异常表达「资源耗尽」 |
| `list.pop()` | O(1) 取尾 |
| 列表推导式过滤 | `[s for s in running if not s.is_finished]` |

---

## 设计取舍

**为什么抢占只丢 KV、不丢已生成的 token？**

`seq.data` 保留（用户已经"看到"的字不能消失），只把 `num_computed_tokens` 归零。重算时会用同样的 `data` 重新推一遍，理论上结果完全一致（测试 `test_preemption_still_produces_correct_tokens` 就在验证这一点）。

**为什么 `waiting` 是 FIFO 而不是按长度排序（SJF）？**

SJF 能降平均延迟，但会让长请求永久饥饿。生产引擎通常 FIFO + 优先级（如 vLLM 的 `priority` 字段）。本项目选最简单的 FIFO。

**为什么调度器完全不知道张量？**

因为调度是「纯逻辑」，能 100% 在 CPU 上测。整个 `tests/test_scheduler.py` 不需要 torch、不需要模型、毫秒级跑完——这是把复杂度分层带来的直接收益。

---

## 常见误解

* 「`max_num_batched_tokens` 是批大小上限」——不，是**每 step 的 token 总数**上限。批的大小上限是 `max_num_seqs`。
* 「decode 优先是因为 decode 更快」——不，是因为 decode 的**延迟**能被用户直接感知（打字卡顿），而 prefill 的结果没人看到。
* 「分块 prefill 是随便切的」——必须按 block 对齐，否则前缀缓存登记不了。
* 「抢占是丢数据」——丢的是可重算的中间结果（KV），不是输出。
* 「前缀缓存命中越多越好」——100% 命中要特殊处理（`-1`），否则死锁。

---

## 动手练习

1. 把 `_schedule_running` 移到 `_schedule_waiting` **之后**，跑 `tests/test_scheduler.py::test_decode_is_scheduled_before_new_prefill`，看它怎么挂。
2. 把 `_chunk_size` 里的 block 对齐去掉（直接 `return n`），跑 `tests/test_engine_e2e.py::test_chunked_prefill_gives_the_same_tokens`，观察结果是否仍然一致（提示：数学上一致，但前缀缓存命中会掉）。
3. 把 `waiting.appendleft(victim)` 改成 `append`，在 `bench_static_vs_continuous` 里加大并发并观察 p99 延迟。
4. 把抢占策略从 `running.pop()` 改成 `running.pop(0)`（牺牲最老的），跑 `test_preemption_still_produces_correct_tokens`，确认仍然正确但 `num_preemptions` 变化。
