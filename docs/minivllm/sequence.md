# `minivllm/sequence.py` 详解

## 这个文件干什么

定义**一条请求是什么**。整个引擎里所有模块都在读写这个对象。

一句话抽象：**一条序列就是一个不断变长的 token 流，外加一张页表和三个计数。**

```python
len(seq.data)                            # 一共有多少 token（prompt + 已生成）
seq.num_computed_tokens                  # 其中多少个的 KV 已经在缓存里
len(seq.data) - seq.num_computed_tokens  # 这一步需要算多少个
```

第三个数是整个调度器的核心：**它是 0 就说明没活干（已结束或已完成 prefill），是 1 就是一次 decode，大于 1 就是一次 prefill / chunked prefill。**

---

## 逐块讲解

### `SequenceStatus`：为什么用 Enum + auto

```python
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
```

* `Enum` 保证值唯一、可比较、打印出来是 `SequenceStatus.RUNNING` 而不是数字，调试时极友好。
* `auto()` 自动编号，不关心具体值。
* **关键设计**：把「是否结束」做成枚举上的属性，而不是在别处写 `if status in (A, B, C)`。这样**新增一个结束状态时，只需改这一处**。

对比一下如果用字符串 `"finished_stopped"`：拼错不报错、IDE 不会提示、`if status.startswith("FINISHED")` 这种脆弱写法会蔓延——这就是为什么引擎里状态一律用枚举。

### `Sequence` 的字段

```python
@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    prompt: str = ""
    data: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
```

前三个**没有默认值**，所以必须传。后三个有默认值。

#### `data` 为什么用 `field(default_factory=list)`

```python
data: list[int] = field(default_factory=list)   # ✅
data: list[int] = []                            # ❌
```

写成 `[]` 是 Python 的经典陷阱：**默认值在类定义时只求值一次，所有实例共享同一个 list**。给一个实例 append，别的实例也会变。

`field(default_factory=list)` 让每个实例调用一次 `list()`，得到各自的空列表。
（顺带一提：Python 3.11+ 对 `list`/`dict`/`set` 的默认值会直接报 `ValueError`，但 3.10 不会，所以这个坑必须自己防。）

#### `arrival_time` 的写法很巧

```python
arrival_time: float = field(default_factory=time.perf_counter)
```

把函数本身作为 factory —— 每个实例创建时调用一次 `time.perf_counter()`，得到「此刻」的单调时钟。

为什么不用 `time.time()`？因为 `perf_counter` 是**单调递增**的，不受系统时间校准/NTP 跳变影响。测耗时必须用它。

### `__post_init__`：把 prompt 复制进 `data`

```python
def __post_init__(self) -> None:
    if not self.data:
        self.data = list(self.prompt_token_ids)
```

`data` 是「prompt + 已生成的 token」的全部历史。初始化时它就是 prompt。

`list(...)` 复制一份，不能直接 `self.data = self.prompt_token_ids`——否则两者共享同一个 list，后面 `append_token` 会把生成的 token 也写进 `prompt_token_ids`，导致 `num_output_tokens` 永远算出 0。

> 这是一个**故意允许覆盖**的设计：测试里可以传 `data=...` 预置一条已经生成了一部分的序列，`if not self.data` 保证不覆盖调用者的意图。

### 页表相关字段

```python
block_table: list[int] = field(default_factory=list)
"""逻辑块索引 -> 物理块 id。这个 list 本身就是页表。"""
block_hashes: list[bytes | None] = field(default_factory=list)
"""与 block_table 平行；块填满后记录其内容哈希。"""
num_computed_tokens: int = 0
"""其 KV 已确定在缓存里的前缀长度。"""
```

* `block_table` 的下标是**逻辑**块号（第 0 块、第 1 块……），值是**物理**块号（池子里的第几块）。`block_table[2] = 9` 意思是「我的第 3 块数据存在池子的第 9 块里」。
* `block_hashes` 与 `block_table` 一一对应，记录每块内容的哈希。**它目前只用于调试观察**，真正的前缀匹配逻辑走的是 `BlockManager.hash_to_block` 和 `PhysicalBlock.tokens`。看到这里别被迷惑：哈希的**权威存储**在 `PhysicalBlock` 上，`Sequence` 上这份是给「想知道这条序列每块的指纹」的场景留的。
* `num_computed_tokens` 是最容易理解错的一个，见下一节。

### `num_computed_tokens` 到底是什么

**定义：这条序列的前 `num_computed_tokens` 个 token，其 KV 已经在显存里且是正确的。**

它不等于 `len(data)`，原因有三种：

| 场景 | `len(data)` | `num_computed_tokens` | 下一步算几个 |
|---|---|---|---|
| 刚进来，还没 prefill | 100 | 0 | 100（一次 prefill） |
| 分块 prefill 算了两块 | 100 | 32 | 32（下一块） |
| prefill 完成 | 100 | 100 | 1（decode） |
| decode 后追加 1 个 token | 101 | 100 | 1 |
| 前缀缓存命中 64 token | 100 | 64 | 36 |
| 被抢占后重置 | 100 | 0 | 100（重算） |

所以：

```python
@property
def num_new_tokens(self) -> int:
    return len(self.data) - self.num_computed_tokens
```

**这个属性的值直接决定这一步要干什么**，是理解 [scheduler.md](scheduler.md) 和 [model_runner.md](model_runner.md) 的钥匙。

### 尺寸相关属性

```python
@property
def num_prompt_tokens(self) -> int:
    return len(self.prompt_token_ids)

@property
def num_output_tokens(self) -> int:
    return len(self.data) - self.num_prompt_tokens

@property
def num_blocks_required(self, block_size: int) -> int: ...
```

`num_output_tokens` 用「总长 - prompt 长」算，而不是单独维护一个计数器——**少一个状态就少一个可能不一致的地方**。

### 状态变更：`append_token` / `mark_finished`

```python
def append_token(self, token_id: int) -> None:
    self.data.append(token_id)

def mark_finished(self, status: SequenceStatus) -> None:
    self.status = status
    self.finish_time = time.perf_counter()
```

`append_token` **不**碰 `num_computed_tokens`。为什么？因为刚追加的这个 token 的 KV 还没算——它的 KV 是在**下一次**前向（作为 query）时写进去的。这个"延迟一步"的语义必须由调用方维护：

```python
# llm_engine.py :: _postprocess
seq.num_computed_tokens += n       # 这一步算的 n 个 token 的 KV 现在有了
...
seq.append_token(token_id)         # 新 token 还没有 KV
```

顺序不能反，也不能漏。

### 指标属性

```python
@property
def ttft(self) -> float | None:
    if self.first_token_time is None:
        return None
    return self.first_token_time - self.arrival_time

@property
def latency(self) -> float | None:
    if self.finish_time is None:
        return None
    return self.finish_time - self.arrival_time
```

两个都返回 `None` 而不是 0。**「还没有值」和「值是 0」是两件事**，混用会让统计结果假性偏低。

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `@dataclass` | 自动 `__init__` |
| `field(default_factory=...)` | 每个实例独立默认值（防可变默认值陷阱） |
| `__post_init__` | 初始化后钩子 |
| `Enum` + `auto()` | 具名常量 |
| `@property` | 派生值，不占存储 |
| `time.perf_counter` | 单调时钟，测耗时专用 |
| `str.removeprefix` | Python 3.9+，`"FINISHED_STOPPED"` → `"stopped"` |

---

## 设计取舍

**为什么 `data` 用 Python `list[int]` 而不是 `torch.Tensor`？**

因为 token 是**逐个追加**的。`list.append` 是 O(1)；`torch.cat` 每次都复制整个张量，做到第 n 个 token 时总开销是 O(n²)。真正的张量化发生在 `_build_inputs`：一次性把本 step 需要的切片转成张量。

**为什么不给 `Sequence` 加一个 `is_prefill` 布尔字段？**

因为 `num_new_tokens > 1` 已经完整表达了这件事，而且不会出现「`is_prefill=True` 但 `num_new_tokens=1`」这种自相矛盾的状态。**派生状态永远优于冗余状态。**

**为什么 `num_preemptions` 放在 Sequence 上而不是全局计数？**

因为要能回答「**这条**请求被抢占了几次」——它能解释某条请求为什么特别慢。全局计数在 `Scheduler` 里也有一份，两者用途不同。

---

## 常见误解

* 「`num_computed_tokens` 等于已生成的 token 数」——不，它是**已算过 KV 的前缀长度**，prefill 阶段它可以小于 `len(data)`。
* 「`block_table` 里存的是偏移量」——存的是**物理块 id**。偏移量在 `slot_mapping` 里算：`block_id * block_size + (t % block_size)`。
* 「`prompt_token_ids` 和 `data` 是一回事」——prompt 是只读的原始输入，`data` 会一直增长。

---

## 动手练习

1. 在 `Sequence` 里加一个不存在的 `max_new_tokens`（故意拼错）字段，然后 `Sequence(seq_id=0, prompt_token_ids=[1], sampling_params=SamplingParams())`，看 dataclass 怎么报错。
2. 把 `data` 的默认值改成 `[]`（而不是 `field(default_factory=list)`），运行下面这段，观察两个实例变成同一个 list：
   ```python
   a = Sequence(seq_id=0, prompt_token_ids=[1], sampling_params=SamplingParams(), data=[1])
   b = Sequence(seq_id=1, prompt_token_ids=[2], sampling_params=SamplingParams())
   a.append_token(9); print(b.data)
   ```
   （提示：`a` 传了非空 `data` 所以不走 `__post_init__`；重点看 `b` 是否受影响，以及为什么 `__post_init__` 里的 `if not self.data` 掩盖了这个 bug。把它改成无条件复制就能暴露。）
3. 造一条 `data` 长 100、`num_computed_tokens` 为 64 的序列，手算 `num_new_tokens` 和 `num_blocks_required(16)`。
