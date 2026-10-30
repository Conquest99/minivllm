# `minivllm/config.py` 详解

## 这个文件干什么

所有可调参数的**唯一真相来源**。整个引擎没有任何魔法数字藏在函数体里，全部集中在这里，并附带约束检查。

文件很小（73 行），但它是理解整个系统调优空间的入口。

---

## 逐块讲解

### 头部：为什么用 `@dataclass`

```python
from dataclasses import dataclass

@dataclass
class EngineConfig:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "float16"
    ...
```

`@dataclass` 会自动生成 `__init__`、`__repr__`、`__eq__`。等价于手写：

```python
class EngineConfig:
    def __init__(self, model="Qwen/...", dtype="float16", ...):
        self.model = model
        self.dtype = dtype
        ...
```

好处：加一个参数只写一行。

注意所有参数**都有默认值**，所以 `EngineConfig()` 直接可用，也可以只覆盖想改的：

```python
EngineConfig(block_size=32, enable_prefix_caching=False)
```

### 第一组：模型

```python
model: str = "Qwen/Qwen2.5-0.5B-Instruct"
dtype: str = "float16"
device: str = "cuda"
seed: int = 0
use_chat_template: bool = True
```

| 字段 | 含义 | 注意 |
|---|---|---|
| `model` | 可以是 HF hub id，也可以是本地目录 | 本项目默认填本地路径 `models/...` |
| `dtype` | 字符串，不是 `torch.float16` 对象 | 因为放进 dataclass 里要可序列化；`ModelRunner` 用 `getattr(torch, cfg.dtype)` 转 |
| `device` | `"cuda"` / `"cpu"` | 测试时设 `cpu` |
| `seed` | 采样随机种子 | 只有 `temperature > 0` 时才起作用 |
| `use_chat_template` | 是否套 chat template | 不套的话模型不知道「谁在说话」，输出会明显变差 |

> `dtype` 存字符串而不是 `torch.dtype` 对象，是个实用选择：`torch.dtype` 不能 JSON 序列化，而配置经常要 dump 进 benchmark 结果里。

### 第二组：分页

```python
block_size: int = 16
num_gpu_blocks: int | None = None
gpu_memory_utilization: float = 0.80
```

* **`block_size = 16`**：每页装 16 个 token。这是 vLLM 的默认值，也是本项目的默认。
  * 小 → 内部碎片少，但页表长、gather 索引大、每页的 kernel 启动摊薄效果差。
  * 大 → 反之。
  * 16 是经验上比较均衡的点。测试里验证过 `1 / 8 / 32` 三种都给出**数值相同**的结果（只有性能差别）。
* **`num_gpu_blocks = None`**：`None` 表示「启动时按可用显存自动算」。压测想要可复现就显式指定，比如 `num_gpu_blocks=1024`。
* **`gpu_memory_utilization = 0.80`**：注意它乘的是**当前空闲显存**，不是总量。所以桌面上开着 QQ/浏览器时会自动缩水——这既是安全设计，也会让两次测量的池子大小不同。这一点在 [model_runner.md](model_runner.md) 的 `_profile_num_blocks` 里详细讲。

### 第三组：调度

```python
max_num_seqs: int = 8
max_num_batched_tokens: int = 1024
max_model_len: int = 2048
```

* **`max_num_seqs`**：同时在跑的请求数上限（序列数上限）。它决定 batch 的**宽度**。
* **`max_num_batched_tokens`**：每个 step 最多处理多少个 **token**（不是多少条请求）。这是理解连续批处理的关键：
  * 1 条 1000 token 的 prefill → 花掉 1000 预算
  * 8 条各 1 token 的 decode → 只花 8 预算
  * **同一个预算池，按 token 计费**，所以 prefill 和 decode 才能混在一个 batch 里而不撑爆显存。
* **`max_model_len`**：prompt + 生成长度的硬上限。它同时决定了 `max_blocks_per_seq`（页表最大宽度），进而限定 `num_gpu_blocks` 的上限。

### 第四组：chunked prefill

```python
enable_chunked_prefill: bool = True
long_prefill_token_threshold: int = 0
```

* **`enable_chunked_prefill`**：一个 3000 token 的 prompt 如果一次算完，会产生一次很长的 prefill，期间**所有正在 decode 的请求都被卡住**（ITL 尖峰）。开启后，它被切成若干块，穿插在 decode 之间。
* **`long_prefill_token_threshold = 0`**：`0` 表示不启用「长 prompt 独占 step」策略。设成非 0 时，超过该长度的 prompt 会等 running 清空后才被接纳——这是 vLLM 处理超长请求的做法，避免一条请求长期霸占预算导致其它请求饿死。

### 第五组：前缀缓存 / 压测开关

```python
enable_prefix_caching: bool = True

static_batching: bool = False
```

* `enable_prefix_caching`：关掉时 [block_manager.py](block_manager.md) 的 `match_prefix` 直接返回空列表，其余逻辑完全不变——这让「开/关」的 A/B 对比非常干净。
* `static_batching`：**只给 benchmark 用**。打开后调度器模拟旧式引擎（一批跑到底才换新请求）。它的存在让 [bench_static_vs_continuous.py](../scripts/bench_static_vs_continuous.py) 能只翻转一个开关就得到对比，而不是拿两套代码比。

### 参数校验：`__post_init__`

```python
def __post_init__(self) -> None:
    if self.block_size <= 0:
        raise ValueError("block_size must be positive")
    if self.max_num_batched_tokens < self.block_size:
        raise ValueError("max_num_batched_tokens must be >= block_size")
    if self.long_prefill_token_threshold and (
        self.long_prefill_token_threshold < self.max_num_batched_tokens
    ):
        raise ValueError("long_prefill_token_threshold must be >= max_num_batched_tokens")
```

`__post_init__` 是 dataclass 的钩子：自动生成的 `__init__` 在**赋完所有字段之后**调用它。用它做校验是最自然的位置。

三条规则的理由：

1. `block_size > 0`：否则除零。
2. `max_num_batched_tokens >= block_size`：分块 prefill 是按 block 对齐切的（见 [scheduler.md](scheduler.md) 的 `_chunk_size`），预算比一块还小时**永远切不出任何一块**，会死锁。
3. `long_prefill_token_threshold >= max_num_batched_tokens`：阈值比单步预算还小的话，任何 prefill 都会被判为「长请求」，等于永久拒收。

> 这些检查是**快速失败**思想：宁可在构造配置时就报错，也不要跑到第 500 个 step 才发现调度器死锁。

### 派生属性

```python
@property
def max_blocks_per_seq(self) -> int:
    return (self.max_model_len + self.block_size - 1) // self.block_size
```

向上取整的惯用写法：`ceil(a / b) == (a + b - 1) // b`。本项目到处是这个式子（算需要几页）。

`@property` 让它每次动态算，而不是存成一个字段——这样改 `block_size` 之后不会出现「缓存了旧值」的不一致。

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `@dataclass` | 自动生成 `__init__`/`__repr__`/`__eq__` |
| `__post_init__` | 字段初始化后的钩子，用来校验/派生 |
| `int \| None` | PEP 604 联合类型（Python 3.10+）。注意本项目 `requires-python = ">=3.10"` 就是因为这个语法 |
| `@property` | 把方法伪装成属性，调用时不写括号 |
| `from __future__ import annotations` | 让类型注解变字符串，避免运行时求值（对 `X \| Y` 在旧版本上很关键） |

---

## 设计取舍

**为什么参数是扁平的一个大 dataclass，而不是分组嵌套？**

嵌套（`cfg.cache.block_size`）在参数多时更清晰，但会让测试和 CLI 传参更啰嗦。本项目参数不到 20 个，扁平的最省事。vLLM 真实实现就拆成了 `CacheConfig`、`SchedulerConfig`、`ModelConfig` 三组——规模问题，不是设计问题。

**为什么用字符串 `dtype` 而不是 `torch.float16`？**

因为 `config` 不该依赖 `torch`。现在 `config.py` 只 import 了 `dataclasses`，可以被纯 CPU 环境（甚至无 torch 环境）加载，测试分配器时完全不需要装 torch。

---

## 常见误解

* 「`max_num_batched_tokens` 是 batch size」——**不是**。batch size 的上限是 `max_num_seqs`；`max_num_batched_tokens` 是「这个 step 总共能处理多少 token」。
* 「`gpu_memory_utilization` 是总显存的 80%」——是**当前空闲**显存的 80%。
* 「`max_model_len` 只影响报错」——它还决定页表宽度上限和显存池大小的上限。

---

## 动手练习

1. 设 `max_num_batched_tokens=8, block_size=16`，构造 `EngineConfig()`，观察第三条校验怎么报错。
2. 把 `block_size` 从 16 改成 32，跑 `python -m pytest tests/test_model_runner.py -k block_size -q`，确认结果数值不变。
3. 在 `max_blocks_per_seq` 里故意去掉 `- 1`，跑 `tests/test_scheduler.py`，找出哪条测试会挂（提示：正好整除时）。
