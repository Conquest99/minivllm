# `scripts/bench_fragmentation.py` 详解

## 这个文件干什么

**只用分配器，不用模型**，量出「预留显存」和「分页」两种策略的浪费差异。

这是整个项目里**最便宜也最有说服力**的 benchmark：纯 CPU、毫秒级、不加载任何权重，却能直接给出「为什么需要 PagedAttention」的数字。

```
KV 池: 4096 tokens (256 blocks × 16 tokens)，64 个请求

                                  预留          分页
最大并发请求数                          2            20
实际占用 tokens                      4096          3836
浪费比例                            90.0%          4.7%
```

---

## 1. 为什么不需要模型

因为碎片问题**完全发生在分配层**：给定「每条请求要 N 个 token」和「池子有 M 个 slot」，能同时服务几条？

这个问题和 attention 一点关系都没有。所以脚本只 import：

```python
from minivllm.block_manager import AllocStatus, BlockManager
from minivllm.sequence import Sequence
from minivllm.sampling import SamplingParams
```

**没有 `import torch`**。这本身就是分层良好的证明。

---

## 2. `fabricate`：不靠 tokenizer 造假序列

```python
def fabricate(seq_id: int, num_tokens: int) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=[seq_id * 10000 + i for i in range(num_tokens)],
        sampling_params=SamplingParams(),
    )
```

关键设计：`[seq_id * 10000 + i for i in range(num_tokens)]`

* 每条序列的 token **全局唯一**（`seq_id * 10000` 错开）。
* 这样**前缀缓存一定不会误命中**，测的就是纯粹的碎片问题。

如果不这么造而是用 `[1,2,3,...]`，所有序列前缀相同，`match_prefix` 会疯狂命中，结果全乱。

> 注意 `num_tokens` 上限是 10000，否则相邻序列的区间会重叠。这里最大 32+256=288，安全。

---

## 3. 对照组一：预留式分配

```python
blocks_per_reservation = (args.max_model_len + bs - 1) // bs
reserve_concurrency = pool_blocks // blocks_per_reservation
reserve_used_tokens = reserve_concurrency * args.max_model_len
```

* `blocks_per_reservation`：每条请求要占多少块 —— `ceil(2048/16) = 128` 块。
* `reserve_concurrency`：池子能塞几条 —— `256 // 128 = 2` 条。
* `reserve_used_tokens`：这 2 条实际占了 `2 × 2048 = 4096` 个 slot。

**为什么老引擎必须预留？** 因为它要求每条请求的 KV 是一块连续内存，而生成过程中长度不可知，所以只能按最坏情况切。中途扩容 = 拷贝全部历史 KV。

---

## 4. 实验组：分页分配

```python
bm = BlockManager(pool_blocks, bs, enable_prefix_caching=False)
admitted = 0
used_tokens = 0
for i, n in enumerate(lengths):
    seq = fabricate(i, n)
    if bm.can_allocate(seq, n) != AllocStatus.OK:
        break
    bm.allocate(seq, n)
    admitted += 1
    used_tokens += n
```

* 依次尝试接纳，`can_allocate` 说不行就停（模拟「池满后排队」）。
* **`allocate(seq, n)` 传的是真实长度 `n`，不是 `max_model_len`** —— 这就是分页的本质。
* 注意这里没有 `free`，模拟「所有请求同时存活」的最坏情况。

结果：**20 条**（而不是 2 条）。

---

## 5. 内部碎片

```python
slot_fragmentation = sum((n + bs - 1) // bs * bs - n for n in lengths[:admitted])
```

每一项是 `ceil(n/bs)*bs - n`，即「分到的 slot 数 - 实际需要的」。

举例（bs=16）：

| n | 占块数 | 占 slot | 浪费 |
|---:|---:|---:|---:|
| 48 | 3 | 48 | 0 |
| 50 | 4 | 64 | 14 |
| 33 | 3 | 48 | 15 |

平均浪费约 `bs/2 = 8` 个 slot。20 条约 160 个 slot，占总用量 3836 的 **4.7%**。

**这 4.7% 是分页无法消除的代价**（除非 block_size=1，但那样页表会爆炸）。而 90% 是预留策略独有的、完全可以消除的浪费。

一句面试话术：

> 「分页把外部碎片降到 0，只剩内部碎片；内部碎片的上界是每条序列 `block_size-1` 个 slot，而外部碎片的上界是整个池子。」

---

## 6. 报告与展示

```python
report["reserve_waste_pct"] = 100 * (
    1 - report["reserve"]["tokens_actually_needed"] / max(report["reserve"]["tokens_held"], 1)
)
report["paged_waste_pct"] = 100 * slot_fragmentation / max(used_tokens, 1)
```

两个 `max(..., 1)` 都是防除零（当没有任何请求被接纳时 `used_tokens=0`）。

**注意两个浪费的定义不同**：

* `reserve_waste_pct` = 「预留了但用不上」的比例 → 90%
* `paged_waste_pct` = 「分配了但装不满」的比例 → 4.7%

这是**故意的**，因为两者的浪费本质不同，不能简单相加。

### 打印对齐

```python
print(f"{'':<28}{'reserve':>14}{'paged':>14}")
print(f"{'max concurrent requests':<28}{reserve_concurrency:>14}{admitted:>14}")
```

`<28` 左对齐宽度 28，`>14` 右对齐宽度 14。手工排版的好处：不依赖任何表格库，复制到微信/markdown 里都不乱。

### `--dump`：存档

```python
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
```

`parents=True` 建整条目录链，`exist_ok=True` 避免目录已存在时报错。

`indent=2` 让 JSON 可读——因为它是给人看的（git diff 也友好）。

---

## 7. 参数怎么影响结论

| 参数 | 调大之后 | 为什么 |
|---|---|---|
| `--kv-pool-tokens` | 两种策略的差距**比例不变** | 浪费率只取决于 `max_model_len / 实际长度` |
| `--block-size` | 内部碎片变大 | 最多浪费 `bs-1` 个 slot |
| `--max-model-len` | 预留策略的并发**线性下降** | 每条要吃更多块 |
| `--prompt-tokens`/`--min/max-output` | 实际长度变大 → 预留策略的浪费比例**下降** | 越接近 `max_model_len` 越划算 |

**最后一条最值得注意**：如果所有请求都跑满 2048 token，预留策略一点都不浪费。**分页的优势来自「长度分布不均且远小于上限」——这正是真实负载的样子。**

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `argparse` | CLI 参数 |
| `json.dumps(..., indent=2)` | 可读 JSON |
| `Path.mkdir(parents=True, exist_ok=True)` | 建目录 |
| `Path.write_text(...)` | 一行写完文件 |
| `f"{x:<28}"` / `f"{x:>14}"` | 手工表格对齐 |
| `sum(generator)` | 内部碎片求和 |
| `enumerate` + `break` | 模拟「池满即停」 |

---

## 设计取舍

**为什么不用真模型跑这个实验？**

因为真模型跑一次要 20 秒，而且引入噪声（GPU 抖动、显存动态分配）。**纯分配器版本 0.1 秒出结果，且结论完全一致**——因为碎片是数学问题，不是性能问题。

**为什么分页实验里不 `free`？**

测的是「同时在线的最大并发」，所以让所有请求都活着。真实系统里短请求会先结束，并发可以更高——但那属于调度问题，不是分配问题。两个实验要隔离变量。

---

## 常见误解

* 「浪费 90% 是因为模型太大」——不，是因为**预留了用不上的空间**。
* 「分页之后完全没有浪费」——有 4.7% 的内部碎片。
* 「block_size 越小越好」——block_size=1 时内部碎片为 0，但页表长度 = 序列长度，gather 索引和元数据开销爆炸。
* 「这个数字和真实部署一致」——真实部署里请求长度分布不同，90% 会变；但「预留策略浪费极大」这个结论是稳健的。

---

## 动手练习

1. 把 `--prompt-tokens` 和 `--max-output` 都设成很大（比如 `--prompt-tokens 1000 --max-output 1000`），让实际长度接近 2048，观察浪费率掉到多少。
2. 把 `--block-size` 从 16 改成 1 和 128 各跑一次，观察 `paged_waste_pct`。
3. 打开 `enable_prefix_caching=True` 并去掉 `fabricate` 的唯一性（改成 `[i for i in range(n)]`），看并发能涨到多少——这就是前缀缓存的威力。
4. 加一个 `--free-probability`，随机释放已完成的请求，模拟更真实的在线负载。
