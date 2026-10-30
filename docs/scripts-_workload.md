# `scripts/_workload.py` 详解

## 这个文件干什么

**所有 benchmark 共用的请求集生成器 + 统计工具**。

文件名以下划线开头，表示「内部模块，不是入口脚本」——它被其它脚本 `import`，不单独运行。

存在的意义：让三个 benchmark 测的是**同一批请求**，数字才能互相比较。

---

## 1. 路径注入：为什么需要那两行

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

运行 `python scripts/bench_xxx.py` 时，Python 把 `scripts/` 放进 `sys.path[0]`，但**仓库根目录不在里面**，所以 `import minivllm` 会失败。

`Path(__file__).resolve().parents[1]` 从脚本文件本身反推仓库根：

```
__file__            = D:\...\mini_vllm\scripts\_workload.py
.resolve()          = 绝对路径（解析符号链接）
.parents[0]         = D:\...\mini_vllm\scripts
.parents[1]         = D:\...\mini_vllm          ← 仓库根
```

插到 `sys.path[0]`（最前面）保证优先于任何已安装的同名包。

> 正规做法是 `pip install -e .` 后直接 `import minivllm`（本项目 `pyproject.toml` 已支持）。这两行是「没装也能跑」的兜底。

---

## 2. `PROMPTS`：为什么是这 12 条

```python
PROMPTS = [
    "Hi",
    "What is paged attention?",
    "Explain the difference between continuous batching and static batching in one paragraph.",
    "Write a short story about a cache that never ran out of memory.",
    ...
]
```

几条设计考虑：

1. **长度跨度大**：从 `"Hi"`（2 token）到一段 20 词的句子（约 25 token）。真实服务的请求长度就是这种长尾分布。
2. **内容有实际语义**：短随机串会让模型立刻输出 EOS，压测就变成测"空转"。这些 prompt 都能稳定生成一段文字。
3. **主题相关**：输出内容本身就是这些概念，肉眼能看出模型没坏。
4. **数量 12**：足够让 32 条请求循环使用而不至于全是重复。注意**重复的 prompt 会命中前缀缓存**——这是刻意的，也提醒你：**测吞吐时前缀缓存的收益会混进来**。

> 如果想测「纯无缓存」性能，用 `--seed` 打乱 + `enable_prefix_caching=False`，或者像 `bench_prefix_caching.py` 那样显式对比。

---

## 3. `build_workload`：两种分布

```python
def build_workload(n, seed=0, min_output=16, max_output=256, distribution="uniform"):
    rng = random.Random(seed)
    workload = []
    for i in range(n):
        u = rng.random()
        if distribution == "heavy_tail":
            u = u**3
        workload.append({
            "prompt": PROMPTS[i % len(PROMPTS)],
            "max_tokens": max(1, min_output + int(u * (max_output - min_output))),
        })
    return workload
```

### `random.Random(seed)` 而不是全局 `random.seed`

建一个**独立的** RNG 实例。理由和 `torch.Generator` 一样：不污染全局状态，别的库调用 `random.random()` 也不会影响本脚本的可复现性。

### `u**3` 为什么能造出重尾分布

`u ~ U(0,1)`，取三次方之后：

| u 区间 | 概率 | u³ 落点 |
|---|---|---|
| 0 ~ 0.5 | 50% | 0 ~ 0.125（很短） |
| 0.5 ~ 0.8 | 30% | 0.125 ~ 0.51 |
| 0.8 ~ 1.0 | 20% | 0.51 ~ 1.0（长） |

**大多数请求落在前半段，少数拉满。** 这正是真实聊天负载的形状（大部分人问一句话，少数人贴一篇文档）。

这个分布对静态批处理特别致命：一个 500 token 的请求会把一批短请求拖住 500 步。实测浪费从均匀分布的 41% 涨到 56%。

### `i % len(PROMPTS)`

取模循环用 prompt。好处是**同一个 prompt 会重复出现**，前缀缓存有机会命中；坏处是测出来的吞吐会略高于纯随机 prompt。写报告时要说明这一点。

---

## 4. `summarise`：分位数为什么要自己算

```python
def summarise(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values = sorted(values)
    n = len(values)

    def pct(p: float) -> float:
        return values[min(n - 1, int(p * n))]

    return {"mean": ..., "p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99), "max": values[-1]}
```

* **`if not values: return {}`** 防空。空列表求 `values[0]` 会 IndexError，空列表求 `sum/n` 会除零。
* **`values = sorted(values)`** 覆盖原变量（这里安全，因为是新 list）。
* **`min(n - 1, ...)`** 防越界：`p99` 在 n=6 时算出 `int(5.94) = 5`，下标 5 合法；但如果写成 `p=1.0` 就会是 6 → 越界。
* **`int(p * n)` 的偏差**：这是「最近秩」的粗略实现，n 很小时不精确。生产环境用 `numpy.percentile` 或 `statistics.quantiles`。

> 为什么不用 `numpy`？为了让这个文件在**没装 numpy 的环境**也能跑（其实 torch 装了就有 numpy……但保持轻量是好事）。

### 为什么必须看 p99 而不是只看均值

延迟分布通常**长尾**。假设 8 个请求里 7 个 1 秒、1 个 9 秒：

```
mean = (7×1 + 9)/8 = 2 秒      ← 看起来还行
p99  = 9 秒                     ← 那一个用户的实际体验
```

**用户体验由尾部决定，不是均值。** 这也是为什么 `bench_static_vs_continuous.py` 一定同时打印 mean 和 p99。

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `sys.path.insert(0, ...)` | 手动加模块搜索路径 |
| `Path(__file__).resolve().parents[N]` | 从文件位置反推目录 |
| `random.Random(seed)` | 独立 RNG 实例 |
| 嵌套函数 + 闭包 | `pct` 捕获 `values` 和 `n` |
| `sorted()` | 返回新 list，不改原对象 |

---

## 设计取舍

**为什么 `workload` 里存 `max_tokens` 而不是 `min/max` 区间？**

因为下游脚本需要的是「这条请求要生成多少 token」这个**确定值**——它是算吞吐的分母，必须可复现。在 `build_workload` 里一次性定死，避免每个脚本各自随机。

**为什么不用 datasets 库造真实请求？**

因为那要下载数据、依赖网络、还引入不确定性。12 条手写 prompt 已经足够暴露调度行为，且完全离线可复现。

---

## 常见误解

* 「`heavy_tail` 是帕累托分布」——不是严格的，只是用 `u**3` 造的近似长尾。真正的重尾（如对数正态）会有更极端的离群值。
* 「`summarise` 的 p99 是统计意义上的 p99」——样本少时它只是「第 99% 位置的那个值」，近似而已。
* 「seed 相同结果就一样」——只有「请求集合」一样；模型侧的随机性由 `SamplingParams.temperature`（贪心为 0）决定。

---

## 动手练习

1. 把 `u**3` 改成 `u**6`，跑 `bench_static_vs_continuous.py --distribution heavy_tail`，观察「静态批浪费比例」和「吞吐增益」怎么变化。
2. 把 `PROMPTS` 减到 2 条，重跑，看前缀缓存命中如何把连续批处理的优势放大（提示：所有请求共享前缀，`prefix_hit_blocks` 会暴涨）。
3. 给 `summarise` 加 `p50 == median` 的断言，用 `statistics.median` 交叉验证。
