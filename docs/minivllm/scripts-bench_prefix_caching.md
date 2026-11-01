# `scripts/bench_prefix_caching.py` 详解

## 这个文件干什么

量出**前缀缓存对首 token 延迟（TTFT）的收益**。

场景：所有请求共享一个 3285 token 的 system prompt，只有末尾的问题不同。

```
                              prefix off     prefix on
wall time (s)                     19.410        13.807
mean TTFT (s)                      0.833         0.214
warm TTFT (s)                      0.779         0.077
shared blocks served                   0          1020

mean TTFT gain 3.9x   |   warm-request TTFT gain 10.2x
```

---

## 1. 为什么必须**串行**跑

这是本脚本最重要的一个设计决定，也是我第一版做错的地方。

第一版把 6 条请求一次性丢进 `engine.generate(prompts)`：

```
                              prefix off     prefix on
mean TTFT (s)                      0.088         0.076     ← 几乎没差别！
shared blocks served                   0           210     ← 命中了很多块
```

原因：**前缀缓存只对「前缀已经算完并登记之后才到达」的请求有用。**

一次性全部投进去时，6 条请求在**同一个 step** 里被接纳：

* 第 1 步：6 条一起 prefill → 谁都没有缓存可命中
* 全部算完之后才登记进缓存 → 但这时 6 条都已经在 decode 了

缓存命中了 210 块（说明机制本身是对的），但**没有任何请求因此受益**。

改成一条一条跑：

```python
for prompt in prompts:
    out = engine.generate([prompt], ...)[0]
```

第 1 条付全价（它负责"填缓存"），第 2~6 条几乎立刻开始 decode。

**这条经验在真实系统里同样成立**：前缀缓存的价值取决于**请求的时间分布**。突发流量（所有请求同时到达）几乎吃不到收益；持续流量（请求陆续到达）才能受益。

---

## 2. 为什么 system prompt 要重复 120 遍

```python
SYSTEM_PROMPT = ("You are a meticulous distributed systems tutor. ... ")
def build_prompts(n, repeats):
    shared = SYSTEM_PROMPT * repeats
    return [f"{shared}\n\nQuestion {i}: ..." for i in range(n)]
```

默认 `--prefix-repeats 120` → 3285 token 的共享前缀。

为什么这么长？因为收益和前缀长度成正比：

| 前缀长度 | 命中块 | warm TTFT 提升 |
|---|---:|---|
| 693 token | 210 | 1.15x（测不出来） |
| 3285 token | 1020 | **10.2x** |

693 token 时 prefill 只要 ~70ms，和固定开销（kernel 启动、采样）混在一起，信噪比太低。**3285 token 时 prefill 要 ~780ms，收益就非常清晰了。**

> 这也解释了为什么长 system prompt 场景（RAG、agent、多轮对话）前缀缓存收益最大：这些场景的前缀动辄几千 token。

`repeats` 用「重复同一句」而不是一段长文本，是为了让 tokenizer 的输出可预测（长度线性增长），方便调参。

---

## 3. 关键指标：`ttft` vs `ttft_warm`

```python
return {
    "ttft": summarise(ttfts),              # 所有请求
    "ttft_warm": summarise(ttfts[1:]),     # 跳过第一条
    ...
}
```

`ttfts[1:]` 是**列表切片**，去掉第一个元素。

为什么要分开？

* **`ttft`（全部）**：包含"填缓存"的那一条，是**冷启动包含在内**的诚实数字。
* **`ttft_warm`（跳过第 1 条）**：稳态下每个请求的体验，这才是缓存长期运行时的真实收益。

两者都报，既诚实又说明问题：

```
mean TTFT gain 3.9x   |   warm-request TTFT gain 10.2x
```

第一版只报全部，得到 3.9x。但真实服务运行几小时后，缓存是热的，用户感受到的是 10.2x。**只报 3.9x 其实低估了自己**。

---

## 4. 单条请求的 TTFT 序列最能说明问题

```
TTFT per request (off): [0.806, 0.704, 0.724, 0.736, 0.782, 0.776]
TTFT per request (on) : [1.014, 0.238, 0.080, 0.068, 0.082, 0.082]
```

开启缓存后：

* **第 1 条更慢了**（0.806 → 1.014）：因为它要额外做哈希计算和登记。
* **第 2 条** 0.238：命中了一部分（可能有些块还没登记完）。
* **第 3~6 条** ~0.08：几乎只剩采样开销。

**第 1 条变慢是必然的**——「没有免费的缓存」。但摊到 6 条上完全值得。这个细节在面试里很能体现「真的量过」。

---

## 5. `reset_cache` 的位置

```python
def run_sequential(engine, prompts, max_tokens):
    engine.scheduler.reset_cache()     # 每次测量前清空
    ...
```

必须清空，否则：

* `prefix off` 那一轮跑完，缓存里有东西
* `prefix on` 那一轮如果复用同一个 engine，起点就不公平

本脚本更彻底——**每一轮都新建一个 `LLMEngine`**：

```python
for enabled in (False, True):
    cfg = EngineConfig(..., enable_prefix_caching=enabled)
    engine = LLMEngine(cfg)
    ...
    del engine
```

这样连「块池的初始状态」都完全一致。代价是加载两次模型（约 10 秒），值得。

`del engine` 是为了尽快释放显存——不 `del` 的话两个 engine 同时持有 KV 池，8G 卡会 OOM。

---

## 6. `--max-model-len 8192` 为什么需要

3285 token 的 prompt + 32 token 输出 = 3317。默认 `max_model_len = 2048` 会直接拒绝（`add_request` 里的校验）。

所以脚本把 `max_model_len` 提到 8192，同时把 `max_num_batched_tokens` 提到 2048（否则一次 prefill 装不下 3285 token，会被切成两块，测的就不是纯 prefill 了）。

**这两个参数是耦合的**：`max_model_len` 决定"允许多长"，`max_num_batched_tokens` 决定"一次最多算多少"。前者必须 ≥ 后者，否则长 prompt 永远切不完。

---

## 7. `max_num_seqs=1`

```python
max_num_seqs=1,  # 一次只跑一条：隔离 prefill，不受排队影响
```

如果允许多条并发，第 2 条请求的 TTFT 会包含「等第 1 条 prefill 算完」的排队时间——那是调度开销，不是 prefill 开销。设成 1 之后，TTFT 干净地等于「prefill + 首 token」。

**这就是控制变量**：这个实验要测的是「前缀缓存对 prefill 的影响」，不是「并发下的整体表现」。

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `list[1:]` 切片 | 跳过第一条，算 warm 指标 |
| `del engine` | 显式释放显存 |
| `argparse` | 参数化前缀长度/请求数 |
| `f"{x:<26}{a:>14.3f}{b:>14.3f}"` | 三列对齐 |
| `summarise` | 复用 `_workload.py` 的分位数工具 |

---

## 设计取舍

**为什么不直接测「用了多少块」而要测 TTFT？**

块数是**机制指标**（中间变量），TTFT 是**用户指标**。面试时讲「我命中了 1020 个块」不如讲「首字延迟从 0.78 秒降到 0.077 秒」。**永远把内部指标翻译成用户能感知的量。**

**为什么不做「多个不同长度前缀」的矩阵？**

因为结论是单调的（前缀越长收益越大），扫一遍不增加洞察。真要做，应该扫「请求到达间隔」——那个才是决定命中率的关键变量。

---

## 常见误解

* 「前缀缓存命中就等于加速」——不，还得调度器推进 `num_computed_tokens`（见 [scheduler.md](scheduler.md)）。
* 「并发投递时缓存没用」——是**这一批**没用。下一批请求会受益。
* 「前缀越长越好」——长到超过 `max_model_len` 就直接被拒了。
* 「第一条变慢说明缓存有害」——摊还之后仍然大幅净赚。

---

## 动手练习

1. 把 `--prefix-repeats` 从 120 调到 30 和 300，画出「前缀长度 → warm TTFT 增益」曲线。
2. 改成一次投递 6 条（去掉 `for prompt in prompts` 循环），亲自复现「命中 210 块但 TTFT 无收益」的现象。
3. 每两轮之间插入一次 `reset_cache` 但不重建 engine，观察与重建 engine 的差异（提示：块池状态不一致）。
4. 把 system prompt 改成 6 条**互不相同**的，验证 `prefix_hit_blocks` 归零——这是对「哈希链正确性」的负向验证。
