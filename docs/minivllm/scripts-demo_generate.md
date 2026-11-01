# `scripts/demo_generate.py` 详解

## 这个文件干什么

**最小可运行示例**：加载引擎、生成几句话、打印统计。

它的定位是「跑通全流程」，不是「压测」。想量性能看 `bench_*.py`。

---

## 1. 为什么需要它

引擎有问题时，`pytest` 能告诉你「哪条断言挂了」，但不能告诉你「**生成的文字看起来对不对**」。

这个脚本提供后一种检查：模型如果因为 KV cache 的 bug 把「巴黎」答成乱码，测试可能仍然通过（因为对拍用的是同一套 bug），但人一眼就能看出来。

**先跑 demo 看输出，再跑 benchmark 看数字，最后跑测试看边界。** 这个顺序最省时间。

---

## 2. 参数设计

```python
parser.add_argument("--prompts", nargs="+", default=[...])
parser.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
parser.add_argument("--max-tokens", type=int, default=128)
parser.add_argument("--block-size", type=int, default=16)
parser.add_argument("--max-num-seqs", type=int, default=8)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--no-prefix-cache", action="store_true")
```

* **`nargs="+"`**：`--prompts "a" "b" "c"` 收成一个 list。注意 PowerShell 里带空格的引号要保留。
* **`action="store_true"`**：`--no-prefix-cache` 出现就是 `True`，不出现就是 `False`。用于布尔开关（因为 `--flag True/False` 太啰嗦）。
* **`--temperature` 默认 0**：贪心，结果可复现。想看多样性就传 0.7。

这些参数直接映射到 `EngineConfig`，没有任何隐藏逻辑。

---

## 3. 加载与自检

```python
engine = LLMEngine(cfg)
print(engine.model_runner.describe())
print(f"kv pool: {engine.block_manager.stats()}\n")
```

`describe()` 打印出来长这样：

```
model=models/Qwen2.5-0.5B-Instruct dtype=float16 layers=24 heads=14/2kv head_dim=64 |
kv_blocks=1024 block_size=16 kv_cache=0.19GiB
```

**这行是排障的第一现场**，能立刻看出：

* `14/2kv` —— GQA 生效了（不是 14/14）
* `kv_blocks=1024` —— 池子按「可引用上限」算的（如果显示 25873 就是显存被白占了）
* `kv_cache=0.19GiB` —— 8G 卡上留足余量

`block_manager.stats()` 打印的是**启动瞬间**的状态：全是 0，只有 `free_blocks` 满。这是基线。

---

## 4. 生成与输出

```python
params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
outputs = engine.generate(args.prompts, params)

for out in outputs:
    print("=" * 80)
    print(f"[seq {out.seq_id}] {out.prompt}")
    print("-" * 80)
    print(out.text)
    print(f"-- finish={out.finish_reason} tokens={out.num_output_tokens} "
          f"ttft={out.ttft:.3f}s latency={out.latency:.3f}s preemptions={out.num_preemptions}")
```

三条 prompt 是**一起**交给 `generate` 的，所以它们会在同一个 batch 里被连续批处理混跑。因此：

* 三条的 `ttft` 会**几乎相同**（因为第一步 prefill 就处理了三条）
* `latency` 差别取决于谁先生成完

第一版跑出来是这样：

```
-- finish=length_capped tokens=64 ttft=5.966s latency=9.255s preemptions=0
-- finish=stopped       tokens=53 ttft=5.965s latency=8.657s preemptions=0
-- finish=stopped       tokens=14 ttft=5.965s latency=6.779s preemptions=0
```

三条 TTFT 都是 5.96s，一模一样——这**证明它们确实被合批了**。如果三条 TTFT 递增（5.9 / 6.5 / 7.1），说明哪里的批处理没生效。

这 5.9 秒里大部分是**首次 CUDA kernel 编译/预热**（第一次前向要初始化 cuBLAS、cuDNN 句柄）。所以 benchmark 里都有 `--warmup`。

---

## 5. 统计输出

```python
for key, value in engine.stats().items():
    print(f"{key:>22}: {value:.4f}" if isinstance(value, float) else f"{key:>22}: {value}")
```

`{key:>22}` 是**右对齐、宽度 22**，让冒号对齐成列。这种格式化在控制台报告里很实用。

`isinstance(value, float)` 分别处理两种：浮点保留 4 位，整数原样打印。`stats()` 返回的 dict 混装了两种类型，直接统一 `:.4f` 会把 `num_blocks=1024` 打成 `1024.0000`。

指标怎么读：

| 字段 | demo 里的值 | 含义 |
|---|---|---|
| `num_steps: 64` | 生成最长的序列有 64 token | 一个 step 生成一个 token（对 decode 而言） |
| `throughput_tok_s: 14.13` | 8.4 秒里 119 token | **含 5.9s 冷启动**，所以偏低 |
| `tokens_per_step: 2.05` | 平均每步只有 2 个有效 token | 因为三条序列长短差很多（64/53/14），后期只剩 1-2 条在跑 |
| `peak_running: 3` | 最多同时 3 条 | 正好是三条 prompt |
| `cached_blocks: 12` | 12 个块进了前缀缓存 | 三条 prompt 的公共前缀（chat template）被复用了 |

**`tokens_per_step: 2.05` 是这里最值得注意的数字**：batch 上限是 8，实际只有 2。原因是三条序列长度差异大——长的还在跑，短的已经结束。这正好说明「连续批处理的价值取决于**有没有新请求补充进来**」；demo 里没有补充源，所以它退化成静态批处理。

想要看到高 `tokens_per_step`，得用 `bench_static_vs_continuous.py`（那里有 32 条排队请求）。

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `argparse` + `nargs="+"` | 接受多个值 |
| `action="store_true"` | 布尔开关 |
| `f"{x:>22}"` | 右对齐宽度 22 |
| `f"{x:.4f}"` | 保留 4 位小数 |
| `f"{x:.2f}GiB"` | 带单位格式化 |
| `isinstance(v, float)` | 分支格式化 |
| `dict.items()` | 遍历键值 |

---

## 设计取舍

**为什么不在这里加 warmup？**

demo 的目的是「看到输出」，不是「量准性能」。加 warmup 会让第一次跑的人等更久，而且打印出的 TTFT 会"太好看"（不含冷启动），反而误导。

benchmark 脚本里都有 warmup，那是为了测准。

**为什么默认三条 prompt 而不是一条？**

一条看不出批处理的效果。三条能同时验证：合批（TTFT 相同）、各自结束时间不同、`peak_running` 等于 3。

---

## 常见误解

* 「`ttft` 应该随请求顺序递增」——合批之后三条是同时开始的，所以几乎相等。
* 「`throughput_tok_s=14` 说明引擎很慢」——主要贡献是 5.9 秒冷启动。热起来之后单条 decode 约 45-50 ms/token。
* 「`finish=stopped` 表示出错了」——只是模型自然结束（输出了 EOS）。`length_capped` 才是被 `max_tokens` 截断。

---

## 动手练习

1. 加 `--max-tokens 8` 跑一次，让三条都在同一时间结束，观察 `tokens_per_step` 是否接近 3。
2. 用 `--temperature 0.7` 连跑两次，验证输出不同（`Generator` 用的是引擎种子，但每次 `generate` 会消耗随机状态）。
3. 用 `--block-size 1` 和 `--block-size 64` 各跑一次，确认输出文字完全一致（只影响性能）。
4. 把三条 prompt 改成完全相同的，观察 `prefix_hit_blocks` 的变化（注意：同一批同时到达时命中率是 0，因为前缀还没算完就都进来了——这也解释了为什么 `bench_prefix_caching.py` 要串行跑）。
