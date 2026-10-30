# `scripts/bench_static_vs_continuous.py` 详解

## 这个文件干什么

同一个模型、同一份代码，**只翻转 `static_batching` 一个开关**，量出连续批处理相对静态批处理的收益。

```
                                     静态         连续
总耗时 (s)                          94.992          59.921
有效 tokens/s                       60.900          96.544
每 step 产出 tokens                  8.000           6.026
平均延迟 (s)                        54.177          24.638
p99 延迟 (s)                        94.981          59.914
平均 TTFT (s)                       30.628          13.164

静态批处理白算了 56% 的生成 token
```

---

## 1. 为什么「只翻一个开关」这件事很重要

很多 benchmark 拿「HF `generate`」对比「vLLM」，那其实是**两套实现**的对比：kernel 不同、KV 管理不同、采样实现不同……你无法归因到某一个机制。

这个脚本的做法是：**同一份代码、同一个模型、同一批请求，只改调度策略。**

```python
engine.cfg.static_batching = static
```

`static_batching` 在 [scheduler.py](../minivllm/scheduler.py) 里做了两件事：

1. `running` 非空时**不接收**新请求
2. 新批次里所有请求的 `max_tokens` **拉齐到最长那条**

所以差异 100% 来自「批处理策略」，这是**可归因的测量**。面试里讲这个比讲「我们比 XX 快 3 倍」有说服力得多。

---

## 2. `run_engine`：统一入口

```python
def run_engine(engine: LLMEngine, workload: list[dict], static: bool) -> dict:
    engine.cfg.static_batching = static
    engine.scheduler.reset_cache()
    engine.metrics = EngineMetrics()
    if engine.scheduler.running or engine.scheduler.waiting:
        raise RuntimeError("engine must be idle before a run")
    ...
```

三个前置动作，缺一不可：

| 动作 | 为什么 |
|---|---|
| `static_batching = static` | 唯一的自变量 |
| `scheduler.reset_cache()` | **清空前缀缓存**，否则第二次跑会因缓存而变快（不公平） |
| `metrics = EngineMetrics()` | 重置计数器，否则 `wall_time` 会累加（`generate` 里是 `+=`） |
| `running/waiting` 检查 | 防止上一次没跑完就覆盖测量 |

### 计时边界

```python
t0 = time.perf_counter()
ids = [engine.add_request(...) for item in workload]
done = {}
while engine.scheduler.has_unfinished():
    for out in engine.step():
        done[out.seq_id] = out
wall = time.perf_counter() - t0
```

计时的**起点在 add_request 之前**，所以包含了 tokenize 和入队开销。这正是用户视角的端到端延迟。

**没有 `torch.cuda.synchronize()`** —— 这是个已知的测量偏差。GPU 是异步的，`perf_counter()` 可能在 kernel 还没算完时就返回了。

不过在**几百次 step** 的量级下，最后一次的误差（几十毫秒）相对总时长（几十秒）是 0.1% 量级，可以忽略。单步测时（比如 `ms/step`）就必须 `synchronize`，否则测的是「发命令的时间」而不是「算完的时间」。

### `useful` 与 `generated` 的区别

```python
useful = sum(item["max_tokens"] for item in workload)       # 用户真正要的
generated = sum(o.num_output_tokens for o in outputs)       # 实际算了多少
```

* **静态批处理**：`generated > useful`（短请求被迫陪跑，多生成的那些是"白算"）
* **连续批处理**：`generated == useful`（每条到自己的上限就停）

这个差值就是「浪费比例」：

```python
waste = 1 - static["useful_tokens"] / static["generated_tokens"]   # 56%
```

**为什么静态批的 `tokens_per_step` 反而更高（8.0 > 6.0）？**

因为它的 batch 永远满员——只是有 56% 在算没人要的东西。**GPU 利用率高 ≠ 有效吞吐高**，这是最容易被误读的指标。

---

## 3. `run_hf`：第三列参照物

```python
if args.with_hf:
    hf = run_hf(engine, workload, args.batch_size)
```

用 `model.generate` 跑同一批请求，提供「成熟实现」的参照线。

要点：

* **`padding_side = "left"`**：decoder-only 模型必须左 padding，否则 `generate` 会在右侧补的 pad 上继续生成，结果错乱。
* **`eos_token_id=None`**：禁用 EOS 停止，保证长度可控（否则会和我们的 `ignore_eos` 不一致）。
* **`do_sample=False`**：贪心，和 `temperature=0` 对齐。
* **latency/TTFT 置 0**：HF 的 `generate` 不流式输出，调用者直到整批结束才拿到东西，所以「逐请求延迟」无意义。**这一列只比吞吐，不比延迟**。

> 这一列**不是** apples-to-apples（不同 kernel、不同 KV 策略），但回答了最自然的追问：「你自己写的东西能打过成熟库吗？」实测 batch=8 时我们的引擎 120 tok/s，HF 105 tok/s——虽然我们的 gather 有额外拷贝，但连续调度赢回了更多。

---

## 4. `run_hf` 的坑：`apply_chat_template`

```python
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": c["prompt"]}], add_generation_prompt=True, tokenize=False
    )
    for c in chunk
]
```

必须和引擎走**完全一样**的模板，否则比的是两个不同的输入。第一版这里我手写了一个 `<|im_start|>user\n...` 字符串，结果和引擎的模板差了几个特殊 token，输出分布都不同——测出来的吞吐不可比。

**这类「基准测试里的输入不一致」是最隐蔽的错误**：数字看起来很正常，但结论是错的。

---

## 5. 表格打印

```python
names = list(rows)
col = 26
header = f"{'':<{col}}" + "".join(f"{n:>16}" for n in names)
print(header)
print("-" * len(header))

def line(label: str, getter) -> None:
    print(f"{label:<{col}}" + "".join(f"{getter(rows[n]):>16.3f}" for n in names))

line("wall time (s)", lambda r: r["wall_time"])
line("useful tokens/s", lambda r: r["throughput_tok_s"])
...
```

用 `getter` 函数 + `lambda` 让每行代码只写一次。好处：**加一列（比如 `--with-vllm`）不需要改 7 处打印**。

`"-" * len(header)` 画分隔线，长度自动跟随表头——比写死 60 个减号更稳。

---

## 6. 怎么读这张表

| 指标 | 静态 | 连续 | 解读 |
|---|---:|---:|---|
| 有效 tok/s | 60.9 | 96.5 | **吞吐 1.59x** |
| 每 step token | 8.0 | 6.0 | 静态批更"满"，但一半是废的 |
| 平均延迟 | 54.2s | 24.6s | **2.2x** |
| p99 延迟 | 95.0s | 59.9s | 尾部也改善 |
| 平均 TTFT | 30.6s | 13.2s | **2.3x** |

两个反直觉但重要的点：

1. **连续批处理的 batch 更"空"（6.0 < 8.0），却更快。** 因为它不浪费。
2. **TTFT 也变好了**，虽然连续批处理看起来"更忙"。原因是静态批处理要等**整批**跑完才换新请求，新来的请求排队时间极长。

> 如果只看 GPU 利用率，你会得出「静态批处理更好」的错误结论。这正是要同时报告「有效吞吐 / 延迟 / TTFT」三个维度的原因。

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `lambda r: r["..."]` | 行取值器 |
| `"".join(f"..." for n in names)` | 动态列 |
| `f"{'':<{col}}"` | 嵌套宽度 |
| `argparse` 的 `choices=[...]` | 枚举参数校验 |
| `json.dumps(..., indent=2)` | 存档 |
| `zip(..., strict=False)` | 长度不等时静默取短 |

---

## 设计取舍

**为什么用同一个 `engine` 跑两种模式，而不是两个进程？**

省去模型加载时间（约 5 秒），也避免两次进程的显存状态差异。代价是必须先 `reset_cache` + 重置 metrics，脚本里显式做了。

**为什么 `--batch-size` 同时控制两种模式的 `max_num_seqs`？**

因为要控制变量：如果静态批用 8、连续批用 16，多出来的吞吐就说不清是「调度」还是「批更大」的功劳。

**为什么 workload 用重尾分布而不是均匀？**

因为均匀分布下静态批的浪费只有 41%，重尾下是 56%——**更接近真实聊天流量**。真实系统里大部分人问一句话，少数人贴长文档。

---

## 常见误解

* 「静态批处理就是 `batch_size` 大」——不是。静态批是**批次生命周期固定**，连续批是**每 step 重排**。
* 「连续批处理一定更快」——如果所有请求长度完全相同，两者几乎没差别（没有"陪跑"可省）。
* 「`tokens_per_step` 越高越好」——见上，静态批反而更高。
* 「HF 那一列说明我们的 kernel 更好」——不。我们慢在 gather，快在调度，两者相互抵消后仍略胜。

---

## 动手练习

1. 用 `--distribution uniform` 和 `--distribution heavy_tail` 各跑一次，比较浪费比例（41% vs 56%）。
2. 把所有请求的 `max_tokens` 固定成同一个值（改 `_workload.py`），看两种模式的差距是否消失——这是对「结论正确性」的验证。
3. 加 `torch.cuda.synchronize()` 到计时边界，比较总耗时变化（应该只差最后一次 step）。
4. 把 `--batch-size` 从 8 调到 16，观察连续批的收益是否变小（批越大，"陪跑"的相对损失越小）。
