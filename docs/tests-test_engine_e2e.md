# `tests/test_engine_e2e.py` 详解

## 这个文件干什么

端到端测试整个引擎，验证的核心是一条**极强的不变量**：

> **贪心采样下，一条请求生成的 token 序列，与它是否被合批、是否被分块、是否被抢占、是否命中前缀缓存完全无关。**

如果分页、调度、COW、前缀缓存里有任何一处簿记错误，这条性质就会破。

这比「结果和某个快照一致」强得多——快照测试只能证明「没变」，这条证明「**在任意调度路径下都对**」。

---

## 1. `make_engine`：可控的引擎工厂

```python
def make_engine(**overrides) -> LLMEngine:
    kwargs = dict(
        model=require_local_model(),
        dtype=DTYPE,
        device=DEVICE,
        block_size=16,
        max_num_seqs=8,
        max_num_batched_tokens=256,
        max_model_len=512,
        num_gpu_blocks=64,
    )
    kwargs.update(overrides)
    return LLMEngine(EngineConfig(**kwargs))
```

* `DTYPE`/`DEVICE` 从环境变量来（见 [conftest.md](tests-conftest.md)），默认 CPU/fp32。
* `max_model_len=512` 而不是默认 2048：让「超长」测试用更小的数字就够。
* `max_num_batched_tokens=256`：足够一次 prefill 短 prompt，又能触发分块。
* `num_gpu_blocks=64`：**故意小**——64 × 16 = 1024 slot，足够 7 条几十 token 的请求，但小到能触发抢占。

```python
def greedy(max_tokens: int = 8) -> SamplingParams:
    return SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
```

**两个参数都必须固定**：

* `temperature=0.0` → 贪心，才能做「结果必须相同」的断言
* `ignore_eos=True` → 长度可控，否则模型可能 2 个 token 就停，测不出差异

**这两个设置是整个文件的正确性基础。** 只要有采样，就无法断言「两次运行结果相同」。

---

## 2. `test_generate_returns_one_output_per_prompt`

```python
outputs = engine.generate(PROMPTS, greedy(6))
assert [o.prompt for o in outputs] == PROMPTS
for out in outputs:
    assert out.num_output_tokens == 6
    assert out.finish_reason == "length_capped"
    assert out.ttft is not None and out.latency is not None
    assert out.ttft <= out.latency
```

三个层次的断言：

1. **顺序**：返回按输入顺序（`generate` 里用 `ids` 重排过）
2. **完整性**：长度、结束原因、指标都填上了
3. **自洽性**：`ttft <= latency`（首 token 不可能晚于结束）

第 3 条是「不变量式断言」的典型——**它不检查具体数值，只检查关系必须成立**。这类断言不需要维护，也不容易被实现细节打破。

`finish_reason == "length_capped"`：因为 `ignore_eos=True`，只能靠长度结束。

---

## 3. `test_batched_output_matches_solo_output` ★

```python
solo = [engine.generate([p], greedy(6))[0].token_ids for p in PROMPTS]
batched = [o.token_ids for o in engine.generate(PROMPTS, greedy(6))]
assert batched == solo
```

**核心不变量的最直接形式**：3 条一起跑 == 各自单独跑。

对比 `token_ids` 而不是 `text`：文本解码可能受 tokenizer 的 normalize 影响（比如连续空格），token id 是精确的。

这条测试能抓到：

* KV cache 串号（一条序列读到另一条的 KV）
* padding 污染（短序列看到长序列的内容）
* 前缀缓存误命中

---

## 4. `test_small_batch_budget_changes_nothing`

```python
engine.cfg.max_num_seqs = 1
try:
    serial = [o.token_ids for o in engine.generate(PROMPTS, greedy(6))]
finally:
    engine.cfg.max_num_seqs = 8
assert serial == [engine.generate([p], greedy(6))[0].token_ids for p in PROMPTS]
```

把并发压到 1，**强制串行**，结果必须和「逐条跑」完全一致。

`try/finally` 保证即使断言失败也会恢复配置——因为这个 `engine` 是 `module` 级 fixture，被后面 3 个测试共用。**测试之间污染是隐蔽的 bug 源**。

> 直接改 `cfg` 有点"脏"（配置本该不可变）。更干净的做法是新建一个 engine，但那样每个测试要多花 3-5 秒加载模型。这里选择了「共享引擎 + 严格恢复」的折中。

---

## 5. `test_chunked_prefill_gives_the_same_tokens`

```python
long_prompt = "Explain, step by step and in detail, how a block table maps logical to physical pages."
one_shot = make_engine(enable_chunked_prefill=False, max_num_batched_tokens=256)
chunked = make_engine(enable_chunked_prefill=True, max_num_batched_tokens=32)

expected = one_shot.generate([long_prompt], greedy(6))[0].token_ids
actual = chunked.generate([long_prompt], greedy(6))[0].token_ids
assert actual == expected
```

**分块与不分块，结果必须相同。**

配置对比：

* `one_shot`：预算 256，一次算完
* `chunked`：预算 32，切成 2 块（block_size=16，按块对齐）

这里真正被验证的是「**分块 prefill 写进去的 KV 是正确的**」。因为生成 6 个 token 的 decode 阶段会不断读这些 KV，一旦有错，token 序列就偏。

> 这条测试用的是**同一个模型的两个 runner 实例**——加载两次，1 秒左右。所以整条测试比其它的慢。

---

## 6. `test_prefix_caching_gives_the_same_tokens`

```python
with_cache = make_engine(enable_prefix_caching=True, max_num_seqs=1)
without = make_engine(enable_prefix_caching=False, max_num_seqs=1)
prompts = ["You are a careful assistant. " + p for p in PROMPTS]

cached = [o.token_ids for o in with_cache.generate(prompts, greedy(6))]
plain = [o.token_ids for o in without.generate(prompts, greedy(6))]
assert cached == plain
assert with_cache.block_manager.num_prefix_hit_blocks > 0
```

两件事：

1. **正确性**：缓存命中不能改变结果
2. **有效性**：必须真的命中了（否则这条测试只证明了「关闭缓存时也正常」）

### 为什么 `max_num_seqs=1` 是必须的

这是修 bug 修出来的。

最初的版本用默认 `max_num_seqs=8`，结果 `num_prefix_hit_blocks == 0`，测试失败。原因：

```
3 条请求同一 step 被接纳
→ 它们的公共前缀（"You are a careful assistant. "）此刻谁都没算完
→ 前缀缓存里什么都没有
→ 命中 0 次
```

缓存只在「前缀已经算完并登记」之后才对**后来者**有用。所以在同一个 step 里全接纳 = 谁都用不上。

设成 `max_num_seqs=1` 之后：第 1 条跑完并登记 → 第 2 条到达时就能命中。

**这个限制本身就是一条重要的结论**：前缀缓存的收益取决于请求的**到达时间分布**，突发流量几乎吃不到。

---

## 7. `test_preemption_still_produces_correct_tokens`

```python
tight = make_engine(num_gpu_blocks=4, max_num_seqs=3, max_num_batched_tokens=64)
roomy = make_engine(num_gpu_blocks=64, max_num_seqs=3, max_num_batched_tokens=64)

expected = [o.token_ids for o in roomy.generate(prompts, greedy(10))]
actual = [o.token_ids for o in tight.generate(prompts, greedy(10))]

assert actual == expected
assert tight.scheduler.num_preemptions > 0, "the pool is too small not to preempt"
```

**抢占后重算，结果必须不变。**

* `tight`：4 块 = 64 slot，3 条请求各要 10-15 token → 中途必然不够
* `roomy`：64 块 = 1024 slot，绝不抢占

两个断言各有作用：

1. `actual == expected`：正确性
2. `num_preemptions > 0`：**确认真的发生了抢占**

第 2 条极其重要。如果池子意外地够大，第 1 条会"通过"但什么都没测到——**测试要能证明自己走到了预期的代码路径**。

> 这是「防假阳性断言」：不仅要验结果，还要验过程。

### recompute 为什么能保证结果不变

因为抢占只清 `num_computed_tokens = 0`，`seq.data` 完整保留。下次调度时从头 prefill，算出**完全相同**的 KV（同样的权重、同样的输入、同样的 RoPE 位置）→ 后续 decode 完全一致。

**这是「重算」策略在数学上安全的原因**——它不引入任何近似。

---

## 8. `test_metrics_are_consistent`

```python
engine.metrics = type(engine.metrics)()
outputs = engine.generate(PROMPTS, greedy(5))
assert engine.metrics.num_generation_tokens == sum(o.num_output_tokens for o in outputs)
assert engine.metrics.throughput > 0
stats = engine.stats()
assert 0.0 <= stats["utilization"] <= 1.0
```

`type(engine.metrics)()` 是「用同样的类型新建一个空实例」——等价于 `EngineMetrics()`，但不 import 那个名字。

它守的是**统计口径一致**：

* 引擎记的生成 token 总数 == 各请求输出 token 之和
* 利用率在 [0, 1] 区间

这类「度量自洽」的测试在工程上很有价值——统计口径错了会让所有 benchmark 数字失去意义，而这种错误**不会有任何功能异常**。

---

## 覆盖矩阵

| 性质 | 测试 |
|---|---|
| 输出完整 + 指标自洽 | 2、8 |
| **合批 == 单独跑** | 3 ★ |
| **串行 == 逐条跑** | 4 |
| **分块 == 一次算完** | 5 ★ |
| **前缀缓存 == 无缓存** | 6 ★ |
| **抢占重算 == 不抢占** | 7 ★ |

这就是「batching independence」的五个面。

---

## 用到的 Python / pytest 特性

| 特性 | 说明 |
|---|---|
| `@pytest.fixture(scope="module")` | 模块内共享引擎（只加载一次模型） |
| `try/finally` | 保证配置恢复 |
| `type(obj)()` | 同类型新建 |
| 断言带中文消息 | 失败时说明违反了哪条设计 |
| `**overrides` | 局部覆盖默认配置 |

---

## 设计取舍

**为什么每条测试都要新建 engine 而不是复用一个？**

因为这些测试改的是**引擎级配置**（`max_num_seqs`、`num_gpu_blocks`、`enable_prefix_caching`），而配置在构造时决定 `k_cache` 形状和 block manager 的行为。复用就得改内部状态，风险更大。

只有「不改配置」的 4 条（2、3、8）共用 module 级 fixture。**权衡点：加载模型 3 秒 vs 状态污染风险。**

**为什么用 `token_ids` 比较而不是 `text`？**

`text` 要经过 `tokenizer.decode(skip_special_tokens=True)`，可能有空白/Unicode 归一化的差异。`token_ids` 是引擎的**直接输出**，比较它才能定位到真正的差异。

**为什么不测采样（temperature > 0）？**

因为采样有随机性，无法断言「必须相同」。要测的话得用固定 `torch.Generator` 种子 + 固定调度顺序，而调度顺序本身受请求到达时间影响——太脆弱。**贪心是唯一能做严格断言的模式。**

---

## 常见误解

* 「这些测试证明性能好」——只证明**正确性**。性能看 benchmark。
* 「`num_preemptions > 0` 是废话断言」——不，它是防止「测试没走到目标路径」的守门员。
* 「`max_num_seqs=1` 是性能测试」——不是，是为了让前缀缓存有机会命中（请求串行到达）。
* 「抢占会损失精度」——不会，重算是精确重现。

---

## 动手练习

1. 把 `test_prefix_caching_gives_the_same_tokens` 的 `max_num_seqs=1` 去掉，观察 `num_prefix_hit_blocks` 变成 0——亲自复现「突发流量吃不到缓存收益」。
2. 把 `test_preemption_still_produces_correct_tokens` 的 `num_gpu_blocks=4` 改成 64，看第二个断言怎么挂（提示：没有抢占发生）。
3. 把 `COW` 逻辑删掉（`ensure_capacity` 里不做复制），跑 `test_prefix_caching_gives_the_same_tokens`，观察结果如何变错——这是「没有 COW 会静默出错」的直接证据。
4. 加一条测试：`ignore_eos=False` 时，三条请求的结束原因是 `"stopped"` 或 `"length_capped"`（都合法），验证 `_postprocess` 的判定优先级。
