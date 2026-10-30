# `tests/test_model_runner.py` 详解

## 这个文件干什么

**和 `transformers` 逐 token 对拍**：分页 KV cache 的实现，必须和 HF 的「无缓存完整前向」给出相同 logits。

这是整个项目**最有说服力**的一组测试——因为标准答案来自一个完全独立的、全世界在用的实现。

---

## 1. 三个辅助函数

```python
def make(cache: BlockManager, seq_id: int, ids: list[int]) -> Sequence:
    seq = Sequence(seq_id=seq_id, prompt_token_ids=ids, sampling_params=SamplingParams())
    cache.allocate(seq, len(seq.data))
    return seq
```

建序列 + 立刻分配页，省掉每个测试重复两行。

```python
def reference_logits(runner, ids: list[int]) -> torch.Tensor:
    with torch.no_grad():
        out = runner.hf_model(torch.tensor([ids], device=runner.device))
    return out.logits[0, -1].float()
```

**标准答案**：把整条序列一次喂给 HF，取最后一个位置的 logits。

特点：

* **不用 KV cache**（每次重算全部）→ 完全独立于我们的分页逻辑
* `torch.no_grad()` → 省显存
* `.float()` → 统一到 fp32 便于比较

**为什么可以这么"浪费"？** 因为测试里的序列最长几十个 token，重算全部对 CPU 也就几百毫秒。**用简单换确定性**是测试的正确取舍。

```python
def assert_close(actual, expected, label, atol):
    diff = (actual - expected).abs().max().item()
    assert torch.allclose(actual, expected, atol=atol), f"{label}: max |diff| = {diff:.3e}"
```

封装了「失败时打印最大误差」。这比裸 `assert allclose` 有用得多——**看到 `max |diff| = 2.030e+01` 就知道是逻辑错，看到 `1e-05` 就是浮点噪声**。

---

## 2. `test_prefill_matches_transformers`

```python
cache = BlockManager(runner.num_gpu_blocks, runner.cfg.block_size, enable_prefix_caching=False)
ids = runner.tokenize("The capital of France is")
seq = make(cache, 0, ids)

logits = runner.execute_model([seq], [len(ids)])
assert_close(logits[0], reference_logits(runner, ids), "prefill", atol=1e-4)
```

最简单的一条：一次 prefill 的 logits 必须等于 HF。

**这条测试只能证明「一次前向是对的」**，而一次前向里「先写 KV 再读 KV」发生在同一层内部，即使层维度写错（24 层共用一片）也看不出来。

所以它必要但不充分——真正的守门员是下一条。

---

## 3. `test_incremental_decode_matches_transformers` ★最关键

```python
logits = runner.execute_model([seq], [len(ids)])
seq.num_computed_tokens = len(ids)

for step in range(8):
    token = int(logits[0].argmax())
    seq.append_token(token)
    cache.ensure_capacity(seq, len(seq.data))
    logits = runner.execute_model([seq], [1])
    seq.num_computed_tokens += 1

    expected = reference_logits(runner, seq.data)
    assert_close(logits[0], expected, f"decode step {step}", atol=2e-3)
```

这条循环模拟了真实的生成过程：每步取 argmax、追加、再前向一次。

**它抓住的 bug：KV cache 少 layer 维。**

第一版实现里 `k_cache` 是 `[blocks, bs, kv, d]`，24 层共用同一片槽位：

* prefill 时：第 i 层写完立刻读，读到的是自己刚写的 → **结果正确**
* decode 时：第 i 层从槽位读 K，但槽位里是第 i+1 层刚写进去的 → **完全错**

表现为 `max |diff| = 20.297`（不是 1e-3 量级）。所以下面的 `atol` 才关键。

### `atol=2e-3` 为什么比 prefill 宽松 20 倍

因为 decode 是**误差累积**：8 步之后，每步的小误差都会影响下一步的输入（因为 token 是自回归生成的）。

但注意：**如果实现有逻辑错误，误差是 10 量级；如果只是浮点噪声，误差是 1e-3 量级。** 两者差三个数量级，所以阈值设在 2e-3 既能容忍噪声，又能抓住真 bug。

> 这个数量级判断很重要：**调阈值不是"让测试通过"，而是"在噪声和 bug 之间划线"**。线要画在两者中间。

`atol` 在 CPU/fp32 下本可以更小，但为了同一套断言也能在 `MINIVLLM_TEST_DEVICE=cuda`（fp16）下跑，取了 2e-3。

---

## 4. `test_chunked_prefill_matches_single_shot`

```python
logits_one_shot = runner.execute_model([seq], [len(ids)])[0]
assert_close(logits_one_shot, reference_logits(runner, ids), "one-shot prefill", atol=1e-4)

# 另起一个池子，分块 prefill
seq2 = make(cache2, 0, ids)
chunk = runner.cfg.block_size * 2
while seq2.num_computed_tokens < len(ids):
    n = min(chunk, len(ids) - seq2.num_computed_tokens)
    runner.execute_model([seq2], [n])
    seq2.num_computed_tokens += n

seq2.append_token(int(logits_one_shot.argmax()))
cache2.ensure_capacity(seq2, len(seq2.data))
logits_chunked = runner.execute_model([seq2], [1])[0]

assert_close(logits_chunked, reference_logits(runner, seq2.data), "chunked prefill", atol=1e-4)
```

设计上有个细节：**分块之后不能直接比「分块 prefill 的最后一块 logits」**，因为 `execute_model` 只在「刚好算完」的那次才产出有意义的 logits。

所以做法是：

1. 分块 prefill 完（此时所有 token 的 KV 都在缓存里，但没有可用的 logits）
2. 用 one-shot 那次的 argmax 作为下一个 token（保证两条路径输入一致）
3. 再跑一次 decode，拿它的 logits 和 HF 对拍

**这里比的是「分块写进去的 KV 是否正确」**，通过「后续 decode 的 logits」间接验证。这是最干净的验证路径。

`while seq2.num_computed_tokens < len(ids)` 这个循环自己就是对 `_chunk_size` 语义的一次非正式验证。

---

## 5. `test_ragged_batch_matches_individual_runs`

```python
prompts = ["Hello", "What is the difference between paging and swapping...?", "1 2 3"]
seqs = [make(cache, i, runner.tokenize(p)) for i, p in enumerate(prompts)]

batched = runner.execute_model(seqs, [len(s.data) for s in seqs])

for i, seq in enumerate(seqs):
    assert_close(batched[i], reference_logits(runner, seq.data), f"batch row {i}", atol=1e-4)
```

三条长度差异极大的 prompt 一起 prefill。**每一行都必须等于它单独跑的结果。**

它守的是 **padding 的正确性**：

* 三条长度不同（约 3 / 20 / 4 token）
* `_build_inputs` 会把它们首尾相接摊平，页表补 0
* 如果 gather 索引或 mask 算错，短序列会"看到"长序列的 KV

注意三条 prompt **长度差异大**是刻意选的——如果都是 3 个 token，padding 就不发生，测不出问题。

---

## 6. `test_prefill_and_decode_can_share_a_batch`

```python
warm = make(cache, 0, runner.tokenize("Once upon a time"))
runner.execute_model([warm], [len(warm.data)])
warm.num_computed_tokens = len(warm.data)
warm.append_token(int(reference_logits(runner, warm.data).argmax()))
cache.ensure_capacity(warm, len(warm.data))

fresh = make(cache, 1, runner.tokenize("Attention is all you need"))

out = runner.execute_model([warm, fresh], [1, len(fresh.data)])
```

**这条测试的存在就是为了证明「静态批处理做不到的事，连续批处理能做到」。**

同一批里：序列 0 在 decode（1 个 token），序列 1 在 prefill（5 个 token）。这种「混合批」正是连续批处理的核心能力，也是 `_build_inputs` 里那套 ragged 逻辑存在的理由。

如果实现假设「一个 batch 里所有序列的 `query_len` 相同」，这条测试立刻失败。

---

## 7. `test_prefix_cache_reuses_kv_and_keeps_logits_identical`

```python
ids = runner.tokenize("You are a helpful assistant that explains memory management ... (long)")
assert len(ids) > 2 * runner.cfg.block_size, "prompt 必须跨至少两个满块"

first = make(cache, 0, ids)
reference = runner.execute_model([first], [len(ids)])[0]
first.num_computed_tokens = len(ids)
cache.register_computed_blocks(first)
cache.free(first)

second = Sequence(seq_id=1, prompt_token_ids=list(ids), sampling_params=SamplingParams())
hits = cache.allocate(second, len(second.data))
assert hits > 0, "expected the prefix cache to serve part of this prompt"

logits = runner.execute_model([second], [len(ids)])[0]
assert_close(logits, reference, "prefix-cache replay", atol=1e-4)
```

它验证一个强性质：**用共享页重放一遍完整 prefill，结果必须和第一次完全一样。**

这意味着「共享的 KV 内容确实是正确的」——如果 COW 或 hash 逻辑有 bug（比如共享了不该共享的页），logits 会偏。

### `assert len(ids) > 2 * block_size`

这条前置断言是**修 bug 修出来的**。

第一版 prompt 只有 27 个 token，`block_size=16` → 只有 **1 个满块**，前缀缓存能登记的太少，`hits` 可能为 0，测试直接挂。

一开始我以为是缓存逻辑坏了，排查半天才发现是「prompt 太短」。现在把它写成断言，**下次失败时会直接告诉你「是测试数据不够长」而不是「实现有问题」**。

> 这是写测试的一个通用技巧：**把隐含前提写成断言**，让失败信息指向真正的原因。

---

## 8. `test_block_size_does_not_change_results`（参数化 3 次）

```python
@pytest.mark.parametrize("block_size", [1, 8, 32])
def test_block_size_does_not_change_results(runner, block_size):
    cfg = EngineConfig(model=runner.cfg.model, dtype=runner.cfg.dtype,
                       device=runner.cfg.device, block_size=block_size, num_gpu_blocks=256)
    other = ModelRunner(cfg)
    ids = other.tokenize("Does block size change the math?")
    cache = BlockManager(other.num_gpu_blocks, block_size, enable_prefix_caching=False)
    seq = make(cache, 0, ids)
    logits = other.execute_model([seq], [len(ids)])[0]
    assert_close(logits, reference_logits(other, ids), f"block_size={block_size}", atol=1e-4)
```

**这条测试回答一个必然会被问的问题**：「block_size 该选多少？」

答案：**只影响性能，不影响正确性。** 1（每块 1 token，页表最长）、8、32（每块 32 token，内部碎片最大）都给出相同结果。

所以 `block_size` 的选择是纯工程权衡：

| block_size | 内部碎片 | 页表长度 | gather 索引 |
|---|---|---|---|
| 1 | 0 | 最长 | 最大 |
| 16 | 小 | 中 | 中 |
| 32 | 大 | 短 | 小 |

### 为什么这里要新建一个 `ModelRunner`

因为 `block_size` 是**构造时决定**的（影响 `k_cache` 的形状）。不能改现有 `runner` 的。

代价是每个参数都要重新加载一次模型（CPU 上约 3 秒 × 3 = 9 秒）。这是这条测试比较慢的原因。

`num_gpu_blocks=256` 而不是默认的 64：`block_size=32` 时，64 块只有 2048 个 slot，够用；但给 256 更保险（28 token 的 prompt 在 block_size=1 时要 28 块）。

---

## 覆盖矩阵

| 属性 | 测试 |
|---|---|
| 单次 prefill 正确 | 2 |
| **增量 decode 正确** | 3 ★ |
| 分块 prefill 的 KV 正确 | 4 |
| ragged batch 的 padding 正确 | 5 |
| prefill + decode 混批 | 6 |
| 前缀缓存重放正确 | 7 |
| block_size 只影响性能 | 8 |

**没覆盖**：fp16 下的一致性（需要 `MINIVLLM_TEST_DEVICE=cuda`）、GQA 的数值（在 `test_paged_attention` 里）、超长序列。

---

## 用到的 PyTorch / pytest 特性

| 特性 | 说明 |
|---|---|
| `torch.no_grad()` | 参考前向不建图 |
| `torch.allclose(a, b, atol=)` | 近似相等 |
| `(a - b).abs().max().item()` | 最大绝对误差（会同步，但测试里无所谓） |
| `logits.argmax()` | 贪心取 token |
| `@pytest.mark.parametrize` | 一次写三种 block_size |
| `with pytest.skip`（在 fixture 里） | 没模型就跳过 |

---

## 设计取舍

**为什么用 HF 的「无缓存全量前向」当 oracle，而不是 HF 的「带 cache 增量生成」？**

因为全量前向**完全独立**于任何 KV cache 逻辑。用带 cache 的版本对比，等于两个都可能错在同一个地方。

**为什么每条测试自己建 `BlockManager` 而不是共享？**

因为块表、引用计数、缓存都是**有状态**的。共享会导致测试之间相互污染（前一条测试的页表影响后一条）。`runner`（无状态的权重）共享，`BlockManager`（有状态的元数据）独立 —— 这是正确的粒度划分。

**为什么阈值不统一？**

| 测试 | 阈值 | 理由 |
|---|---|---|
| prefill | 1e-4 | 单次前向，无累积 |
| decode 8 步 | 2e-3 | 自回归累积 |
| 前缀缓存重放 | 1e-4 | 单次前向（虽然数据来自缓存，但计算量相同） |

---

## 常见误解

* 「atol 1e-4 说明实现有误差」——不，fp32 下两个**数学等价**的实现也会差 1e-6 量级（求和顺序不同）。
* 「prefill 通过就够了」——不够。**层维度的 bug 只在 decode 暴露**。
* 「block_size 会影响结果精度」——不影响，只是把更多的 token 装进同一页。
* 「前缀缓存重放必须 new 一条序列才算测」——是的，必须走「allocate 时命中缓存」的路径，不能用原来的 `seq`。

---

## 动手练习

1. 把 `k_cache` 的形状改回 4 维（去掉 layer 维），跑这条文件，记录 `test_incremental_decode` 的 `max |diff|`——你会看到 2e+01 而不是 1e-03。
2. 把 `test_incremental_decode` 的 `atol` 改成 1e-1，验证它会"通过"——这说明**阈值过松会让测试失去意义**。
3. 把 `test_ragged_batch` 的三条 prompt 都改成 3 个 token 的短句，确认测试仍然通过（因为 padding 不发生），从而理解为什么原来的长句是必要的。
4. 加一条测试：`MINIVLLM_TEST_DEVICE=cuda` 下跑同一套（用 `pytest.mark.skipif` 控制），观察 fp16 的误差量级。
