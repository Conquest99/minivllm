# `tests/test_paged_attention.py` 详解

## 这个文件干什么

**证明「按页表取 KV」和「连续 KV」算出来的 attention 完全一样。**

这是整个项目最重要的一条测试。如果它成立，那么所有分页的复杂度（页表、dummy 页、padding、COW）都只是**性能/内存的取舍**，不会改变数学结果。

---

## 1. 参照实现：`dense_attention`

```python
def dense_attention(q, k, v, scale):
    """参考实现：q [Lq, H, D]，k/v [Lk, Hkv, D]，causal。"""
    lq, heads, _ = q.shape
    lk = k.shape[0]
    reps = heads // k.shape[1]
    k = k.repeat_interleave(reps, dim=1).transpose(0, 1)    # [H, Lk, D]
    v = v.repeat_interleave(reps, dim=1).transpose(0, 1)
    scores = (q.transpose(0, 1) @ k.transpose(1, 2)) * scale # [H, Lq, Lk]
    offset = lk - lq
    mask = torch.arange(lk)[None, :] <= torch.arange(lq)[:, None] + offset
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v).transpose(0, 1).reshape(lq, heads * q.shape[-1])
```

这是**故意写得很笨**的版本：显式矩阵乘、显式 mask、显式 softmax，**不用 SDPA**。

为什么？因为「用 SDPA 当参考实现」等于「用同一个函数验证自己」——万一 SDPA 的 mask 语义理解错了，两边会一起错。

这个版本只依赖最基础的线性代数：

```
scores[i,j] = q_i · k_j * scale          → q.transpose @ k.transpose
mask        = 只能看 j ≤ i + offset
probs       = softmax(scores)
out[i]      = Σ_j probs[i,j] * v_j
```

### `offset = lk - lq` 是什么

当 query 只覆盖 KV 序列的**尾部**时（chunked prefill 的情形），query 的位置 `i` 对应的真实位置是 `i + (lk - lq)`。

比如 `lk=11, lq=5`：query 覆盖位置 6..10，所以 `offset=6`，`mask[i,j] = j <= i + 6`。

### 为什么这里可以用 `repeat_interleave`

因为这是**参照实现**，每秒只跑几次，清晰比性能重要。被测代码才需要 `enable_gqa`。

---

## 2. 测试工具类：`PagedCache`

```python
class PagedCache:
    def __init__(self, num_blocks, block_size, kv_heads, head_dim, seed=0):
        torch.manual_seed(seed)
        self.k_cache = torch.zeros(num_blocks + 1, block_size, kv_heads, head_dim)
        self.v_cache = torch.zeros(num_blocks + 1, block_size, kv_heads, head_dim)
        self.dummy_slot = num_blocks * block_size
        self.seqs = []

    def add(self, block_ids, k, v):
        n = k.shape[0]
        slots = [block_ids[t // self.block_size] * self.block_size + t % self.block_size
                 for t in range(n)]
        write_kv_cache(self.k_cache, self.v_cache, torch.tensor(slots), k, v)
        self.seqs.append({"blocks": block_ids, "k": k, "v": v, "len": n})
```

它做了两件事：

1. **用真实的 `write_kv_cache` 写 KV**，并**同时保留原始 K/V**（`self.seqs` 里的 `k`/`v`）——这份就是「标准答案」。
2. **允许指定物理块号**（`block_ids`），所以可以故意用 `[3, 5]` 这种**非连续、非单调**的块，正是分页要处理的情况。

`num_blocks + 1` 多出来的那块是 dummy 页，和真实实现一致。

`torch.manual_seed(seed)` 在 `__init__` 里设，让 `random_kv` 每次得到相同数据——失败时可复现。

```python
def block_tables(self):
    width = max(len(s["blocks"]) for s in self.seqs)
    rows = [s["blocks"] + [0] * (width - len(s["blocks"])) for s in self.seqs]
    return torch.tensor(rows, dtype=torch.long)

def kv_lens(self):
    return torch.tensor([s["len"] for s in self.seqs], dtype=torch.long)
```

这正是 `ModelRunner._build_inputs` 里那段 padding 逻辑的**测试版**：不同长度的页表补 0 到等长。

---

## 3. 测试 1：写进去再读出来

```python
def test_write_and_read_roundtrip():
    cache = PagedCache(num_blocks=8, block_size=4, kv_heads=2, head_dim=4)
    k, v = random_kv(6, 2, 4)
    cache.add([3, 5], k, v)          # 故意用非单调的物理块

    tables = torch.tensor([[3, 5]])
    kv_lens = torch.tensor([6])
    idx, valid = build_gather_index(tables, kv_lens, block_size=4, dummy_slot=cache.dummy_slot)
    gathered_k = cache.k_cache.view(-1, 2, 4).index_select(0, idx.reshape(-1))[:6]

    assert torch.allclose(gathered_k, k, atol=1e-6)
    assert valid[0, :6].all() and not valid[0, 6:].any()
```

守的是**最底层的双向映射**：

```
write(slot_mapping) → 散点写
build_gather_index  → 从页表反推 slot
index_select        → 按 slot 取回
结果必须等于原始 k
```

`[:6]` 是关键：`idx` 展平后有 `max_len = 6` 个元素（因为 `kv_lens.max() = 6`），取前 6 个正好是这条序列的有效部分。

> 第一版这里写成 `[0, :6]`（多了一个维度），实际 `index_select` 返回的是 1 维 `[6, kv, d]`。测试代码写错和产品代码写错一样常见。

---

## 4. 测试 2：乱序 batch 的 decode

```python
def test_decode_matches_dense_for_ragged_batch():
    lens = [6, 3, 11]
    blocks = [[0, 1], [2], [3, 4, 5]]
    for n, b in zip(lens, blocks):
        k, v = random_kv(n, kv_heads, head_dim)
        cache.add(b, k, v)

    q = torch.randn(len(lens), heads, head_dim)
    out = paged_attention_decode(q, cache.k_cache, cache.v_cache,
                                 cache.block_tables(), cache.kv_lens(),
                                 scale=head_dim**-0.5, dummy_slot=cache.dummy_slot)

    for i, seq in enumerate(cache.seqs):
        expected = dense_attention(q[i:i+1], seq["k"], seq["v"], scale)
        assert torch.allclose(out[i], expected[0], atol=1e-5)
```

三个刻意设计的难点：

1. **长度不同**（6/3/11）→ 必须 padding，短序列要正确 mask 掉 padding 部分
2. **物理块不连续**（0,1 / 2 / 3,4,5）→ 必须真的走页表翻译
3. **11 不是 4 的倍数** → 最后一页只有 3 个有效 slot

只要有任何一处错了（比如 mask 写反、slot 算错、padding 读到别人的数据），`allclose` 立刻失败。

`atol=1e-5` 是给 float32 累加误差的余量。

---

## 5. 测试 3：prefill 的因果 mask 位置

```python
def test_prefill_matches_dense():
    layouts = [(11, [0, 1, 2]), (5, [3, 4])]
    for n, blocks in layouts:
        k, v = random_kv(n, kv_heads, head_dim)
        cache.add(blocks, k, v)

    q_lens = [5, 2]         # 这两条序列本步分别要「再算」5 个和 2 个 token
    q = torch.randn(sum(q_lens), heads, head_dim)
    out = paged_attention_prefill(q, ..., q_lens, cache.kv_lens(), scale)

    for seq, q_len in zip(cache.seqs, q_lens):
        expected = dense_attention(q[offset:offset+q_len], seq["k"], seq["v"], scale)
        assert torch.allclose(...)
```

测试的语义是 **chunked prefill**：KV 里已经有 11 个 token，这一 step 再算 5 个。

关键点：`dense_attention` 内部用 `offset = lk - lq` 自动把 query 放到尾部。如果被测代码把 `q_pos` 写成从 0 开始（即 `arange(q_len)`），那它认为 query 在位置 0..4，而参考实现认为在 6..10 → **mask 位置整体错位**，结果必然不等。

**这条测试就是专门为那个 bug 写的。** 它之所以重要，是因为这个 bug 在真实模型上表现为「长 prompt 的输出变成胡言乱语」，非常难定位。

---

## 6. 测试 4：`copy_blocks` 真的复制了

```python
def test_copy_blocks_duplicates_kv():
    k, v = random_kv(4, 2, 4)
    cache.add([1], k, v)
    copy_blocks(cache.k_cache, cache.v_cache, [(1, 6)], block_size=4)

    assert torch.allclose(cache.k_cache[6], cache.k_cache[1])
    assert torch.allclose(cache.v_cache[6], cache.v_cache[1])
    cache.k_cache[1].zero_()
    assert not torch.allclose(cache.k_cache[6], cache.k_cache[1])
```

最后两行是精髓：**清空源块，目标块必须不受影响**。

如果 `copy_blocks` 只是「让两个索引指向同一块」（比如误用了 view/别名），清零操作会同时改变两边，`not allclose` 就失败。

这就是「复制」和「共享」的区别——而 COW 的正确性完全依赖「真的复制了」。

> `zero_()` 是**原地**操作（方法名以下划线结尾是 PyTorch 的原地约定）。`zeros()` 是新建张量。

---

## 7. 测试 5：GQA 参数化

```python
@pytest.mark.parametrize("num_heads,kv_heads", [(4, 4), (4, 1), (8, 2)])
def test_gqa_head_expansion(num_heads, kv_heads):
    ...
    expected = dense_attention(q, k, v, scale)
    assert torch.allclose(out, expected, atol=1e-5)
```

三种组合覆盖三种情况：

| 组合 | 情况 | 走的分支 |
|---|---|---|
| 4/4 | MHA（相等） | `k.shape[-3] == q.shape[-3]` → 直接 SDPA |
| 4/1 | MQA（1 个 KV head） | `enable_gqa=True` |
| 8/2 | GQA（Qwen 的形状） | `enable_gqa=True` |

`@pytest.mark.parametrize` 会把这个函数**执行三遍**，每遍注入不同的参数值。测试报告里显示为 `test_gqa_head_expansion[4-1]` 等。

**为什么必须三种都测**：`_sdpa` 里有三个分支（相等 / 支持 GQA / 回退到 `repeat_interleave`），不参数化就只会覆盖第一条。

> 如果跑在不支持 `enable_gqa` 的 torch 上，4/1 和 8/2 会走 `repeat_interleave` 分支——同一套断言仍然成立，这正是「特征探测」的价值。

---

## 覆盖了什么、没覆盖什么

| 覆盖 | 测试 |
|---|---|
| 写/读 slot 映射 | 1 |
| ragged decode + 非连续块 | 2 |
| chunked prefill 的 mask 位置 | 3 |
| COW 复制是真复制 | 4 |
| MHA / MQA / GQA 三态 | 5 |

**没覆盖**：空 batch、`kv_len = 0`、`block_size = 1`、bf16 精度。

---

## 用到的 PyTorch / pytest 特性

| 特性 | 说明 |
|---|---|
| `torch.allclose(a, b, atol=)` | 逐元素近似相等 |
| `torch.manual_seed` | 可复现随机 |
| `x.transpose(0, 1)` / `@` | 矩阵乘两种写法 |
| `masked_fill(~mask, -inf)` | 参考实现里的屏蔽 |
| `x.zero_()` | 原地清零 |
| `index_select(0, idx)` | 按下标取 |
| `@pytest.mark.parametrize` | 参数化用例 |
| `zip(a, b, strict=False)` | 并行遍历 |

---

## 设计取舍

**为什么参考实现不直接用 `F.scaled_dot_product_attention`？**

同源验证没有意义。用最朴素的 `@` + `softmax` 重写一遍，等于用「另一条独立思路」交叉检验。**测试的价值来自独立性，不来自简洁。**

**为什么把 `PagedCache` 写成辅助类而不是每个测试重复建？**

三个测试都需要「建池子 + 写 KV + 保留标准答案 + 造页表」这套动作。抽出来之后，每个测试只剩「断言」部分，读起来更聚焦。

**为什么块号故意不连续（`[3, 5]`）？**

因为连续块是**退化情况**——即使页表映射写错了（比如直接用 `i * bs + offset`），块号连续时也能得到正确结果。只有乱序才能真正验证「走了页表」。

---

## 常见误解

* 「这些测试通过说明性能好」——只说明**正确性**。性能要看 benchmark。
* 「`atol=1e-5` 可以随便改大」——改大到 1e-2 时，`q_pos` 错位的 bug 也能"通过"，测试就失效了。
* 「decode 不需要 mask」——需要 `valid`（长度）mask；只是不需要 causal mask。
* 「`copy_blocks` 只是换指针」——不是，是真的 memcpy，测试 4 专门守这一点。

---

## 动手练习

1. 把 `paged_attention_prefill` 里的 `q_pos` 改成 `torch.arange(q_len)`，跑测试 3，记下 `max |diff|` 的量级。
2. 把 `paged_attention_decode` 的 `mask` 换成 `None`，跑测试 2，观察短序列（长度 3）的结果怎么被污染。
3. 把 `build_gather_index` 里 `torch.where(valid, slots, dummy_slot)` 的 `dummy_slot` 改成 `0`，看测试是否仍然通过（因为 mask 还在）——进而理解双保险的意义。
4. 给 `test_copy_blocks_duplicates_kv` 加一个多块、多层的用例（用 5 维 cache），验证 `num_slots` 那段循环的正确性。
