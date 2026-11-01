# `minivllm/paged_attention.py` 详解 ★核心

## 这个文件干什么

把「散落在池子里的 KV」变成 attention 能用的连续张量，然后算 attention。

如果 [block_manager.py](block_manager.md) 是「决定哪几页归谁」，这个文件就是「**真的按那几页去读写显存**」。

五个函数，职责清晰：

| 函数 | 一句话 | 何时调用 |
|---|---|---|
| `write_kv_cache` | 把本 step 新算的 K/V 散点写进池子 | 每层、每 step |
| `copy_blocks` | 执行 COW：整页复制 | 仅在调度器给出 `cow_ops` 时 |
| `build_gather_index` | 页表 → 线性 slot 索引（含 padding） | 每次 decode |
| `gather_kv` | 按索引把 K/V 取成连续张量 | 每次 decode |
| `paged_attention_decode` / `_prefill` | 真正的 attention | 每层 |

---

## 1. 物理布局：一个 slot 公式搞定一切

```python
k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
```

Qwen2.5-0.5B 具体数字：`[1024, 16, 2, 64]`，fp16。

**第 `b` 块、第 `o` 个槽位、第 `h` 个 head 的 K 向量**，存储位置就是：

```
k_cache[b, o, h, :]
```

把它摊平成一维看（`k_cache.view(-1, kv_heads, head_dim)`），第 `b` 块第 `o` 槽对应的行号是：

```
slot = b * block_size + o
```

**这一个公式是整个文件的地基。** 页表做的唯一一件事就是：给定逻辑块号 `i`，查出物理块号 `block_table[i]`，于是逻辑第 `t` 个 token 的 slot 是：

```
slot(t) = block_table[t // block_size] * block_size + (t % block_size)
```

代码里就这一行（`model_runner.py::_build_inputs`）：

```python
slots.append(seq.block_table[t // bs] * bs + (t % bs))
```

> 举例：`block_table = [7, 2, 9]`，`bs = 16`。逻辑第 20 个 token（`20 // 16 = 1`，`20 % 16 = 4`）→ 物理块 2 的第 4 个槽 → `slot = 2*16 + 4 = 36`。

### 为什么不是「逻辑块号连续 → 物理也连续」

这正是分页的意义：物理上不连续，代价就是每次读都要经过页表翻译。

---

## 2. `write_kv_cache`：散点写入

```python
def write_kv_cache(k_cache, v_cache, slot_mapping, key, value) -> None:
    flat_k = k_cache.view(-1, *k_cache.shape[2:])
    flat_v = v_cache.view(-1, *v_cache.shape[2:])
    flat_k[slot_mapping] = key
    flat_v[slot_mapping] = value
```

调用方（`model_runner.py`）：

```python
write_kv_cache(self.k_cache[layer_idx], self.v_cache[layer_idx], inp["slot_mapping"], k, v)
```

* 传进来的是**单层**的 cache，形状 `[blocks, bs, kv, d]`。
* `view(-1, kv, d)` 把它变成 `[blocks*bs, kv, d]`，即「所有 slot 排成一行」。
* `flat_k[slot_mapping] = key`：`slot_mapping` 是 `[num_tokens]` 的长整型张量，`key` 是 `[num_tokens, kv, d]`。这是**高级索引赋值**，逐行 scatter。
* 因为 `flat_k` 是 `view`（共享内存），赋值直接落到 `k_cache` 上，**不需要赋回去**。

### 为什么 `slot_mapping` 是乱序的

一个 step 里可能同时有：
- 序列 A 的 16 个 prefill token（slot 连续）
- 序列 B 的 1 个 decode token（slot 在另一个块）
- 序列 C 的 1 个 decode token（slot 又在别处）

所以 `slot_mapping` 长这样：`[0,1,...,15, 48, 96]`。**写操作天然是 scatter，读操作天然是 gather**，这就是分页 attention 的两个基本动作。

### 一个易错点

`slot_mapping` 里不能有重复值。语义上「同一位置写两次」是未定义的（取决于硬件实现）。本项目不会发生，因为一条序列在一个 step 里每个位置只算一次；但如果你自己动手改代码，务必保持这个不变量。

---

## 3. `copy_blocks`：COW 的显存侧

```python
def copy_blocks(k_cache, v_cache, mapping, block_size) -> None:
    if not mapping:
        return
    device = k_cache.device
    src = torch.tensor([s for s, _ in mapping], device=device)
    dst = torch.tensor([d for _, d in mapping], device=device)
    offsets = torch.arange(block_size, device=device)
    src_slots = (src[:, None] * block_size + offsets[None, :]).reshape(-1)
    dst_slots = (dst[:, None] * block_size + offsets[None, :]).reshape(-1)

    num_slots = 1
    for dim in k_cache.shape[:-2]:
        num_slots *= dim
    flat_k = k_cache.reshape(-1, num_slots, *k_cache.shape[-2:])
    flat_v = v_cache.reshape(-1, num_slots, *v_cache.shape[-2:])
    flat_k[:, dst_slots] = flat_k[:, src_slots]
    flat_v[:, dst_slots] = flat_v[:, src_slots]
```

### `src_slots` 是怎么算的

假设 `block_size = 4`，要复制 `src=5 → dst=9`：

```
src[:, None]            → [[5]]
offsets[None, :]        → [[0, 1, 2, 3]]
相加                     → [[5, 6, 7, 8]]
reshape(-1)             → [5, 6, 7, 8]      ← 4 个 slot
```

即「第 5 块占用的 4 个连续 slot」。多对映射就是多行拼起来。

### `num_slots` 那段为什么写成循环

```python
num_slots = 1
for dim in k_cache.shape[:-2]:
    num_slots *= dim
```

因为它要同时支持两种布局：

| 传入形状 | `shape[:-2]` | `num_slots` | 结果 |
|---|---|---|---|
| `[blocks, bs, kv, d]` | `(blocks,)` | `blocks` | `[1, blocks, kv, d]` |
| `[layers, blocks, bs, kv, d]` | `(layers, blocks)` | `layers*blocks` | `[1, layers*blocks, kv, d]` |

这样 `flat_k[:, dst_slots] = flat_k[:, src_slots]` 就同时覆盖「所有层、所有目标块」——**每层的复制完全一样，只是 slot 号不同**，所以一次广播搞定。

> 为什么不用 `math.prod`？因为它 3.8+ 才有……其实够用。写循环只是照顾可读性（`prod` 这个名字在「元素个数」语境下不如直接乘清楚）。

### 那句注释在说什么

```python
# 右手边的花式索引已经物化出一份拷贝，所以即使 src 和 dst 重叠，原地 scatter 也是安全的
flat_k[:, dst_slots] = flat_k[:, src_slots]
```

如果 `src_slots` 和 `dst_slots` 有交集（理论上不该有，但防御一下），Python 语义保证**先算完右边再赋值**。如果换成 `index_copy_` 之类的原地接口，重叠就可能读到已被覆盖的数据。

---

## 4. `build_gather_index`：页表 → 索引

decode 时 batch 里每条序列长度不同（比如 `[6, 3, 11]`），但张量必须是矩形。怎么办？

```python
def build_gather_index(block_tables, kv_lens, block_size, dummy_slot):
    batch, _ = block_tables.shape
    max_len = int(kv_lens.max().item())
    offsets = torch.arange(block_size, device=block_tables.device)
    slots = (block_tables[:, :, None] * block_size + offsets[None, None, :]).reshape(batch, -1)
    slots = slots[:, :max_len]
    valid = torch.arange(max_len, device=block_tables.device)[None, :] < kv_lens[:, None]
    return torch.where(valid, slots, dummy_slot), valid
```

### 第一步：所有块展开成 slot

`block_tables` 是 `[B, max_blocks]`，比如 `[[0,1], [2,0], [3,4,5]]` 补齐成 `[[0,1,0],[2,0,0],[3,4,5]]`。

```python
(block_tables[:, :, None] * block_size + offsets[None, None, :])
```

* `block_tables[:, :, None]` → `[B, max_blocks, 1]`
* `offsets[None, None, :]` → `[1, 1, block_size]`
* 广播相加 → `[B, max_blocks, block_size]`
* `.reshape(batch, -1)` → `[B, max_blocks * block_size]`

直观理解：每个「块号」被替换成「这一块的所有 slot 号」。`3`（bs=4）变成 `[12, 13, 14, 15]`。

### 第二步：截到最长长度

`max_len = kv_lens.max()`。比最长序列还长的部分不需要，直接切掉，**省一次无用的 gather**。

> 这里有个 `.item()`（GPU→CPU 同步）。它每层都会被调用一次，但因为值在 `_build_inputs` 里就算过、且只有 1 个标量，开销可以忽略。真要抠，可以把这个值算进 Python 侧并传进来——就是我们在 `query_lens` 上做的事。

### 第三步：有效性 mask

```python
valid = torch.arange(max_len)[None, :] < kv_lens[:, None]
#        [1, max_len]              <  [B, 1]      →  [B, max_len]
```

序列 0 长度 6、`max_len = 11` 时：`valid[0] = [T,T,T,T,T,T,F,F,F,F,F]`。

### 第四步：padding 指向 dummy 页

```python
torch.where(valid, slots, dummy_slot)
```

无效位置的 slot 被换成 `dummy_slot`，这是池子末尾**额外分配的一页**，永远全 0，永不被写。

**为什么需要它？** 两个理由：

1. `index_select` 需要一个合法下标。如果填 0，就会读到真实数据（可能属于别的序列）。
2. 即使读到 0，attention 也算错了——所以还要 `valid` mask 把它屏蔽掉。**索引合法 + mask 屏蔽，双保险**。

`dummy_slot` 的定义（`model_runner.py`）：

```python
shape = (self.num_layers, self.num_gpu_blocks + 1, bs, nkv, hd)   # 注意 +1
self.dummy_slot = self.num_gpu_blocks * bs                         # 额外那页的第一个 slot
```

---

## 5. `gather_kv`：一次 `index_select` 拿到整个 batch

```python
def gather_kv(k_cache, v_cache, gather_index):
    head_dim = k_cache.shape[-1]
    kv_heads = k_cache.shape[-2]
    flat_k = k_cache.view(-1, kv_heads, head_dim)
    flat_v = v_cache.view(-1, kv_heads, head_dim)
    idx = gather_index.reshape(-1)
    batch, max_len = gather_index.shape
    k = flat_k.index_select(0, idx).view(batch, max_len, kv_heads, head_dim)
    v = flat_v.index_select(0, idx).view(batch, max_len, kv_heads, head_dim)
    return k, v
```

流程：

```
gather_index [B, max_len]  →  reshape(-1)  →  [B*max_len]
        ↓ index_select（一次 kernel，取 B*max_len 行）
flat_k         [slots, kv, d]
        ↓
              [B*max_len, kv, d]
        ↓ view
              [B, max_len, kv, d]
```

**关键点：整个 batch 只调一次 `index_select`**，而不是「每条序列一次」或「每个块一次」。这是这个实现在不写 kernel 的前提下能做到的最好情况。

### 代价（必须诚实地说）

`index_select` 会**真的复制** `B × max_len × kv_heads × head_dim` 个元素。

举例：B=8，max_len=2000，kv=2，d=64，fp16：

```
8 × 2000 × 2 × 64 × 2 B = 4.1 MB / 层 / step
× 24 层 = 98 MB / step
```

这些数据本来在显存里躺着，现在被读一遍再写一遍。**这就是「不写 kernel」的账单**，也是本项目在长上下文下不可能追上 vLLM 的根本原因。

> 面试标准答法：「vLLM 的 kernel 在算 attention 时按 `block_table` 现算物理地址，不需要连续内存；我这里用 `index_select` 先物化一份连续 KV。功能等价、数值一致，代价是多一次全量拷贝，序列越长越明显。」

---

## 6. `_sdpa`：GQA 的正确打开方式

```python
def _detect_gqa_support() -> bool:
    try:
        q = torch.zeros(1, 2, 1, 4)
        kv = torch.zeros(1, 1, 1, 4)
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
    except TypeError:
        return False
    return True

_SDPA_SUPPORTS_GQA = _detect_gqa_support()

def _sdpa(q, k, v, mask, scale):
    if k.shape[-3] == q.shape[-3]:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    if _SDPA_SUPPORTS_GQA:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale, enable_gqa=True)
    reps = q.shape[-3] // k.shape[-3]
    return F.scaled_dot_product_attention(
        q, k.repeat_interleave(reps, dim=-3), v.repeat_interleave(reps, dim=-3),
        attn_mask=mask, scale=scale,
    )
```

### 为什么用「特性探测」而不是版本号判断

第一版我写的是 `inspect.signature(F.scaled_dot_product_attention)`，结果直接崩：

```
ValueError: no signature found for builtin <built-in function scaled_dot_product_attention>
```

因为 SDPA 是 **C 实现的内建函数**，Python 拿不到它的签名。改成「试着调一次，捕获 TypeError」——这也叫 duck typing 的实用版：**问「你能做吗」比问「你是谁」可靠**。

### 为什么 `repeat_interleave` 是坏主意

`[.., 2, L, D]` → `[.., 14, L, D]`，L=2000 时每个 step 每层多出：

```
12 × 2000 × 64 × 2 B = 3 MB 的额外读写
```

而且这 3 MB 里 6/7 是重复数据。`enable_gqa=True` 让内核内部按组广播，**一个字节都不多占**。

实际效果：改之前 batch=16 时 63.4 ms/step，改之后 48.6 ms/step。

---

## 7. `paged_attention_decode`：整个 batch 一个 query token

```python
def paged_attention_decode(query, k_cache, v_cache, block_tables, kv_lens, scale, dummy_slot):
    batch, num_heads, head_dim = query.shape
    gather_index, valid = build_gather_index(block_tables, kv_lens, k_cache.shape[-3], dummy_slot)
    k, v = gather_kv(k_cache, v_cache, gather_index)          # [B, L, kv, d]

    q = query.unsqueeze(2)                                     # [B, H, 1, d]
    mask = valid[:, None, None, :]                             # [B, 1, 1, L]
    out = _sdpa(q, k.transpose(1, 2), v.transpose(1, 2), mask, scale)
    return out.transpose(1, 2).reshape(batch, num_heads * head_dim)
```

形状演进（B=8, H=14, kv=2, d=64, L=2000）：

```
query                 [8, 14, 64]
unsqueeze(2)          [8, 14, 1, 64]      ← 1 个 query
k (gather)            [8, 2000, 2, 64]
transpose(1,2)        [8, 2, 2000, 64]    ← head 在倒数第三维（SDPA 要求）
mask                  [8, 1, 1, 2000]     ← 广播到 [8, 14, 1, 2000]
─────────────────────────────────────────
SDPA 输出             [8, 14, 1, 64]
transpose(1,2)        [8, 1, 14, 64]
reshape               [8, 896]            ← 摊平成 hidden
```

### 为什么 decode 不需要 causal mask

因为唯一的那个 query 位于序列**末尾**，它可以看全部历史，不存在「未来」。所以只需要 `valid`（长度 padding 屏蔽）。

**这是 decode 和 prefill 最大的一处差异**，也是最容易写错的地方——如果 decode 也加三角 mask，所有序列的第 0 个 token 都会看不到任何内容。

### `k_cache.shape[-3]` 为什么是 block_size

传入的 `k_cache` 是**单层**的 `[blocks+1, bs, kv, d]`，所以 `-3` 正好是 `bs`。

> 这个用法很脆：如果哪天有人把 5 维的完整 cache 传进来，`-3` 就变成 `bs` 之外的维度，直接算错（而且是静默算错）。重构这类代码时务必看形状注释。真要稳妥，应该把它作为显式参数传。

---

## 8. `paged_attention_prefill`：按序列循环

```python
def paged_attention_prefill(query, k_cache, v_cache, block_tables, query_lens, kv_lens, scale):
    num_heads, head_dim = query.shape[1], query.shape[2]
    block_size = k_cache.shape[-3]
    flat_k = k_cache.reshape(-1, *k_cache.shape[-2:])
    flat_v = v_cache.reshape(-1, *v_cache.shape[-2:])
    kv_lens = kv_lens.tolist()

    outputs = []
    q_offset = 0
    for i in range(block_tables.shape[0]):
        q_len = query_lens[i]
        kv_len = kv_lens[i]
        q_i = query[q_offset : q_offset + q_len]
        q_offset += q_len

        n_blocks = (kv_len + block_size - 1) // block_size
        offsets = torch.arange(block_size, device=block_tables.device)
        flat_idx = (block_tables[i, :n_blocks, None] * block_size + offsets[None, :]).reshape(-1)[:kv_len]
        k, v = _gather(flat_k, flat_v, flat_idx)

        q_pos = torch.arange(kv_len - q_len, kv_len, device=query.device)[:, None]
        kv_pos = torch.arange(kv_len, device=query.device)[None, :]
        mask = kv_pos <= q_pos

        out = _sdpa(q_i.unsqueeze(0).transpose(1, 2), k.transpose(0, 1).unsqueeze(0),
                    v.transpose(0, 1).unsqueeze(0), mask, scale)
        outputs.append(out.transpose(1, 2).reshape(q_len, num_heads * head_dim))

    if not outputs:
        return query.new_zeros((0, num_heads * head_dim))
    return torch.cat(outputs, dim=0)
```

### 为什么必须按序列循环

batch 里可能同时有：序列 A 的 200 token prefill、序列 B 的 1 token decode。

* 如果 padding 成 `[B, max_q_len=200, H, d]`：B 只有 1 个 token，却被 padding 出 199 个空位，白算 199 倍。
* 所以每个序列单独切片、单独算。**一个 Python 循环换取「不浪费算力」**。

### 因果 mask 的位置

```python
q_pos = torch.arange(kv_len - q_len, kv_len)[:, None]
```

为什么起点是 `kv_len - q_len`？

因为 query 只覆盖序列的**最后 q_len 个位置**（前面已有 KV，不重算）。比如 `kv_len = 100, q_len = 20`，query 对应位置 `80..99`。

```
kv_pos  →  0 ... 99
q_pos   →  80 ... 99   （转置成列）
mask    →  kv_pos <= q_pos
```

结果是个「右下角的梯形」：位置 80 的 query 能看 0..80，位置 99 的能看 0..99。

**如果写成 `kv_pos <= q_pos` 但用 `arange(q_len)` 当 q_pos**（即 0..19），mask 会完全错位——位置 80 的 query 会以为自己在位置 0，只能看第一个 token。这个 bug 在长序列上表现为「输出胡言乱语」，很难 debug。测试 `test_prefill_matches_dense` 就是专门守这一点的。

### `kv_lens = kv_lens.tolist()` 的作用

一次同步把所有长度搬到 CPU，之后循环里 `kv_lens[i]` 是纯 Python 整数索引，**零同步**。

对比第一版写法（循环里 `int(kv_lens[i].item())`）：24 层 × B 条序列次同步，每次几十微秒。修掉这一步是 48 ms vs 63 ms 差距的主要来源之一。

### `query_lens` 为什么是 Python list 而不是张量

同一个原因。签名里显式写 `list[int]` 就是为了让「不传张量」成为接口契约，而不是靠注释提醒。

### 为什么 prefill 用 `k.transpose(0, 1)` 而 decode 用 `k.transpose(1, 2)`

因为形状不同：

| | k 的形状 | 需要变成 |
|---|---|---|
| decode | `[B, L, kv, d]` | `[B, kv, L, d]` → `transpose(1, 2)` |
| prefill | `[L, kv, d]`（无 batch 维） | `[1, kv, L, d]` → `transpose(0, 1).unsqueeze(0)` |

---

## 9. 数值正确性是怎么保证的

这个文件的所有复杂度（页表、dummy 页、padding）本质上都只影响**性能**，不影响**数学**。所以验证策略是：

> 用页表 gather 出来的 KV，必须和「同一批 KV 直接拼成一个连续张量」算出的 attention **逐元素相等**。

实现见 [tests-test_paged_attention.md](tests-test_paged_attention.md)：

```python
def test_decode_matches_dense_for_ragged_batch():
    # 3 条不同长度、物理块故意乱序的序列
    # 期望：paged_attention_decode 的输出 == dense_attention(参考实现)
```

再加一层端到端对拍（[tests-test_model_runner.md](tests-test_model_runner.md)）：同一条 prompt，分页实现 vs `transformers` 的 logits，`max |diff| < 1e-4`。

**两层验证的意义**：

* 第一层证明「gather 逻辑对」——不依赖模型，跑得飞快。
* 第二层证明「接到真实模型上也对」——能抓出 RoPE、层索引、dtype 之类的问题。

---

## 用到的 PyTorch 特性（本文件）

| API | 作用 | 注意 |
|---|---|---|
| `tensor.view(-1, *shape[2:])` | 摊平前两维 | 需连续 |
| `tensor.reshape(...)` | 摊平任意前缀维 | 会按需复制 |
| `tensor[long_tensor] = value` | 散点写 | 下标不能重复 |
| `tensor.index_select(0, idx)` | 按行取 | 一次 kernel |
| `torch.arange(n, device=)` | 序列 | device 必须和目标一致，否则隐式拷贝 |
| `x[:, None, None, :]` | 造广播维 | mask 的固定套路 |
| `torch.where(cond, a, b)` | 条件选择 | 用标量 `b` 会自动广播 |
| `x.transpose(i, j)` | 换两维 | 之后不能直接 `view` |
| `x.repeat_interleave(n, dim)` | 复制展开 | 有真实内存代价 |
| `torch.cat(list, dim=0)` | 拼结果 | prefill 用 |
| `query.new_zeros(shape)` | 同 dtype/device 的空张量 | 空 batch 兜底 |
| `F.scaled_dot_product_attention` | 融合 attention | mask `True`=允许 |

---

## 设计取舍

**为什么不写 Triton/CUDA kernel？**

三个理由，按重要性排序：

1. **Windows 上没有可靠的 Triton/flash-attn 轮子**，要自己编译，这个项目的可移植性就没了。
2. 面试问的是「为什么这么设计」，不是「你会不会写 kernel」。**能讲清 gather 的代价，比写一个半吊子 kernel 更有价值。**
3. 纯 torch 版本能被 `pytest` 直接验证（CPU 上也能跑），这对学习更重要。

**为什么 decode 和 prefill 分成两个函数？**

因为它们的 mask 语义完全不同（无 causal vs 有 causal），且形状不同（有 batch 维 vs 无），混在一个函数里会变成一堆 `if`。分开写让两处都能被独立测试。

**为什么 prefill 不也走 batch 化的 `build_gather_index`？**

可以，但 query 维是 ragged 的，padding 到 `max_q_len` 会浪费（见前面分析）。真实 kernel 用「flat query + 累积长度指针」解决，Python 里用循环最直白。

---

## 常见误解

* 「paged attention 和普通 attention 算出来不一样」——**数学完全一样**，区别只在 KV 的存放方式。所有测试都在验证这一点。
* 「`dummy_slot` 是为了对齐」——不是对齐，是为了让 `index_select` 有合法下标，真正屏蔽靠 mask。
* 「decode 也要 causal mask」——不需要，query 只有 1 个且在末尾。
* 「`transpose` 会复制数据」——不会，只改 stride。所以 `transpose` 之后不要接 `view`（会报错），要接 `reshape`。
* 「`enable_gqa=True` 是精度选项」——不，是**内存/带宽**选项，数值上等价。

---

## 动手练习

1. 把 `paged_attention_decode` 里的 `mask` 去掉（传 `None`），跑 `test_decode_matches_dense_for_ragged_batch`，看短序列的结果如何被长序列的 padding 污染。
2. 把 `build_gather_index` 里 `torch.where(valid, slots, dummy_slot)` 的 `dummy_slot` 改成 `0`，观察是否仍然「侥幸」正确（因为 mask 还在），从而理解「为什么需要双保险」。
3. 把 prefill 的 `q_pos` 改成 `torch.arange(q_len)`，跑 `test_prefill_matches_dense`，读它报的最大误差。
4. 在 `gather_kv` 前后加 `torch.cuda.synchronize()` + 计时，把 `L` 从 128 扫到 2048，画出 gather 耗时曲线——你会亲眼看到「不写 kernel 的代价」。
