# PyTorch 基础补课（对照本项目）

这篇不是 PyTorch 教程的复刻，而是**只讲本项目真正用到的东西**，并且每条都指回代码里的位置。读完你应该能看懂 [paged_attention.py](../minivllm/paged_attention.py) 里的每一行。

---

## 1. 张量到底是什么

一个 `torch.Tensor` 有 4 个属性：

| 属性 | 例子 | 含义 |
|---|---|---|
| `shape` | `(8, 16, 2, 64)` | 每一维的大小 |
| `dtype` | `torch.float16` | 每个元素占几字节（fp16 = 2） |
| `device` | `cuda:0` | 数据在显存还是内存 |
| `stride` | `(2048, 128, 64, 1)` | **走到下一维要跨过多少个元素**（绝大多数人忽略，但它决定 view 能不能用） |

`stride` 是理解后面所有「形状操作」的钥匙。举个最小例子：

```python
x = torch.arange(6).reshape(2, 3)
# tensor([[0, 1, 2],
#         [3, 4, 5]])
x.shape   # torch.Size([2, 3])
x.stride()# (3, 1)   ← 行内跳 1 步到下一个元素，换行跳 3 步
```

张量只是「一块内存 + 一组 stride」。所以：

* 改 shape **不一定要搬数据**（只要 stride 描述得出来）
* 有些操作会让 stride 变得「不规整」，此时很多操作就失效了

---

## 2. `view` / `reshape` / `contiguous`——最容易踩的坑

### 2.1 三者区别

| 方法 | 行为 | 失败时 |
|---|---|---|
| `view(shape)` | 只改 shape 元信息，**绝不复制** | 内存不连续时**直接报错** |
| `reshape(shape)` | 能 view 就 view，不能就**悄悄复制** | 基本不报错 |
| `contiguous()` | 强制复制成连续内存 | — |

### 2.2 什么时候会不连续

`transpose` / `permute` 只交换 stride，不搬数据：

```python
x = torch.randn(2, 3)
x.is_contiguous()          # True
y = x.transpose(0, 1)      # shape (3,2)，stride 变成 (1,3)
y.is_contiguous()          # False
y.view(-1)                 # ❌ RuntimeError: view size is not compatible
y.reshape(-1)              # ✅ 内部先 contiguous 再 view（会复制）
```

### 2.3 本项目里的用法

```python
# paged_attention.py :: write_kv_cache
flat_k = k_cache.view(-1, *k_cache.shape[2:])
```

`k_cache` 是从 `torch.zeros(...)` 新建的，天然连续，所以用 `view` 安全（**不复制**，只是换了「二维视角」）。`-1` 的意思是「这一维你自己算」：总元素数 ÷ 每行元素数。

```python
# paged_attention.py :: copy_blocks
flat_k = k_cache.reshape(-1, num_slots, *k_cache.shape[-2:])
```

这里用 `reshape` 而不是 `view`，因为它要同时兼容 4 维 `[blocks, bs, kv, d]` 和 5 维 `[layers, blocks, bs, kv, d]` 两种布局——`reshape` 更宽容。

### 2.4 一个反直觉的点

`flat_k = k_cache.view(...)` 得到的 `flat_k` **和 `k_cache` 共享同一块内存**。所以：

```python
flat_k[slot_mapping] = key   # 这一行真的把 KV 写进了 k_cache
```

不需要 `k_cache = flat_k.reshape(...)` 赋回去。这是本项目「零拷贝写入 KV cache」的基础。

> 检验方法：`flat_k.data_ptr() == k_cache.data_ptr()` 为 `True` 就说明共享。

---

## 3. 形状操作速查（附本项目实例）

| 操作 | 作用 | 本项目实例 |
|---|---|---|
| `x.unsqueeze(d)` / `x[:, None]` | 在第 d 维插一个长度 1 的维 | `q = query.unsqueeze(2)` 把 `[B,H,D]` 变 `[B,H,1,D]` |
| `x.squeeze(d)` | 去掉长度 1 的维 | 本项目没直接用 |
| `x.transpose(i, j)` | 交换两维（只改 stride） | `k.transpose(1, 2)` 把 `[B,L,H,D]` 变 `[B,H,L,D]`（SDPA 要求 heads 在倒数第三维） |
| `x.permute(...)` | 任意重排多维 | 未使用（等价于多次 transpose） |
| `x.reshape(...)` | 改形状，必要时复制 | `out.transpose(1, 2).reshape(B, H*D)` |
| `x.expand(...)` | 沿长度 1 的维**广播展开**（不复制） | `position_ids[:, None, :].float().expand(...)`（在 transformers 内部） |
| `x.repeat_interleave(n, dim)` | **真的复制** n 份（有内存代价） | 只在 `_sdpa` 的兼容分支里用 |
| `torch.cat(list, dim)` | 拼接 | `torch.cat(outputs, dim=0)` 在 prefill 里拼各序列结果 |

### `expand` 与 `repeat_interleave` 的差距为什么重要

GQA 场景：14 个 query head，2 个 KV head。要让 `[.., 2, L, D]` 的 K 匹配 `[.., 14, L, D]` 的 Q：

* `repeat_interleave(7, dim=-3)` → 真的生成一份 7 倍大的张量。序列长 4000、head_dim 64 时，K 是 `14 × 4000 × 64 × 2字节 ≈ 7 MB`，每层每批都多算一遍。
* `enable_gqa=True` → 交给 SDPA 内部广播，**一个字节都不多占**。

这就是 [paged_attention.py](../minivllm/paged_attention.py) 里 `_sdpa()` 存在的理由。修掉它之后 batch=16 的 step 时间从 63 ms 降到 48 ms。

---

## 4. 广播（broadcasting）——mask 能这么写的原因

规则只有 3 条（从最后一维往前比）：

1. 维数不同 → 在**前面**补 1
2. 该维长度为 1 → 可以拉长到对方长度
3. 否则必须相等，不然报错

### 4.1 本项目最典型的一处

```python
# paged_attention.py :: paged_attention_decode
mask = valid[:, None, None, :]   # valid 是 [B, L] → mask 是 [B, 1, 1, L]
out = _sdpa(q, k.transpose(1, 2), v.transpose(1, 2), mask, scale)
#           q: [B, H, 1, D]   k: [B, Hkv, L, D]   v: 同上
```

SDPA 内部算的是 `scores[..., Lq, Lk] = q @ k^T * scale`，即 `[B, H, 1, L]`。mask 是 `[B, 1, 1, L]`，按规则：

```
scores [B, H, 1, L]
mask   [B,  1, 1, L]   ← H 和 1 这两维被广播
```

一个 mask 就同时屏蔽了「所有 head」和「唯一的那个 query 位置」。这就是为什么不需要显式复制 mask。

### 4.2 prefill 的因果 mask

```python
q_pos  = torch.arange(kv_len - q_len, kv_len)[:, None]   # [q_len, 1]
kv_pos = torch.arange(kv_len)[None, :]                    # [1, kv_len]
mask   = kv_pos <= q_pos                                  # [q_len, kv_len]
```

用广播造出一个下三角矩阵，**不需要 `torch.tril`**，也不需要写循环。含义：位置 `i` 的 query 只能看 `kv_pos <= 它自己位置` 的 key。

> `arange(...)[:, None]` 出 `[n,1]`，`arange(...)[None, :]` 出 `[1,m]`，两者 `<=` 得 `[n,m]`。这是造 mask 的通用套路，背下来。

---

## 5. 索引的三个档次

### 5.1 基础索引（切片）——返回 view

```python
query[q_offset : q_offset + q_len]   # [q_len, H, D]
inp["input_ids"]                     # 整个张量
seq.data[start : start + n]          # Python list，不是张量
last = hidden[inp["last_indices"]]   # 见下
```

### 5.2 高级索引（用张量/list 当下标）——返回**副本**

```python
hidden[inp["last_indices"]]   # 等价于 hidden.index_select(0, idx)
```

`last_indices` 是 `[B]` 的长整型张量，比如 `[3, 11, 17]`，结果就是「挑出第 3、11、17 行」。这是 `[B, hidden]`，正好喂给 `lm_head` → `[B, vocab]`。

**为什么要挑行？** 一个 step 里 3 条序列可能一共 25 个 token（1 条 prefill 23 个 + 2 条 decode 各 1 个），但下一步只需要每条的**最后一个** token 的 logits。不挑的话，`lm_head` 要对 25 个 token 各算一次 `[hidden] × [hidden, 151936]`，纯浪费。

### 5.3 `index_select` —— 明确、批量、快

```python
flat_k.index_select(0, idx)     # idx 是 1-D LongTensor
```

跟 `flat_k[idx]` 等价，但语义更明确，且**保证走的是 gather 路径**（不会退化成 Python 循环）。本项目所有「按页表取 KV」都靠它。

### 5.4 高级索引**赋值** = scatter（这是写入 KV 的唯一手法）

```python
# paged_attention.py :: write_kv_cache
flat_k[slot_mapping] = key
```

* `slot_mapping` 形如 `[0, 1, 2, ..., 15, 48, 49]`（跨 block，物理上不连续）
* `key` 形如 `[num_tokens, kv_heads, head_dim]`
* 这一行就是把每个 token 的 K 向量**散点写入**它该在的槽位

等价写法：`flat_k.index_copy_(0, slot_mapping, key)`。用中括号是因为更好读。

> ⚠️ 前提：`slot_mapping` 里不能有重复下标（同一位置写两次结果不确定）。本项目不会出现，因为一条序列在一个 step 里每个 token 位置只算一次。

---

## 6. `torch.where` / `masked_fill` —— 两个「按条件选值」

```python
# build_gather_index：不合法位置指向 dummy 页
torch.where(valid, slots, dummy_slot)
#  valid 为 True 取 slots 对应元素，否则取 dummy_slot（标量自动广播）
```

```python
# sampling.py：把低于阈值的 logits 打成 -inf
logits = logits.masked_fill(logits < kth, float("-inf"))
```

区别：

| | `where` | `masked_fill` |
|---|---|---|
| 输入 | 条件 + 两个候选张量 | 条件 + **一个常数** |
| 结果 | 新张量 | 新张量（非原地；原地版是 `masked_fill_`） |
| 典型用途 | 在两个张量间选择 | 屏蔽某些位置 |

**为什么用 `-inf` 而不是 0？** 因为后面要 `softmax`。`softmax` 里 `e^{-inf} = 0`，这些位置的权重精确为 0；如果填 0，`e^0 = 1`，反而变成「允许」了。

---

## 7. `softmax` / `cumsum` / `topk` / `multinomial` —— 采样四件套

```python
# sampling.py :: sample()
if params.top_k > 0:
    kth = torch.topk(logits, k).values[-1]      # 第 k 大的值
    logits = logits.masked_fill(logits < kth, float("-inf"))

if params.top_p < 1.0:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    cutoff = cumulative - probs > params.top_p   # 累计概率「已经超过」的位置
    ...

probs = torch.softmax(logits, dim=-1)
torch.multinomial(probs, num_samples=1, generator=generator)
```

逐个说：

* **`topk(x, k)`** 返回 `(values, indices)` 两个张量，`values[-1]` 是第 k 大的值。
* **`sort`** 返回排序后的值和**原始下标**（后面要靠 `scatter` 把顺序还原）。
* **`softmax(dim=-1)`** 必须指明沿哪一维归一化。词表维在最后，所以总是 `-1`。忘了写会警告并默认 dim=0，那是错的。
* **`cumsum`** 累计和。`cumulative - probs` 是「**不包含自己**的累计概率」，所以判据是「加入这个候选后才会超过 top_p」，这保证了 nucleus 至少保留 1 个 token。
* **`multinomial`** 按概率抽样。传 `generator` 是为了可复现（`torch.Generator` 自己管种子）。

**注意 `cumulative - probs > top_p`**：如果写成 `cumulative > top_p`，那么 top_p=1.0 时最后一个位置可能被判掉，导致概率和小于 1。这是个经典 off-by-one。

---

## 8. `scaled_dot_product_attention`（SDPA）完全解析

这是本项目唯一一个「深度学习算子」，值得单独讲。

### 8.1 它算什么

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d  + mask ) · V
```

代码里等价于：

```python
scores = (q @ k.transpose(-1, -2)) * scale
scores = scores.masked_fill(~mask, -inf)
out = scores.softmax(-1) @ v
```

但 SDPA 会挑一个**融合实现**（FlashAttention / memory-efficient / cuDNN / 纯数学），不把 `scores` 这个 `[.., Lq, Lk]` 的大矩阵写进显存。长序列时这是数量级的差别。

### 8.2 形状约定（记牢）

```
query:      [..., Lq,  E]      E = head_dim
key:        [..., Lk,  E]
value:      [..., Lk, Ev]      Ev 通常 = E
attn_mask:  可广播到 [..., Lq, Lk]
输出:       [..., Lq, Ev]
```

本项目里 `...` 是 `[B, H]`。所以：

```python
q_i.unsqueeze(0).transpose(1, 2)   # [q_len, H, D] → [1, H, q_len, D]
```

`unsqueeze(0)` 补 batch 维，`transpose(1, 2)` 把 head 挪到 Lq 前面。这两步是 SDPA 的固定仪式。

### 8.3 `attn_mask` 的语义（最容易搞错的地方）

**在 SDPA 里，`attn_mask` 为 `True` 表示「允许参与 attention」，为 `False` 表示被屏蔽。**

```python
mask = kv_pos <= q_pos     # True = 可以看到
```

⚠️ 这与 `nn.MultiheadAttention` 的 `key_padding_mask` **语义相反**（那个 `True` 表示「这是 padding，要屏蔽」）。两者混用是高频 bug。

> bool 类型走「True 保留」；float 类型则**直接加到 scores 上**（0 表示不屏蔽，-inf 表示屏蔽）。本项目全用 bool。

### 8.4 `scale` 参数

默认是 `1/sqrt(E)`。为什么要缩放：点积随维度增大而方差变大，softmax 会饱和到 one-hot，梯度消失。除 `√d` 让方差回到 O(1)。

本项目显式传 `scale=self.scale`，因为 `head_dim` 是 `hidden // num_heads`，而且有些模型的 `head_dim` 不等于这个（Qwen2 就是显式写在 config 里的）。

### 8.5 `enable_gqa`

torch ≥ 2.5 才有。要求 `H % Hkv == 0`，内部按组广播。我们的 `_sdpa` 里做了**特性探测**而不是版本判断：

```python
def _detect_gqa_support() -> bool:
    try:
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
    except TypeError:
        return False
    return True
```

为什么不用 `inspect.signature`？因为 SDPA 是 **C 实现的内建函数，拿不到签名**（会抛 `ValueError: no signature found for builtin`）——这是我实际踩过的坑。探测式写法还有个好处：将来改名/换实现也不会误判。

---

## 9. 推理模式：`no_grad` vs `inference_mode` vs `detach`

| | 作用 | 本项目位置 |
|---|---|---|
| `torch.no_grad()` | 不建计算图 | 测试里对拍 HF 时用 |
| `@torch.inference_mode()` | 更彻底：连 version counter 都不跟踪，更快 | `execute_model`、`sample` 的装饰器 |
| `x.detach()` | 从计算图里摘出来（共享内存） | 调试脚本里存中间结果用 |

推理引擎没有反向传播，所以整个前向都该在 `inference_mode` 下。忘记加不会报错，只是慢一点、显存多一点——**这类隐性浪费在写引擎时最致命**。

---

## 10. GPU 同步：`.item()` / `.tolist()` / `print` 的隐藏成本

CUDA 是**异步**的：Python 发一条 kernel 命令就立刻返回，不等它算完。

但下面这些操作**必须等**，于是会阻塞整条流水线（pipeline stall）：

| 操作 | 为什么阻塞 | 本项目怎么处理 |
|---|---|---|
| `tensor.item()` | 要把一个标量拷回 CPU | 循环外一次性算好 |
| `tensor.tolist()` | 要把整个张量拷回 CPU | 每层调用一次不够，要提到外面 |
| `bool(tensor.all())` | 要把 bool 拷回 CPU | 改成 Python 侧判断 `all(n == 1 ...)` |
| `print(tensor)` | 要格式化，所以会拷回 CPU | 不要留在热路径 |
| `.cpu()`、`.numpy()` | 同 `.tolist()` | 只在收尾时用 |

真实教训：最初 `paged_attention_prefill` 在**每个序列、每一层**都调一次 `query_lens[i].item()` 和 `kv_lens[i].item()`。24 层 × 8 序列 = 384 次同步，每次约几十微秒，白白吃掉十几毫秒/step。

修法：在 `_build_inputs` 里就把 `query_lens` 留一份 Python `list`（`query_lens_list`），并把「是否全是 decode」算成 Python `bool`（`all_decode`），层循环里直接用。

```python
# model_runner.py :: _build_inputs
"query_lens_list": query_lens,          # Python list，不触发同步
"all_decode": all(n == 1 for n in query_lens),
```

---

## 11. 显存：怎么算、怎么查

```python
# model_runner.py :: _allocate_kv_cache
shape = (self.num_layers, self.num_gpu_blocks + 1, bs, nkv, hd)
self.k_cache = torch.zeros(shape, dtype=self.dtype, device=self.device)
```

字节数：

```python
self.k_cache.numel() * self.k_cache.element_size()
# numel()        → 元素总数
# element_size() → 每个元素几字节（fp16 = 2）
```

查显存：

```python
free, total = torch.cuda.mem_get_info(device)   # 单位是 byte
```

注意它返回的是**驱动视角的当前空闲量**，包含其它进程占用的部分。所以「8G 卡但只有 4.6G 可用」是正常的。

`per_block_bytes` 的算法值得记住：

```python
2 * num_layers * block_size * num_kv_heads * head_dim * element_size
↑
K 和 V 各一份
```

对 Qwen2.5-0.5B：`2 × 24 × 16 × 2 × 64 × 2 B = 196,608 B ≈ 192 KiB/块`。**注意这里用的是 `num_kv_heads=2` 而不是 `num_heads=14`**——GQA 让 KV cache 小了 7 倍，这是 0.5B 模型能在 8G 卡上跑几千 token 上下文的原因。

---

## 12. 本项目用到的所有 torch API 速查表

| API | 一句话 | 出现在 |
|---|---|---|
| `torch.zeros(shape, dtype, device)` | 全 0 张量 | `_allocate_kv_cache` |
| `torch.arange(n, device=)` | 0..n-1 | 造 slot / mask |
| `torch.tensor(list, dtype=torch.long, device=)` | list → 张量 | `_build_inputs` |
| `torch.where(cond, a, b)` | 条件选择 | `build_gather_index` |
| `torch.cat(list, dim)` | 拼接 | prefill 输出拼接 |
| `torch.Generator(device)` | 随机数发生器 | 采样可复现 |
| `tensor.view/reshape(...)` | 改形状（后者会复制） | 到处 |
| `tensor.transpose(i, j)` | 换两维（不动数据） | 送进 SDPA 前 |
| `tensor.unsqueeze(d)` / `[:, None]` | 插一个长度 1 的维 | mask / q |
| `tensor.index_select(0, idx)` | 按下标取行 | `gather_kv` |
| `tensor[idx] = value` | 散点写入 | `write_kv_cache` |
| `tensor.masked_fill(cond, val)` | 按条件填常数 | top-k 过滤 |
| `tensor.repeat_interleave(n, dim)` | 复制展开 | `_sdpa` 兼容分支 |
| `tensor.numel()` / `.element_size()` | 元素数 / 元素字节 | 显存统计 |
| `tensor.float()` | 升到 fp32 | 采样、logits |
| `tensor.tolist()` / `.item()` | 取回 Python 值（**会同步**） | 注意热路径别用 |
| `torch.softmax(x, dim)` | 归一化 | 采样 |
| `torch.cumsum(x, dim)` | 累计和 | top-p |
| `torch.topk(x, k)` | 前 k 大 | top-k |
| `torch.sort(x, descending=True)` | 排序 + 下标 | top-p |
| `torch.multinomial(p, n, generator=)` | 按概率抽样 | `sample` |
| `torch.argmax(x)` | 最大值下标 | 贪心采样 |
| `torch.cuda.mem_get_info(d)` | (空闲, 总量) 字节 | 定池子大小 |
| `torch.inference_mode()` | 推理模式装饰器 | `execute_model` |
| `F.scaled_dot_product_attention(...)` | 融合 attention | `_sdpa` |

---

## 13. 自测：能回答这 8 题就算过关

1. `x.view(-1)` 在什么情况下会报错？为什么 `reshape` 不会？
2. `k_cache.view(-1, ...)` 得到的张量改内容，`k_cache` 会变吗？
3. SDPA 的 `attn_mask` 里 `True` 表示「允许」还是「屏蔽」？
4. 为什么 mask 可以用 `[B, 1, 1, L]` 去配 `[B, H, 1, L]` 的 scores？
5. 为什么屏蔽要填 `-inf` 而不是 `0`？
6. `cumulative - probs > top_p` 里为什么要减掉 `probs`？
7. 为什么 `repeat_interleave` 在 GQA 里是个坏主意？
8. 为什么层循环里禁止出现 `.item()`？

答案都在上面 12 节里，找不到就回头看对应小节。
