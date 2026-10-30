# `minivllm/model_runner.py` 详解

## 这个文件干什么

**跑一次 transformer 前向**，但把 KV 的存/取换成「按页表」。

关键设计：**不复刻模型权重和结构，直接复用 `transformers` 的 Qwen2 实现**，只重写 attention 的簿记部分。这样：

* 权重加载、tokenizer、RoPE、RMSNorm、MLP 都不用自己写（少写少错）
* 要讲清楚的那部分（KV 放哪、从哪读）完全是本项目自己的代码
* 可以直接拿 HF 的结果当标准答案对拍

---

## 1. `__init__`：从 HF 借来的东西

```python
self.hf_model = (
    AutoModelForCausalLM.from_pretrained(cfg.model, dtype=self.dtype, low_cpu_mem_usage=True)
    .to(self.device)
    .eval()
)
self.model = self.hf_model.model        # embedding + 24 层 decoder + final norm
self.lm_head = self.hf_model.lm_head
```

### `.to(device).eval()` 链式调用

* `.to(self.device)`：把权重搬到 GPU。
* `.eval()`：切到推理模式（关掉 dropout）。**必须调用**——`nn.Module` 默认是 training 模式，dropout 会随机丢弃激活值，导致同一输入两次结果不同。
* `low_cpu_mem_usage=True`：用 `safetensors` 的延迟加载，避免「先整个读进 CPU 内存再传 GPU」的 2 倍峰值。

### 为什么拆成 `model` 和 `lm_head`

`AutoModelForCausalLM` 的结构是：

```
hf_model
├── model          ← Qwen2Model：embedding + layers + norm
│   ├── embed_tokens
│   ├── layers[24]
│   ├── norm
│   └── rotary_emb
└── lm_head        ← 词表投影（151936 维）
```

我们需要手动控制每一层（在层间插入自己的 attention），所以直接拿 `model` 的子模块跑，`lm_head` 单独留到最后用。

> 注意 `tie_word_embeddings: true`（Qwen2.5-0.5B）：`lm_head.weight` 和 `embed_tokens.weight` 是同一份数据，所以「0.5B」参数在显存里实际只有约 0.36B 份。

### 从 config 里读出模型形状

```python
hf_cfg = self.hf_model.config
self.num_layers = hf_cfg.num_hidden_layers          # 24
self.num_heads = hf_cfg.num_attention_heads         # 14
self.num_kv_heads = hf_cfg.num_key_value_heads      # 2   ← GQA
self.hidden_size = hf_cfg.hidden_size               # 896
self.head_dim = getattr(hf_cfg, "head_dim", None) or self.hidden_size // self.num_heads
self.scale = self.head_dim**-0.5
```

`head_dim` 那一行是兼容写法：有的模型 config 里显式写了 `head_dim`（Qwen2 写了 64），有的没写，只能 `hidden // heads` 推。**不能想当然地认为 `896/14 = 64`**——虽然这里恰好整除，但很多模型不整除（比如 hidden=4096, heads=32 → 128 恰好；但有些是 hidden=2880, heads=40 → 72，也是整除……总之别假设）。

`scale = head_dim**-0.5` 就是 `1/√64 = 0.125`，见 [02-attention-kv-cache.md](02-attention-kv-cache.md#为什么除-headdim)。

### EOS token 可能有多个

```python
def _resolve_eos_ids(self) -> set[int]:
    ids = set()
    eos = self.hf_model.generation_config.eos_token_id
    if eos is not None:
        ids.update(eos if isinstance(eos, (list, tuple)) else [eos])
    tok_eos = self.tokenizer.eos_token_id
    if tok_eos is not None:
        ids.add(tok_eos)
    return ids
```

有的模型有多个结束符（Qwen 有 `<|im_end|>` 和 `<|endoftext|>`），有的 `generation_config` 里是 list。用 `set` 收集、判断时 `token_id in ids`，不遗漏。

> 用 `set` 而不是 `list`：判断从 O(n) 变 O(1)，而且自动去重。

### 显存池大小的两次封顶

```python
def _profile_num_blocks(self) -> int:
    useful = max(1, self.cfg.max_num_seqs * self.cfg.max_blocks_per_seq)
    if self.device.type == "cpu":
        return useful
    free, _total = torch.cuda.mem_get_info(self.device)
    budget = int(free * self.cfg.gpu_memory_utilization)
    return max(1, min(useful, budget // self.per_block_bytes))
```

第一版这里只按显存算，8G 卡给出 **25873 块**（4.74 GiB KV cache）。问题是：调度器的 `max_num_seqs × max_blocks_per_seq` 最多只能引用 1024 块，剩下 24849 块**永远用不到，却实打实占了 4.7 GB 显存**。桌面上还会因此变得卡顿。

所以现在取两者最小值：

```
useful = max_num_seqs(8) × max_blocks_per_seq(2048/16=128) = 1024 块
budget = 空闲显存 × 0.8 / 每块字节数
num_gpu_blocks = min(useful, budget // per_block)
```

**教训：显存按「可能被引用的上限」分配，而不是「显存能装多少」。**

`per_block_bytes` 的算法：

```python
@property
def per_block_bytes(self) -> int:
    element = torch.tensor([], dtype=self.dtype).element_size()
    return 2 * self.num_layers * self.cfg.block_size * self.num_kv_heads * self.head_dim * element
```

`torch.tensor([], dtype=torch.float16).element_size()` 是一种取「这种 dtype 一个元素几字节」的写法（返回 2）。因为 `torch.float16` 的 `torch.finfo` 不方便直接拿字节数，而 `torch.tensor([])` 的 `.element_size()` 恰好就是它。

展开（Qwen2.5-0.5B, bs=16）：

```
2(K和V) × 24(层) × 16(每块token) × 2(KV head) × 64(head_dim) × 2(byte) = 196,608 B ≈ 192 KiB/块
```

---

## 2. KV cache 的实际分配（含 `+1` 的秘密）

```python
def _allocate_kv_cache(self) -> None:
    bs, nkv, hd = self.cfg.block_size, self.num_kv_heads, self.head_dim
    shape = (self.num_layers, self.num_gpu_blocks + 1, bs, nkv, hd)
    self.k_cache = torch.zeros(shape, dtype=self.dtype, device=self.device)
    self.v_cache = torch.zeros(shape, dtype=self.dtype, device=self.device)
    self.dummy_slot = self.num_gpu_blocks * bs
```

三层信息：

1. **为什么有 `layers` 维？**

   这是本项目最大的一个 bug 修出来的。第一版写成 `[blocks, bs, kv, hd]`，24 层共用同一片槽位。prefill 时看不出来（同一层内「写完立刻读」），但 decode 时第 5 层读到的 K 其实是第 6 层刚写进去的——**logits 和正确值差 20**。

   教训：**prefill 的 logits 对得上，不等于实现是对的**。对拍必须覆盖「增量 decode」。

2. **为什么 `+1`？**

   多出来的那一块是 dummy 页，永远全 0、永不被写。ragged batch 里 padding 出来的位置都指向它，保证 `index_select` 的下标合法。详见 [paged_attention.md](paged_attention.md#4-build_gather_index页表--索引)。

3. **为什么 `dummy_slot = num_gpu_blocks * bs`？**

   因为它是「第 `num_gpu_blocks` 块的第 0 个 slot」，而这块正好是那块多出来的。要放进 `view(-1, kv, d)` 的合法下标范围内，必须是真实存在的 slot。

> 一次性 `torch.zeros` 的另一个好处：**池子地址固定，永不复用/迁移**。所有块的地址在整个生命周期内不变，所以页表里存的整数永远不会失效。如果用动态分配，就得处理「张量被回收后指针悬空」的问题。

---

## 3. `_build_inputs`：把 Python 侧的决策翻译成张量

这是整个文件最密的一段。它的职责：把**一组异构的 (序列, 要算几个 token)** 拼成一个**矩形张量批**，并算出 7 个辅助张量。

```python
for seq, n in zip(seqs, num_new_tokens, strict=False):
    start = seq.num_computed_tokens
    input_ids.extend(seq.data[start : start + n])
    positions.extend(range(start, start + n))
    for t in range(start, start + n):
        slots.append(seq.block_table[t // bs] * bs + (t % bs))
    query_lens.append(n)
    kv_lens.append(start + n)
    last_indices.append(len(input_ids) - 1)
    n_blocks = (start + n + bs - 1) // bs
    table_rows.append(seq.block_table[:n_blocks])
    max_blocks = max(max_blocks, n_blocks)
```

逐项解释：

| 变量 | 含义 | 谁用 |
|---|---|---|
| `input_ids` | 本 step 所有要算的 token，**首尾相接摊平** | `embed_tokens` |
| `positions` | 每个 token 的**绝对位置**（RoPE 用） | `rotary_emb` |
| `slots` | 每个 token 的 KV 该写到哪个 slot | `write_kv_cache` |
| `query_lens` | 每条序列贡献几个 token | prefill 的循环边界 |
| `kv_lens` | 每条序列算完后总共有多少 KV | mask + gather 长度 |
| `last_indices` | 每条序列**最后一个** token 在摊平后的下标 | 取 logits |
| `table_rows` | 每条序列的页表（截到实际用到的块数） | gather 索引 |

### 三个必须理解的细节

**① `start = seq.num_computed_tokens`，不是 0**

只算「还没算过的后缀」。这就是增量 prefill / decode 的实现。

**② `last_indices.append(len(input_ids) - 1)`**

注意是在 `extend` **之后**取 `len - 1`。因为 logits 只需要每条序列最后一个 token 的，而它此刻在摊平张量里的位置就是「刚加进去的最后一项」。

用 B=3、token 数 `[23, 1, 1]` 举例：

```
input_ids 长度变化: 23 → 24 → 25
last_indices = [22, 23, 24]
```

**③ `kv_lens.append(start + n)`**

这是「算完这一步之后，这条序列一共有多少 KV」。**不是** `len(seq.data)`——因为 `seq.data` 里可能有还没算的 token（chunked prefill 的后续部分）。

### padding 页表

```python
padded_tables = [[*row, *([0] * (max_blocks - len(row)))] for row in table_rows]
```

不同序列的页表长度不同，必须补成矩形。**补 0 是安全的**，因为超出 `kv_len` 的部分会被 mask 掉（并且 gather 索引在那里用的是 dummy slot 或者被切掉）。

> 为什么用 `[*row, *([0] * n)]` 这种写法而不是 `row + [0]*n`？纯粹是风格，两者等价。星号解包在构造字面量时更直观。

### 一次性 hoist 出 Python 值

```python
"query_lens_list": query_lens,                        # Python list
"all_decode": all(n == 1 for n in query_lens),        # Python bool
```

这两个字段**不是张量**，是「提前算好、避免在层循环里同步」的缓存。详见 [01-pytorch-basics.md](01-pytorch-basics.md#10-gpu-同步item--tolist--print-的隐藏成本)。

这行 `all_decode` 还有个语义用途：**用 Python 分支决定走 decode 还是 prefill 路径**，而不是在每层里对张量做 `(query_lens == 1).all()`（那会触发一次同步）。

---

## 4. `execute_model`：一层层跑

```python
@torch.inference_mode()
def execute_model(self, seqs, num_new_tokens) -> torch.Tensor:
    inp = self._build_inputs(seqs, num_new_tokens)

    hidden = self.model.embed_tokens(inp["input_ids"])            # [T, 896]
    cos, sin = self.model.rotary_emb(hidden.unsqueeze(0), inp["positions"].unsqueeze(0))
    cos, sin = cos[0], sin[0]                                      # [T, 64]

    for layer_idx, layer in enumerate(self.model.layers):
        hidden = self._forward_layer(layer_idx, layer, hidden, cos, sin, inp)

    hidden = self.model.norm(hidden)
    return self.lm_head(hidden[inp["last_indices"]]).float()
```

`T` = 本 step 所有 token 总数（不是 batch size）。

### `unsqueeze(0)` 是为了满足 HF 的接口

`rotary_emb.forward(x, position_ids)` 期望 `position_ids` 有 batch 维（`[B, L]`）。我们这里是摊平的 `[T]`，所以补一个假的 batch 维 `[1, T]`，算完再 `[0]` 去掉。

### `cos, sin` 的形状

`[T, head_dim]`，即每个 token 一个 64 维的余弦/正弦向量。RoPE 用它旋转 Q 和 K。

> 注意 RoPE 用的是**绝对位置** `positions`，而不是「这个 token 在张量里排第几」。分页之后物理位置是乱的，靠位置索引会全错。

### `hidden[inp["last_indices"]]`：只对最后一个 token 做词表投影

`lm_head` 是 `[896, 151936]` 的矩阵。对 T=25 个 token 全做要 `25 × 896 × 151936 ≈ 34 亿` 次乘加；只对 3 个要做 `4 亿`。**省 8 倍**，而且完全不影响结果（其它位置的 logits 没人要）。

`.float()` 把 fp16 logits 升到 fp32：采样时需要足够精度，fp16 在 151936 个值上分布很密，容易出现精度问题。

---

## 5. `_forward_layer`：一层的完整计算

```python
def _forward_layer(self, layer_idx, layer, hidden, cos, sin, inp):
    attn = layer.self_attn
    num_tokens = hidden.shape[0]

    residual = hidden
    x = layer.input_layernorm(hidden)

    q = attn.q_proj(x).view(num_tokens, self.num_heads, self.head_dim)
    k = attn.k_proj(x).view(num_tokens, self.num_kv_heads, self.head_dim)
    v = attn.v_proj(x).view(num_tokens, self.num_kv_heads, self.head_dim)

    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    write_kv_cache(self.k_cache[layer_idx], self.v_cache[layer_idx], inp["slot_mapping"], k, v)
    out = self._attend(q, inp, layer_idx)

    hidden = residual + attn.o_proj(out)
    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden
```

对照标准 transformer 一层：

```
x = x + Attention(LayerNorm(x))     ← Pre-Norm 结构
x = x + MLP(LayerNorm(x))
```

代码一一对应。

### `.view(num_tokens, heads, head_dim)`

`q_proj` 输出是 `[T, 896]`，896 = 14 × 64。`view` 把它「看成」`[T, 14, 64]`，即每个 token 的 14 个 head 各 64 维。**零拷贝**，因为 896 = 14×64 且内存连续。

### `apply_rotary_pos_emb(..., unsqueeze_dim=1)`

从 `transformers` 直接 import。`unsqueeze_dim=1` 告诉它「cos/sin 要在第 1 维（head 维）广播」。

* 我们的 Q/K 形状：`[T, heads, head_dim]`
* cos/sin 形状：`[T, head_dim]`
* 需要广播成：`[T, 1, head_dim]` → `unsqueeze_dim=1` 正是干这个

如果形状是 `[B, heads, L, D]`（HF 内部用法），就要 `unsqueeze_dim=2`。

### `write_kv_cache` 在 attention **之前**

顺序很重要：先把本 step 的 K/V 写进池子，再算 attention——这样 attention 的 gather 才能读到刚写的这批（新 token 也要参与自己的 attention）。

### 残差和 MLP 完全用 HF 的模块

```python
hidden = residual + attn.o_proj(out)
hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
```

`residual` 是 attention 之前的 hidden（Pre-Norm）。这里**没有 `dropout`**，因为 `.eval()` 已经关掉了；也**没有手写 RMSNorm**，直接用 `layer.input_layernorm`。

> 为什么不用 `layer(...)` 直接调用整层？因为 `Qwen2DecoderLayer.forward` 内部会自己处理 KV cache（`past_key_values`），那就绕过了我们的分页机制。**我们必须自己接管 attention 的 K/V 流向**，所以只能手写这一层的组装。

---

## 6. `_attend`：选哪条 attention 路径

```python
def _attend(self, q, inp, layer_idx):
    k_cache = self.k_cache[layer_idx]
    v_cache = self.v_cache[layer_idx]
    if inp["all_decode"]:
        return paged_attention_decode(q, k_cache, v_cache, inp["block_tables"],
                                      inp["kv_lens"], self.scale, self.dummy_slot)
    return paged_attention_prefill(q, k_cache, v_cache, inp["block_tables"],
                                   inp["query_lens_list"], inp["kv_lens"], self.scale)
```

`inp["all_decode"]` 是 Python bool，所以这个分支**不触发 GPU 同步**。

注意 `k_cache = self.k_cache[layer_idx]` 是**索引切片**，得到 `[blocks+1, bs, kv, d]` 的视图（不复制），正好是 `paged_attention.py` 期望的形状。

---

## 用到的 PyTorch / transformers 特性

| API | 作用 | 注意 |
|---|---|---|
| `AutoModelForCausalLM.from_pretrained(..., low_cpu_mem_usage=True)` | 低峰值加载 | `dtype=` 而不是 `torch_dtype=`（4.56+ 改名） |
| `.to(device).eval()` | 搬设备 + 推理模式 | **必须 eval**，否则 dropout 生效 |
| `nn.Module.__call__` / 子模块直接调用 | 前向 | 不走 `forward` 也能调，但推荐直接调用模块 |
| `model.embed_tokens(ids)` | token → 向量 | `[T] → [T, 896]` |
| `model.rotary_emb(x, pos)` | 算 cos/sin | 需要 batch 维 |
| `apply_rotary_pos_emb(q,k,cos,sin,unsqueeze_dim)` | 施加旋转 | `unsqueeze_dim` 由布局决定 |
| `hidden[idx_tensor]` | 挑行 | 返回副本 |
| `torch.tensor([], dtype=).element_size()` | dtype 字节数 | fp16 → 2 |
| `torch.cuda.mem_get_info(device)` | (空闲, 总) 字节 | 是「当前空闲」 |
| `torch.inference_mode()` | 推理装饰器 | 比 `no_grad` 更彻底 |
| `x.view(..., heads, dim)` | 拆 head | 需连续 |
| `.float()` | 升 fp32 | logits 用于采样 |

---

## 设计取舍

**为什么复用 HF 的权重而不是自己写模型？**

因为本项目的教学目标是**KV cache 管理**，不是 transformer 本身。自己写一遍 Qwen2 只会引入大量无关 bug，而 `transformers` 的实现正好可以当**标准答案**（测试直接拿它当 oracle）。

**为什么 `_build_inputs` 返回一个 dict 而不是 dataclass？**

因为它是一次性的中间产物，字段多且只在 `model_runner` 内部流转。dict 够用，还能混装张量和 Python list（`query_lens_list`、`all_decode`）——这一点用带类型的 dataclass 反而别扭。

**为什么每层都 `self.k_cache[layer_idx]` 而不是把整个 cache 传下去？**

让 `paged_attention` 的函数只管一层，形状简单（4 维），不用到处判断「有没有 layer 维」。代价是 `copy_blocks` 那个函数要额外兼容两种布局。

---

## 常见误解

* 「`num_tokens` 是 batch size」——是**本 step 所有 token 的总数**。B=3，token 数 `[23,1,1]` 时 `T=25`。
* 「`positions` 可以省略，用 `arange(T)` 代替」——绝对不行。分页之后 token 在张量里的顺序和它的真实位置无关。
* 「logits 长度是 T」——是 `[B, vocab]`，只有每条序列最后一个。
* 「`eval()` 只是习惯」——不调用的话 dropout 会随机丢弃激活，结果不可复现。
* 「KV cache 分配得越多越好」——按「能被引用的上限」分配，多出来的纯属浪费。

---

## 我踩过的两个坑

1. **KV cache 少了 layer 维**（最严重的一个）。prefill 单次前向完全正常，decode 时读到别的层的 KV，logits 差 20。测试 `test_incremental_decode_matches_transformers` 就是抓这个的。
2. **`torch_dtype=` 参数在 transformers 4.57 被废弃**，运行时打 warning。改成 `dtype=` 并顺手把 `pyproject.toml` 的版本下限提到 `>=4.56`。

---

## 动手练习

1. 注释掉 `.eval()`，跑 `tests/test_engine_e2e.py`，观察贪心解码的结果是否变成不确定（可能要多跑几次）。
2. 把 `last_indices` 改成对所有 token 都算 logits（`inp["last_indices"]` 换成 `torch.arange(T)`），用 `torch.cuda.Event` 计时，看 step 时间涨多少。
3. 把 `_profile_num_blocks` 里的 `min(useful, ...)` 去掉，启动后 `torch.cuda.memory_allocated()` 会从 ~0.2 GB 跳到 ~5 GB。
4. 把 `kv_lens.append(start + n)` 改成 `kv_lens.append(len(seq.data))`，跑 `tests/test_scheduler.py` + `test_model_runner.py`，看 chunked prefill 的用例怎么挂。
