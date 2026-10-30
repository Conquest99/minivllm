# 概念篇：Attention、KV Cache、Prefill / Decode

代码里很多「为什么要这么折腾」的问题，答案在这一层。这篇不写代码，只讲必要的心智模型。

---

## 1. Attention 在算什么

一句话：**每个 token 去问前面所有 token「你们和我有多相关」，然后按相关性加权求和它们的值。**

```
输入：每个 token 的隐藏向量 x_t            (hidden = 896)
       ↓ 三个线性投影（可训练矩阵）
Q = x · Wq    K = x · Wk    V = x · Wv      (各自 (heads × head_dim) = 14 × 64 = 896)
       ↓
score(t, s) = Q_t · K_s / √head_dim          ← 相关度（点积）
       ↓ softmax（沿 s 归一化，causal 时只在 s ≤ t 上算）
attn(t) = Σ_s softmax(score)(t,s) · V_s      ← 加权求和
       ↓ 输出投影
out_t = attn(t) · Wo
```

拆开成 head 是为了并行看「不同的关系」（一个 head 学语法、一个学指代……），最后拼回去。

### 为什么除 √head_dim

`Q·K` 是 64 个随机数相乘再求和，方差 ≈ 64。不缩放的话 logits 很大，softmax 直接饱和成 one-hot：梯度几乎为 0，也没法「软」地组合多个来源。除 `√64 = 8` 后方差回到 1。

---

## 2. 为什么需要 KV Cache

看上面公式：token `t` 的 attention 需要**所有** `s ≤ t` 的 K、V。

生成是自回归的，一个一个往外吐：

```
step 1: 算 token 1   → 需要 K1 V1
step 2: 算 token 2   → 需要 K1 V1, K2 V2      ← K1 V1 和上一步一模一样！
step 3: 算 token 3   → 需要 K1..K3
...
step n: 算 token n   → 需要 K1..Kn
```

关键观察：**K_s、V_s 只取决于第 s 个 token 及其前缀，一旦算出来就永远不变。**
所以把它们存起来，每步只算新的那一个。这就是 KV cache。

不用 cache 的话，生成第 n 个 token 要把前面 n 个 token 全部重算一遍 → 总计算量 O(n²) 变成 O(n³)。生成 500 token 时差 500 倍。

### 代价：显存

每个 token 每一层都要存 K 和 V：

```
每 token 字节数 = 2(K和V) × layers × kv_heads × head_dim × dtype字节
```

Qwen2.5-0.5B（24 层，2 个 KV head，head_dim 64，fp16）：

```
2 × 24 × 2 × 64 × 2 = 12,288 字节 ≈ 12 KB / token
```

1000 token 的请求 = 12 MB。100 条并发 = 1.2 GB。**这就是为什么 KV cache 的管理方式能决定一个引擎能服务多少并发。**

> 对比一下：如果是 14 个 KV head（MHA，不做 GQA），同样配置是 84 KB/token，7 倍。GQA 就是拿一点质量换 7 倍显存。

---

## 3. Prefill 与 Decode：两种完全不同的计算形态

| | Prefill | Decode |
|---|---|---|
| 输入 | 整个 prompt（L 个 token） | 1 个 token |
| Q 的形状 | `[L, heads, d]` | `[1, heads, d]` |
| 需要算的 K/V | L 个 | 1 个 |
| 读 KV cache | 不需要（自己就是） | 读全部历史 |
| 计算特征 | **计算密集**（大矩阵乘大矩阵） | **带宽密集**（读权重和 KV，只做一点点乘加） |
| 用户感知 | 决定 TTFT | 决定「打字是否流畅」（ITL） |

### 为什么 decode 是带宽瓶颈

batch=1 时，为了产出 1 个 token，要把**整个模型权重**从显存读到计算单元一次：

```
权重 0.5B × 2 字节 = 1 GB
产出 1 个 token 的浮点运算 ≈ 2 × 0.5B = 1 GFLOP
算术强度 ≈ 1 FLOP/Byte
```

而 4070 的算力是几十 TFLOP/s、带宽是几百 GB/s——比值在 ~100 FLOP/Byte 才算「喂饱」。**1 FLOP/Byte 意味着算力只被用了个零头，瓶颈在搬数据。**

推论：

* 把 GPU 换成算力 2 倍但带宽不变的，**decode 速度几乎不变**。
* 把 batch 从 1 加到 16，同一份权重被 16 个 token 摊薄 → 吞吐接近线性上涨（实测：20.8 → 328.9 tok/s）。
* 再往上加，KV cache 的读取量开始占主导，收益递减。

这也解释了为什么「连续批处理」这么重要：**它让每 step 的 batch 尽量不空转**，从而把带宽摊销掉。

---

## 4. 为什么静态批处理浪费

假设一个 batch 有 8 条请求，输出长度分别是 `[400, 20, 20, 20, 20, 20, 20, 20]`。

* 静态批处理：一个 batch 要跑到**最长的那条**结束。第 21 步时，7 条已经完成，但它们的位置还占着——GPU 每步仍然算 8 个 token，其中 7 个是废的。
* 连续批处理：第 21 步就把 7 条踢出去，把队列里的新请求拉进来，8 个位置全部有效。

实测（32 请求、8 槽、重尾分布）：静态批 **56% 的生成 token 是白算的**，吞吐只有 60.9 tok/s；连续批 96.5 tok/s。

---

## 5. 分页：把操作系统的思路搬过来

### 5.1 老办法的问题

KV cache 要为每条请求准备一块**连续**内存。但两条约束打架：

1. 生成长度**事先不知道**（模型可能吐 5 个 token，也可能吐 500 个）
2. 连续内存要么预留够、要么中途搬家（搬家 = `torch.cat` 复制全部历史 KV，O(n²) 总拷贝）

结果大家都选「按 `max_model_len` 预留」：

```
pool 有 4096 token 的空间，max_model_len = 2048
→ 最多只能同时服务 2 条请求
→ 而这两条实际可能只用了 32 + 200 = 232 token
→ 利用率 5.7%
```

实测浪费 **90%**。

### 5.2 分页的解法

把池子切成固定大小的**物理块**（page），每条请求持有一张**页表**（block table）：

```
逻辑块 0,1,2,3  →  物理块 7, 2, 9, 5     ← 不需要连续
```

好处：

* **按需分配**：写到第 17 个 token 才要第 2 块，绝不提前预留。
* **零拷贝增长**：新块从池子里拿，历史 KV 原地不动。
* **可共享**：两条请求前缀相同 → 页表指向同一批物理块（引用计数）。这就是 prefix caching。
* **可回收**：请求结束，块立即归还池子，给下一个请求用。

只剩**内部碎片**：一条 33 token 的请求占 3 块 = 48 slot，尾巴的 15 个 slot 浪费。`block_size` 越小浪费越少（但页表越长）。实测这一项只占 **4.7%**。

### 5.3 与操作系统的对应关系

| OS 概念 | 本项目 |
|---|---|
| 物理页帧 | `PhysicalBlock`（`k_cache` 里的一块） |
| 页表 | `seq.block_table`（list，下标=逻辑块号，值=物理块号） |
| 缺页 | 新块从 free list 取 |
| 引用计数 | `PhysicalBlock.ref_count` |
| 写时复制 fork | `BlockManager.fork` + `copy_blocks` |
| 页面置换 | `_evict`（前缀缓存的缓存项被覆盖时失效） |

**唯一的区别**：OS 的 page fault 由硬件 MMU 处理是「隐式」的；这里必须在 Python 里显式算 `slot_mapping` 并把 KV 搬到位。

---

## 6. 为什么 attention 要「按页表读」

分页之后，一条序列的 KV 在显存里是**散落**的：第 2 块可能在物理块 9，第 3 块在物理块 2。

attention 需要 `K = [k_0, k_1, ..., k_{n-1}]` 这个逻辑顺序。两种做法：

1. **写 kernel**（vLLM 的做法）：kernel 里每次读 K/V 时现算物理地址 `block_table[i // bs] * bs + i % bs`。不需要连续内存，零额外拷贝。
2. **先 gather 成连续**（本项目的做法）：用 `index_select` 把散落的 KV 抄进一个连续张量，然后调用普通 attention。

第 2 种简单、可读、能在 Windows 上跑（不需要编译 CUDA），代价是**多一次全量拷贝**。序列越长，这次拷贝越贵。这就是为什么本项目在长上下文时不可能追上 vLLM —— 这是有意接受的取舍，也是面试时「你为什么不写 kernel」的标准答案。

---

## 7. RoPE：位置信息藏在哪

attention 本身是**无序**的（点积对位置置换不变），所以位置信息必须注入。现代模型基本都用 RoPE（旋转位置编码）。

做法：把 Q、K 的每一对维度看成二维平面上的点，按位置 `p` 旋转一个角度 `θ_i × p`：

```
q' = q·cos(pθ) + rotate_half(q)·sin(pθ)
```

妙处在于：两个向量各自旋转后，**点积只依赖它们的角度差**（即相对位置 `p_q - p_k`）。所以模型学到的是「相隔几个 token」，而不是「绝对第几个」。

代码里对应 [model_runner.py](../minivllm/model_runner.py)：

```python
cos, sin = self.model.rotary_emb(hidden.unsqueeze(0), inp["positions"].unsqueeze(0))
q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
```

⚠️ **RoPE 只作用在 Q 和 K 上，不作用在 V 上**（V 只是被加权求和，不需要方向信息）。这也是为什么 cache 里 K 是「旋转后」的、V 是原始投影。

⚠️ 还有一个关键点：**cos/sin 必须按 token 的绝对位置算**。分页之后 token 在显存里的物理位置是乱的，所以位置要靠 `positions` 显式传，绝不能靠「它在张量里排第几」。

---

## 8. GQA / MQA：KV head 比 Q head 少

| 方案 | Q heads | KV heads | KV cache 大小 |
|---|---|---|---|
| MHA | 14 | 14 | 1× |
| GQA（Qwen2.5-0.5B） | 14 | 2 | **1/7** |
| MQA | 14 | 1 | 1/14 |

做法：每 7 个 query head 共享一组 K/V。效果是显存降 7 倍，质量只掉一点点（这是现在所有主流模型的选择）。

对代码的影响：attention 里 Q 是 `[.., 14, L, D]`，K 是 `[.., 2, L, D]`，必须广播。这是 `_sdpa(..., enable_gqa=True)` 干的事。

---

## 9. 一图总结

```
                     ┌──────────── 一次请求的一生 ────────────┐
prompt "今天天气"     │                                        │
   ↓ tokenize        │  [1234, 5678, 9012]                    │
   ↓ prefill         │  一次前向算完 3 个 token 的 KV          │
   ↓                 │  KV 写进 block_table 指向的物理块       │
首 token            │  ↓  取最后一个位置的 logits             │
   ↓ decode × N      │  每次 1 个 token，读全部历史 KV         │
   ↓                 │  每 16 个 token 多占一块                │
EOS / 长度上限       │  ↓                                     │
   ↓                 │  归还所有块（引用计数减到 0 → free）    │
结束                 └────────────────────────────────────────┘
```

读完 [02] 这两篇，再去看 [block_manager.md](block_manager.md) 和 [paged_attention.md](paged_attention.md)，每个函数名应该都能对应上「它在解决上面哪一段」。
