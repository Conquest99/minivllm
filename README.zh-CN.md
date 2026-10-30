# minivllm

用**纯 PyTorch** 从零重写 **PagedAttention** 与 **Continuous Batching** 的最小实现。工业引擎把这两个机制藏进融合 CUDA kernel 和几千行调度器里，这里把每一处都摊开：看得见、能单测、能测出数字。

不是 vLLM 的竞品。目标只有一个——面试被问「*为什么这么设计*」时，能用**代码和数字**回答，而不是背博客。

> 英文版见 [README.md](README.md)。

```
                    ┌───────────────────────────────────────────────┐
  prompts ─────────▶│ Scheduler  调度器                              │
                    │  · 先 decode 后 prefill（共享 token 预算）      │
                    │  · 抢占 = 丢弃 KV + 重算                        │
                    │  · chunked prefill（分块预填充）                │
                    └───────────────┬───────────────────────────────┘
                                    │  [(seq, n_tokens)], cow_ops
                    ┌───────────────▼───────────────────────────────┐
                    │ BlockManager   物理页分配器                     │
                    │  · 空闲链表 + 引用计数                          │
                    │  · 写时复制 copy-on-write                       │
                    │  · 按 block 内容哈希的 prefix cache             │
                    └───────────────┬───────────────────────────────┘
                                    │  每序列一个 block_table（页表）
                    ┌───────────────▼───────────────────────────────┐
                    │ ModelRunner                                    │
                    │  k_cache[layers, num_blocks, block_size,        │
                    │          kv_heads, head_dim]                    │
                    │  按 slot_mapping 写，按 block_table 读          │
                    └───────────────────────────────────────────────┘
```

## 已实现的内容

| 机制 | 代码位置 | 为什么重要 |
|---|---|---|
| 分页 KV cache | [block_manager.py](minivllm/block_manager.py)、[paged_attention.py](minivllm/paged_attention.py) | 不为请求预留显存；内存按 16 token 一页增长 |
| 写时复制 COW | `BlockManager.ensure_capacity`、`fork` | 请求间共享整页，只在真正分叉的那一页复制 |
| 前缀缓存 | `BlockManager.match_prefix` / `register_computed_blocks` | 共享的 system prompt 只 prefill 一次，不是 N 次 |
| 连续批处理 | [scheduler.py](minivllm/scheduler.py) | 每个 step 都能进出请求，而不是整批进出 |
| 分块预填充 | `Scheduler._chunk_size` | 一个长 prompt 无法冻结整条 decode 流 |
| 抢占（重算） | `Scheduler._schedule_running` | 显存吃紧时不 OOM、不 swap |
| GQA / MQA | `_sdpa`（`enable_gqa`） | 0.5B Qwen 有 14 个 query head，但只有 2 个 KV head |
| Paged Attention 数学 | [paged_attention.py](minivllm/paged_attention.py) | 用测试证明：页表 `index_select` ≡ 连续 KV |

## 安装

```powershell
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
```

模型（约 1 GiB）：

```powershell
$env:HTTPS_PROXY = "http://127.0.0.1:10081"   # huggingface.co 不通时
python scripts\download_model.py
```

## 运行

```powershell
python scripts\demo_generate.py
python scripts\bench_fragmentation.py --kv-pool-tokens 4096 --requests 64
python -m pytest tests\test_block_manager.py tests\test_scheduler.py tests\test_paged_attention.py -q
python -m pytest tests\ -q                       # 追加与 HF 对拍的数值测试
python scripts\bench_static_vs_continuous.py --num-requests 32 --batch-size 8 `
    --min-output 8 --max-output 512 --distribution heavy_tail
python scripts\bench_prefix_caching.py --requests 6 --max-tokens 32
```

下面引用的数字就是这组命令跑出来的，脚本会把 JSON 写到 `benchmarks/results/`。

数值测试默认跑在 **CPU / float32**（不占显存、没有数值噪声）；设 `MINIVLLM_TEST_DEVICE=cuda` 可以把同一套断言跑在 GPU 路径上。

## 本机实测

RTX 4070 Laptop（8 GiB，且与桌面共享显存），Qwen2.5-0.5B-Instruct，fp16。

绝对值是笔记本量级，**真正有意义的是比值**。每个数字都能由旁边标出的脚本重新生成。

### 1. 省下来的显存到底来自哪里 —— `bench_fragmentation.py`

同一个池子、同一批请求，只换分配器。「reserve」是 PagedAttention 出现之前的做法：给每个被接纳的请求直接切一块 `max_model_len` 大小的连续 buffer。

```
KV 池: 4096 tokens (256 blocks × 16 tokens)，64 个请求，32 + 8..256 tokens

                                  预留          分页
最大并发请求数                          2            20
实际占用 tokens                      4096          3836
浪费比例                            90.0%          4.7%
```

并发提升 10 倍，剩余浪费只剩每条序列最后一页的尾巴。分页**彻底消除外部碎片**；内部尾巴则是 `block_size` 与页表宽度之间的权衡。

### 2. 连续批处理为什么赢过静态批处理 —— `bench_static_vs_continuous.py`

同一个模型、同一份代码，只翻转 `static_batching` 一个开关。32 个请求、8 个槽位，输出长度在 8..512 之间**重尾分布**（大多数短、少数极长——真实聊天负载的形状）。

```
                                     静态         连续
总耗时 (s)                          94.992          59.921
有效 tokens/s                       60.900          96.544
每 step 产出 tokens                  8.000           6.026
平均延迟 (s)                        54.177          24.638
p99 延迟 (s)                        94.981          59.914
平均 TTFT (s)                       30.628          13.164

静态批处理白算了 56% 的生成 token
吞吐提升 1.59x   延迟 2.2x   TTFT 2.3x
```

这 56% 是**真的算出来了**，不是空转：一个 400 token 的请求会把同批 7 个短请求硬拖 400 步。连续批处理让短请求一完成就离场，把槽位让给队列里的新请求。

### 3. 前缀缓存值多少 —— `bench_prefix_caching.py`

6 个请求共享同一个 3285 token 的 system prompt，**逐个串行跑**（一次性打进批次是看不出效果的——缓存只对「前缀已算完并注册之后才到达」的请求有帮助）。

```
                              关闭前缀缓存    开启
平均 TTFT (s)                      0.833         0.214
预热后 TTFT (s)                    0.779         0.077
命中共享页数                            0          1020

平均 TTFT 提升 3.9x   |   预热后请求 TTFT 提升 10.2x
```

第一个请求付全价（页是它填的），之后每个请求几乎立刻开始 decode。注意这里调度器必须配合：**光共享页不够**，`num_computed_tokens` 必须从缓存前缀之后开始，否则引擎会欣然重算一遍结果已在显存里的活。

## 这个仓库回答的三个问题

**1. 省下的显存究竟从哪来？**
`bench_fragmentation.py` 只测分配器。按 `max_model_len` 为每个请求预留，在真实序列长度下会让 KV cache 约 **90% 空着**；分页消除了外部碎片，只剩每条序列最后一页的尾巴，而它随 `block_size` 线性缩小。

**2. 连续批处理为什么比静态批处理快？**
静态批处理要跑到**最长成员**结束；期间每个短请求都一直占着槽位**和它的 KV 显存**。连续批处理每个 step 重新决策，完成的请求立刻把槽位让给队列。上面的实测：吞吐 1.6x、平均延迟降低 2.2x，代价是批次稍微没那么满（每 step 6.0 vs 8.0 tokens）。

**3. 为什么 decode 阶段是显存带宽瓶颈？**
batch=1 时，GPU 要把**整个权重矩阵**读一遍才产出 1 个 token，算术强度约 1 FLOP/byte。批处理把这个读取代价摊到 B 个 token 上。这就是为什么「把 GPU 换更快」不解决延迟，也是为什么 `tokens/s` 会随 batch 增大而增长，直到 KV cache 的访存成为新瓶颈。

## 面试时值得守住的几个设计点

* **先 decode 后 prefill。** prefill 是一团没人看得见的计算爆发，decode 才是用户感知的流。优先排空 decode 让 ITL 保持平稳；分块 prefill 则避免一个 4k prompt 插进 200 ms 的卡顿。
* **用重算代替 swap。** 抢占一条序列就是丢掉它的 KV，之后重跑 prefill。对小模型这是一次很快的前向；而 swap 要走一次 D2H + H2D，在 PCIe 上通常更亏。
* **gather 就是「不写 kernel」的代价。** `gather_kv` 通过 `index_select` 把 KV 拼成连续视图，等于把融合 PagedAttention kernel 本可以原地读的显存复制了一遍。它是每批一次 kernel launch（不是每 block 一次），但带宽开销是真实的——见 `paged_attention.py` 的模块注释。
* **block size 是权衡，不是常数。** 块越小尾巴浪费越少，但页表越长、gather 索引越大。`test_block_size_does_not_change_results` 证明这个选择只影响性能，**永远不影响正确性**。

## 刻意没有实现

* Triton/CUDA 版 PagedAttention kernel（flash-decoding 风格的 split-K 归约）。
* KV swap 到 CPU；`fork` 原语之外的 beam search / 并行采样。
* 投机解码、量化、多卡。
* HTTP server —— 入口就是 [scripts/demo_generate.py](scripts/demo_generate.py)。

这些都是实打实的优化，但没有一个会改变本仓库想讲清楚的**数据流**。

## 踩过的坑（这一节比上面都值钱）

1. **KV cache 必须带 layer 维度。** 最初写成 `[blocks, block_size, kv_heads, head_dim]`，24 层共用同一片槽位、互相覆盖。**单次 prefill 完全看不出问题**（同层刚写完就读），一旦进入 decode 就会读到别的层的旧值，logits 差 20。改成 `[layers, blocks, ...]` 后对拍通过。
   - 教训：能对上 prefill 的 logits，**不等于**实现是对的。对拍必须覆盖「增量 decode」和「chunked prefill」。
2. **前缀缓存只共享页是不够的。** 一开始只共享了物理页，调度器仍然把整个 prompt 重算一遍，收益接近 0。还要把 `num_computed_tokens` 推进到缓存前缀之后。
3. **层循环里不能有 `.item()`。** 每层读一次 CUDA tensor 标量就是一次流水线停顿，24 层 × N 条序列全部放大。把 `query_lens` 提到循环外、用 Python list 传进去。
4. **GQA 别用 `repeat_interleave`。** 2 个 KV head 直接展开成 14 个，等于把 KV 显存访问量放大 7 倍。torch ≥ 2.5 用 SDPA 的 `enable_gqa=True`，让 kernel 内部广播。
   - 修完这两条，batch=16 时本引擎 **48.6 ms/step**，反超 HF `generate` 的 **52.0 ms/step**。

## Windows 注意事项

* `flash-attn` 和 Triton 在 Windows 上没有可靠的 wheel —— 这里用纯 torch 写 attention，正是整个东西能在 Windows 跑起来的原因。
* KV 池按**当前空闲**显存（`gpu_memory_utilization`）自动定大小，所以压测前关掉其它吃显存的程序，或者显式设 `num_gpu_blocks` 以获得可复现的数字。
* 实验模型锁在 0.5B–1.5B；8 GiB 显存扣掉桌面占用后，fp16 的 7B 放不下。
