# 代码详解 · 索引

这套文档是 [minivllm](../README.zh-CN.md) 的逐文件讲解，目标读者是**想从零补 PyTorch 与推理引擎基础**的人。

每个源文件都有一份对应文档，结构统一：

| 小节 | 你会得到什么 |
|---|---|
| 这个文件干什么 | 一句话定位 + 它在整条链路中的位置 |
| 逐块讲解 | 引用真实代码，说明「这段在解决什么问题」 |
| 用到的 PyTorch/Python 特性 | 这个函数是干什么的、参数什么含义、有什么坑 |
| 关键设计取舍 | 为什么这么写，换个写法会怎样 |
| 常见误解 | 容易想错的地方（含我自己踩过的坑） |
| 动手练习 | 改一行代码就能验证理解的实验 |

---

## 建议阅读顺序

```
第 0 天  docs/01-pytorch-basics.md        张量、形状、广播、索引、SDPA
         docs/02-attention-kv-cache.md    attention 数学、KV cache、prefill/decode

第 1 天  config.md  →  sequence.md  →  sampling.md
         （三个最轻的文件，先把「请求是什么」搞清楚）

第 2 天  block_manager.md                  分页、引用计数、COW、前缀缓存 ★核心
         paged_attention.md                按页表取 KV 做 attention ★核心

第 3 天  model_runner.md                   把 HF 的权重接到分页 cache 上
         scheduler.md                      连续批处理、chunked prefill、抢占 ★核心

第 4 天  llm_engine.md                     串起来的 step 循环
         scripts-*.md                      三个实验脚本 + demo

第 5 天  tests-*.md                        41 个测试为什么这么设计 ★面试素材
```

★ 标记的三个文件是面试真正会被追问的地方，其余是支撑细节。

---

## 文件地图

| 源文件 | 行数 | 难度 | 一句话作用 | 对应文档 |
|---|---:|---|---|---|
| [minivllm/__init__.py](../minivllm/__init__.py) | 14 | — | 包门面，导出 3 个名字 | [init.md](init.md) |
| [minivllm/config.py](../minivllm/config.py) | 73 | ★ | 所有可调参数与它们的约束 | [config.md](config.md) |
| [minivllm/sequence.py](../minivllm/sequence.py) | 115 | ★ | 一条请求 = 一个增长的 token 流 + 一张页表 | [sequence.md](sequence.md) |
| [minivllm/sampling.py](../minivllm/sampling.py) | 62 | ★ | 从 logits 里挑下一个 token | [sampling.md](sampling.md) |
| [minivllm/block_manager.py](../minivllm/block_manager.py) | 290 | ★★★ | 物理页池：分配、引用计数、COW、前缀缓存 | [block_manager.md](block_manager.md) |
| [minivllm/paged_attention.py](../minivllm/paged_attention.py) | 223 | ★★★ | 按页表把散落的 KV 取出来做 attention | [paged_attention.md](paged_attention.md) |
| [minivllm/model_runner.py](../minivllm/model_runner.py) | 238 | ★★★ | 一层层跑 transformer，KV 写进分页池 | [model_runner.md](model_runner.md) |
| [minivllm/scheduler.py](../minivllm/scheduler.py) | 202 | ★★★ | 每个 step 决定「谁跑、跑几个 token」 | [scheduler.md](scheduler.md) |
| [minivllm/llm_engine.py](../minivllm/llm_engine.py) | 207 | ★★ | 把调度 → 前向 → 采样 → 回收串成循环 | [llm_engine.md](llm_engine.md) |
| [scripts/_workload.py](../scripts/_workload.py) | 75 | ★ | 统一的压测请求集 + 分位数统计 | [scripts-_workload.md](scripts-_workload.md) |
| [scripts/download_model.py](../scripts/download_model.py) | 41 | ★ | 走镜像/代理拉权重 | [scripts-download_model.md](scripts-download_model.md) |
| [scripts/demo_generate.py](../scripts/demo_generate.py) | 67 | ★ | 最小可运行示例 | [scripts-demo_generate.md](scripts-demo_generate.md) |
| [scripts/bench_fragmentation.py](../scripts/bench_fragmentation.py) | 121 | ★★ | 只测分配器：预留 vs 分页的浪费 | [scripts-bench_fragmentation.md](scripts-bench_fragmentation.md) |
| [scripts/bench_static_vs_continuous.py](../scripts/bench_static_vs_continuous.py) | 241 | ★★ | 静态批 vs 连续批，只翻转一个开关 | [scripts-bench_static_vs_continuous.md](scripts-bench_static_vs_continuous.md) |
| [scripts/bench_prefix_caching.py](../scripts/bench_prefix_caching.py) | 136 | ★★ | 前缀缓存对 TTFT 的收益 | [scripts-bench_prefix_caching.md](scripts-bench_prefix_caching.md) |
| [tests/conftest.py](../tests/conftest.py) | 45 | ★ | 共享 fixture：一个 CPU/fp32 的 ModelRunner | [tests-conftest.md](tests-conftest.md) |
| [tests/test_block_manager.py](../tests/test_block_manager.py) | 146 | ★★ | 分配器的 9 条不变量 | [tests-test_block_manager.md](tests-test_block_manager.md) |
| [tests/test_paged_attention.py](../tests/test_paged_attention.py) | 180 | ★★ | 分页 attention 必须等于稠密 attention | [tests-test_paged_attention.md](tests-test_paged_attention.md) |
| [tests/test_scheduler.py](../tests/test_scheduler.py) | 174 | ★★ | 纯 CPU 的调度策略测试 | [tests-test_scheduler.md](tests-test_scheduler.md) |
| [tests/test_model_runner.py](../tests/test_model_runner.py) | 165 | ★★★ | 与 `transformers` 逐 token 对拍 | [tests-test_model_runner.md](tests-test_model_runner.md) |
| [tests/test_engine_e2e.py](../tests/test_engine_e2e.py) | 125 | ★★ | 「批处理不影响结果」这条最强性质 | [tests-test_engine_e2e.md](tests-test_engine_e2e.md) |

---

## 依赖关系（谁调用谁）

```
                        EngineConfig (config.py)
                              │
LLMEngine (llm_engine.py) ────┼──── Scheduler (scheduler.py)
      │                       │           │
      │                       │           └──> BlockManager (block_manager.py)
      │                       │                     │
      │                       │                     └──> Sequence (sequence.py)
      │                       │                                │
      └──> ModelRunner  ──────┴──> paged_attention.py          └──> SamplingParams (sampling.py)
             │
             └──> transformers (只借权重和 tokenizer)
```

一句话版本：**Engine 驱动 Scheduler 决定批，Scheduler 通过 BlockManager 分配页，ModelRunner 按页表跑前向，Sampling 从 logits 里取 token，回去更新 Sequence。**

---

## 一次 `generate()` 到底发生了什么

拿 3 条 prompt 走一遍，标注每一步跳到哪个文件看：

```
engine.generate(prompts)
│
├─ 1. add_request ×3                      llm_engine.py
│     ├─ 套 chat template → tokenizer
│     └─ 建 Sequence，塞进 scheduler.waiting        sequence.py
│
├─ 2. while scheduler.has_unfinished():
│     └─ engine.step()                            llm_engine.py
│         │
│         ├─ a. scheduler.schedule()               scheduler.py
│         │     ├─ _schedule_running：老的每序列 1 个 token（decode）
│         │     ├─ _schedule_waiting：新的 prefill，受 token 预算限制
│         │     └─ 期间 block_manager.ensure_capacity 可能要 COW
│         │
│         ├─ b. model_runner.apply_cow(cow_ops)    paged_attention.py
│         │     └─ 真的把那几页 KV 在显存里复制一份
│         │
│         ├─ c. model_runner.execute_model()       model_runner.py
│         │     ├─ _build_inputs：拼 input_ids / positions / slot_mapping
│         │     ├─ 24 层循环：
│         │     │    ├─ q/k/v 投影 + RoPE
│         │     │    ├─ write_kv_cache  ← 按 slot_mapping 写进池子
│         │     │    └─ _attend → paged_attention_decode / _prefill
│         │     └─ 只对 last_indices 做 lm_head → logits
│         │
│         └─ d. _postprocess                       llm_engine.py
│               ├─ sample() 取 1 个 token            sampling.py
│               ├─ 判断 EOS / 长度 / stop 串
│               └─ scheduler.free_finished()
│                     ├─ register_computed_blocks（把满页登记进前缀缓存）
│                     └─ block_manager.free（引用计数减一）
│
└─ 3. 按输入顺序返回 RequestOutput
```

**整个过程没有一次 `torch.cat` 扩容 KV cache，也没有一次为某条请求「预留」显存**——这两件事正是 PagedAttention 要消灭的。

---

## 术语表

| 术语 | 含义 | 在本项目里的名字 |
|---|---|---|
| prefill | 一次性把整个 prompt 过一遍，产出 KV 和首 token | `query_len > 1` 的那次前向 |
| decode | 每次只算 1 个新 token，复用已有 KV | `query_len == 1` |
| TTFT | Time To First Token，从请求到首 token 的时间 | `Sequence.ttft` |
| ITL / TPOT | Inter-Token Latency，相邻 token 间隔 | `step_times` 的均值 ≈ ITL |
| KV cache | 缓存每层每 token 的 K、V 向量，避免重复计算 | `k_cache` / `v_cache` |
| block / page | KV cache 的一个固定大小物理块（默认 16 token） | `PhysicalBlock`、`block_size` |
| block table | 逻辑块号 → 物理块号的映射，就是页表 | `seq.block_table` |
| slot | 池子里一个 token 的 KV 位置 = `block_id * block_size + offset` | `slot_mapping` |
| COW | 写时复制：共享页在被写入前先复制一份 | `copy_blocks`、`ensure_capacity` |
| prefix caching | 相同 token 前缀复用已算好的 KV 页 | `match_prefix` |
| preemption | 显存不够时把某条请求踢回队列，丢弃其 KV | `_schedule_running` |
| chunked prefill | 长 prompt 分几个 step 算，避免卡住 decode | `_chunk_size` |
| GQA | 多个 query head 共享一组 KV head | `num_kv_heads`、`enable_gqa` |
| token budget | 每个 step 最多处理多少个 token（不是多少条请求） | `max_num_batched_tokens` |

---

## 关于「为什么要读代码而不是读论文」

论文告诉你 *是什么*，代码告诉你 *代价在哪*。这份实现里，有三处只有写了代码才会懂：

1. `gather_kv` 的 `index_select` —— 论文说「kernel 直接按页表读」，你不写一遍不会意识到**不写 kernel 就得先复制一份连续内存**，而这在长上下文时是真实带宽开销。
2. `_chunk_size` 的 block 对齐 —— 分块 prefill 不能随便切，切在半个 block 中间会让前缀缓存无法登记，也会让 `slot_mapping` 变得别扭。
3. `num_computed_tokens` —— 前缀缓存**光共享页是不够的**，调度器不知道「前 N 个 token 已经算过」，就会照样重算一遍。

这三条都在对应文档的「踩过的坑」里。
