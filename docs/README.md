# 逐文件代码讲解

两个部分，各自独立：

| 目录 | 内容 | 入口 |
|---|---|---|
| [minivllm/](minivllm/00-index.md) | 推理引擎：PagedAttention + 连续批处理 | [minivllm/00-index.md](minivllm/00-index.md) |
| [miniagent/](miniagent/00-index.md) | Agent：手写 ReAct 循环 + 工具调用 | [miniagent/00-index.md](miniagent/00-index.md) |

---

## 第一部分 · minivllm（21 个源文件）

先读两篇概念，再读代码，最后读测试。

```
minivllm/00-index.md              链路追踪、文件地图、术语表
minivllm/02-attention-kv-cache.md 为什么需要 KV cache、prefill/decode 差异
minivllm/01-pytorch-basics.md     view/reshape、广播、SDPA、GPU 同步陷阱
        ↓
minivllm/config.md  →  sequence.md  →  sampling.md      （最轻的三个）
        ↓
minivllm/block_manager.md   ★ 分页、引用计数、COW、前缀缓存
minivllm/paged_attention.md ★ 按页表取 KV 做 attention
        ↓
minivllm/model_runner.md  →  scheduler.md  →  llm_engine.md
        ↓
minivllm/scripts-*.md   （要复现数字时再看）
minivllm/tests-*.md     （面试弹药：每条断言在守什么）
```

★ 标记的两篇是面试真正会被追问的地方。

## 第二部分 · miniagent（7 个源文件）

```
miniagent/00-index.md      循环伪代码、四条护栏、和 LangGraph 的对照
        ↓
miniagent/types.md         全链路唯一的消息格式
miniagent/tools.md         从函数签名推 JSON Schema
miniagent/parser.md        模型文本 → 工具调用，4 种方言
        ↓
miniagent/agent.md      ★ ReAct 循环
        ↓
miniagent/backends.md   →  builtin_tools.md  →  scripts-agent_demo.md
        ↓
miniagent/tests-*.md       控制流全部分支 + 安全边界
```

---

## 每篇的统一结构

| 小节 | 你会得到什么 |
|---|---|
| 这个文件干什么 | 一句话定位 + 它在整条链路中的位置 |
| 逐块讲解 | 引用真实代码，说明「这段在解决什么问题」 |
| 用到的 Python / PyTorch 特性 | 函数作用、参数含义、坑（表格） |
| 设计取舍 | 为什么这么写，换个写法会怎样 |
| 常见误解 | 容易想错的地方 |
| 动手练习 | 改一行代码就能验证理解的实验 |

大部分文件还有一节「我踩过的坑」——那是文档里最值钱的部分：每个坑都有现象、根因和修法。

---

## 统计

| | 文件数 | 行数 |
|---|---:|---:|
| `docs/minivllm/` | 24 | 6,767 |
| `docs/miniagent/` | 11 | 3,380 |
| 合计 | 35 | 10,147 |

对应源代码：`minivllm/` + `scripts/` + `tests/` 共 21 个文件，`miniagent/` 7 个文件，全部有对应文档。
