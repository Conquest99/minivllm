# miniagent —— 手写 Agent 循环

> **第二部分 · Agent**（miniagent）。推理引擎部分见 [../minivllm/00-index.md](../minivllm/00-index.md)。
> 全部文档总览见 [../README.md](../README.md)。

## 一句话

**Agent 就是一个 `while` 循环加几条护栏**，不是框架。

```
messages = [system, user]
loop:
    reply = model(messages, tools)        # 文本，可能含工具调用
    calls = parse(reply)                  # 文本 -> 结构化调用
    if 没有调用:  return reply            # 模型认为做完了
    messages += assistant(reply, calls)   # 保留模型说的话
    for call in calls:
        result = tools.execute(call)      # 永不抛异常
        messages += tool(result)          # 观察结果
    if 情况不对: break                    # 见下面的护栏
```

其余全是护栏。**护栏才是 demo 和能上线的东西之间的差别。**

---

## 文件分工

| 文件 | 职责 | 关键点 | 文档 |
|---|---|---|---|
| [types.py](../../miniagent/types.py) | `Message` / `ToolCall` / `Step` / `AgentResult` | 全链路只用一种消息格式 | [types.md](types.md) |
| [tools.py](../../miniagent/tools.py) | `@tool` 装饰器 + JSON Schema 生成 + 执行边界 | 从函数签名和 docstring **自动推出** schema | [tools.md](tools.md) |
| [parser.py](../../miniagent/parser.py) | 把模型输出的文本解析成工具调用 | 兼容 4 种方言，坏输出不抛异常 | [parser.md](parser.md) |
| [builtin_tools.py](../../miniagent/builtin_tools.py) | 计算器、文本统计、沙箱读写 | 不用 `eval`；读是默认能力、写要显式授出 | [builtin_tools.md](builtin_tools.md) |
| [backends.py](../../miniagent/backends.py) | `complete(messages, tools) -> str` | 脚本 / 可调用 / 本地 mini-vllm / OpenAI 兼容 | [backends.md](backends.md) |
| [agent.py](../../miniagent/agent.py) | ReAct 循环 + 六条护栏 | 循环检测、步数上限、解析重试、一轮一改、改后必读、错误回灌 | [agent.md](agent.md) |
| [scripts/agent_demo.py](../../scripts/agent_demo.py) | 逐步打印 trace | 三种后端，从确定性到真实模型 | [scripts-agent_demo.md](scripts-agent_demo.md) |
| [scripts/agent_files_demo.py](../../scripts/agent_files_demo.py) | 真的建/改文件，并对比磁盘内容 | 小模型改错行还自信报成功的现场 | [scripts-agent_files_demo.md](scripts-agent_files_demo.md) |

测试（93 个，纯 CPU、无需模型）：

| 文件 | 验证什么 | 文档 |
|---|---|---|
| [test_agent_tools.py](../../tests/test_agent_tools.py) | schema 推导、参数校验、计算器安全、沙箱逃逸、写入回执行号 | [tests-tools.md](tests-tools.md) |
| [test_agent_parser.py](../../tests/test_agent_parser.py) | 4 种方言 + 容错 + 从不抛异常 | [tests-parser.md](tests-parser.md) |
| [test_agent_loop.py](../../tests/test_agent_loop.py) | 控制流全部分支 + 六条护栏 | [tests-loop.md](tests-loop.md) |

**建议阅读顺序**：本页（伪代码）→ `types` → `tools` → `parser` → `agent` → `backends` → `builtin_tools` → 两个 demo → 三个测试。

---

## 三个设计决定（面试会问）

### 1. 工具 schema 是「推」出来的，不是手写的

```python
@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression safely.

    Args:
        expression: The arithmetic expression, for example "(23 * 7 + 11) / 4".
    """
```

这个装饰器用 `inspect.signature` + `get_type_hints` + docstring 生成：

```json
{
  "type": "function",
  "function": {
    "name": "calculator",
    "description": "Evaluate an arithmetic expression safely.",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {"type": "string", "description": "The arithmetic expression, ..."}
      },
      "required": ["expression"]
    }
  }
}
```

**这就是 OpenAI function calling 的 payload 生成方式**——没有魔法，只有内省。手写 schema 迟早和函数签名不一致；推导出来的不可能不一致。

### 2. 工具错误是「观察」，不是「异常」

```python
try:
    value = t.func(**call.arguments)
except Exception as exc:
    return fail(f"{type(exc).__name__}: {exc}")
```

模型传了个错参数名，会发生什么？

```
step 0  call   calculator({"expr": "23 * 7"})
        !! result ERROR: unexpected argument(s) ['expr'] for calculator; allowed: expression

step 1  call   calculator({"expression": "23 * 7"})
        -> result 161
```

**如果把异常抛出去，这次运行就废了；把它变成一个观察，模型自己就改正了。** 这是 agent 和脚本的本质区别之一。

注意错误消息的写法：`allowed: expression` 直接告诉模型正确拼写，比 `TypeError: unexpected keyword argument` 有用得多。

### 3. 解析器必须容忍「坏输出」

模型不返回结构体，它返回**恰好包含结构体的文本**。而且每个模型方言不同：

| 方言 | 例子 |
|---|---|
| Qwen / Hermes | `<tool_call>{"name": "add", "arguments": {"a": 1}}</tool_call>` |
| JSON 代码块 | ` ```json {...} ``` ` |
| 裸 JSON | 整个回复就是一个对象 |
| ReAct | `Action: add` / `Action Input: {"a": 1}` |

所以 `parse()` 返回 `ParseResult(text, calls, errors)`——**三种都可能同时非空**：

* `calls` 非空 → 执行
* `errors` 非空 → 把错误回灌给模型让它重试
* `text` 是「去掉工具调用块之后剩下的散文」

最后一条容易忽略：模型可能说「让我算一下。」然后才发工具调用。**这句散文属于 assistant 消息，丢掉就丢了上下文。**

---

## 四条护栏

| 护栏 | 触发条件 | 为什么需要 |
|---|---|---|
| `max_steps` | 超过步数上限 | 模型可能永远调工具不停 |
| `max_identical_calls` | 同一工具 + **同一参数** 重复超过 N 次 | 观察没帮上忙，再试也一样 |
| `max_parse_errors` | 累计解析失败超过 N 次 | 坏格式可重试，但不能无限重试 |
| 工具异常 → `ok=False` | 工具内部报错 | 栈追踪教不会模型，可读的错误消息可以 |

循环检测用的是「工具名 + 参数」的稳定哈希（`ToolCall.key()`），**不是**「工具名」——同一个工具用不同参数反复调用是正常的（比如连续算好几步）。

---

## 后端：为什么抽象出 `complete()`

```python
def complete(messages, *, tools=None, max_tokens=None) -> str
```

一个方法换来三件事：

1. **可测**。`ScriptedBackend` 回放固定台词，`CallableBackend` 根据对话内容决定下一句——**循环的正确性可以完全脱离模型验证**（65 个测试，CPU 上 0.2 秒跑完）。
2. **本地**。`MiniVLLMBackend` 直接套用上一章的 `LLMEngine`，这就是为什么这个仓库里有一个推理引擎。
3. **可换**。`OpenAICompatBackend` 什么都能接（DeepSeek、vLLM server、Ollama、LM Studio）。

### 本地后端的一个关键细节

Qwen2.5 的 chat template **原生认 `tools=` 参数**，会自己渲染出官方的工具声明和 `<tool_call>` 格式：

```python
prompt = tokenizer.apply_chat_template(
    [m.to_hf() for m in messages],
    tools=tools or None,          # ← 模板自己插入 # Tools 段落
    add_generation_prompt=True,
    tokenize=False,
)
```

所以**不用手拼 prompt 字符串**，模型看到的就是它训练时的布局。

因为 `apply_chat_template` 只能由后端调用，构造 `EngineConfig` 时要设 `use_chat_template=False`，否则 `LLMEngine` 会再套一层 user 模板，把整个对话塞进一条 user 消息里。

---

## 实测

### 脚本后端（确定性，无需 GPU）

```
step 0   call  calculator({"expression": "23 * 7"})        -> 161
step 1   call  calculator({"expression": "(161 + 11) / 4"}) -> 43
step 2   model: The answer is 43.
```

### 本地 0.5B 模型（真实工具调用）

问题：`Calculate 23 * 7 with the calculator, then tell me the answer.`

```
step 0 (1268 ms)  call  calculator({"expression": "23 * 7"}) -> 161
step 1 (587 ms)   model: The result of 23 * 7 is 161.
stopped: final_answer   tool calls: 1 (0 failed)   wall: 1.86s
```

**0.5B 能正确发出 `<tool_call>` 格式并利用观察结果。**

但它算术不可靠。换一个需要保留括号的问题：

```
问题: What is (23 * 7 + 11) / 4?
step 0  call calculator({"expression": "23 * 7 + 11 / 4"})  -> 163.75   ← 括号丢了
step 1  model: ... is 163.75.                                          ← 自信地答错
```

**循环是对的，工具是对的，模型把表达式抄错了。** 这就是为什么 0.5B 只能用来验证机制，不能用来做真实任务——要上生产就换 API 或 ≥7B 的模型。

> 这个对比本身就是很好的面试素材：「我验证了循环的正确性，也验证了 0.5B 的可靠性边界」。

---

## 和 LangGraph / LangChain 的关系

手写这一遍之后，再去看框架就只是「它替我省了哪些样板」：

| 手写版 | 框架提供 |
|---|---|
| `while` 循环 | 图 / 状态机（可分支、可并行） |
| `messages` 列表 | 持久化 checkpoint、断点续跑 |
| `max_steps` | 递归上限、超时、人工介入（interrupt） |
| 自己 print trace | 可观测性面板 |
| — | 多 agent 协作、流式事件 |

**但没有任何框架会替你回答**：工具错误该怎么回灌、解析失败该重试几次、什么情况下该判定死循环。这些是这个文件存在的理由。

---

## 动手练习

1. 把 `max_identical_calls` 从 2 改成 5，跑 `--backend scripted-error`，观察需要几次才触发循环检测。
2. 在 `builtin_tools.py` 里加一个 `@tool def now() -> str`（返回当前时间），用 `--backend local` 问「现在几点」，看 0.5B 能不能挑对工具。
3. 把 `ToolRegistry.execute` 里 `except Exception` 改成 `except ValueError`，跑 `test_tool_exception_becomes_an_error_result`，看它如何挂掉——这就是「工具异常必须兜住」的反证。
4. 给 `Agent` 加一个 `max_tool_errors`：累计失败超过 N 次就停。想想这个护栏和 `max_identical_calls` 有什么区别（提示：模型可以每次都换参数但每次都失败）。
