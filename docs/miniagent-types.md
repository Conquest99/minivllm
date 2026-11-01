# `miniagent/types.py` 详解

## 这个文件干什么

定义**全链路唯一的一种消息格式**。

Agent 循环只认这 6 个类型：`Role` / `ToolCall` / `ToolResult` / `Message` / `Step` / `AgentResult`（加一个 `StopReason`）。

没有任何逻辑——只有数据结构。但这恰恰是最该先想清楚的地方：**框架千差万别，中间表示只有一个。**

---

## 1. `Role`：为什么继承 `str`

```python
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    def __str__(self) -> str:
        return self.value
```

继承 `str` 有两个实际好处：

1. **可以直接当字符串用**。`{"role": role}` 不需要 `.value`，JSON 序列化也不用手动转换。
2. **`Role.USER == "user"` 为真**。写测试时不用在两个表示之间来回换。

重写 `__str__` 是因为 `Enum` 默认打印 `Role.USER`，而 `f"Message({self.role}, ...)"` 里你想要的显然是 `user`。

> 不继承 `str` 的 `Enum`（即 `class Role(Enum)`）在 3.11+ 有 `StrEnum`，但项目要求 3.10，所以用 `str, Enum` 混合。

---

## 2. `ToolCall.key()`：循环检测的地基

```python
def key(self) -> str:
    return f"{self.name}:{json.dumps(self.arguments, sort_keys=True, ensure_ascii=False)}"
```

护栏要用「同一工具 + 同一参数」判断死循环。三个细节：

* **`sort_keys=True`**：`{"a":1,"b":2}` 和 `{"b":2,"a":1}` 是同一个调用，不排序会被当成两次不同的调用，循环检测失效。
* **`ensure_ascii=False`**：中文参数不会被转义，key 可读，调试时能直接看出模型在重复什么。
* **为什么不用 `hash()`**：Python 的 `str.__hash__` 每次进程启动都不同（PYTHONHASHSEED 随机化），但它只在进程内用，无所谓。用字符串是因为 `Counter` 的 key 要可读——报错时会打印出来。

---

## 3. `ToolResult.to_observation()`

```python
def to_observation(self) -> str:
    return f"[{self.name}] {self.content}"
```

看起来只是加了个前缀，但这是**被 Qwen 模板逼出来的**。

Qwen2.5 的 chat template 渲染 `role: "tool"` 时只保留 `content`：

```jinja
{{- '\n<tool_response>\n' }}{{- message.content }}{{- '\n</tool_response>' }}
```

`name` 和 `tool_call_id` 全丢了。如果一个 step 里调了两个工具，模型就分不清哪条结果属于哪个调用。所以在正文里把名字重复一遍。

> OpenAI 的协议是靠 `tool_call_id` 对应的，Qwen 的模板不支持。**这是「协议差异」的典型例子：同样的语义，不同的载体。**

---

## 4. `Message.to_hf()`：唯一的格式转换点

```python
def to_hf(self) -> dict[str, Any]:
    if self.role is Role.ASSISTANT and self.tool_calls:
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ],
        }
    if self.role is Role.TOOL:
        return {"role": "tool", "content": self.content}
    return {"role": self.role.value, "content": self.content}
```

三个分支，对应模板里的三种渲染路径：

| 分支 | 模板行为 |
|---|---|
| assistant + tool_calls | 渲染 `content` + 每个 `<tool_call>{...}</tool_call>` |
| tool | 包成 `<tool_response>`，且**连续的 tool 消息合并进同一个 user 轮** |
| 其它 | 普通 `<|im_start|>role\ncontent<|im_end|>` |

关键点：**`arguments` 必须是 dict，不能是 JSON 字符串**。模板里是 `{{ tool_call.arguments | tojson }}`，如果传字符串会得到双重编码的 `"{\"a\": 1}"`。

**整个项目只有这一个地方产出 HF 格式**。想换后端（比如 Anthropic 的格式），只改这一个方法。

---

## 5. `Step`：可观测性的最小单位

```python
@dataclass
class Step:
    index: int
    assistant_text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0
```

一个 step = **一次模型调用 + 它触发的所有工具执行**。

`summary()` 是给控制台看的：

```python
def summary(self) -> str:
    if self.parse_errors:
        return f"step {self.index}: {len(self.parse_errors)} parse error(s)"
    if not self.tool_calls:
        return f"step {self.index}: final answer"
    parts = []
    for call, result in zip(self.tool_calls, self.tool_results, strict=False):
        status = "ok" if result.ok else "error"
        parts.append(f"{call.name}({status})")
    return f"step {self.index}: {', '.join(parts)}"
```

输出形如 `step 0: calculator(ok), text_stats(error)`。**没有 `duration_s` 之外的任何数字，但足够一眼看出这步干了什么。**

> 为什么工具结果单独一个字段，而不塞进某个日志字符串？因为 `AgentResult.num_tool_errors` 要统计它，测试也要断言它。**结构化优先于字符串。**

---

## 6. `AgentResult`：三个派生属性

```python
@property
def num_tool_calls(self) -> int:
    return sum(len(s.tool_calls) for s in self.steps)

@property
def num_tool_errors(self) -> int:
    return sum(1 for s in self.steps for r in s.tool_results if not r.ok)

@property
def succeeded(self) -> bool:
    return self.stopped_reason is StopReason.FINAL_ANSWER and bool(self.answer)
```

三条都是**从 `steps` 算出来的，不存字段**。理由和 `minivllm/sequence.py` 一样：**派生状态永远优于冗余状态**，存两份迟早不一致。

`succeeded` 的定义值得注意：不只是「正常结束」，还要求 `answer` 非空。**空答案不算成功**——模型可能返回一个空字符串然后停止。

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `class Role(str, Enum)` | 字符串枚举，可直接当 dict 值 |
| 重写 `__str__` | 让 `f"{role}"` 输出 `user` 而不是 `Role.USER` |
| `@dataclass` + `field(default_factory=list)` | 每个实例独立的 list（避免可变默认值陷阱） |
| `@property` | 派生量不占存储 |
| `json.dumps(..., sort_keys=True)` | 参数顺序无关的稳定键 |
| `zip(..., strict=False)` | 长度不等时静默取短 |

---

## 设计取舍

**为什么不用 dict 传消息？**

`dict` 能省掉这 140 行，但会丢掉：IDE 补全、拼写检查、类型提示，以及「哪些字段是合法的」这个约束。链路上会经过 6 个模块，任何一个手滑写错 key 都要跑到运行时才发现。

**为什么 `Message` 里同时有 `tool_calls` 和 `content`？**

因为模型可能**先说一句话再调工具**：「让我算一下。`<tool_call>...</tool_call>`」。两个都要保留——散文属于上下文，丢了下一轮模型就不知道自己说过什么。

**为什么 `StopReason` 是枚举而不是字符串？**

因为调用方要据此分支（重试 / 报错 / 展示）。字符串拼错不报错，枚举拼错立刻 `AttributeError`。

---

## 常见误解

* 「`ToolCall.arguments` 可以是 JSON 字符串」——在自己的类型里必须是 dict。字符串只在解析阶段短暂存在（见 `parser.py` 会重新解析）。
* 「`Step` 和一次模型调用一一对应」——对，但一个 step 可能有 0 个（最终回答）或 N 个工具调用。
* 「`succeeded` 等价于 `stopped_reason == FINAL_ANSWER`」——不等价，还要求答案非空。
* 「`to_hf` 是通用的」——它只针对 Qwen 的模板。换模型要确认模板是否接受 `tool_calls` 这个 key。

---

## 动手练习

1. 删掉 `ToolCall.key()` 里的 `sort_keys=True`，把 `test_identical_calls_trip_the_loop_guard` 改成两个参数顺序不同的相同调用，看循环检测是否失效。
2. 把 `to_observation` 的前缀去掉，用 `--backend local` 问一个需要调两个工具的问题，观察模型是否会搞混结果。
3. 给 `AgentResult` 加一个 `tool_names_used` 属性，返回去重后的工具名列表。
4. 把 `ToolCall.arguments` 的类型注解改成 `str`，跑 `mypy`（如果装了）或直接看 `to_hf` 里 `| tojson` 会发生什么。
