# `miniagent/backends.py` 详解

## 这个文件干什么

**回答「token 从哪来」这个问题。**

循环不关心对面是 0.5B 的本地权重还是托管 API，所以唯一的契约是：

```python
def complete(messages, *, tools=None, max_tokens=None) -> str
```

一个方法换来三件事：

| 收益 | 怎么来的 |
|---|---|
| **可测** | `ScriptedBackend` 回放台词、`CallableBackend` 按对话内容回应 → 循环的正确性完全脱离模型 |
| **本地** | `MiniVLLMBackend` 套用上一章的 `LLMEngine` |
| **可换** | `OpenAICompatBackend` 接任何 OpenAI 兼容端点 |

---

## 1. `Protocol` 而不是 ABC

```python
class LLMBackend(Protocol):
    name: str
    def complete(self, messages: list[Message], *, tools=None, max_tokens=None) -> str: ...
```

`typing.Protocol` 是**结构化类型**（鸭子类型的静态版）：只要一个类有 `name` 和 `complete`，它就是 `LLMBackend`，**不需要继承**。

好处：

* 测试用的 `ScriptedBackend` 不用 `import` 任何基类
* 第三方后端可以独立演进
* `mypy` 照样能检查

对比 `ABC`：`class ScriptedBackend(LLMBackend)` 强制继承，多一层耦合，而且容易变成「为了通过继承而写空实现」。

> `Protocol` 是 3.8+ 的标准库。`@runtime_checkable` 才能用 `isinstance`，这里不需要——静态检查足够。

### 为什么 `tools` 和 `max_tokens` 是 keyword-only（`*`）

```python
def complete(self, messages, *, tools=None, max_tokens=None) -> str
```

`*` 之后只能按关键字传。理由是**防止位置参数错位**：如果签名是 `(messages, tools, max_tokens)`，调用方很容易写成 `complete(msgs, 512)` 把 512 当成 tools 传进去。关键字强制让意图显式。

这是写库的一般习惯：**超过一个可选参数就用 keyword-only**。

---

## 2. `ScriptedBackend`：回放固定台词

```python
class ScriptedBackend:
    name = "scripted"

    def __init__(self, replies):
        self._replies = deque(replies)
        self.requests: list[list[Message]] = []

    def complete(self, messages, *, tools=None, max_tokens=None) -> str:
        self.requests.append(list(messages))
        if not self._replies:
            raise RuntimeError(
                f"ScriptedBackend ran out of replies after {len(self.requests) - 1} turn(s); "
                "the agent asked for one more"
            )
        return self._replies.popleft()
```

三个细节：

**① `deque` + `popleft()`**

`list.pop(0)` 是 O(n)，`deque.popleft()` 是 O(1)。测试里只有几步，差别无所谓，但这是正确习惯。

**② `self.requests` 记录每一次发送的对话**

让测试能断言「agent 到底给模型看了什么」：

```python
second_turn = backend.requests[1]
assert any(m.role is Role.TOOL and "43" in m.content for m in second_turn)
```

**没有这个记录，"观察结果真的回灌了吗" 就无法验证**——而这是 agent 最核心的机制。

注意 `list(messages)` 复制了一份。不复制的话，agent 后续 `append` 会同时改到记录里的列表，测试看到的就永远是最终状态。

**③ 台词用完了要抛异常，不能返回空串**

如果返回 `""`，`parse("")` 得到「无调用」→ agent 认为模型给出了空答案 → 停在 `FINAL_ANSWER`。**测试会静默通过，但什么都没测到。**

抛异常能让测试立刻失败，并且错误信息直接告诉你「agent 多请求了一轮」——这正是你需要的诊断信息（比如 `max_steps` 是不是设太大）。

---

## 3. `CallableBackend`：按对话内容决定

```python
class CallableBackend:
    name = "callable"

    def __init__(self, fn: Callable[[list[Message]], str]) -> None:
        self._fn = fn
        self.requests: list[list[Message]] = []

    def complete(self, messages, *, tools=None, max_tokens=None) -> str:
        self.requests.append(list(messages))
        return self._fn(list(messages))
```

比固定列表强的地方：**能读取工具结果并据此决定下一步**。这才是模拟「一个真的会推理的模型」。

`test_two_step_chain` 用它验证多步链：

```python
def decide(messages):
    tool_messages = [m for m in messages if m.role is Role.TOOL]
    if not tool_messages:
        return tool_call("calculator", expression="23*7")
    if len(tool_messages) == 1:
        first = tool_messages[0].content.split()[-1]
        return tool_call("calculator", expression=f"({first}+11)/4")
    return "The result is 43."
```

传 `list(messages)` 而不是 `messages`，同样是为了防止闭包里的函数改动 agent 的状态。

---

## 4. `MiniVLLMBackend`：本地引擎

### 关键：`use_chat_template=False`

```python
cfg = EngineConfig(
    model=str(model or DEFAULT_MODEL),
    ...
    # The backend applies the chat template itself, because only it can
    # pass tools= to apply_chat_template.
    use_chat_template=False,
)
self.engine = LLMEngine(cfg)
```

`LLMEngine._encode_prompt` 默认会把 prompt 包成 `[{"role": "user", "content": prompt}]` 再套模板。但 agent 要传的是**整段多轮对话 + tools**，所以模板必须由后端来套。

如果不关掉，会发生：后端套一遍完整模板 → `LLMEngine` 把结果当成 user 内容再套一遍 → 模型看到的是「一条 user 消息里塞着整个对话」，工具声明也被埋在正文里。**输出会明显退化，但不会报错**——这类 bug 最难查。

### 为什么 `apply_chat_template` 能直接传 `tools`

Qwen2.5 的模板里有这段：

```jinja
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    ... 你的 system prompt ...
    {{- "\n\n# Tools\n\nYou may call one or more functions ...\n<tools>" }}
    {%- for tool in tools %}{{- "\n" }}{{- tool | tojson }}{%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object ... <tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    ... 普通 system ...
{%- endif %}
```

**模板自己是知道工具调用格式的。** 所以传 `tools=schemas` 就够了，不需要手拼一个 `# Tools` 段落。

这是个重要认知：**「function calling」在开源模型里大量是靠 chat template 实现的**，不是模型天生就会。`tools` 参数通过 Jinja 变成文本，模型再按文本格式生成。

### 三个参数为什么这么设

```python
SamplingParams(
    temperature=self.temperature,   # 默认 0.0 → 贪心，可复现
    max_tokens=max_tokens or 512,   # 一次工具调用需要几十个 token，64 不够
    ignore_eos=False,               # 需要 <|im_end|> 正常截断
)
```

`max_tokens=512` 而不是引擎默认的 64：一个 `<tool_call>{"name": "calculator", "arguments": {"expression": "..."}}</tool_call>` 大概 40-60 个 token，加上前后散文，64 会截断在中间，导致解析失败。

### 懒导入

```python
def __init__(self, model=None, *, dtype="float16", ...):
    from minivllm.config import EngineConfig
    from minivllm.llm_engine import LLMEngine
```

`minivllm` 的导入是**放在函数体里的**。这样 `import miniagent` 不需要 torch——测试 parser 和 tools 的 65 个用例可以在没有 GPU、甚至没装 torch 的环境跑。

这是「延迟导入」的标准用法：**只在真正需要重量级依赖的入口处导入。**

---

## 5. `OpenAICompatBackend`：接任何兼容端点

```python
def complete(self, messages, *, tools=None, max_tokens=None) -> str:
    body = json.dumps(self._payload(messages, tools, max_tokens)).encode("utf-8")
    request = urllib.request.Request(
        f"{self.base_url}/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
        },
        method="POST",
    )
    ...
```

### 为什么用 `urllib` 而不是 `requests`

`requests` 确实更好用，但它是个额外依赖。这个文件只需要发一个 POST 收一个 JSON，标准库够用。

```python
**({"Authorization": ...} if self.api_key else {})
```

用字典解包做条件头。等价的写法是 if 判断后赋值，但这行更紧凑。**没有 API key 就不带这个头**——本地 Ollama / LM Studio 不需要认证。

### 结构化 `tool_calls` 转回文本

```python
message = data["choices"][0]["message"]
text = (message.get("content") or "").strip()
calls = message.get("tool_calls") or []
if not calls:
    return text

rendered = []
for call in calls:
    function = call.get("function", {})
    raw_args = function.get("arguments") or "{}"
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except json.JSONDecodeError:
        args = {"_raw": raw_args}
    rendered.append(format_tool_call(function.get("name", ""), args))
return "\n".join(filter(None, [text, *rendered]))
```

**这是本文件最重要的一个决定。**

OpenAI 协议的 `tool_calls` 是结构化的（不是文本）。如果让它直接穿到 agent 里，那 `parse()` 就用不上了——下游会有两条完全不同的代码路径。

所以这里把它**渲染回 `<tool_call>` 文本**，让整个 agent 只处理一种表示。

代价是多一次序列化/反序列化（微不足道）。收益是：

* `parse()` 只有一条路径要维护
* 测试可以统一用文本
* 换协议时只改这一个方法

`filter(None, [text, *rendered])` 去掉空字符串——当模型只返回工具调用、没有散文时，`text` 是 `""`，不该产生一个空行。

---

## 6. `build_backend` 工厂

```python
def build_backend(kind: str, **kwargs) -> LLMBackend:
    if kind == "local":
        return MiniVLLMBackend(**kwargs)
    if kind in {"api", "openai"}:
        return OpenAICompatBackend(**kwargs)
    raise ValueError(f"unknown backend {kind!r}; choose 'local' or 'api'")
```

给 CLI 用的。`agent_demo.py` 里另有一套 `describe_backend`，因为它还要处理 `scripted` / `scripted-error`。

> 两处重复了类似逻辑，是有意的：`build_backend` 是给「真后端」用的库函数，demo 里的 `describe_backend` 多了两个测试用的假后端。合并的话 demo 就得往库里加测试专用的东西。

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `typing.Protocol` | 结构化类型，无需继承 |
| keyword-only 参数（`*`） | 防止位置参数错位 |
| `collections.deque` | O(1) 左端弹出 |
| `collections.abc.Callable` / `Iterable` | 类型提示 |
| `urllib.request.Request` | 标准库 HTTP |
| `urllib.error.HTTPError` | 捕获并读取响应体 |
| 字典解包做条件头 | `**({...} if cond else {})` |
| `filter(None, ...)` | 去掉空字符串 |
| 函数内懒导入 | 重量级依赖按需加载 |

---

## 设计取舍

**为什么协议只有一个 `complete()`，不支持流式？**

流式（`yield` token）对 SSE 服务很重要，但会让协议复杂一倍（要处理「半个工具调用」的边界）。这个项目要的是**看清结构**，一次性返回最直白。

真要加，应该是第二个方法 `stream()`，而不是改 `complete()`。

**为什么 `ScriptedBackend` 不检查 `tools`？**

因为它是假的。让它什么都不做，测试才能专注在循环逻辑上。要验证"工具被正确宣传给模型"，单独写一个 spy（`test_tools_are_advertised_to_the_backend` 就是这么做的）。

**为什么 `OpenAICompatBackend` 不重试？**

重试策略属于上层（agent 还是调用方？）。这里失败就抛，`Agent.run` 会捕获并返回 `BACKEND_ERROR`。**把重试放在一个地方**，比散落在每个后端里好。

---

## 常见误解

* 「`Protocol` 需要在运行时检查」——不需要，`mypy` 静态检查就够。要 `isinstance` 才加 `@runtime_checkable`。
* 「`use_chat_template=False` 会丢掉聊天格式」——不会，后端自己套了完整模板，只是不再套第二层。
* 「`tools=` 是 API 参数，本地模型用不上」——Qwen 的 chat template 认它，会渲染成文本提示。
* 「`ScriptedBackend` 是玩具」——它是这个项目里 65 个 agent 测试的基础设施。
* 「`OpenAICompatBackend` 只支持 OpenAI」——任何实现该协议的都能用（DeepSeek、vLLM、Ollama、vLLM 的 `--api-key` 模式）。

---

## 动手练习

1. 把 `MiniVLLMBackend` 的 `use_chat_template` 改回 `True`，用 `--backend local` 跑，观察输出如何退化（模型可能开始重复 `<|im_start|>` 字样）。
2. 把 `ScriptedBackend` 台词用完时改成 `return ""`，跑 `tests/test_agent_loop.py`，看哪些测试会「假通过」。
3. 给 `OpenAICompatBackend` 加一个 `stream=False` 参数，用 vLLM 的 OpenAI server（`python -m vllm.entrypoints.openai.api_server`）实测一次。
4. 写一个新的 `EchoBackend`，把收到的最后一条消息原样返回，然后想：它能用来测循环的哪一部分？
