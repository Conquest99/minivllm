# `miniagent/parser.py` 详解

## 这个文件干什么

**把模型输出的文本，变成结构化的工具调用。**

模型不返回 struct，它返回**恰好含有 struct 的文本**。这个文件是适配器，也是真实 agent 最常失去鲁棒性的地方——因为每个模型都有自己的方言。

| 方言 | 例子 |
|---|---|
| Qwen / Hermes | `<tool_call>{"name": "add", "arguments": {"a": 1}}</tool_call>` |
| JSON 代码块 | ` ```json {"name": "add", ...} ``` ` |
| 裸 JSON | 整个回复就是一个对象 |
| ReAct | `Action: add` / `Action Input: {"a": 1}` |

---

## 1. 输出为什么是三个字段

```python
@dataclass
class ParseResult:
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
```

**三者可能同时非空**，这是刻意的：

* `calls` → 去执行
* `errors` → 回灌给模型让它重试
* `text` → 去掉工具调用块之后剩下的散文

第三条最容易忽略。模型可能说：

```
让我算一下。
<tool_call>{"name": "calculator", "arguments": {"expression": "23*7"}}</tool_call>
```

那句「让我算一下。」**属于 assistant 消息**。丢掉它，下一轮模型就不知道自己说过什么，上下文就断了。

### 为什么不用异常

坏 JSON 是**正常事件**，不是异常情况。抛出去 = 一次可恢复的抖动变成一次运行失败。

```python
result = parse(text)
if result.errors:
    messages.append(...)   # 告诉模型哪里错了，重试
```

`max_parse_errors` 护栏负责在重试太多次时喊停。**能用返回值表达的失败，就不要用异常。**

---

## 2. 四个正则

```python
_TOOL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)\s*```", re.DOTALL)
_REACT = re.compile(
    r"Action\s*:\s*(?P<name>[A-Za-z_][\w.\-]*)\s*(?:\r?\n)+\s*Action\s*Input\s*:\s*"
    r"(?P<args>\{[^\n]*\}|[^\n]+)",
    re.IGNORECASE,
)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
```

### `re.DOTALL` 的必要性

`.*?` 默认不跨行。`<tool_call>\n{...}\n</tool_call>` 里有换行，不加 `DOTALL` 就匹配不到。**所有「截取一段可能含换行的内容」的正则都要 `DOTALL`。**

### `.*?` 非贪婪

`<tool_call>(.*?)</tool_call>` 用非贪婪，这样多个块能分别匹配：

```
<tool_call>A</tool_call> <tool_call>B</tool_call>
```

贪婪版会一次性吞掉 `A</tool_call> <tool_call>B`。

### ReAct 的 `\{[^\n]*\}` 为什么不是 `\{.*?\}`

这是踩过的坑。对于 `{"query": "x", "opts": {"limit": 3}}`：

| 写法 | 匹配结果 |
|---|---|
| `\{.*?\}` | `{"query": "x", "opts": {"limit": 3}` ← **少一个 `}`** |
| `\{[^\n]*\}` | `{"query": "x", "opts": {"limit": 3}}` ✅ |
| `\{.*\}` + DOTALL | 会跨行吞掉后面的 `Observation:` |

非贪婪会停在**第一个** `}`，嵌套对象就断了。所以用「贪婪但不跨行」——`[^\n]*` 是贪婪的，配合 `\}` 会匹配到**本行最后一个** `}`。

---

## 3. `_load_object`：容错解析

```python
def _load_object(blob: str) -> dict | None:
    blob = blob.strip()
    if not blob:
        return None
    if blob.startswith("```"):
        inner = _FENCE.search(blob)
        if inner:
            blob = inner.group(1).strip()

    for candidate in (blob, _TRAILING_COMMA.sub(r"\1", blob)):
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            try:
                value = ast.literal_eval(candidate)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                continue
        return value if isinstance(value, dict) else None
    return None
```

三层容错，逐层降级：

| 尝试 | 能处理的 |
|---|---|
| `json.loads` | 标准 JSON |
| `_TRAILING_COMMA` + `json.loads` | `{"a": 1,}` 这种尾逗号 |
| `ast.literal_eval` | 单引号 `{'a': 1}`、Python 字面量 |

### 为什么 `ast.literal_eval` 是安全的

它**只**接受字面量：数字、字符串、元组、列表、字典、集合、布尔、None。**不能求值、不能调函数、不能属性访问**。

```python
ast.literal_eval("__import__('os')")     # ValueError
ast.literal_eval("{'a': 1}")             # {'a': 1}
```

这正是我们需要的：把模型写的「差不多是 JSON」的东西解析出来，同时不给它执行代码的能力。

> 注意 `ast.literal_eval('{"a": true}')` 会失败——`true` 不是 Python 字面量（是 `True`）。所以 **必须 json 先试**。

### 为什么捕获 `MemoryError` / `RecursionError`

`ast.literal_eval` 在遇到极深嵌套时会爆栈。模型输出是不可信输入，**不该让一个畸形字符串拖垮进程**。这是个容易被忽略的防御点。

---

## 4. `_normalise`：接纳模型的各种拼写

```python
_NAME_KEYS = ("name", "tool", "tool_name", "function")
_ARG_KEYS = ("arguments", "args", "parameters", "arguments_json", "input")

def _normalise(obj, raw):
    name = None
    for key in _NAME_KEYS:
        if key in obj:
            name = obj[key]
            break
    if isinstance(name, dict):
        name = name.get("name")

    args = {}
    for key in _ARG_KEYS:
        if key in obj:
            args = obj[key]
            break

    if isinstance(args, str):
        parsed = _load_object(args)
        args = parsed if parsed is not None else {"_raw": args}
    if args is None:
        args = {}
    if not isinstance(args, dict):
        args = {"_raw": args}

    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool call has no usable 'name' field")

    return ToolCall(name=name.strip(), arguments=args, raw=raw)
```

四种情况都覆盖了：

1. `{"name": "calc", "arguments": {...}}` —— 标准
2. `{"function": {"name": "calc"}, "parameters": {...}}` —— OpenAI 嵌套式
3. `{"name": "calc", "arguments": "{\"x\": 1}"}` —— **参数是字符串**（模型很爱这么干）
4. `{"name": "calc", "arguments": [1, 2]}` —— 参数是数组 → 包成 `{"_raw": [...]}`

第 3 种值得单独说：模型经常把 arguments 序列化成字符串。必须再 `_load_object` 一次，否则工具会收到一个字符串而不是 dict。

第 4 种兜底 `{"_raw": ...}` 是刻意的：**让它走到 `validate` 报错**，而不是在这里静默丢弃。错误信息会告诉模型「unexpected argument(s) ['_raw']」，它就知道格式不对了。

---

## 5. 主流程：四级降级

```python
def parse(text: str) -> ParseResult:
    calls, errors, leftover = [], [], text

    # 1) <tool_call> 标签
    for match in _TOOL_TAG.finditer(text):
        ...
    if calls or errors:
        leftover = _TOOL_TAG.sub("", text)

    # 2) ```json 代码块
    if not calls and not errors:
        ...

    # 3) 整个回复是一个 JSON 对象
    if not calls and not errors:
        ...

    # 4) ReAct
    if not calls and not errors:
        ...
```

**顺序不是随意的**：

1. `<tool_call>` 最优先，因为它是 Qwen 的官方格式，也是唯一**明确无歧义**的（有标签包着）。
2. 代码块次之——也是明确的标记。
3. 裸 JSON 再次之——整个回复都是 JSON 才认，否则容易误判（正常文本里可能含 `{`）。
4. ReAct 最后，因为它最容易误匹配（任何含 `Action:` 的文本）。

每一级都用 `if not calls and not errors` 门控，**一旦某一级有结果就停**。避免同一个调用被解析两遍。

---

## 6. 一个具体的警告

`_resolve_lone_json` 要求**整个文本**就是 JSON：

```python
def _resolve_lone_json(text: str) -> list[ToolCall] | None:
    obj = _load_object(text)
    if obj is None:
        return None
    if not any(k in obj for k in _NAME_KEYS):
        return None
    return [_normalise(obj, text)]
```

如果一个正常回答恰好是 `{"name": "something"}` 形式的 JSON（比如用户让它输出 JSON），会被误判成工具调用。

**这是取舍，不是 bug**：为了兼容更多方言，牺牲了一点精确性。真要严格，就要求模型必须用 `<tool_call>` 标签——但那样对不支持标签的模型就不工作了。

---

## 7. `format_tool_call`：反向渲染

```python
def format_tool_call(name: str, arguments: dict, *, style: str = "qwen") -> str:
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    if style == "qwen":
        return f"<tool_call>\n{payload}\n</tool_call>"
    return payload
```

谁在用？`OpenAICompatBackend`。当 API 返回结构化的 `tool_calls` 时，把它**转回文本形式**，这样整个下游只需要处理一种表示。

> 反向设计：如果让 `parse` 直接产出结构化结果、`format` 再转回去，就形成一个闭环。测试 `test_format_tool_call_round_trips` 就是在验证这个闭环。

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `re.DOTALL` | 让 `.` 匹配换行 |
| `.*?` 非贪婪 | 多个块分别匹配 |
| `(?P<name>...)` | 命名捕获组 |
| `re.finditer` | 遍历所有匹配 |
| `re.sub` | 删除匹配（剥离标签） |
| `json.loads` | 标准 JSON |
| `ast.literal_eval` | 安全的 Python 字面量求值 |
| `str.partition` | 三分字符串 |
| `@dataclass` | ParseResult |

---

## 设计取舍

**为什么不用正规的 JSON 流式解析器？**

因为模型输出不是合法 JSON（前面有散文、后面有标签）。需要的是「从文本里挖出 JSON」而不是「解析 JSON」。正则 + `json.loads` 是最直接的组合。

**为什么把 ReAct 放最后一级？**

它最容易误判，而且和其他三种格式互斥（ReAct 的模型不会同时输出 `<tool_call>`）。放在最后意味着「只有前面都不成立时才试它」。

**为什么 `errors` 是列表而不是单个字符串？**

一个回复可能包含多个坏块：`<tool_call>bad1</tool_call><tool_call>bad2</tool_call>`。全部报告比只报第一个更有用。

---

## 常见误解

* 「解析失败说明模型坏了」——不一定，可能是格式方言没覆盖。
* 「`ast.literal_eval` 等于 `eval`」——完全不同，它不能执行任意代码。
* 「`text` 字段就是原始输入」——是**剥掉工具调用块之后**的剩余部分。
* 「一个回复只能有一个工具调用」——`calls` 是列表，测试里验证过两个。
* 「尾逗号修复对所有情况有效」——只修 `,}` 和 `,]` 两种。

---

## 动手练习

1. 构造 `<tool_call>{"name": "a", "arguments": {}}</tool_call>` 和一个不闭合的 `<tool_call>{"name": "b"}`，看 `parse` 返回什么。

   **答案**：闭合的正常返回 `[('a', {})]`；不闭合的返回 `calls=[]`、`errors=[]`、
   `text='<tool_call>{"name": "b"}'` —— 也就是说**没有任何东西报错**。

   而这个形状在真实场景里几乎必然出现：**`max_tokens` 把回复砍在半路**。`_TOOL_TAG`
   要求闭合标签，所以一个都不匹配，`errors` 也没机会被填；而 `Agent.run` 判断「模型答完
   了」的条件正是「`calls` 和 `errors` 都空」——

   ```python
   if not parsed.calls and not parsed.errors:      # ← 两个都空
       return result(parsed.text.strip() or reply.strip(), reason)   # 半截调用当成答案
   ```

   实测（用 `EchoBackend` 回显一段截断的调用）：

   ```
   stopped : final_answer
   answer  : '<tool_call>{"name": "calculator", "arguments": {"expressio'
   tools   : 0 | errors: 0
   ```

   修法是让 `parse` 在发现**未闭合**的 `<tool_call>` 时明确报错（并给出「重发一个完整
   块」的提示），于是 agent 走修复分支，而不是把半截调用当答案。现在同一个输入得到
   `stopped: too_many_errors`。

   顺带挖出两个更深的坑：

   * `ast.literal_eval` 的官方契约包含 **`TypeError`**，而原代码只捕
     `(ValueError, SyntaxError, MemoryError, RecursionError)`。`{"a": {{1, 2}}}` 会让它
     去建「装 set 的 set」，`TypeError: unhashable type: 'set'` **一路穿出 `parse`** ——
     直接违反模块的「坏输出永不抛异常」契约。
   * 「整段就是一个 `{...}` 但解析不了」原本静默变成了最终答案。现在会报错，因为整段被
     一对花括号包着是这个意图的强信号（只在**整段**成立时才判，句子里夹一个 `{x}` 不算）。

2. 把 `_REACT` 里的 `\{[^\n]*\}` 改成 `\{.*?\}`，跑 `test_react_with_nested_json_arguments`，看它如何失败。
3. 写一个回复 `我现在调用工具 {"name": "calculator", "arguments": {}}`（前后有中文），确认裸 JSON 分支**不会**误判（因为没有标签包裹且不是整段 JSON），然后想想要不要补一个「任意位置 JSON」的分支。
4. 把 `_load_object` 里 `ast.literal_eval` 那层删掉，跑 `test_single_quotes_are_tolerated`，看它失败。
