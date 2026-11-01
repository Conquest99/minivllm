# `tests/test_agent_parser.py` 详解

## 这个文件干什么

**证明 `parse()` 能吃下模型可能吐出的任何方言，并且永远不抛异常。**

17 个用例，纯 CPU，毫秒级。

它的价值不是「覆盖率高」，而是**列出了格式的所有已知变体**——你以后换模型时，对着这个文件加一条用例就知道新方言有没有覆盖。

---

## 1. 四种方言，各一条

### Qwen 标签

```python
def test_qwen_tool_call_tag():
    payload = '{"name": "calculator", "arguments": {"expression": "1+1"}}'
    text = f"Let me compute.\n<tool_call>\n{payload}\n</tool_call>"
    result = parse(text)

    assert result.has_calls
    assert not result.errors
    assert result.text == "Let me compute."
    assert result.calls[0].arguments == {"expression": "1+1"}
```

**`result.text == "Let me compute."` 这条最容易漏**。它验证的是「散文被保留、标签被剥离」。

如果实现是「整段返回」，`text` 会包含标签；如果是「返回空」，散文就丢了。两种都会让下一轮上下文出错。

用 `payload` 变量而不是内联字符串，是因为原来的内联版本超过 110 字符被 ruff 拦了（E501）。**这个重构顺手让「哪个是 JSON」更清楚。**

### JSON 代码块

```python
def test_json_fence():
    text = 'Checking.\n```json\n{"name": "calculator", "arguments": {"expression": "2*3"}}\n```'
    result = parse(text)
    assert result.has_calls
    assert result.calls[0].arguments == {"expression": "2*3"}
    assert result.text == "Checking."
```

### 裸 JSON

```python
def test_bare_json_object():
    result = parse('{"name": "calculator", "arguments": {"expression": "9-4"}}')
    assert result.has_calls
```

整个回复就是 JSON。**注意没有 `assert result.text == ""`**——实现里确实设了 `leftover = ""`，但这条测试没断言它，属于遗漏（可以补）。

### ReAct

```python
def test_react_style():
    text = "I need to multiply.\nAction: calculator\nAction Input: {\"expression\": \"23*7\"}\n"
    result = parse(text)
    assert result.has_calls
    assert result.calls[0].name == "calculator"
    assert result.calls[0].arguments == {"expression": "23*7"}
```

---

## 2. 三个最容易崩的边界

### 嵌套 JSON（`test_react_with_nested_json_arguments`）

```python
def test_react_with_nested_json_arguments():
    text = 'Action: search\nAction Input: {"query": "x", "opts": {"limit": 3}}'
    result = parse(text)
    assert result.calls[0].arguments == {"query": "x", "opts": {"limit": 3}}
```

**这条是我实际踩坑之后补的。**

`_REACT` 里如果用 `\{.*?\}`（非贪婪），会匹配到第一个 `}`：

```
{"query": "x", "opts": {"limit": 3}
                              ↑ 停在这里，少一个 }
```

JSON 解析失败 → 整条回复被当成解析错误。而实际模型经常输出嵌套对象。

修法是 `\{[^\n]*\}`：贪婪但**不跨行**，所以会匹配到本行最后一个 `}`。

> 为什么不能直接用 `\{.*\}` + DOTALL？因为会跨行吞掉后面的 `Observation:`，导致参数里混进无关文本。

### ReAct 的非 JSON 输入

```python
def test_react_with_non_json_input_is_reported_not_raised():
    result = parse("Action: calculator\nAction Input: 23*7")
    assert not result.calls
    assert result.errors and "must be a JSON object" in result.errors[0]
```

经典 ReAct 允许 `Action Input: 23*7` 这种裸值。我们不支持——**明确报错而不是猜**。

为什么？因为猜错了（比如自动包成 `{"input": "23*7"}`）会让工具收到一个不存在的参数，然后报一个**更难理解的错**。直接说「要用 JSON」更好。

注意 `assert not result.calls`——错误和调用是互斥的（这一级解析失败就停）。

### 参数是字符串

```python
def test_arguments_as_a_json_string_are_reparsed():
    text = '<tool_call>{"name": "calculator", "arguments": "{\\"expression\\": \\"5+5\\"}"}</tool_call>'
    result = parse(text)
    assert result.calls[0].arguments == {"expression": "5+5"}
```

模型很爱把 arguments 序列化成字符串再塞进去。`_normalise` 里的这段负责解开：

```python
if isinstance(args, str):
    parsed = _load_object(args)
    args = parsed if parsed is not None else {"_raw": args}
```

字符串里那堆 `\\"` 是 Python 源码里的转义；实际文本是 `"arguments": "{\"expression\": \"5+5\"}"`。

---

## 3. 容错宽容度

### 单引号

```python
def test_single_quotes_are_tolerated():
    result = parse("<tool_call>{'name': 'calculator', 'arguments': {'expression': '1+1'}}</tool_call>")
    assert result.has_calls
```

靠 `ast.literal_eval` 兜住。`json.loads` 会失败，`literal_eval` 能处理 Python 字面量。

### 尾逗号

```python
def test_trailing_comma_is_tolerated():
    result = parse('<tool_call>{"name": "calculator", "arguments": {"expression": "1+1",}}</tool_call>')
    assert result.has_calls
```

靠 `_TRAILING_COMMA = re.compile(r",\s*([}\]])")` + `sub(r"\1", ...)`。

### 多种键名

```python
def test_alternate_key_spellings():
    assert parse('<tool_call>{"tool": "a", "args": {"x": 1}}</tool_call>').calls[0].name == "a"
    nested = parse('<tool_call>{"function": {"name": "b"}, "parameters": {"y": 2}}</tool_call>')
    assert nested.calls[0].name == "b"
    assert nested.calls[0].arguments == {"y": 2}
```

`_NAME_KEYS = ("name", "tool", "tool_name", "function")`、`_ARG_KEYS = ("arguments", "args", "parameters", ...)`。

第二个用例覆盖 `function` 是**嵌套对象**的情况（OpenAI 的形状）：

```python
if isinstance(name, dict):
    name = name.get("name")
```

---

## 4. 三类失败

### 完全不是 JSON

```python
def test_malformed_block_is_an_error_not_an_exception():
    result = parse("<tool_call>not json at all</tool_call>")
    assert not result.calls
    assert result.errors
    assert "not a JSON object" in result.errors[0]
```

### 缺名字

```python
def test_missing_name_is_an_error():
    result = parse('<tool_call>{"arguments": {"x": 1}}</tool_call>')
    assert not result.calls
    assert "no usable 'name'" in result.errors[0]
```

### 纯散文

```python
def test_plain_prose_has_no_calls():
    result = parse("The answer is 42.")
    assert not result.has_calls
    assert not result.errors       # ← 注意：不是错误
    assert result.text == "The answer is 42."
```

**这条最关键**：散文不是错误，是「模型认为做完了」。`errors` 为空 → agent 走 `FINAL_ANSWER` 分支。

如果把「没有工具调用」也当成 `errors`，agent 会陷入无限重试。

---

## 5. 往返测试

```python
@pytest.mark.parametrize("style", ["qwen", "bare"])
def test_format_tool_call_round_trips(style):
    text = format_tool_call("calculator", {"expression": "1+2"}, style=style)
    result = parse(text)
    assert result.has_calls
    assert result.calls[0].name == "calculator"
    assert result.calls[0].arguments == {"expression": "1+2"}
```

验证 `format` ↔ `parse` 是闭环。这个性质被 `OpenAICompatBackend` 依赖：它把结构化的 `tool_calls` 用 `format_tool_call` 转成文本，然后 agent 再用 `parse` 转回结构化。

**如果往返不闭合，接 API 的时候就会静默丢调用。**

---

## 覆盖矩阵

| 方言 / 情况 | 测试 |
|---|---|
| `<tool_call>` 标签 | 1 |
| 多个标签 | 1 |
| ```json 代码块 | 1 |
| 裸 JSON | 1 |
| ReAct | 2 |
| ReAct 嵌套 JSON | 1 |
| ReAct 非 JSON | 1 |
| arguments 是字符串 | 1 |
| 键名变体 | 1 |
| 单引号 | 1 |
| 尾逗号 | 1 |
| 坏 JSON | 1 |
| 缺名字 | 1 |
| 散文 | 1 |
| 空输入 | 1 |
| 往返 | 2（参数化） |

---

## 用到的 pytest / Python 特性

| 特性 | 说明 |
|---|---|
| `@pytest.mark.parametrize("style", [...])` | 单参数参数化 |
| `pytest.raises`（本文件没用） | 因为 parser 不抛异常 |
| 三引号 vs 转义 | 含引号的 JSON 用转义更清楚 |
| `f-string` 拼 payload | 让 JSON 和包装分离 |

---

## 设计取舍

**为什么不测「模型输出的真实样本」？**

因为真实样本又长又不可控（还依赖具体模型版本）。用最小构造的字符串，**每个用例只测一个格式特征**，失败时一眼能定位。

想测真实样本，应该放到集成测试里（需要 GPU），而不是这个纯 CPU 的单元测试。

**为什么 `parse("")` 单独一条？**

空字符串是 `not text` 的提前返回路径。不测的话，将来有人加一个 `text[0]` 就会崩。

**为什么没有「错误消息全文」的断言？**

`match=` 只用于 `pytest.raises`，而这里 `errors` 是**返回值**。所以用 `in` 断言关键词。

---

## 常见误解

* 「`parse` 失败会抛异常」——不会，`errors` 是返回值。
* 「散文也算错误」——不算。散文 = 模型完成。
* 「`text` 是原始输入」——是剥离工具块之后的剩余。
* 「一个回复只能有一个调用」——`calls` 是列表，`test_multiple_tool_calls_in_one_reply` 验证过。
* 「裸 JSON 分支会误伤正常回答」——只在**整段**是 JSON 且含 `name` 键时才触发，但确实存在误判可能（见实现文档的讨论）。

---

## 动手练习

1. 给 `test_bare_json_object` 补一句 `assert result.text == ""`，确认实现确实清空了散文。
2. 加一条用例：`'<tool_call>{"name": "a", "arguments": {}}</tool_call>'` 后面跟一段散文，验证 `text` 只包含散文。
3. 加一条用例测「未闭合的 `<tool_call>`」（`'<tool_call>{"name": "a"}'`），观察当前实现返回什么——然后想想该不该支持。
4. 构造一个 10 层嵌套的 JSON，测 `_load_object` 的 `RecursionError` 兜底是否生效。
