# `miniagent/tools.py` 详解

## 这个文件干什么

把**一个普通 Python 函数**变成模型能用的工具，并负责安全地执行它。

两件事：

1. **`@tool` 装饰器** —— 从函数签名 + docstring 推出 JSON Schema
2. **`ToolRegistry.execute`** —— 校验参数、调用、把一切异常变成 `ToolResult(ok=False)`

这是「function calling 到底怎么实现的」这个问题的完整答案。

---

## 1. `@tool` 装饰器做了什么

```python
@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression safely.

    Supports + - * / // % ** parentheses, and a small set of functions:
    abs, round, sqrt, log, log2, log10, exp, floor, ceil, min, max.

    Args:
        expression: The arithmetic expression, for example "(23 * 7 + 11) / 4".
    """
```

产出的 schema：

```json
{
  "type": "function",
  "function": {
    "name": "calculator",
    "description": "Evaluate an arithmetic expression safely. Supports + - * / ...",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "The arithmetic expression, for example \"(23 * 7 + 11) / 4\"."
        }
      },
      "required": ["expression"]
    }
  }
}
```

**这些信息全部来自代码本身**：`name` 来自函数名，`description` 来自 docstring 第一段，`type` 来自类型注解，`required` 来自「有没有默认值」，参数描述来自 `Args:` 段。

手写 schema 的后果是「代码改了、schema 忘了改」——模型于是传给函数一个不存在的参数。推导出来的**不可能不一致**。

---

## 2. 类型映射：`_json_schema`

```python
_PY_TO_JSON = {
    str: "string", int: "integer", float: "number",
    bool: "boolean", list: "array", dict: "object", type(None): "null",
}

def _json_schema(tp):
    if tp is inspect.Parameter.empty:
        return {"type": "string"}
    origin = get_origin(tp)

    if _is_union(origin):                     # X | None
        members = [a for a in get_args(tp) if a is not type(None)]
        if len(members) > 1:
            return {"anyOf": [_json_schema(m) for m in members]}
        return _json_schema(members[0]) if members else {"type": "string"}

    if origin in (list, tuple, set, frozenset):
        args = get_args(tp)
        item = _json_schema(args[0]) if args else {}
        return {"type": "array", "items": item}

    if origin is dict:
        return {"type": "object"}

    if isinstance(tp, type) and tp in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[tp]}

    return {"type": "string"}
```

### 为什么需要 `get_origin`

`list[str]` 不是一个类型，是一个**泛型别名**。`get_origin(list[str])` 返回 `list`，`get_args(list[str])` 返回 `(str,)`。

所以要分两步：先看「外壳是什么」，再看「里面装什么」。

### `X | None` 的坑

`int | None` 在 3.10 里 `get_origin` 返回 `types.UnionType`，而 `typing.Optional[int]` 返回 `typing.Union`。**两个不是同一个对象**：

```python
def _is_union(origin):
    return origin is typing.Union or origin is types.UnionType
```

只判 `typing.Union` 的话，`int | None` 会掉到最后的兜底分支，被当成字符串——模型于是给 `limit` 传 `"3"` 而不是 `3`。

**兜底为什么是 `string` 而不是报错？** 因为工具可能用了自定义类型（`Path`、`Decimal`）。给个宽松的 `string` 至少能用，报错会让整个工具注册失败。

---

## 3. docstring 解析：`_docstring_summary` 和 `_param_docs`

### `_docstring_summary`

```python
def _docstring_summary(fn):
    doc = inspect.getdoc(fn) or ""
    lines = []
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.endswith(":") and stripped.rstrip(":").lower() in _DOC_SECTIONS:
            break
        lines.append(line)
    return " ".join(" ".join(lines).split())
```

逐行累积，遇到 `Args:` / `Returns:` / `Examples:` 这类段落标题就停。最后两重 `join` + `split` 是把换行和连续空格压成单个空格——**JSON 里的 description 应该是干净的一行**。

`_DOC_SECTIONS` 是个常量集合，包含两种风格（Google 的 `Args:` 和 Sphinx 的 `:param x:`）的段落名。

### `_param_docs`

```python
def _param_docs(fn):
    doc = inspect.getdoc(fn) or ""
    out = {}
    current = None
    in_args = False
    for line in doc.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.rstrip(":").lower() in {"args", "arguments", "parameters"}:
            in_args = True
            continue
        if not in_args:
            continue
        if stripped.endswith(":") and " " not in stripped.rstrip(":"):
            break                          # 下一个段落标题
        name, sep, desc = stripped.partition(":")
        bare = name.strip()
        if sep and bare and " " not in bare:
            current = bare
            out[current] = desc.strip()    # 新参数
        elif current:
            out[current] = f"{out[current]} {stripped}".strip()   # 续行
    return out
```

三种行要区分：

| 行 | 判定 | 动作 |
|---|---|---|
| `Args:` | 段落标题 | 进入 Args 模式 |
| `expression: The expression...` | `::` 前无空格 | 新参数 |
| `    continued text` | 没有 `:` | 拼到上一个参数后面 |
| `Returns:` | 以 `:` 结尾且前面无空格 | 退出 |

**为什么用 `" " not in bare` 判断**：`Returns: the sum` 这行也含 `:`，但它前面是 `Returns`，没有空格——所以会被误判成新参数。不过它在 `in_args` 之外所以进不来；真正靠这个判据过滤的是 `expression: xxx`（前面有空格？没有）和 `for example: (23 * 7)`（前面有空格）。

> 这是一个启发式解析器，不是完整的 docstring 语法树。够用，但别指望它处理所有写法。要严格就得引入 `docstring_parser` 库。

---

## 4. `Tool.validate`：调用前先查参数

```python
def validate(self, arguments: dict) -> str | None:
    props = self.parameters.get("properties", {})
    unknown = [k for k in arguments if k not in props]
    if unknown:
        allowed = ", ".join(props) or "(none)"
        return f"unexpected argument(s) {unknown} for {self.name}; allowed: {allowed}"
    missing = [k for k in self.required if k not in arguments]
    if missing:
        return f"missing required argument(s) {missing} for {self.name}"
    return None
```

**为什么不能靠 `t.func(**arguments)` 自己报错**：

Python 的报错是 `TypeError: calculator() got an unexpected keyword argument 'expr'`。这句话对模型**几乎没用**——它不知道正确的是什么。

而 `allowed: expression` 直接给出了答案。实测中模型看完这条错误，下一步就改对了：

```
step 0  call   calculator({"expr": "23 * 7"})
        !! result ERROR: unexpected argument(s) ['expr'] for calculator; allowed: expression
step 1  call   calculator({"expression": "23 * 7"})   -> 161
```

**错误消息的措辞是 agent 可靠性的重要一环**，这是很多人忽略的地方。

注意 `validate` 只检查**键**，不检查**类型**。`{"expression": 123}` 会通过校验，然后在 `ast.parse` 里报错——那也是可接受的，因为错误信息同样可读。

---

## 5. `ToolRegistry.execute`：唯一的执行边界

```python
def execute(self, call: ToolCall) -> ToolResult:
    started = time.perf_counter()

    def fail(message: str) -> ToolResult:
        return ToolResult(
            call_id=call.call_id, name=call.name, ok=False,
            content=f"ERROR: {message}", error=message,
            duration_s=time.perf_counter() - started,
        )

    t = self.get(call.name)
    if t is None:
        return fail(f"unknown tool {call.name!r}; available: {', '.join(self.names) or '(none)'}")

    problem = t.validate(call.arguments)
    if problem:
        return fail(problem)

    try:
        value = t.func(**call.arguments)
    except Exception as exc:
        return fail(f"{type(exc).__name__}: {exc}")

    return ToolResult(..., ok=True, content=str(value), ...)
```

四条规则，每条对应一个真实故障：

| 情况 | 处理 | 不这样做的后果 |
|---|---|---|
| 工具不存在 | 返回错误 + **列出可用工具** | 模型猜名字，越猜越偏 |
| 参数不对 | 返回错误 + **给出正确拼写** | 模型陷入同样的错误 |
| 工具抛异常 | 捕获，返回 `类型: 消息` | 整个 agent 运行崩溃 |
| 成功 | `str(value)` | — |

### `except Exception` 是不是太宽

是，但**在这里是对的**。工具执行是「外部代码」，行为不可信：可能是用户写的，可能读文件时遇到编码错误，可能除零。**对不可信代码的边界就该兜住一切。**

反过来说：`BaseException`（`KeyboardInterrupt`、`SystemExit`）不该拦，`Exception` 正好。

### 为什么 `content=f"ERROR: {message}"` 还要单独存 `error`

* `content` 是**给模型看的**（要带 `ERROR:` 前缀，模型才知道这不是正常结果）
* `error` 是**给程序用的**（统计、断言、日志）

一个字段两种用途会打架，所以分两个。

---

## 6. `Tool.__call__`

```python
def __call__(self, **kwargs):
    return self.func(**kwargs)
```

让 `Tool` 实例可以直接 `calculator(expression="1+1")`。测试里用 `calculator.func(...)` 则更明确地表示「绕过 schema 直接调」。

> `ToolRegistry` 实现了 `__contains__` / `__len__` / `__iter__` / `names`，所以 `"calculator" in registry` 和 `len(registry)` 都能用。

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `inspect.signature` | 拿到参数名、默认值、种类 |
| `inspect.getdoc` | 拿 docstring（自动 dedent） |
| `typing.get_type_hints` | 解析注解（含字符串形式的 forward ref） |
| `typing.get_origin` / `get_args` | 拆泛型 |
| `types.UnionType` | `X \| Y` 的真实类型（3.10+） |
| `inspect.Parameter.empty` | 「没有默认值 / 没有注解」的哨兵 |
| `str.partition(":")` | 比 `split(":", 1)` 更好读，总返回三元组 |
| `@dataclass` | Tool 定义 |
| `time.perf_counter` | 单调时钟，测工具耗时 |

---

## 设计取舍

**为什么不用 Pydantic 生成 schema？**

Pydantic 更强大（嵌套模型、校验器、错误信息更细），但会引入一个依赖，而且会把「schema 从哪来」这个问题藏起来。用 `inspect` 写的 40 行是**透明的**——面试时你能指着代码说清楚每个字段的来源。

**为什么 `Tool` 是 dataclass 而不是普通类？**

它就是个数据容器（`func` 也是字段）。用 dataclass 自动生成 `__init__` 和 `__repr__`，`repr` 在调试时很有用。

**为什么拒绝 `*args` / `**kwargs`**

```python
if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
    raise TypeError(f"{func.__name__}: *args/**kwargs are not supported in tool signatures")
```

因为 JSON Schema 没办法表达「任意参数」。与其生成一个错误的 schema 让模型瞎猜，不如在注册时**立刻失败**。

---

## 常见误解

* 「`description` 会把整段 docstring 塞进去」——只取第一段，`Args:` 之后全部丢弃。参数描述单独放在 `properties[x].description` 里。
* 「`required` 是所有参数」——只有**没有默认值**的才是 required。
* 「工具抛异常会让 agent 崩」——不会，被 `execute` 兜住了。
* 「schema 校验能防住类型错误」——不能，只查键的存在性。类型问题交给函数自己报。
* 「`Tool` 和 `ToolResult` 是一回事」——`Tool` 是定义（注册一次），`ToolResult` 是每次执行的结果。

---

## 动手练习

1. 给一个工具加 `tags: list[str]` 参数，打印 `to_schema()`，确认生成的是 `{"type": "array", "items": {"type": "string"}}`。
2. 把 `validate` 里 `unknown` 的检查删掉，跑 `--backend local` 让模型故意传错参数名，看它会陷入循环还是能自己恢复。
3. 把 `except Exception` 改成 `except ValueError`，跑 `tests/test_agent_tools.py::test_tool_exception_becomes_an_error_result`，看它如何失败。
4. 写一个带多行参数描述的 docstring，验证 `_param_docs` 能把续行拼回去。
