# `tests/test_agent_tools.py` 详解

## 这个文件干什么

验证 4 件事，全部纯 CPU：

1. **schema 真的从签名推出来**（含可选参数、列表、docstring）
2. **参数校验报错可读**（不是 `TypeError`）
3. **计算器不能执行代码**（安全边界）
4. **沙箱不能被逃逸**（`..`、绝对路径）

51 个用例，跑完不到 0.3 秒。

---

## 1. schema 生成

```python
def test_schema_is_derived_from_the_signature():
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    assert add.name == "add"
    assert add.description == "Add two integers."
    assert add.parameters["properties"]["a"]["type"] == "integer"
    assert add.parameters["required"] == ["a", "b"]
```

**在测试函数内部定义工具**，不是模块级。好处是：

* 不污染全局注册表
* 每个测试的工具名可以重名
* 读测试时能看到完整定义

### 可选参数

```python
def test_optional_parameters_are_not_required():
    @tool
    def greet(name: str, greeting: str = "hello") -> str:
        """Greet someone."""
        return f"{greeting} {name}"

    assert greet.parameters["required"] == ["name"]
```

守的是「有默认值 → 不在 required 里」。这个如果错了，模型会被迫每次都传 `greeting`。

### `list` 和 `int | None`

```python
def test_list_and_optional_annotations():
    @tool
    def summarise(items: list[str], limit: int | None = None) -> str:
        """Summarise items."""
        return ",".join(items[: limit or len(items)])

    props = summarise.parameters["properties"]
    assert props["items"] == {"type": "array", "items": {"type": "string"}, "description": "items"}
    assert props["limit"]["type"] == "integer"
```

这条同时覆盖两个坑：

* `list[str]` 要拆成 `{"type": "array", "items": {"type": "string"}}`
* `int | None` 要去掉 `None` 只留 `integer`

第二句断言 `props["limit"]["type"] == "integer"` 直接验证了 `_is_union` 那段。**如果只判 `typing.Union` 不判 `types.UnionType`，这里会得到 `"string"`**（兜底分支）。

> 这条断言我在写的时候特意让它同时覆盖，因为它比单独写两个测试更紧凑，失败时也能立刻看出是哪个注解没处理。

### docstring 的参数描述

```python
def test_docstring_args_become_parameter_descriptions():
    @tool
    def divide(numerator: float, denominator: float) -> float:
        """Divide one number by another.

        Args:
            numerator: The number on top.
            denominator: The number underneath.
        """
        return numerator / denominator

    props = divide.parameters["properties"]
    assert props["numerator"]["description"] == "The number on top."
```

验证 `_param_docs` 能正确识别 `Args:` 段、剥离缩进、在下一个段落标题处停止。

### 拒绝 `*args`

```python
def test_varargs_are_rejected():
    with pytest.raises(TypeError, match="not supported"):
        @tool
        def bad(*args):
            """Nope."""
```

`pytest.raises(..., match=...)` 用正则匹配异常消息。这里匹配 `"not supported"`，因为完整消息是 `bad: *args/**kwargs are not supported in tool signatures`。

**为什么这算一个测试而不是实现细节**：因为它防止别人「顺手」加上 `*args` 支持，然后生成一个语义错误的 schema。

---

## 2. 注册表与执行

### 重名拒绝

```python
def test_duplicate_tool_names_are_rejected():
    @tool
    def twice(text: str) -> str:
        """Twice."""
        return text

    registry = ToolRegistry([twice])
    with pytest.raises(ValueError, match="duplicate tool name"):
        registry.register(twice)
```

重名会让 `get(name)` 返回不确定的那个。**在注册时失败**比在运行时行为诡异好得多。

### 正常的执行

```python
def test_execute_returns_ok_and_records_timing():
    result = ToolRegistry([calculator]).execute(
        ToolCall(name="calculator", arguments={"expression": "2 + 2"})
    )
    assert result.ok and result.content == "4"
    assert result.duration_s >= 0
```

`duration_s >= 0` 而不是 `> 0`：极快的操作可能测出 0（`perf_counter` 精度足够，但精确等于 0 理论上可能）。

### 三类错误

```python
def test_unknown_tool_is_an_error_but_not_an_exception():
    result = ToolRegistry([calculator]).execute(ToolCall(name="nope", arguments={}))
    assert not result.ok
    assert "unknown tool" in result.content
    assert "calculator" in result.content   # 列出了可用工具


def test_unexpected_argument_is_reported_readably():
    result = ToolRegistry([calculator]).execute(ToolCall(name="calculator", arguments={"expr": "2+2"}))
    assert not result.ok
    assert "unexpected argument" in result.content
    assert "allowed: expression" in result.content


def test_missing_required_argument_is_reported():
    result = ToolRegistry([calculator]).execute(ToolCall(name="calculator", arguments={}))
    assert not result.ok
    assert "missing required argument" in result.content
```

**最后那条 `assert "allowed: expression"` 是核心**——它验证的不是「有错误」，而是「错误里包含了修复所需的信息」。

这是产品级断言。`"unexpected argument"` 只证明校验发生了；`"allowed: expression"` 证明它**有用**。

### 工具异常被兜住

```python
def test_tool_exception_becomes_an_error_result():
    @tool
    def explode() -> str:
        """Always fails."""
        raise RuntimeError("boom")

    result = ToolRegistry([explode]).execute(ToolCall(name="explode", arguments={}))
    assert not result.ok
    assert result.error == "RuntimeError: boom"
    assert result.content.startswith("ERROR:")
```

断言三件事：`ok=False`、`error` 字段精确、`content` 带前缀（给模型看的版本）。

---

## 3. 计算器安全（11 个用例）

### 正常算术

```python
@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2+2", "4"),
        ("(23 * 7 + 11) / 4", "43"),
        ("2 ** 10", "1024"),
        ("17 // 5", "3"),
        ("17 % 5", "2"),
        ("-3 + 1", "-2"),
        ("sqrt(16)", "4"),
        ("max(3, 9)", "9"),
        ("round(3.14159, 2)", "3.14"),
    ],
)
def test_calculator_evaluates_arithmetic(expression, expected):
    assert calculator.func(expression) == expected
```

用 `calculator.func(...)` 而不是 `calculator(...)`：**绕过 schema，直接测函数**。schema 校验在别的测试里测。

`"(23 * 7 + 11) / 4"` → `"43"` 而不是 `"43.0"`，验证了 `is_integer()` 的整数化处理。

### 拒绝一切非算术

```python
@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__",
        "[x for x in range(3)]",
        "lambda: 1",
        "a + 1",
    ],
)
def test_calculator_refuses_anything_that_is_not_arithmetic(expression):
    with pytest.raises(ValueError):
        calculator.func(expression)
```

六个攻击面：

| 表达式 | 攻击类型 | 被哪个分支拦住 |
|---|---|---|
| `__import__('os')...` | 任意代码执行 | `ast.Call` → 不在白名单 |
| `open(...)` | 文件读取 | `ast.Call` → 不在白名单 |
| `().__class__.__bases__` | 反射/沙箱逃逸 | `ast.Attribute` → 兜底拒绝 |
| `[x for x in ...]` | 列表推导 | `ast.ListComp` → 兜底拒绝 |
| `lambda: 1` | 匿名函数 | `ast.Lambda` → 兜底拒绝 |
| `a + 1` | 未定义变量 | `ast.Name` → 兜底拒绝 |

**这六个都不是靠字符过滤拦住的**，是靠「只放行 `Constant` / `BinOp` / `UnaryOp` / 白名单 `Call`」这个结构性设计。

### 拒绝服务防护

```python
def test_calculator_bounds_exponents():
    with pytest.raises(ValueError, match="too large"):
        calculator.func("9 ** 9999")
```

`9 ** 9999` 不违法，但会算很久（大整数运算）。这条测试守的是**资源边界**，不是安全边界。两者经常被混为一谈。

---

## 4. 沙箱

### 允许内部路径

```python
def test_sandbox_allows_paths_inside_the_root(tmp_path):
    box = Sandbox(tmp_path)
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    read_file, list_dir = make_file_tools(box)

    assert read_file.func("notes.txt") == "hello"
    assert "notes.txt" in list_dir.func(".")
```

`tmp_path` 是 pytest 内置 fixture，每个测试给一个独立的临时目录。**不用 `tempfile` 手动清理**，pytest 自动处理。

### 拒绝 `..` 逃逸

```python
def test_sandbox_refuses_to_escape(tmp_path):
    box = Sandbox(tmp_path / "root")
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")

    with pytest.raises(PermissionError, match="outside the sandbox"):
        box.resolve("../secret.txt")
```

**注意 `tmp_path / "secret.txt"` 在 `root/` 之外**，所以是真正的越界目标。

### 拒绝绝对路径

```python
def test_sandbox_refuses_absolute_escapes(tmp_path):
    box = Sandbox(tmp_path / "root")
    with pytest.raises(PermissionError):
        box.resolve(str(tmp_path / "elsewhere.txt"))
```

这条特别重要：**绝对路径里没有 `..`**，所以「过滤 `..`」的实现在这里会全线失守。

> 两条测试（`..` 和绝对路径）合起来才证明了「基于 `resolve()` 的判定」是必要的。只测 `..` 的话，字符串过滤的实现也能通过。

### 截断

```python
def test_read_file_truncates(tmp_path):
    box = Sandbox(tmp_path)
    (tmp_path / "big.txt").write_text("x" * 50, encoding="utf-8")
    read_file, _ = make_file_tools(box)
    out = read_file.func("big.txt", max_chars=10)
    assert out.startswith("x" * 10)
    assert "truncated" in out
```

`"truncated" in out` 验证了提示信息存在——模型看到它才知道内容不完整。

---

## 5. 默认工具集

```python
def test_default_tool_set_has_the_expected_names():
    names = {t.name for t in default_tools()}
    assert {"calculator", "text_stats", "read_file", "list_dir"} <= names
```

用 `<=`（子集）而不是 `==`：将来加工具不会打破这条测试。这是**向前兼容的断言**写法。

---

## 覆盖矩阵

| 属性 | 测试数 |
|---|---|
| schema 从签名推导 | 5 |
| 注册表错误处理 | 4 |
| 工具异常兜底 | 1 |
| 计算器正确性 | 9（参数化） |
| 计算器安全性 | 6（参数化） |
| 计算器资源边界 | 1 |
| 沙箱 | 5 |
| 默认工具集 | 1 |

**没覆盖**：`_docstring_summary` 遇到 `Returns:` 的情况、多行参数描述、`anyOf` 分支。

---

## 用到的 pytest 特性

| 特性 | 说明 |
|---|---|
| `@pytest.mark.parametrize` | 一个函数跑多组数据 |
| `("expr", "expected")` 元组参数化 | 多参数 vs 单参数 |
| `pytest.raises(Exc, match="regex")` | 断言异常类型 + 消息 |
| `tmp_path` fixture | 每测试独立临时目录 |
| 测试内定义工具 | 不污染全局 |

---

## 设计取舍

**为什么这些测试不需要 torch？**

因为 `miniagent.tools` / `builtin_tools` 不 import torch。`backends.py` 里 `MiniVLLMBackend` 的 import 是**惰性的**（在 `__init__` 里）。这个分层让 51 个安全测试能在 0.3 秒内跑完——**快到你愿意每次都跑**。

**为什么安全测试用 `pytest.raises` 而不是断言返回值？**

因为 `calculator` 的契约就是「不合法 → 抛 `ValueError`」，由装饰器的 `_json_schema`... 不，由 `ToolRegistry.execute` 兜住。直接测函数时应该测它**原始的失败方式**。

**为什么不断言具体的错误消息全文？**

消息会随实现变化。断言关键词（`"outside the sandbox"`）既够精确又不容易因为改措辞而误报。

---

## 常见误解

* 「参数化测试算一个测试」——算 N 个。上面 9 组就是一个函数生成 9 个用例。
* 「测了 `..` 就测了路径安全」——绝对路径是另一条绕过路径。
* 「`ValueError` 就够用了」——如果哪天改成返回 `None`，测试会失败，这是好事。
* 「`<=` 不如 `==` 严格」——这里故意宽松，为了向前兼容。

---

## 动手练习

1. 在 `test_calculator_refuses_anything_that_is_not_arithmetic` 里加 `"1 if True else 2"`（三元表达式），看它被哪个分支拒绝，然后确认这符合预期。
2. 把 `_is_union` 里的 `types.UnionType` 去掉，跑 `test_list_and_optional_annotations`，看它报什么。
3. 写一条新测试：`Sandbox` 对指向外部的**符号链接**也要拒绝（提示：Windows 上建符号链接需要管理员权限，可以用 `pytest.mark.skipif` 处理）。
4. 给 `Tool.validate` 加类型检查（比如 `expression` 必须是 str），然后补对应的测试，再想想这样做会不会误伤合法的 `int` 输入。
