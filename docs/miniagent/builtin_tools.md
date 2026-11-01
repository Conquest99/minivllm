# `miniagent/builtin_tools.py` 详解

## 这个文件干什么

三个可以直接用的工具，顺带演示**工具的安全边界该怎么划**：

| 工具 | 说明 |
|---|---|
| `calculator` | 走 AST 的算术求值，**不调用 `eval`** |
| `text_stats` | 字符/词/行统计 |
| `read_file` / `list_dir` | 锁在沙箱目录内的文件访问 |

这些工具本身很无聊。**值得看的是它们怎么守边界**——因为工具是 agent 里唯一能造成实际后果的部位。

---

## 1. 为什么计算器不能用 `eval`

朴素写法：

```python
def calculator(expression):
    return eval(expression)     # ❌
```

`eval` 会执行**任意 Python 代码**：

```python
calculator("__import__('os').system('rm -rf ~')")
calculator("open('/etc/passwd').read()")
```

而 `expression` 这个参数是**模型生成的**，模型可能被 prompt 注入操纵。等于把 shell 交给了一个可以被说服的组件。

### 正确做法：走 AST

```python
def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"only numeric literals are allowed, got {value!r}")
        return value

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator {type(node.op).__name__} is not allowed")
        ...

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ValueError(f"function calls are limited to: {', '.join(sorted(_FUNCTIONS))}")
        if node.keywords:
            raise ValueError("keyword arguments are not supported")
        return _FUNCTIONS[node.func.id](*[_eval_node(a) for a in node.args])

    raise ValueError(f"{type(node).__name__} is not allowed in an arithmetic expression")
```

`ast.parse(expression, mode="eval")` 把字符串变成语法树，然后**只放行白名单里的节点类型**。

关键点：**不是「过滤危险字符」，而是「默认拒绝、逐个放行」**。

```python
calculator("__import__('os').system('echo hi')")
# ValueError: function calls are limited to: abs, ceil, ...
```

注意错误来自 `ast.Call` 分支（`__import__` 不在 `_FUNCTIONS` 里），**不是**因为字符串里有 `__import__`。即使换个名字、加空格、用 Unicode 混淆，只要结构是「调用一个非白名单函数」，就过不去。

> 这是安全设计的通用原则：**allowlist 而不是 blocklist**。黑名单永远有漏，白名单漏不了。

### 白名单内也有边界

```python
_MAX_EXPONENT = 100
_MAX_RESULT = 1e18

if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
    raise ValueError(f"exponent {right} is too large (limit {_MAX_EXPONENT})")
result = op(left, right)
if isinstance(result, complex) or abs(result) > _MAX_RESULT:
    raise ValueError("result is out of the allowed range")
```

`9 ** 9999` 不会崩，但会**算很久**（或者生成一个巨大的整数占满内存）。`2 ** 0.5` 会得到复数——数学上合法，但工具语义上不该出现。

**这是「拒绝服务」防御**：不是让模型做坏事，而是让一个畸形输入不能拖死进程。agent 里的工具都可能被模型用奇怪参数调用，这类护栏必不可少。

### `bool` 要单独排除

```python
if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(...)
```

因为 `isinstance(True, int)` 是 `True`（Python 里 `bool` 继承 `int`）。不排除的话 `True + True` 会得到 `2`，虽然无害但语义混乱。

---

## 2. `text_stats`：最简单的一个

```python
@tool
def text_stats(text: str) -> str:
    """Count characters, words and lines in a piece of text.

    Args:
        text: The text to analyse.
    """
    words = text.split()
    return f"chars={len(text)} words={len(words)} lines={len(text.splitlines()) or 1}"
```

`text.splitlines() or 1` 处理空字符串：`"".splitlines()` 是 `[]`，`len([]) = 0`，但空文本应该算 1 行。

它存在的意义是**给模型一个「不需要推理」的工具**，用来对比「模型自己数」和「工具数」的差别。0.5B 数词数经常错。

---

## 3. `Sandbox`：路径逃逸的正确防法

```python
class Sandbox:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError(f"{relative!r} resolves outside the sandbox ({self.root})")
        return candidate
```

### 为什么不在字符串里过滤 `..`

错误做法：

```python
if ".." in path:   # ❌
    raise PermissionError
```

会被这些绕过：

```
....//          （某些解析器）
a/../../etc     （部分过滤逻辑）
%2e%2e/         （URL 编码）
\..\            （Windows 分隔符）
C:\Windows      （绝对路径，没有 ..）
```

正确做法只有一种：**先把路径解析成真实的绝对路径，再判断它是否在根目录之内。**

```
(root / "../../etc/passwd").resolve()   →   /etc/passwd
                                          ↑ 不在 root 之下 → 拒绝
```

`.resolve()` 会把 `..`、符号链接、Windows 的盘符全部折叠成最终真实路径。判断 `root in candidate.parents` 就是判断「它是不是 root 的后代」。

`candidate != self.root` 这个额外条件让 `"."` 合法（沙箱根本身是允许的）。

### 为什么 `resolve()` 而不是 `absolute()`

`Path.absolute()` 只是拼接当前目录，**不解析 `..` 和符号链接**。一个指向 `/etc` 的符号链接会被 `absolute()` 放过。必须 `resolve()`。

---

## 4. `make_file_tools`：闭包绑定沙箱

```python
def make_file_tools(sandbox: Sandbox | None = None) -> list[Tool]:
    box = sandbox or Sandbox(Path.cwd() / "agent_workspace")

    @tool(name="read_file")
    def read_file(path: str, max_chars: int = 4000) -> str:
        """Read a UTF-8 text file from the workspace.

        Args:
            path: Path relative to the workspace root.
            max_chars: Truncate the content to this many characters.
        """
        target = box.resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            return text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"
        return text

    @tool(name="list_dir")
    def list_dir(path: str = ".") -> str:
        ...

    return [read_file, list_dir]
```

三个设计点：

**① `@tool(name="read_file")` 显式命名**

函数名本来就叫 `read_file`，为什么还要指定？因为这是**闭包里的局部函数**，`func.__name__` 虽然是 `read_file`，但显式写出来更能防止将来重命名时 schema 静默变化。而且它让「模型看到的工具名」在代码里是可搜索的。

**② 闭包捕获 `box`**

沙箱通过闭包传进去，所以工具函数**签名里没有 `root` 参数**——模型改不了它。如果写成 `read_file(path, root)`，模型完全可以传 `root="/"` 把沙箱废掉。

**这是工具设计的关键原则：安全边界必须是闭包/配置，绝不能是参数。**

**③ `max_chars` 是参数，因为它无害**

`max_chars` 只控制返回多少内容，不影响访问范围，所以可以暴露给模型。默认 4000 防止一次把整个大文件塞进上下文。

`errors="replace"` 是防御：二进制文件被当文本读时不会抛 `UnicodeDecodeError`，而是显示替换字符。**工具面对不可信输入时要尽量不抛异常**（虽然抛了也会被 `execute` 兜住，但那会浪费一轮）。

### `list_dir` 的排序

```python
entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
```

- `(p.is_file(), p.name)` 先按「是不是文件」排：`False`（目录）在前。**目录优先**符合直觉。
- `p.is_file()` 只返回 bool，不能区分符号链接指向目录的情况，够用。
- `entries[:200]` 截断，防止一个巨大目录撑爆上下文。

---

## 5. `default_tools`：默认工具集

```python
def default_tools(workspace: str | Path | None = None) -> list[Tool]:
    return [calculator, text_stats, *make_file_tools(Sandbox(workspace) if workspace else None)]
```

一次拿到 4 个工具。`demo_generate` 和 `agent_demo` 都用它。

注意 `calculator` 和 `text_stats` 是**模块级单例**（装饰器在导入时就执行了），而文件工具是**每次新建**的（因为要绑定不同的沙箱）。

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `ast.parse(..., mode="eval")` | 只解析表达式，不解析语句 |
| `ast.NodeVisitor` 风格的手写递归 | 比 `NodeVisitor` 更直观，能提前返回 |
| `isinstance(node, ast.X)` | 按节点类型分派 |
| `Path.resolve()` | 折叠 `..` 和符号链接 |
| `Path.parents` | 所有祖先目录 |
| 闭包捕获 | 把沙箱藏进工具，不进签名 |
| `str.splitlines()` | 比 `split("\n")` 正确处理 `\r\n` |
| `errors="replace"` | 解码失败不抛异常 |

---

## 设计取舍

**为什么 `tool` 里 `_MAX_RESULT = 1e18` 而不是更大？**

因为超过 1e18 的 float 精度已经很差（`float` 只有 53 位尾数，约 1.8e16 之后整数不再精确）。**与其给出错误答案，不如拒绝。**

**为什么沙箱默认是 `cwd/agent_workspace` 而不是 `cwd`？**

如果默认是当前目录，agent 就能读到你项目的任何文件。默认值应该是**最安全的那个**。要用别的目录就显式传。

**为什么没有 `write_file` / `python_exec`？**

因为它们的风险等级完全不同：读是幂等的、可审计的；写会破坏数据，执行会运行任意代码。**要加的话应该是单独一个显式开关，不能混在默认工具集里。** 这是刻意的缺席。

---

## 常见误解

* 「过滤掉 `..` 就安全了」——绝对路径、符号链接、URL 编码都能绕过。必须 `resolve()` 后比较。
* 「`ast.literal_eval` 不安全」——它只认字面量，非常安全。不安全的是 `eval`。
* 「`calculator` 支持变量」——不支持，`a + 1` 会报 `Name is not allowed`。
* 「幂运算太大只是慢一点」——会长时间占用 CPU，属于拒绝服务。
* 「`max_chars` 是安全限制」——不是，它只是防止上下文爆炸；访问范围由沙箱控制。

---

## 动手练习

1. 用 `calculator` 试 `"2**0.5"`（复数）、`"9**9999"`（超限）、`"a+1"`（未定义名），记录三种错误消息。
2. 把 `Sandbox.resolve` 改成 `if ".." in relative: raise`，然后试 `box.resolve(str(tmp_path / "elsewhere.txt"))`（绝对路径），看它被绕过。
3. 把 `read_file` 的 `box` 改成参数 `root: str = "."`，想清楚这对 agent 的安全性意味着什么。
4. 给 `list_dir` 加上「超过 200 项时提示总数」，观察模型是否会因此改变行为。
