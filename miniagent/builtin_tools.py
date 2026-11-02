"""内置工具：一个安全的计算器、一个文本计数器、一个沙箱文件系统。

它们刻意很无聊。把它们发出来的意义，是展示让工具安全的*边界*纪律：

* ``calculator`` 从不调用 :func:`eval`。它遍历 AST，只允许对数字做算术，所以
  ``__import__("os").system("...")`` 不是「被正则挡住了」，而是在结构上就写不出来。
* ``read_file`` / ``list_dir`` 把每个路径相对一个根目录解析，逃出去的一律拒绝。
  ``..`` 不是当字符串过滤掉的，而是比较真实路径。
"""

from __future__ import annotations

import ast
import hashlib
import math
import operator
from pathlib import Path
from typing import Any

from miniagent.tools import Tool, tool

# --------------------------------------------------------------------- 计算器

_BIN_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Any] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "min": min,
    "max": max,
}

_MAX_EXPONENT = 100
_MAX_RESULT = 1e18


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
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponent {right} is too large (limit {_MAX_EXPONENT})")
        result = op(left, right)
        if isinstance(result, complex) or abs(result) > _MAX_RESULT:
            raise ValueError("result is out of the allowed range")
        return result

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unary operator {type(node.op).__name__} is not allowed")
        return op(_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ValueError(f"function calls are limited to: {', '.join(sorted(_FUNCTIONS))}")
        if node.keywords:
            raise ValueError("keyword arguments are not supported")
        return _FUNCTIONS[node.func.id](*[_eval_node(a) for a in node.args])

    raise ValueError(f"{type(node).__name__} is not allowed in an arithmetic expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression safely.

    Supports + - * / // % ** parentheses, and a small set of functions:
    abs, round, sqrt, log, log2, log10, exp, floor, ceil, min, max.

    Args:
        expression: The arithmetic expression, for example "(23 * 7 + 11) / 4".
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"cannot parse {expression!r}: {exc.msg}") from exc
    value = _eval_node(tree)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# ------------------------------------------------------------------ 文本统计


@tool
def text_stats(text: str) -> str:
    """Count characters, words and lines in a piece of text.

    Args:
        text: The text to analyse.
    """
    words = text.split()
    return f"chars={len(text)} words={len(words)} lines={len(text.splitlines()) or 1}"


# --------------------------------------------------------------------- 文件系统


class Sandbox:
    """文件工具不许逃出去的目录。

    它同时也是本次运行「看过什么」的记忆：每次读都会记下文件哈希，而每个改动类工具都拒绝
    去碰一个「已经存在但从未被读过」或者「读完之后变过」的文件。这就是在文件系统上做
    乐观并发控制，也是它拦住模型往一个从没看过的文件后面追加内容的原因。
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._seen: dict[Path, str] = {}

    def resolve(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError(f"{relative!r} resolves outside the sandbox ({self.root})")
        return candidate

    # --- 本次运行观察到过什么 ------------------------------------------
    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def observe(self, path: Path) -> None:
        """记下文件当前内容，好让后面的写操作能发现它被改过。"""
        self._seen[path] = self._digest(path)

    def observed_forget(self, path: Path) -> None:
        """把一个路径从「看过」的记忆里删掉（它已经不存在了）。"""
        self._seen.pop(path, None)

    def check_writable(self, path: Path, tool: str) -> None:
        """拒绝改动一个本次运行没读过、或已被发现变过的文件。"""
        if not path.is_file():
            return  # 新建一个文件永远是安全的
        if path not in self._seen:
            raise PermissionError(
                f"{tool} refuses to modify {path.name}: it already exists and this run has "
                "not read it. Call read_file first, or pick a name that does not exist yet."
            )
        if self._digest(path) != self._seen[path]:
            raise PermissionError(
                f"{tool} refuses to modify {path.name}: it changed after you read it. "
                "Call read_file again to see the current content."
            )


def make_file_tools(sandbox: Sandbox | None = None) -> list[Tool]:
    """构建绑定到某个根目录的沙箱文件工具。"""
    box = sandbox or Sandbox(Path.cwd() / "agent_workspace")

    @tool(name="read_file", reads="path")
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
        box.observe(target)
        if len(text) > max_chars:
            return text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"
        return text

    @tool(name="list_dir")
    def list_dir(path: str = ".") -> str:
        """List entries of a directory in the workspace.

        Args:
            path: Path relative to the workspace root; defaults to the root.
        """
        target = box.resolve(path)
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {path}")
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = [f"{'d' if p.is_dir() else 'f'}  {p.name}" for p in entries[:200]]
        return "\n".join(lines) if lines else "(empty)"

    return [read_file, list_dir]


# ------------------------------------------------------------------ 写入工具


def _receipt(target: Path) -> str:
    """一行描述这个文件 —— 便宜到可以当观察结果返回。"""
    text = target.read_text(encoding="utf-8", errors="replace")
    return f"{len(text.splitlines())} lines, {target.stat().st_size} bytes"


def _preview(text: str, limit: int = 48) -> str:
    """为回执裁剪一段片段；换行由调用方的 ``repr`` 负责转义。"""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _change_summary(before: str, old: str, new: str) -> str:
    """说清这次编辑落在*哪里*，用模型真正需要的形式：行号。

    没有它，回执就只是「edited todo.txt (3 lines, 46 bytes)」，分不清「我改了第 2 行」
    和「我改了第 1 行」—— 而这个区别正是整个任务本身。
    """
    index = before.index(old)
    first = before.count("\n", 0, index) + 1
    span = old.count("\n") + 1
    where = f"line {first}" if span == 1 else f"lines {first}-{first + span - 1}"
    return f"{where}: {_preview(old)!r} -> {_preview(new)!r}"


def make_write_tools(sandbox: Sandbox | None = None) -> list[Tool]:
    """构建*会改动*的文件系统工具，绑定到某个根目录。

    刻意与 :func:`make_file_tools` 分开：读是无害的，所以读工具默认就发；写是一种能力，
    要你主动交出去。这里的每个工具守三条规矩：

    * 路径必须过 :meth:`Sandbox.resolve`，所以 ``..`` 和绝对路径是被**比较**拒绝的，
      从来不是字符串过滤；
    * 失败是一个能读懂的 ``ValueError`` —— 注册表会把它变成观察结果，所以消息必须说清
      模型下一步该做什么；
    * 返回值是回执，不是文件内容：模型已经知道自己写了什么，而每条观察都要花上下文。

    ``write_lines`` 之所以存在，是因为「一串短字符串的 JSON 数组」是小模型唯一稳定写对的
    形状；把 ``\n`` 塞进 JSON 字符串里通常就是它翻车的地方。
    """

    box = sandbox or Sandbox(Path.cwd() / "agent_workspace")

    @tool(name="write_file", mutates=True, affects="path")
    def write_file(path: str, content: str, mode: str = "overwrite") -> str:
        """Create a text file, or overwrite / append to an existing one.

        Prefer write_lines for anything with more than one line.

        Args:
            path: Path relative to the workspace root; missing parent folders are created.
            content: The text to write. Write line breaks as \\n.
            mode: "overwrite" replaces the whole file, "append" adds to the end.
        """
        if mode not in ("overwrite", "append"):
            raise ValueError(f"mode must be 'overwrite' or 'append', got {mode!r}")
        target = box.resolve(path)
        if target.is_dir():
            raise IsADirectoryError(f"{path} is a directory, not a file")
        box.check_writable(target, "write_file")
        created = not target.is_file()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a" if mode == "append" else "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
        box.observe(target)
        if not created:
            verb = "appended to" if mode == "append" else "overwrote"
        else:
            verb = "created"
        return f"{verb} {path} ({_receipt(target)})"

    @tool(name="write_lines", mutates=True, affects="path")
    def write_lines(path: str, lines: list[str], mode: str = "overwrite") -> str:
        """Write a list of lines to a file, one array element per line.

        The shape a small model gets right: a JSON array of short strings needs no
        escaping, while a multi-line string needs \\n inside the JSON.

        Args:
            path: Path relative to the workspace root.
            lines: The lines to write, without trailing newline characters.
            mode: "overwrite" replaces the whole file, "append" adds to the end.
        """
        content = "".join(f"{line}\n" for line in lines)
        return write_file.func(path, content, mode)

    @tool(name="edit_file", mutates=True, affects="path")
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace exactly one occurrence of `old` with `new` inside a file.

        Use this for small changes; use write_file to rewrite a whole file.

        Args:
            path: Path relative to the workspace root.
            old: The exact text to replace. It must occur exactly once.
            new: What to put in its place.
        """
        if not old:
            raise ValueError("old must not be empty")
        target = box.resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        box.check_writable(target, "edit_file")
        text = target.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            raise ValueError(
                f"{old!r} does not occur in {path}. Read the file first and copy the text "
                "exactly, including spaces and punctuation."
            )
        if count > 1:
            raise ValueError(
                f"{old!r} occurs {count} times in {path}. Include more of the surrounding "
                "text so the match is unique."
            )
        after = text.replace(old, new, 1)
        target.write_text(after, encoding="utf-8", newline="\n")
        box.observe(target)
        return f"edited {path} ({_change_summary(text, old, new)}; {_receipt(target)})"

    @tool(name="delete_file", mutates=True, affects="path")
    def delete_file(path: str) -> str:
        """Delete one file from the workspace. Directories are never removed.

        Args:
            path: Path relative to the workspace root.
        """
        target = box.resolve(path)
        if target.is_dir():
            raise IsADirectoryError(f"{path} is a directory; delete_file only removes files")
        if not target.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        box.check_writable(target, "delete_file")
        target.unlink()
        box.observed_forget(target)
        return f"deleted {path}"

    return [write_file, write_lines, edit_file, delete_file]


def default_tools(workspace: str | Path | None = None) -> list[Tool]:
    """demo 用的工具集：calculator + 文本统计 + 沙箱文件。

    刻意保持只读：``default_tools`` 是每个 demo 和测试的起点，所以它绝不能有能力毁掉任何
    东西。当任务真的需要写时，再额外加上 :func:`make_write_tools`。
    """
    return [calculator, text_stats, *make_file_tools(Sandbox(workspace) if workspace else None)]
