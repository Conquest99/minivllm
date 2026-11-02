"""Built-in tools: a safe calculator, a text counter, and a sandboxed filesystem.

These are deliberately boring.  The point of shipping them is to show the
*boundary* discipline that makes tools safe:

* ``calculator`` never calls :func:`eval`.  It walks the AST and only permits
  arithmetic on numbers, so ``__import__("os").system("...")`` is not "blocked by
  a regex", it is structurally inexpressible.
* ``read_file`` / ``list_dir`` resolve every path against a root and refuse
  anything that escapes it.  ``..`` is not filtered as a string; the real path is
  compared.
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
    """A directory that file tools may not escape from.

    It is also the run's memory of what has been *seen*: every read records the
    file's hash, and every mutating tool refuses to touch a file that already
    exists but was never read, or that changed after it was read.  That is
    optimistic concurrency control over a filesystem, and it is what stops a model
    from appending to a file it never looked at.
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
        """Record the file's current content so a later write can detect drift."""
        self._seen[path] = self._digest(path)

    def observed_forget(self, path: Path) -> None:
        """Drop a path from the memory of observed files (it no longer exists)."""
        self._seen.pop(path, None)

    def check_writable(self, path: Path, tool: str) -> None:
        """Refuse to mutate a file this run has not read, or has seen change."""
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
    """Build the sandboxed filesystem tools bound to one root directory."""
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
    """One-line description of a file -- cheap enough to return as the observation."""
    text = target.read_text(encoding="utf-8", errors="replace")
    return f"{len(text.splitlines())} lines, {target.stat().st_size} bytes"


def _preview(text: str, limit: int = 48) -> str:
    """Trim a snippet for a receipt; ``repr`` in the caller escapes any newlines."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _change_summary(before: str, old: str, new: str) -> str:
    """Say *where* the edit landed, in the form the model actually needs: a line number.

    Without this the receipt is just "edited todo.txt (3 lines, 46 bytes)", which
    cannot distinguish "I changed line 2" from "I changed line 1" -- and that
    distinction was the entire task.
    """
    index = before.index(old)
    first = before.count("\n", 0, index) + 1
    span = old.count("\n") + 1
    where = f"line {first}" if span == 1 else f"lines {first}-{first + span - 1}"
    return f"{where}: {_preview(old)!r} -> {_preview(new)!r}"


def make_write_tools(sandbox: Sandbox | None = None) -> list[Tool]:
    """Build the *mutating* filesystem tools, bound to one root directory.

    Deliberately separate from :func:`make_file_tools`: reading is harmless, so the
    read tools ship by default; writing is a capability you hand out on purpose.
    Every tool here follows three rules:

    * the path goes through :meth:`Sandbox.resolve`, so ``..`` and absolute paths
      are refused by comparison, never by string filtering;
    * a failure is a readable ``ValueError`` -- the registry turns it into an
      observation, so the message has to say what the model should do next;
    * the return value is a receipt, not the file's contents: the model already
      knows what it wrote, and every observation costs context.

    ``write_lines`` exists because a JSON array of short strings is the one shape a
    small model reliably gets right; embedding ``\\n`` inside a JSON string is where
    it usually breaks.
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
    """The tool set used by the demo: calculator + text stats + sandboxed files.

    Read-only on purpose: ``default_tools`` is what every demo and test starts from,
    so it must not be able to destroy anything.  Add :func:`make_write_tools` when a
    task genuinely needs to write.
    """
    return [calculator, text_stats, *make_file_tools(Sandbox(workspace) if workspace else None)]
