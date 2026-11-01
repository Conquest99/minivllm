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
import math
import operator
from pathlib import Path
from typing import Any

from miniagent.tools import Tool, tool

# --------------------------------------------------------------------- calculator

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


# -------------------------------------------------------------------- text stats


@tool
def text_stats(text: str) -> str:
    """Count characters, words and lines in a piece of text.

    Args:
        text: The text to analyse.
    """
    words = text.split()
    return f"chars={len(text)} words={len(words)} lines={len(text.splitlines()) or 1}"


# --------------------------------------------------------------------- filesystem


class Sandbox:
    """A directory that file tools may not escape from."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise PermissionError(f"{relative!r} resolves outside the sandbox ({self.root})")
        return candidate


def make_file_tools(sandbox: Sandbox | None = None) -> list[Tool]:
    """Build the sandboxed filesystem tools bound to one root directory."""
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


def default_tools(workspace: str | Path | None = None) -> list[Tool]:
    """The tool set used by the demo: calculator + text stats + sandboxed files."""
    return [calculator, text_stats, *make_file_tools(Sandbox(workspace) if workspace else None)]
