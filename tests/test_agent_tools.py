"""Schema generation, argument validation, sandboxing and calculator safety."""

from __future__ import annotations

import pytest

from miniagent.builtin_tools import Sandbox, calculator, default_tools, make_file_tools, text_stats
from miniagent.tools import ToolRegistry, tool
from miniagent.types import ToolCall

# --------------------------------------------------------------- schema generation


def test_schema_is_derived_from_the_signature():
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    assert add.name == "add"
    assert add.description == "Add two integers."
    assert add.parameters["properties"]["a"]["type"] == "integer"
    assert add.parameters["properties"]["b"]["type"] == "integer"
    assert add.parameters["required"] == ["a", "b"]


def test_optional_parameters_are_not_required():
    @tool
    def greet(name: str, greeting: str = "hello") -> str:
        """Greet someone."""
        return f"{greeting} {name}"

    assert greet.parameters["required"] == ["name"]
    assert greet.parameters["properties"]["greeting"]["type"] == "string"


def test_list_and_optional_annotations():
    @tool
    def summarise(items: list[str], limit: int | None = None) -> str:
        """Summarise items."""
        return ",".join(items[: limit or len(items)])

    props = summarise.parameters["properties"]
    assert props["items"] == {"type": "array", "items": {"type": "string"}, "description": "items"}
    assert props["limit"]["type"] == "integer"
    assert set(summarise.parameters["required"]) == {"items"}


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
    assert props["denominator"]["description"] == "The number underneath."


def test_varargs_are_rejected():
    with pytest.raises(TypeError, match="not supported"):

        @tool
        def bad(*args):
            """Nope."""


def test_schema_shape_matches_the_openai_tools_payload():
    @tool
    def echo(text: str) -> str:
        """Echo text."""
        return text

    schema = echo.to_schema()
    assert schema["type"] == "function"
    assert set(schema["function"]) == {"name", "description", "parameters"}


# --------------------------------------------------------------------- registry


def test_duplicate_tool_names_are_rejected():
    @tool
    def twice(text: str) -> str:
        """Twice."""
        return text

    registry = ToolRegistry([twice])
    with pytest.raises(ValueError, match="duplicate tool name"):
        registry.register(twice)


def test_execute_returns_ok_and_records_timing():
    result = ToolRegistry([calculator]).execute(
        ToolCall(name="calculator", arguments={"expression": "2 + 2"})
    )
    assert result.ok and result.content == "4"
    assert result.duration_s >= 0


def test_unknown_tool_is_an_error_but_not_an_exception():
    result = ToolRegistry([calculator]).execute(ToolCall(name="nope", arguments={}))
    assert not result.ok
    assert "unknown tool" in result.content
    assert "calculator" in result.content  # lists the alternatives


def test_unexpected_argument_is_reported_readably():
    result = ToolRegistry([calculator]).execute(ToolCall(name="calculator", arguments={"expr": "2+2"}))
    assert not result.ok
    assert "unexpected argument" in result.content
    assert "allowed: expression" in result.content


def test_missing_required_argument_is_reported():
    result = ToolRegistry([calculator]).execute(ToolCall(name="calculator", arguments={}))
    assert not result.ok
    assert "missing required argument" in result.content


def test_tool_exception_becomes_an_error_result():
    @tool
    def explode() -> str:
        """Always fails."""
        raise RuntimeError("boom")

    result = ToolRegistry([explode]).execute(ToolCall(name="explode", arguments={}))
    assert not result.ok
    assert result.error == "RuntimeError: boom"
    assert result.content.startswith("ERROR:")


# -------------------------------------------------------------------- calculator


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


def test_calculator_bounds_exponents():
    with pytest.raises(ValueError, match="too large"):
        calculator.func("9 ** 9999")


def test_text_stats():
    assert text_stats.func("a b\nc") == "chars=5 words=3 lines=2"


# --------------------------------------------------------------------- sandbox


def test_sandbox_allows_paths_inside_the_root(tmp_path):
    box = Sandbox(tmp_path)
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    read_file, list_dir = make_file_tools(box)

    assert read_file.func("notes.txt") == "hello"
    assert "notes.txt" in list_dir.func(".")


def test_sandbox_refuses_to_escape(tmp_path):
    box = Sandbox(tmp_path / "root")
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")

    with pytest.raises(PermissionError, match="outside the sandbox"):
        box.resolve("../secret.txt")


def test_sandbox_refuses_absolute_escapes(tmp_path):
    box = Sandbox(tmp_path / "root")
    with pytest.raises(PermissionError):
        box.resolve(str(tmp_path / "elsewhere.txt"))


def test_read_file_truncates(tmp_path):
    box = Sandbox(tmp_path)
    (tmp_path / "big.txt").write_text("x" * 50, encoding="utf-8")
    read_file, _ = make_file_tools(box)
    out = read_file.func("big.txt", max_chars=10)
    assert out.startswith("x" * 10)
    assert "truncated" in out


def test_default_tool_set_has_the_expected_names():
    names = {t.name for t in default_tools()}
    assert {"calculator", "text_stats", "read_file", "list_dir"} <= names
