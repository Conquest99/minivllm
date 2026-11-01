"""Schema generation, argument validation, sandboxing and calculator safety."""

from __future__ import annotations

import pytest

from miniagent.builtin_tools import (
    Sandbox,
    calculator,
    default_tools,
    make_file_tools,
    make_write_tools,
    text_stats,
)
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


# ------------------------------------------------------------------- write tools


def write_tools(root) -> dict:
    box = root if isinstance(root, Sandbox) else Sandbox(root)
    return {t.name: t for t in make_write_tools(box)}


def test_write_file_creates_a_file(tmp_path):
    out = write_tools(tmp_path)["write_file"].func("notes.txt", "hello")
    assert "created" in out and "notes.txt" in out
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello"


def test_write_file_overwrites_and_appends(tmp_path):
    tools = write_tools(tmp_path)
    tools["write_file"].func("a.txt", "one")
    tools["write_file"].func("a.txt", "two")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "two"

    tools["write_file"].func("a.txt", "three", mode="append")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "twothree"


def test_write_file_creates_missing_parent_directories(tmp_path):
    write_tools(tmp_path)["write_file"].func("sub/dir/x.txt", "x")
    assert (tmp_path / "sub" / "dir" / "x.txt").is_file()


def test_write_file_rejects_an_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="mode must be"):
        write_tools(tmp_path)["write_file"].func("a.txt", "x", mode="replace")


def test_write_lines_joins_an_array(tmp_path):
    write_tools(tmp_path)["write_lines"].func("todo.txt", ["buy milk", "walk the dog"])
    assert (tmp_path / "todo.txt").read_text(encoding="utf-8") == "buy milk\nwalk the dog\n"


def test_edit_file_replaces_exactly_one_occurrence(tmp_path):
    tools = write_tools(tmp_path)
    tools["write_file"].func("a.txt", "one\ntwo\nthree\n")

    out = tools["edit_file"].func("a.txt", "two", "TWO")

    assert "edited" in out
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "one\nTWO\nthree\n"


def test_edit_file_explains_a_missing_match(tmp_path):
    tools = write_tools(tmp_path)
    tools["write_file"].func("a.txt", "one\ntwo\n")
    with pytest.raises(ValueError, match="does not occur"):
        tools["edit_file"].func("a.txt", "three", "THREE")


def test_edit_file_refuses_an_ambiguous_match(tmp_path):
    tools = write_tools(tmp_path)
    tools["write_file"].func("a.txt", "a\na\n")
    with pytest.raises(ValueError, match="occurs 2 times"):
        tools["edit_file"].func("a.txt", "a", "b")


def test_edit_file_rejects_an_empty_needle(tmp_path):
    tools = write_tools(tmp_path)
    tools["write_file"].func("a.txt", "x\n")
    with pytest.raises(ValueError, match="must not be empty"):
        tools["edit_file"].func("a.txt", "", "y")


def test_edit_file_receipt_names_the_line_it_changed(tmp_path):
    """The receipt must distinguish line 1 from line 2 -- that was the whole task."""
    tools = write_tools(tmp_path)
    tools["write_file"].func("a.txt", "one\ntwo\nthree\n")

    out = tools["edit_file"].func("a.txt", "three", "THREE")

    assert "line 3" in out
    assert "'three' -> 'THREE'" in out


def test_edit_file_receipt_reports_a_line_range_for_multi_line_edits(tmp_path):
    tools = write_tools(tmp_path)
    tools["write_file"].func("a.txt", "one\ntwo\nthree\nfour\n")

    out = tools["edit_file"].func("a.txt", "two\nthree", "TWO\nTHREE")

    assert "lines 2-3" in out


def test_edit_file_receipt_trims_a_long_snippet(tmp_path):
    tools = write_tools(tmp_path)
    long_line = "x" * 100
    tools["write_file"].func("a.txt", long_line + "\n")

    out = tools["edit_file"].func("a.txt", long_line, "y")

    assert "…" in out
    assert len(out) < 200


def test_delete_file_removes_a_file_and_refuses_a_directory(tmp_path):
    tools = write_tools(tmp_path)
    tools["write_file"].func("gone.txt", "x")

    assert "deleted" in tools["delete_file"].func("gone.txt")
    assert not (tmp_path / "gone.txt").exists()

    with pytest.raises(IsADirectoryError):
        tools["delete_file"].func(".")


def test_write_tools_cannot_escape_the_sandbox(tmp_path):
    tools = write_tools(tmp_path / "root")
    with pytest.raises(PermissionError):
        tools["write_file"].func("../outside.txt", "nope")
    with pytest.raises(PermissionError):
        tools["delete_file"].func(str(tmp_path / "elsewhere.txt"))


def test_write_tool_schemas_are_derived_like_any_other(tmp_path):
    tools = write_tools(tmp_path)

    lines = tools["write_lines"].parameters["properties"]["lines"]
    assert lines["type"] == "array" and lines["items"] == {"type": "string"}
    assert tools["write_file"].parameters["required"] == ["path", "content"]
    assert tools["edit_file"].parameters["required"] == ["path", "old", "new"]
    assert tools["delete_file"].parameters["required"] == ["path"]


def test_only_the_write_tools_are_marked_as_mutating(tmp_path):
    mutating = {name for name, t in write_tools(tmp_path).items() if t.mutates}
    assert mutating == {"write_file", "write_lines", "edit_file", "delete_file"}

    read_tools = {t.name: t for t in make_file_tools(Sandbox(tmp_path))}
    assert not any(t.mutates for t in read_tools.values())


def test_the_registry_reports_which_paths_a_call_touched(tmp_path):
    box = Sandbox(tmp_path)
    registry = ToolRegistry([*make_file_tools(box), *make_write_tools(box)])

    wrote = registry.execute(ToolCall(name="write_file", arguments={"path": "a.txt", "content": "x"}))
    assert wrote.ok and wrote.affects == ["a.txt"]

    read = registry.execute(ToolCall(name="read_file", arguments={"path": "a.txt"}))
    assert read.ok and read.reads == ["a.txt"]

    failed = registry.execute(ToolCall(name="read_file", arguments={"path": "missing.txt"}))
    assert not failed.ok and failed.reads == ["missing.txt"]


# ---------------------------------------------- read-before-write (optimistic)


def test_appending_to_a_file_this_run_never_saw_is_refused(tmp_path):
    """The exact bug from the demo: `buy milk` + `buy milk` = one glued line."""
    (tmp_path / "todo.txt").write_text("buy milk", encoding="utf-8")

    with pytest.raises(PermissionError, match="has not read it"):
        write_tools(tmp_path)["write_lines"].func("todo.txt", ["walk the dog"], mode="append")


def test_reading_first_makes_the_file_writable(tmp_path):
    (tmp_path / "todo.txt").write_text("buy milk\n", encoding="utf-8")
    box = Sandbox(tmp_path)
    read_file, _ = make_file_tools(box)

    read_file.func("todo.txt")
    write_tools(box)["write_lines"].func("todo.txt", ["walk the dog"], mode="append")

    assert (tmp_path / "todo.txt").read_text(encoding="utf-8") == "buy milk\nwalk the dog\n"


def test_a_file_that_changed_after_it_was_read_is_refused(tmp_path):
    (tmp_path / "a.txt").write_text("v1\n", encoding="utf-8")
    box = Sandbox(tmp_path)
    read_file, _ = make_file_tools(box)
    read_file.func("a.txt")

    (tmp_path / "a.txt").write_text("v2\n", encoding="utf-8")  # somebody else wrote it

    with pytest.raises(PermissionError, match="changed after you read it"):
        write_tools(box)["edit_file"].func("a.txt", "v2", "v3")
