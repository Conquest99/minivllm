"""Parsing every dialect a model might use, and failing gracefully on garbage."""

from __future__ import annotations

import pytest

from miniagent.parser import format_tool_call, parse


def test_qwen_tool_call_tag():
    payload = '{"name": "calculator", "arguments": {"expression": "1+1"}}'
    text = f"Let me compute.\n<tool_call>\n{payload}\n</tool_call>"
    result = parse(text)

    assert result.has_calls
    assert not result.errors
    assert result.text == "Let me compute."
    assert result.calls[0].name == "calculator"
    assert result.calls[0].arguments == {"expression": "1+1"}


def test_multiple_tool_calls_in_one_reply():
    text = (
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>\n'
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
    )
    result = parse(text)
    assert [c.name for c in result.calls] == ["a", "b"]
    assert result.text == ""


def test_json_fence():
    text = 'Checking.\n```json\n{"name": "calculator", "arguments": {"expression": "2*3"}}\n```'
    result = parse(text)
    assert result.has_calls
    assert result.calls[0].arguments == {"expression": "2*3"}
    assert result.text == "Checking."


def test_bare_json_object():
    result = parse('{"name": "calculator", "arguments": {"expression": "9-4"}}')
    assert result.has_calls
    assert result.calls[0].arguments == {"expression": "9-4"}


def test_react_style():
    text = 'I need to multiply.\nAction: calculator\nAction Input: {"expression": "23*7"}\n'
    result = parse(text)
    assert result.has_calls
    assert result.calls[0].name == "calculator"
    assert result.calls[0].arguments == {"expression": "23*7"}


def test_react_with_nested_json_arguments():
    text = 'Action: search\nAction Input: {"query": "x", "opts": {"limit": 3}}'
    result = parse(text)
    assert result.calls[0].arguments == {"query": "x", "opts": {"limit": 3}}


def test_react_with_non_json_input_is_reported_not_raised():
    result = parse("Action: calculator\nAction Input: 23*7")
    assert not result.calls
    assert result.errors and "must be a JSON object" in result.errors[0]


def test_arguments_as_a_json_string_are_reparsed():
    text = '<tool_call>{"name": "calculator", "arguments": "{\\"expression\\": \\"5+5\\"}"}</tool_call>'
    result = parse(text)
    assert result.calls[0].arguments == {"expression": "5+5"}


def test_alternate_key_spellings():
    assert parse('<tool_call>{"tool": "a", "args": {"x": 1}}</tool_call>').calls[0].name == "a"
    nested = parse('<tool_call>{"function": {"name": "b"}, "parameters": {"y": 2}}</tool_call>')
    assert nested.calls[0].name == "b"
    assert nested.calls[0].arguments == {"y": 2}


def test_openai_wire_shape_keeps_its_arguments():
    """``{"function": {"name": …, "arguments": …}}`` 是 OpenAI API 返回的形状。

    以前 name 会在内层 dict 里找到，而 arguments 仍在外层查找，于是调用变成 ``{}`` ——
    一次静默的数据丢失，只在后面以「缺少必需参数」的形式暴露出来。
    """
    text = '<tool_call>{"type": "function", "function": {"name": "add", "arguments": {"a": 1}}}</tool_call>'
    call = parse(text).calls[0]
    assert call.name == "add"
    assert call.arguments == {"a": 1}


def test_single_quotes_are_tolerated():
    result = parse("<tool_call>{'name': 'calculator', 'arguments': {'expression': '1+1'}}</tool_call>")
    assert result.has_calls
    assert result.calls[0].arguments == {"expression": "1+1"}


def test_trailing_comma_is_tolerated():
    result = parse('<tool_call>{"name": "calculator", "arguments": {"expression": "1+1",}}</tool_call>')
    assert result.has_calls
    assert result.calls[0].arguments == {"expression": "1+1"}


def test_malformed_block_is_an_error_not_an_exception():
    result = parse("<tool_call>not json at all</tool_call>")
    assert not result.calls
    assert result.errors
    assert "not a JSON object" in result.errors[0]


def test_missing_name_is_an_error():
    result = parse('<tool_call>{"arguments": {"x": 1}}</tool_call>')
    assert not result.calls
    assert "no usable 'name'" in result.errors[0]


def test_plain_prose_has_no_calls():
    result = parse("The answer is 42.")
    assert not result.has_calls
    assert not result.errors
    assert result.text == "The answer is 42."


def test_empty_input():
    result = parse("")
    assert not result.has_calls
    assert result.text == ""


@pytest.mark.parametrize("style", ["qwen", "bare"])
def test_format_tool_call_round_trips(style):
    text = format_tool_call("calculator", {"expression": "1+2"}, style=style)
    result = parse(text)
    assert result.has_calls
    assert result.calls[0].name == "calculator"
    assert result.calls[0].arguments == {"expression": "1+2"}
