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


def test_a_truncated_tool_call_is_reported_not_silently_answered():
    """``max_tokens`` 砍在半路时，``_TOOL_TAG`` 一个都匹配不上。

    于是 ``calls`` 和 ``errors`` 都空 —— 而 agent 判断「模型答完了」的条件正是「两个都
    空」，半截调用会被当成最终答案原样返回。这里锁住「必须报错」。
    """
    result = parse('<tool_call>{"name": "b"}')
    assert result.calls == []
    assert result.errors, "an unclosed <tool_call> must not look like a finished answer"
    assert "never closed" in result.errors[0]
    assert "<tool_call>" not in result.text


def test_an_unclosed_tag_does_not_hide_a_complete_one():
    text = '<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b"}'
    result = parse(text)
    assert [c.name for c in result.calls] == ["a"]
    assert result.errors, "the second, truncated call must still be reported"


def test_a_bare_object_that_does_not_parse_is_an_error_not_an_answer():
    result = parse('{"name": "a", "arguments": {{1, 2}}}')
    assert result.calls == []
    assert result.errors and "not parseable" in result.errors[0]


def test_prose_with_braces_in_the_middle_is_still_prose():
    """只在**整段**被花括号包着时才判成坏调用，否则会冤枉正常回答。"""
    result = parse("答案是 {x} 这个集合")
    assert result.calls == [] and result.errors == []


def test_literal_eval_raising_typeerror_cannot_escape_parse():
    """``ast.literal_eval`` 的契约里有 ``TypeError``，而原代码没捕它。

    ``{"a": {{1, 2}}}`` 会让它去建「装 set 的 set」，然后
    ``TypeError: unhashable type: 'set'`` 一路穿出 ``parse`` —— 直接违反
    「坏输出永不抛异常」这个模块前提。
    """
    from miniagent.parser import _load_object

    for blob in (
        '{"name": "a", "arguments": {{1, 2}}}',
        "{'name': 'a', 'arguments': {{1, 2}}}",
    ):
        assert _load_object(blob) is None or isinstance(_load_object(blob), dict)


def test_parse_never_raises_on_hostile_input(box=None):
    """把「永不抛异常」当契约来测，而不是当口号。"""
    hostile = [
        "<tool_call>",
        "</tool_call>",
        "<tool_call></tool_call>",
        '<tool_call>{"name": ' + "[" * 200,
        '<tool_call>{"name": "a", "arguments": "{"}</tool_call>',
        "Action: x\nAction Input: \x00\x01",
        "{" * 50,
        "}",
        '{"name": null}',
        '<tool_call>{"function": null}</tool_call>',
    ]
    for text in hostile:
        result = parse(text)  # 抛异常就会红
        assert isinstance(result.calls, list)
        assert isinstance(result.errors, list)


@pytest.mark.parametrize("style", ["qwen", "bare"])
def test_format_tool_call_round_trips(style):
    text = format_tool_call("calculator", {"expression": "1+2"}, style=style)
    result = parse(text)
    assert result.has_calls
    assert result.calls[0].name == "calculator"
    assert result.calls[0].arguments == {"expression": "1+2"}
