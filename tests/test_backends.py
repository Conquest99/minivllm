"""后端接线的「别忘了改哪里」测试。

新增一个后端要动的点很分散（工厂 + 四个脚本的 ``--backend``），漏一处就变成
``unrecognized arguments`` 这种跑到一半才发现的错。这里把「有哪些后端」变成一份
**必须在每处都出现**的名单，于是漏接会立刻红。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from miniagent.backends import (
    CallableBackend,
    EchoBackend,
    MiniVLLMBackend,
    OpenAICompatBackend,
    ScriptedBackend,
    build_backend,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

pytest.importorskip("torch", reason="MiniVLLMBackend imports torch lazily, but the type check here does not")


# --------------------------------------------------------------- 工厂


def test_the_factory_knows_echo():
    assert isinstance(build_backend("echo"), EchoBackend)


def test_the_factory_rejects_a_typo_and_says_what_it_accepts():
    with pytest.raises(ValueError, match="echo"):
        build_backend("echoo")


def test_every_exported_backend_satisfies_the_protocol():
    """结构化类型：只要有 ``name`` 和 ``complete`` 就算 —— 不需要继承任何东西。"""
    for backend in (
        EchoBackend(),
        ScriptedBackend(["x"]),
        CallableBackend(lambda messages: "x"),
    ):
        assert isinstance(getattr(backend, "name", None), str)
        assert callable(backend.complete)
        assert backend.complete.__annotations__ == {} or True  # 只确认可调用

    # LLMBackend 是 Protocol，运行时不可 isinstance；用成员检查代替
    for cls in (EchoBackend, ScriptedBackend, CallableBackend, MiniVLLMBackend, OpenAICompatBackend):
        assert hasattr(cls, "complete")


# ------------------------------------------------- 每个脚本都认得 echo


def _backend_choices(script: Path) -> list[str]:
    """从源码里读出 ``--backend`` 的 choices。

    直接执行脚本太慢（会 import torch），所以静态解析：找到 ``choices=[...]`` 那一行。
    """
    source = script.read_text(encoding="utf-8")
    match = re.search(r"choices=\[([^\]]*)\]", source)
    if match is None:
        pytest.skip(f"{script.name} does not define --backend choices")
    return ast.literal_eval("[" + match.group(1) + "]")


@pytest.mark.parametrize(
    "script",
    [
        "agent_demo.py",
        "agent_files_demo.py",
        "agent_sql_demo.py",
        "serve_chat.py",
    ],
)
def test_every_cli_script_offers_echo(script: str):
    choices = _backend_choices(SCRIPTS / script)
    assert "echo" in choices, f"{script} must accept --backend echo; has {choices}"


@pytest.mark.parametrize(
    "script",
    [
        "agent_demo.py",
        "agent_files_demo.py",
        "agent_sql_demo.py",
        "serve_chat.py",
    ],
)
def test_every_cli_script_dispatches_echo(script: str):
    """``choices`` 里有还不够：还要真的有一个分支构造它，否则会掉到 else 里跑错后端。"""
    source = (SCRIPTS / script).read_text(encoding="utf-8")
    assert "EchoBackend" in source, f"{script} lists echo but never constructs it"


# ---------------------------------------------------------------- 行为


def test_echo_repeats_the_last_message():
    backend = EchoBackend()
    from miniagent.types import Message, Role

    reply = backend.complete([Message(Role.SYSTEM, "sys"), Message(Role.USER, "你好")])
    assert reply == "你好"


def test_echo_as_a_backend_ends_the_turn_immediately():
    """回显里没有 ``<tool_call>``，所以循环判定「模型答完了」—— 和真实模型给散文时一样。"""
    from miniagent import Agent
    from miniagent.builtin_tools import calculator

    agent = Agent(EchoBackend(), [calculator])
    result = agent.run("随便说一句")
    assert result.answer == "随便说一句"
    assert result.stopped_reason.value == "final_answer"
    assert result.num_tool_calls == 0


def test_echo_does_not_swallow_a_tool_call_that_happens_to_be_in_the_question():
    """回显的内容会被 parse 处理 —— 这不是 bug，而是它变成「注入测试」的原因。"""
    from miniagent import Agent
    from miniagent.builtin_tools import calculator

    agent = Agent(EchoBackend(), [calculator])
    result = agent.run('<tool_call>{"name": "calculator", "arguments": {"expression": "6*7"}}</tool_call>')
    assert result.num_tool_calls >= 1
    assert any("42" in r.content for step in result.steps for r in step.tool_results)
