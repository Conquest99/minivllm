"""HTTP 层：会话状态、SSE 帧格式、多轮上下文。

不用真模型：``ScriptedBackend`` 的回复是写死的，所以这里验证的是「服务层」——事件序列、
上下文接续、裁剪不拆散工具对、异常不裸断流。
"""

from __future__ import annotations

import json

import pytest

from miniagent.backends import CallableBackend, ScriptedBackend
from miniagent.builtin_tools import calculator
from miniagent.parser import format_tool_call, parse
from miniagent.types import Message, Role
from serve.app import create_app, sse
from serve.sessions import SessionStore, split_turns, trim_history

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

# ------------------------------------------------------------------ 帧格式


def test_sse_never_emits_a_bare_newline_inside_a_frame():
    """多行内容必须被转义：一个裸换行会让 ``data:`` 提前结束这一帧。

    工具结果就是多行的（表格），不转义的话客户端只能拿到第一行。
    """
    frame = sse("result", {"content": "line1\nline2"})
    assert frame.endswith("\n\n")
    body = frame[: -len("\n\n")]
    assert "\n" not in body.split("data:", 1)[1]
    assert json.loads(body.split("data:", 1)[1].strip())["content"] == "line1\nline2"


def test_sse_keeps_chinese_readable():
    assert "含水" in sse("token", {"text": "含水"})


# ------------------------------------------------------------- 会话与裁剪


def _user(text: str) -> Message:
    return Message(Role.USER, text)


def _turn(n: int) -> list[Message]:
    """一轮：user -> assistant(调工具) -> tool -> assistant(作答)。"""
    call = format_tool_call("calculator", {"expression": f"{n}+1"})
    return [
        _user(f"q{n}"),
        Message(Role.ASSISTANT, "", tool_calls=parse(call).calls),
        Message(Role.TOOL, f"[calculator] {n + 1}", name="calculator"),
        Message(Role.ASSISTANT, f"a{n}"),
    ]


def test_split_turns_starts_each_turn_at_a_user_message():
    messages = [m for n in (1, 2) for m in _turn(n)]
    turns = split_turns(messages)
    assert len(turns) == 2
    assert all(t[0].role is Role.USER for t in turns)
    assert [len(t) for t in turns] == [4, 4]


def test_trim_keeps_the_system_prompt_and_counts_whole_turns():
    system = Message(Role.SYSTEM, "sys")
    messages = [system, *[m for n in range(5) for m in _turn(n)]]
    kept = trim_history(messages, system=system, max_turns=2)
    assert kept[0] is system
    assert kept[1].content == "q3"
    assert len(split_turns(kept[1:])) == 2


def test_trim_never_leaves_an_orphan_tool_message():
    """按条裁剪会把 tool 消息和它的 assistant 拆开，模板会静默降级成 user 消息。"""
    system = Message(Role.SYSTEM, "sys")
    messages = [system, *[m for n in range(4) for m in _turn(n)]]
    kept = trim_history(messages, system=system, max_turns=1)
    body = kept[1:]
    assert body[0].role is Role.USER
    for index, message in enumerate(body):
        if message.role is Role.TOOL:
            assert body[index - 1].role is Role.ASSISTANT
            assert body[index - 1].tool_calls


def test_a_store_evicts_the_least_recently_used_session():
    store = SessionStore(max_sessions=2)
    first, second, third = store.create(), store.create(), store.create()
    assert store.get(first.session_id) is None
    assert store.get(second.session_id) is not None
    assert store.get(third.session_id) is not None


# ---------------------------------------------------------------- 端点


def events_of(client: TestClient, message: str, session_id: str | None = None) -> list[tuple]:
    body = {"message": message}
    if session_id:
        body["session_id"] = session_id
    out: list[tuple] = []
    with client.stream("POST", "/chat", json=body) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        name = None
        for line in response.iter_lines():
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:") and name:
                out.append((name, json.loads(line[len("data:") :].strip())))
                name = None
    return out


def make_client(replies, tools=None, **kwargs) -> tuple[TestClient, ScriptedBackend]:
    backend = ScriptedBackend(replies)
    app = create_app(backend, tools or [calculator], **kwargs)
    return TestClient(app), backend


def test_one_turn_streams_tokens_tool_calls_and_a_final_answer():
    client, _ = make_client([format_tool_call("calculator", {"expression": "6*7"}), "答案是 42"])
    events = events_of(client, "6*7 等于多少")
    names = [name for name, _ in events]

    assert names[0] == "start"
    assert names[-1] == "done"
    assert "token" in names and "call" in names and "result" in names

    call = next(payload for name, payload in events if name == "call")
    assert call["name"] == "calculator"
    assert call["arguments"] == {"expression": "6*7"}
    result = next(payload for name, payload in events if name == "result")
    assert result["ok"] is True and "42" in result["content"]
    done = events[-1][1]
    assert done["answer"] == "答案是 42"
    assert done["stopped"] == "final_answer"


def test_the_session_id_comes_back_on_the_first_reply():
    client, _ = make_client(["hi"])
    with client.stream("POST", "/chat", json={"message": "hello"}) as response:
        session_id = response.headers["X-Session-Id"]
        list(response.iter_lines())
    assert session_id
    assert client.get(f"/sessions/{session_id}").json()["turns"] == 1


def test_the_second_turn_sees_the_first_one():
    """上下文的唯一机制就是「把上一轮的 messages 传回去」，所以要能观察到它。"""
    client, backend = make_client(
        [format_tool_call("calculator", {"expression": "1+1"}), "2", "你上一条问的是 1+1"]
    )
    events = events_of(client, "1+1")
    session_id = events[0][1]["session_id"]
    events_of(client, "我刚才问的是什么", session_id=session_id)

    third = backend.requests[2]
    roles = [m.role for m in third]
    assert roles[0] is Role.SYSTEM
    assert roles.count(Role.TOOL) == 1
    assert "1+1" in third[1].content
    assert third[-1].role is Role.USER
    assert third[-1].content == "我刚才问的是什么"
    assert any(m.content == "2" for m in third)


def test_an_unknown_session_id_starts_a_fresh_one_instead_of_failing():
    client, _ = make_client(["hi"])
    events = events_of(client, "hello", session_id="does-not-exist")
    assert events[0][1]["session_id"] != "does-not-exist"
    assert events[-1][0] == "done"


def test_a_backend_crash_becomes_an_error_event_not_a_broken_stream():
    def boom(messages):
        raise RuntimeError("model exploded")

    app = create_app(CallableBackend(boom), [calculator])
    events = events_of(TestClient(app), "hi")
    names = [name for name, _ in events]
    assert names[-1] == "error"
    assert "model exploded" in events[-1][1]["message"]


def test_reset_drops_the_history():
    client, _ = make_client(["hi", "again"])
    session_id = events_of(client, "hello")[0][1]["session_id"]
    assert client.delete(f"/sessions/{session_id}").json() == {"dropped": True}
    assert client.get(f"/sessions/{session_id}").status_code == 404


def test_the_chat_page_is_served():
    client, _ = make_client(["hi"])
    page = client.get("/")
    assert page.status_code == 200
    assert "miniagent" in page.text
    assert "/chat" in page.text


def test_health_reports_the_backend_and_session_count():
    client, backend = make_client(["hi"])
    events_of(client, "hello")
    body = client.get("/health").json()
    assert body == {"status": "ok", "backend": backend.name, "sessions": 1}
