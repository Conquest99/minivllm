"""把 agent 套成一个 HTTP 服务：一个 ``POST /chat``，SSE 流式返回。

这一层刻意很薄。它要回答的问题只有三个：

* **上下文放哪？**  放服务端会话表（``sessions.py``），浏览器只拿一个 session_id。
  放客户端意味着把整段对话每轮重发一遍，而且客户端可以篡改历史。
* **怎么流式？**  agent 循环是同步阻塞的，所以生成器里用一个工作线程跑它，事件
  通过 ``queue.Queue`` 出来。没有协程魔法，也没有第二份 agent 实现。
* **模型谁独占？**  一个进程一份权重，所以用一个全局锁串行化 ``complete``。
  这是本仓库最该被替换的地方，见文件末尾的说明。
"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from miniagent.agent import DEFAULT_SYSTEM_PROMPT, Agent
from miniagent.backends import LLMBackend
from miniagent.types import Role, Step, StopReason
from serve.sessions import DEFAULT_MAX_TURNS, ChatSession, SessionStore

STATIC = Path(__file__).resolve().parent / "static"

Event = tuple[str, dict[str, Any]]
"""``(event_name, payload)`` —— 直接对应 SSE 的 ``event:`` / ``data:`` 两行。"""


def sse(event: str, payload: dict[str, Any]) -> str:
    """编码一条 SSE 帧。

    用 ``json.dumps`` 而不是手拼，是因为换行会让 ``data:`` 提前结束一帧：一条多行的
    工具结果（表格输出就是多行）会直接切断流。JSON 会把换行转义成 ``\\n``。
    """
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class Turn:
    """把一次 ``agent.run`` 的中间过程翻译成 SSE 事件。

    两个回调恰好覆盖了「正在说」和「说完了」：
    ``on_token`` 在模型生成时被反复调用，``on_step`` 在一轮结束时被调用一次。
    """

    def __init__(self, emit: Callable[[str, dict[str, Any]], None]) -> None:
        self._emit = emit

    def on_token(self, text: str) -> None:
        self._emit("token", {"text": text})

    def on_step(self, step: Step) -> None:
        self._emit("step", {"index": step.index, "duration_s": round(step.duration_s, 3)})
        for call, result in zip(step.tool_calls, step.tool_results, strict=False):
            self._emit("call", {"name": call.name, "arguments": call.arguments})
            self._emit(
                "result",
                {
                    "name": result.name,
                    "ok": result.ok,
                    "content": result.content,
                    "duration_s": round(result.duration_s, 3),
                },
            )
        for err in step.parse_errors:
            self._emit("parse_error", {"message": err})
        for note in step.notes:
            self._emit("note", {"message": note})


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


def build_agent(
    backend: LLMBackend,
    tools: list[Any],
    *,
    system_prompt: str,
    on_token: Callable[[str], None] | None = None,
    on_step: Callable[[Step], None] | None = None,
    max_steps: int = 6,
) -> Agent:
    return Agent(
        backend,
        tools,
        system_prompt=system_prompt,
        max_steps=max_steps,
        on_step=on_step,
        on_token=on_token,
    )


def create_app(
    backend: LLMBackend,
    tools: list[Any],
    *,
    system_prompt: str | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_steps: int = 6,
    store: SessionStore | None = None,
) -> FastAPI:
    """构造 app。后端和工具由调用方注入，所以测试可以用 ScriptedBackend 跑。"""
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    system_message = _system_message(prompt)
    sessions = store or SessionStore()
    model_lock = threading.Lock()
    """一个进程一份权重：同一时刻只让一个 ``complete`` 进去。

    注意粒度是「一次模型调用」，不是「一整轮对话」。所以两个会话可以交错进行 ——
    A 等工具执行时 B 能拿到模型。整轮独占会让第二个用户等到第一个人把 agent 跑完。
    """

    app = FastAPI(title="miniagent chat", version="0.1.0")
    app.state.sessions = sessions

    def serialized_complete(*, on_token, **kwargs):
        with model_lock:
            return backend.complete(on_token=on_token, **kwargs)

    class _Serialized:
        """把全局锁套成一个后端，好让 Agent 完全不知道并发的存在。"""

        name = getattr(backend, "name", "backend")

        def complete(self, messages, *, tools=None, max_tokens=None, on_token=None):
            return serialized_complete(
                messages=messages, tools=tools, max_tokens=max_tokens, on_token=on_token
            )

    agent_backend = _Serialized()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "backend": agent_backend.name, "sessions": len(sessions)}

    @app.post("/sessions")
    def new_session() -> dict[str, str]:
        return {"session_id": sessions.create().session_id}

    @app.get("/sessions/{session_id}")
    def read_session(session_id: str) -> dict[str, Any]:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return {
            "session_id": session.session_id,
            "turns": session.turns,
            "messages": len(session.messages),
            "history": [
                {"role": m.role.value, "content": m.content, "name": m.name} for m in session.messages
            ],
        }

    @app.delete("/sessions/{session_id}")
    def drop_session(session_id: str) -> dict[str, bool]:
        return {"dropped": sessions.drop(session_id)}

    @app.post("/chat")
    def chat(request: ChatRequest) -> StreamingResponse:
        """一轮对话，SSE 流式返回。

        返回的流一定会以 ``done`` 或 ``error`` 结束，所以客户端不必处理「连接正常关闭
        但没收到结论」这种中间状态。
        """
        session = sessions.get_or_create(request.session_id)
        return StreamingResponse(
            _stream_turn(
                session=session,
                message=request.message,
                backend=agent_backend,
                tools=tools,
                system_prompt=prompt,
                system_message=system_message,
                max_turns=max_turns,
                max_steps=max_steps,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Session-Id": session.session_id,
                # 关掉反向代理的缓冲，否则 SSE 会被攒成一坨再吐出来
                "X-Accel-Buffering": "no",
            },
        )

    return app


def _system_message(prompt: str):
    from miniagent.types import Message

    return Message(Role.SYSTEM, prompt)


def _stream_turn(
    *,
    session: ChatSession,
    message: str,
    backend: Any,
    tools: list[Any],
    system_prompt: str,
    system_message: Any,
    max_turns: int,
    max_steps: int,
) -> Iterator[str]:
    """跑一轮，把事件从工作线程搬回这个生成器。

    为什么要多一个线程：``agent.run`` 是阻塞的，而生成器每 yield 一次就把控制权还给
    Starlette。不开线程的话，所有 ``on_token``/``on_step`` 事件只能在 ``run`` 返回后
    一次性吐出 —— 流式就名存实亡了。
    """
    events: queue.Queue[Event | None] = queue.Queue()

    def emit(name: str, payload: dict[str, Any]) -> None:
        events.put((name, payload))

    def work() -> None:
        turn = Turn(emit)
        try:
            with session.lock:  # 同一个会话不能被两轮同时改写
                history = session.messages
                agent = build_agent(
                    backend,
                    tools,
                    system_prompt=system_prompt,
                    on_token=turn.on_token,
                    on_step=turn.on_step,
                    max_steps=max_steps,
                )
                result = agent.run(message, history=history or None)
                session.record(result.messages, system=system_message, max_turns=max_turns)
                if result.stopped_reason is StopReason.BACKEND_ERROR:
                    # agent 自己已经把后端异常收成了一条结果，但发给客户端的应该是
                    # 「这一轮没跑成」，而不是 "done" + 一句看起来像答案的话。
                    emit(
                        "error",
                        {"type": "backend_error", "message": result.answer, "stopped": "backend_error"},
                    )
                    return
                emit(
                    "done",
                    {
                        "answer": result.answer,
                        "stopped": result.stopped_reason.value,
                        "steps": len(result.steps),
                        "tool_calls": result.num_tool_calls,
                        "tool_errors": result.num_tool_errors,
                        "duration_s": round(result.duration_s, 3),
                        "turns": session.turns,
                        "messages": len(session.messages),
                    },
                )
        except Exception as exc:  # 任何异常都要变成一条可读事件，不能让流裸断
            emit("error", {"type": type(exc).__name__, "message": str(exc)})
        finally:
            events.put(None)

    yield sse("start", {"session_id": session.session_id, "turn": session.turns + 1})
    worker = threading.Thread(target=work, name="agent-turn", daemon=True)
    worker.start()
    try:
        while True:
            item = events.get()
            if item is None:
                break
            name, payload = item
            yield sse(name, payload)
    finally:
        worker.join(timeout=1.0)


__all__ = ["ChatRequest", "create_app", "sse", "Turn"]
