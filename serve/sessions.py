"""服务端的会话状态：一段可以接着写的对话，加上它的裁剪规则。

刻意不依赖 FastAPI —— 多轮上下文是纯数据结构问题，能单独测就不该绑在 web 框架上。
``serve/app.py`` 只负责把它接到 HTTP 和 SSE。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from miniagent.types import Message, Role

DEFAULT_MAX_TURNS = 6
"""保留最近几轮对话（一轮 = 一条 user 消息及其后的全部消息）。"""


def split_turns(messages: list[Message]) -> list[list[Message]]:
    """把消息列表切成「轮」，每轮从一条 ``user`` 消息开始。

    为什么按轮而不是按条裁剪：一条 ``tool`` 消息只有在它前面那条带 ``tool_calls``
    的 assistant 消息存在时才有意义。按条裁剪会把它们拆散，而 Qwen 的模板遇到孤立的
    ``tool`` 消息不会报错，而是**静默降级成一条 user 消息** —— 模型于是看到一段
    「用户发来一段 tool_response」，却不知道自己调用过什么。
    """
    turns: list[list[Message]] = []
    for message in messages:
        if message.role is Role.USER or not turns:
            turns.append([message])
        else:
            turns[-1].append(message)
    return turns


def trim_history(
    messages: list[Message],
    *,
    system: Message | None,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> list[Message]:
    """保留 system 消息 + 最近 ``max_turns`` 轮，绝不拆散一轮。"""
    turns = split_turns([m for m in messages if m.role is not Role.SYSTEM])
    kept = turns[-max_turns:] if max_turns > 0 else []
    out: list[Message] = []
    if system is not None:
        out.append(system)
    for turn in kept:
        out.extend(turn)
    return out


@dataclass
class ChatSession:
    """一次浏览器/客户端会话的全部状态。"""

    session_id: str
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    turns: int = 0
    """已经完成的轮数，仅用于展示。"""
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, result_messages: list[Message], *, system: Message, max_turns: int) -> None:
        """把这一轮结束后的完整消息列表收进来，并按轮裁剪。"""
        self.messages = trim_history(result_messages, system=system, max_turns=max_turns)
        self.turns += 1
        self.last_used_at = time.time()

    def reset(self) -> None:
        self.messages = []
        self.turns = 0
        self.last_used_at = time.time()


class SessionStore:
    """有上限的会话表，最久没用的先扔。

    只存在内存里：进程重启就全丢。这对一个教学用的单进程服务是合理取舍，但要写进文档
    —— 「重启后上下文消失」是使用者第一个会踩到的坑。
    """

    def __init__(self, max_sessions: int = 64) -> None:
        self._sessions: OrderedDict[str, ChatSession] = OrderedDict()
        self._lock = threading.Lock()
        self.max_sessions = max_sessions

    def create(self) -> ChatSession:
        with self._lock:
            session = ChatSession(session_id=uuid.uuid4().hex[:12])
            self._sessions[session.session_id] = session
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
            return session

    def get(self, session_id: str) -> ChatSession | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions.move_to_end(session_id)
            return session

    def get_or_create(self, session_id: str | None) -> ChatSession:
        if session_id:
            session = self.get(session_id)
            if session is not None:
                return session
        return self.create()

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


__all__ = [
    "ChatSession",
    "DEFAULT_MAX_TURNS",
    "SessionStore",
    "split_turns",
    "trim_history",
]
