"""agent、工具、后端共用的消息与结果类型。

整个 agent 就是一个在 ``list[Message]`` 上的循环。两个设计选择值得说明：

* **只有一种内部消息格式。**  后端各自负责和自己那家服务商互转（Qwen 的
  ``<tool_call>`` 标签、OpenAI 的 ``tool_calls`` 数组），但循环永远只看见这一种类型。
  这正是循环可以在没有模型的情况下被测试的原因。
* **工具调用挂在 assistant 消息上。**  这与 OpenAI 的 schema 对齐，Qwen 的 chat
  template 渲染的正是这个形状，所以不需要任何自定义的 prompt 改写。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    def __str__(self) -> str:  # 让 f"{role}" 打印 "user"，而不是 "Role.USER"
        return self.value


@dataclass
class ToolCall:
    """模型发出的、用 JSON 参数运行某个具名工具的请求。"""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    raw: str = ""
    """The exact text the model produced, kept for tracing and debugging."""

    def key(self) -> str:
        """(name, arguments) 的稳定标识 —— 用来检测重复调用。"""
        return f"{self.name}:{json.dumps(self.arguments, sort_keys=True, ensure_ascii=False)}"


@dataclass
class ToolResult:
    call_id: str
    name: str
    ok: bool
    content: str
    error: str | None = None
    duration_s: float = 0.0
    affects: list[str] = field(default_factory=list)
    """Paths this call changed.  Filled in by the registry, not by the tool."""
    reads: list[str] = field(default_factory=list)
    """Paths this call observed -- what discharges the read-back duty."""

    def to_observation(self) -> str:
        """模型被允许看到的东西。

        Qwen 的 chat template 会丢掉工具名和调用 id，所以名字在正文里重复一次 —— 否则一步里
        调了两个工具的模型分不清哪条结果属于哪个调用。
        """
        return f"[{self.name}] {self.content}"


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    name: str | None = None
    """Set on ``role=TOOL`` messages so traces can show who produced the result."""

    def to_hf(self) -> dict[str, Any]:
        """渲染成 ``apply_chat_template`` 期望的 dict 形状。"""
        if self.role is Role.ASSISTANT and self.tool_calls:
            return {
                "role": "assistant",
                "content": self.content,
                "tool_calls": [
                    {"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                    for c in self.tool_calls
                ],
            }
        if self.role is Role.TOOL:
            return {"role": "tool", "content": self.content}
        return {"role": self.role.value, "content": self.content}

    def __repr__(self) -> str:
        if self.role is Role.ASSISTANT and self.tool_calls:
            names = ", ".join(c.name for c in self.tool_calls)
            return f"Message(assistant, tool_calls=[{names}])"
        preview = self.content[:48].replace("\n", " ")
        more = "..." if len(self.content) > 48 else ""
        return f"Message({self.role}, {preview!r}{more})"


class StopReason(str, Enum):
    FINAL_ANSWER = "final_answer"
    MAX_STEPS = "max_steps"
    LOOP_DETECTED = "loop_detected"
    TOO_MANY_ERRORS = "too_many_errors"
    BACKEND_ERROR = "backend_error"
    WRITE_INCOMPLETE = "write_incomplete"
    """The model answered while a file change it attempted never happened."""
    UNVERIFIED = "unverified"
    """The answer came before the model read back a file it had changed."""


@dataclass
class Step:
    """一次模型回合，加上它触发的全部工具调用。"""

    index: int
    assistant_text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    """Loop-level remarks about this step, e.g. why an answer was rejected."""
    duration_s: float = 0.0

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def summary(self) -> str:
        if self.parse_errors:
            return f"step {self.index}: {len(self.parse_errors)} parse error(s)"
        if self.notes:
            return f"step {self.index}: {self.notes[0]}"
        if not self.tool_calls:
            return f"step {self.index}: final answer"
        parts = []
        for call, result in zip(self.tool_calls, self.tool_results, strict=False):
            status = "ok" if result.ok else "error"
            parts.append(f"{call.name}({status})")
        return f"step {self.index}: {', '.join(parts)}"


@dataclass
class AgentResult:
    question: str
    answer: str
    steps: list[Step] = field(default_factory=list)
    stopped_reason: StopReason = StopReason.FINAL_ANSWER
    messages: list[Message] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def num_tool_calls(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps)

    @property
    def num_tool_errors(self) -> int:
        return sum(1 for s in self.steps for r in s.tool_results if not r.ok)

    @property
    def succeeded(self) -> bool:
        return self.stopped_reason is StopReason.FINAL_ANSWER and bool(self.answer)
