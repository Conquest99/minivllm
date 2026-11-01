"""Message and result types shared by the agent, the tools and the backends.

The whole agent is a loop over ``list[Message]``.  Two design choices matter:

* **One internal message format.**  Backends translate to/from whatever their
  provider wants (Qwen's ``<tool_call>`` tags, OpenAI's ``tool_calls`` array), but
  the loop only ever sees this one type.  That is what makes the loop testable
  without a model.
* **Tool calls live on the assistant message.**  That mirrors the OpenAI schema,
  and Qwen's chat template renders exactly that shape, so no custom prompt
  rewriting is needed.
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

    def __str__(self) -> str:  # so f"{role}" prints "user", not "Role.USER"
        return self.value


@dataclass
class ToolCall:
    """A request from the model to run a named tool with JSON arguments."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    raw: str = ""
    """The exact text the model produced, kept for tracing and debugging."""

    def key(self) -> str:
        """Stable identity of (name, arguments) -- used to detect repeated calls."""
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
        """What the model is allowed to see.

        Qwen's chat template drops the tool name and the call id, so the name is
        repeated inside the body -- otherwise a model that called two tools in one
        step cannot tell which result belongs to which call.
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
        """Render as the dict shape ``apply_chat_template`` expects."""
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
    """One model turn plus every tool call it triggered."""

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
