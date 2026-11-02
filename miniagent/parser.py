"""把模型的原始输出转成工具调用。

做 function calling 的模型并不返回结构体 —— 它返回的是*文本*，只不过文本里碰巧
装着一个结构体。这个模块就是那层适配，也正是大多数真实 agent 丢掉健壮性的地方，
因为每个模型都自己发明方言：

===============  ==========================================================
方言              例子
===============  ==========================================================
Qwen / Hermes    ``<tool_call>{"name": "add", "arguments": {"a": 1}}</tool_call>``
JSON 围栏         用 ```json 包起来的对象
裸 JSON          整个回复就是一个 JSON 对象
ReAct            ``Action: add`` / ``Action Input: {"a": 1}``
===============  ==========================================================

两个值得辩护的决定：

* **坏输出永不抛异常。**  畸形 JSON 是常态而不是异常：循环把错误回灌给模型让它
  重试。抛异常会把一个可恢复的小磕绊变成一次死掉的运行。
* **把剩下的散文也返回。**  模型可能先说「我查一下」，再发工具调用。那段散文属于
  assistant 消息；丢掉它就丢了下一轮的上下文。
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

from miniagent.types import ToolCall

_TOOL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)\s*```", re.DOTALL)
_REACT = re.compile(
    r"Action\s*:\s*(?P<name>[A-Za-z_][\w.\-]*)\s*(?:\r?\n)+\s*Action\s*Input\s*:\s*"
    r"(?P<args>\{[^\n]*\}|[^\n]+)",
    re.IGNORECASE,
)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")

_NAME_KEYS = ("name", "tool", "tool_name", "function")
_ARG_KEYS = ("arguments", "args", "parameters", "arguments_json", "input")


@dataclass
class ParseResult:
    text: str
    """Assistant prose with the tool-call blocks removed."""
    calls: list[ToolCall] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def has_calls(self) -> bool:
        return bool(self.calls)


def _load_object(blob: str) -> dict | None:
    """Parse a JSON (or JSON-ish) object.  Returns None if it is not an object."""
    blob = blob.strip()
    if not blob:
        return None
    if blob.startswith("```"):
        inner = _FENCE.search(blob)
        if inner:
            blob = inner.group(1).strip()

    for candidate in (blob, _TRAILING_COMMA.sub(r"\1", blob)):
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            try:
                value = ast.literal_eval(candidate)
            except (ValueError, SyntaxError, MemoryError, RecursionError):
                continue
        return value if isinstance(value, dict) else None
    return None


def _first(containers: tuple[dict, ...], keys: tuple[str, ...]) -> object | None:
    """First value found for any of ``keys``, searching ``containers`` in order."""
    for container in containers:
        for key in keys:
            if key in container:
                return container[key]
    return None


def _normalise(obj: dict, raw: str) -> ToolCall:
    """Accept the several key spellings models use and build a ToolCall."""
    # OpenAI 的线上格式把所有东西都嵌深了一层：
    # {"function": {"name": "add", "arguments": {...}}}。先解包：否则
    # name 会在内层 dict 里找到，而 arguments 仍在外层查找，
    # 于是悄悄变成空。
    source = obj
    for key in _NAME_KEYS:
        if isinstance(obj.get(key), dict):
            source = obj[key]
            break
    containers = (source, obj) if source is not obj else (obj,)

    name: object = _first(containers, _NAME_KEYS)
    if isinstance(name, dict):  # 形如 {"name": {"name": "..."}}
        name = name.get("name")

    args: object = _first(containers, _ARG_KEYS)
    if isinstance(args, str):
        parsed = _load_object(args)
        args = parsed if parsed is not None else {"_raw": args}
    if args is None:
        args = {}
    if not isinstance(args, dict):
        args = {"_raw": args}

    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool call has no usable 'name' field")

    return ToolCall(name=name.strip(), arguments=args, raw=raw)


def _resolve_lone_json(text: str) -> list[ToolCall] | None:
    """Handle a reply that is nothing but one JSON object."""
    obj = _load_object(text)
    if obj is None:
        return None
    if not any(k in obj for k in _NAME_KEYS):
        return None
    return [_normalise(obj, text)]


def parse(text: str) -> ParseResult:
    """Extract every tool call from ``text`` and strip the blocks from the prose."""
    if not text:
        return ParseResult(text="")

    calls: list[ToolCall] = []
    errors: list[str] = []
    leftover = text

    # 1) 显式的 <tool_call> 标签 —— Qwen2.5 / Hermes 的约定
    for match in _TOOL_TAG.finditer(text):
        block = match.group(1).strip()
        obj = _load_object(block)
        if obj is None:
            errors.append(f"tool_call block is not a JSON object: {block[:120]!r}")
            continue
        try:
            calls.append(_normalise(obj, match.group(0)))
        except ValueError as exc:
            errors.append(str(exc))
    if calls or errors:
        leftover = _TOOL_TAG.sub("", text)

    # 2) 围栏代码块 ```json
    if not calls and not errors:
        for match in _FENCE.finditer(text):
            block = match.group(1).strip()
            obj = _load_object(block)
            if obj is None or not any(k in obj for k in _NAME_KEYS):
                continue
            try:
                calls.append(_normalise(obj, match.group(0)))
            except ValueError as exc:
                errors.append(str(exc))
        if calls:
            leftover = _FENCE.sub("", text)

    # 3) 整个回复就是一个 JSON 对象
    if not calls and not errors:
        lone = _resolve_lone_json(text.strip())
        if lone:
            calls = lone
            leftover = ""

    # 4) ReAct 风格
    if not calls and not errors:
        match = _REACT.search(text)
        if match:
            raw_args = match.group("args").strip()
            obj = _load_object(raw_args)
            if obj is None:
                errors.append(
                    f"ReAct Action Input must be a JSON object, got {raw_args[:120]!r}. "
                    'Use: Action Input: {"expression": "1+2"}'
                )
            else:
                try:
                    calls.append(
                        ToolCall(
                            name=match.group("name").strip(),
                            arguments=obj,
                            raw=match.group(0),
                        )
                    )
                except ValueError as exc:
                    errors.append(str(exc))
            leftover = (text[: match.start()] + text[match.end() :]).strip()

    return ParseResult(text=leftover.strip(), calls=calls, errors=errors)


def format_tool_call(name: str, arguments: dict, *, style: str = "qwen") -> str:
    """Render a call back into text -- used by backends that return structured calls."""
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    if style == "qwen":
        return f"<tool_call>\n{payload}\n</tool_call>"
    return payload
