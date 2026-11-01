"""Turn raw model output into tool calls.

A function-calling model does not return a struct -- it returns *text* that
happens to contain a struct.  This module is the adapter, and it is where most
real-world agents lose robustness, because every model invents its own dialect:

===============  ==========================================================
dialect          example
===============  ==========================================================
Qwen / Hermes    ``<tool_call>{"name": "add", "arguments": {"a": 1}}</tool_call>``
JSON fence       ```json``{"name": "add", "arguments": {"a": 1}}`````
bare JSON        the whole reply is one JSON object
ReAct            ``Action: add`` / ``Action Input: {"a": 1}``
===============  ==========================================================

Two decisions worth defending:

* **Never raise on bad output.**  Malformed JSON is a normal event, not an
  exception: the loop feeds the error back so the model can retry.  Raising would
  turn a recoverable hiccup into a dead run.
* **Return the leftover text too.**  A model may say "Let me check that." and then
  emit a tool call.  That prose belongs in the assistant message; dropping it
  loses context on the next turn.
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


def _normalise(obj: dict, raw: str) -> ToolCall:
    """Accept the several key spellings models use and build a ToolCall."""
    name: object = None
    for key in _NAME_KEYS:
        if key in obj:
            name = obj[key]
            break
    if isinstance(name, dict):  # {"function": {"name": "..."}}
        name = name.get("name")

    args: object = {}
    for key in _ARG_KEYS:
        if key in obj:
            args = obj[key]
            break

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

    # 1) explicit <tool_call> tags -- the Qwen2.5 / Hermes convention
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

    # 2) fenced ```json blocks
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

    # 3) the whole reply is a JSON object
    if not calls and not errors:
        lone = _resolve_lone_json(text.strip())
        if lone:
            calls = lone
            leftover = ""

    # 4) ReAct style
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
