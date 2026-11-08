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
_OPEN_TAG = re.compile(r"<tool_call>", re.IGNORECASE)
_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)\s*```", re.DOTALL)
_REACT = re.compile(
    r"Action\s*:\s*(?P<name>[A-Za-z_][\w.\-]*)\s*(?:\r?\n)+\s*Action\s*Input\s*:\s*"
    r"(?P<args>\{[^\n]*\}|[^\n]+)",
    # 此处做了尝试修改
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
    """解析一个 JSON（或近似 JSON）对象。不是对象就返回 None。"""
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
            except Exception:
                # ``literal_eval`` 的官方契约是 ValueError / TypeError / SyntaxError /
                # MemoryError / RecursionError，而 **TypeError 不在**上面的元组里 ——
                # 它真的会抛：``{"a": {{1, 2}}}`` 会让它去建一个「装 set 的 set」，然后
                # ``TypeError: unhashable type: 'set'`` 一路穿出 ``parse``。
                #
                # ``parse`` 的整个设计前提是「坏输出永不抛异常」，所以这里必须兜住一切：
                # 模型写出来的畸形文本，没有任何一种值得让整轮运行崩掉。
                continue
        return value if isinstance(value, dict) else None
    return None


def _first(containers: tuple[dict, ...], keys: tuple[str, ...]) -> object | None:
    """按顺序在 ``containers`` 里查找，返回 ``keys`` 中任一键命中的第一个值。"""
    for container in containers:
        for key in keys:
            if key in container:
                return container[key]
    return None


def _normalise(obj: dict, raw: str) -> ToolCall:
    """接受模型用过的几种键名拼法，构建出 ToolCall。"""
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
    """处理「整个回复就是一个 JSON 对象」的情况。"""
    obj = _load_object(text)
    if obj is None:
        return None
    if not any(k in obj for k in _NAME_KEYS):
        return None
    return [_normalise(obj, text)]


def _looks_like_a_bare_object(text: str) -> bool:
    """整段回复被一对花括号包着 —— 强信号：模型想发一个 JSON 工具调用。

    只认「整段」，不认「句子里夹了一个 {...}」：后者是讲解性文本，误判会白白多跑一轮
    修复。宁可漏判一种，也不要冤枉一段正常的回答。
    """
    stripped = text.strip()
    return stripped.startswith("{") and stripped.endswith("}") and len(stripped) > 2


_JSON_DECODER = json.JSONDecoder()


def _find_embedded_call(text: str) -> dict | None:
    """在文本**任意位置**找第一个「带 name 键的 JSON 对象」。

    用 ``raw_decode`` 而不是正则：花括号可以嵌套，``\\{.*?\\}`` 会在第一个 ``}`` 处截断，
    ``{"name": "a", "arguments": {"x": 1}}`` 直接废掉。

    找到就返回，**不执行**。调用方拿它去报错 —— 见 ``parse`` 第 5 步。
    """
    for match in re.finditer(r"\{", text):
        try:
            obj, _end = _JSON_DECODER.raw_decode(text, match.start())
        except ValueError:
            continue  # JSONDecodeError 是 ValueError 的子类
        if isinstance(obj, dict) and any(key in obj for key in _NAME_KEYS):
            return obj
    return None


def parse(text: str) -> ParseResult:
    """从 ``text`` 里取出每一个工具调用，并把调用块从散文里剥掉。"""
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
        try:
            lone = _resolve_lone_json(text.strip())
        except ValueError as exc:
            # ``_normalise`` 会为「有 name 键但不是字符串」抛 ValueError（比如
            # ``{"name": null}``）。这里必须接住 —— 否则它就是一个能让 ``parse``
            # 抛异常、把整轮运行带崩的输入。
            errors.append(str(exc))
            lone = None
            leftover = ""
        if lone:
            calls = lone
            leftover = ""
        elif not errors and _looks_like_a_bare_object(text):
            # 整段就是一个 {...} 但解析不了：这就是一次写坏的工具调用，不是答案。
            # 不报错的话它会变成 final_answer，而模型以为自己调了工具。
            errors.append(
                f"the whole reply looks like a JSON object but is not parseable: {text.strip()[:120]!r}. "
                'Send exactly one <tool_call>{"name": "...", "arguments": {...}}</tool_call> block.'
            )
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

    # 5) 未闭合的 <tool_call> —— 几乎总是 max_tokens 把它砍在半路
    #
    # ``_TOOL_TAG`` 要求闭合，所以截断的输出**一个都匹配不上**，于是 calls 和 errors
    # 都是空的。而 ``Agent.run`` 的「模型答完了」判据正是「两个都空」—— 半截工具调用
    # 会被当成最终答案原样返回。实测：``<tool_call>{"name": "calculator", "arguments":
    # {"expressio`` 会得到 ``stopped=final_answer``、0 个错误。
    #
    # 所以这里必须说话。说不了「该怎么改」，至少要说「被截断了，重发一次」。
    stray = _OPEN_TAG.search(leftover)
    if stray is not None:
        tail = leftover[stray.end() :].strip()
        errors.append(
            "a <tool_call> was opened but never closed"
            + (f" (content: {tail[:120]!r})" if tail else "")
            + "; the reply was probably cut off by max_tokens. Send exactly one complete "
            '<tool_call>{"name": "...", "arguments": {...}}</tool_call> block.'
        )
        leftover = leftover[: stray.start()]

    # 6) 夹在句子中间的「工具调用形状的 JSON」—— 认出意图，但**拒绝执行**
    #
    # ``我现在调用工具 {"name": "calculator", "arguments": {"expression": "1+1"}} 请你执行``
    # 原本 calls 和 errors 都空，于是整句变成了最终答案。用户看到的是半截 JSON。
    #
    # 为什么不直接执行它：误判的代价不对称。实测「删除文件该怎么写？」这种问题里，一段
    # 讲解用的 JSON 会被真的执行 —— 文件被删掉。漏判只是难看，误判是事故。
    #
    # 所以这里是第三条路：告诉模型「你写了调用但没包标签，所以我没有执行」。
    if not calls and not errors:
        embedded = _find_embedded_call(leftover)
        if embedded is not None:
            preview = json.dumps(embedded, ensure_ascii=False)[:120]
            errors.append(
                f"this reply contains {preview} in the middle of the text, but a tool call "
                "was not executed because it was not wrapped in tags. Send exactly "
                '<tool_call>{"name": "...", "arguments": {...}}</tool_call>, or, if you were '
                "quoting an example, describe it in words instead."
            )

    return ParseResult(text=leftover.strip(), calls=calls, errors=errors)


def format_tool_call(name: str, arguments: dict, *, style: str = "qwen") -> str:
    """把一个调用渲染回文本 —— 给那些返回结构化调用的后端用。"""
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    if style == "qwen":
        return f"<tool_call>\n{payload}\n</tool_call>"
    return payload
