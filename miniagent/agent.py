"""agent 循环本体。

"agent" 通常被当成框架来卖，但控制流其实就是这些::

    messages = [system, user]
    loop:
        reply  = model(messages, tools)      # 文本，里面可能藏着工具调用
        calls  = parse(reply)                # 文本 -> 结构化调用
        if no calls:  return reply           # 模型认为自己答完了
        append assistant(reply, calls)       # 记住模型说了什么
        for call in calls:
            result = tools.execute(call)     # 永不抛异常
            append tool(result)              # 观察结果
        if something looks wrong: stop       # 护栏，见下

文件里剩下的全是护栏，而护栏正是「demo」和「敢放到用户面前」之间的区别：

* **max_steps** —— 模型可能无限调工具；给它上界。
* **重复检测** —— 同一个工具配同一份参数来两次，说明观察没帮上忙，第三次也不会。
* **解析错误预算** —— 坏输出是可恢复的（回灌给它），但只给几次机会。
* **工具错误是观察，不是崩溃** —— 栈追踪教不会模型任何东西；
  ``ERROR: missing required argument(s) ['a']`` 能教会很多。
* **一轮只允许一个改动调用** —— 一次回复里两个写操作就是两次*盲写*：
  模型在看到任何一个结果之前，就把两个都定下来了。多余的写会以 ``SKIPPED``
  观察返回，让下一轮基于真实状态再动手。
* **失败的写不许被嘴硬掉** —— 文件改动没发生，而模型用散文宣称改了，这个答案
  会被驳回，并告诉它文件没变。否则 0.5B 会愉快地汇报一个从未发生的成功。
* **写完必须读回** —— 本次运行改过的文件，在接受最终答案前必须被读过：答案要
  建立在磁盘的观察之上，而不是模型的意图之上。这是对付「自信地描述一个其实不是
  它以为的那样的文件」的唯一办法。
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable

from miniagent.backends import LLMBackend
from miniagent.parser import parse
from miniagent.tools import Tool, ToolRegistry
from miniagent.types import (
    AgentResult,
    Message,
    Role,
    Step,
    StopReason,
    ToolResult,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant. Use the provided tools when they help you get a "
    "factual answer. After you have what you need, answer the user directly and "
    "concisely. Never invent a tool result."
)

_REPAIR_HINT = (
    "Your previous reply did not contain a usable tool call. Reply with exactly one "
    'block of the form <tool_call>{{"name": "<tool>", "arguments": {{...}}}}</tool_call>, '
    "or answer the user directly without calling a tool."
)

_WRITE_INCOMPLETE_HINT = (
    "Your previous turn tried to change a file but the change did not happen ({detail}). "
    "The file is unchanged. Do not claim a change you did not make: run the tool again "
    "now, or answer saying what is still missing."
)

_READBACK_HINT = (
    "You changed {paths} but never read {them} back. The file on disk is the only truth now: "
    "call read_file and answer from what it actually shows, not from what you meant to write."
)


class Agent:
    def __init__(
        self,
        backend: LLMBackend,
        tools: Iterable[Tool] | ToolRegistry | None = None,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 8,
        max_tokens: int = 512,
        max_parse_errors: int = 2,
        max_identical_calls: int = 2,
        max_mutating_calls: int = 1,
        max_write_retries: int = 2,
        max_readback_retries: int = 2,
        on_step: Callable[[Step], None] | None = None,
    ) -> None:
        self.backend = backend
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_parse_errors = max_parse_errors
        self.max_identical_calls = max_identical_calls
        self.max_mutating_calls = max_mutating_calls
        self.max_write_retries = max_write_retries
        self.max_readback_retries = max_readback_retries
        self.on_step = on_step
        self.trace: list[Step] = []

    # ------------------------------------------------------------------ 辅助方法
    @property
    def tool_schemas(self) -> list[dict]:
        return self.tools.schemas()

    def reset(self) -> None:
        self.trace.clear()

    def _emit(self, step: Step) -> None:
        self.trace.append(step)
        if self.on_step is not None:
            self.on_step(step)

    def _mutates(self, name: str) -> bool:
        tool = self.tools.get(name)
        return tool is not None and tool.mutates

    @staticmethod
    def _write_incomplete_hint(failed: list[ToolResult]) -> str:
        detail = "; ".join(f"{r.name}: {r.error or r.content[:100]}" for r in failed)
        return _WRITE_INCOMPLETE_HINT.format(detail=detail)

    @staticmethod
    def _readback_hint(paths: list[str]) -> str:
        them = "it" if len(paths) == 1 else "them"
        return _READBACK_HINT.format(paths=", ".join(paths), them=them)

    def _finish(self, **kwargs) -> AgentResult:
        return AgentResult(**kwargs)

    # --------------------------------------------------------------------- 主循环
    def run(self, question: str) -> AgentResult:
        started = time.perf_counter()
        messages: list[Message] = [
            Message(Role.SYSTEM, self.system_prompt),
            Message(Role.USER, question),
        ]
        steps: list[Step] = []
        seen: Counter[str] = Counter()
        parse_errors = 0
        failed_writes: list[ToolResult] = []
        write_retries = 0
        dirty: dict[str, None] = {}
        """Paths changed by this run that have not been read back since."""
        readback_retries = 0

        def result(answer: str, reason: StopReason) -> AgentResult:
            out = self._finish(
                question=question,
                answer=answer,
                steps=steps,
                stopped_reason=reason,
                messages=messages,
                duration_s=time.perf_counter() - started,
            )
            return out

        for index in range(self.max_steps):
            step_started = time.perf_counter()

            try:
                reply = self.backend.complete(
                    messages, tools=self.tool_schemas or None, max_tokens=self.max_tokens
                )
                print(f"DEBUG: backend reply: {reply}")
            except Exception as exc:
                step = Step(index=index, assistant_text="", parse_errors=[f"backend: {exc}"])
                step.duration_s = time.perf_counter() - step_started
                steps.append(step)
                self._emit(step)
                return result(f"backend error: {exc}", StopReason.BACKEND_ERROR)

            parsed = parse(reply)
            step = Step(
                index=index,
                assistant_text=parsed.text,
                tool_calls=parsed.calls,
                parse_errors=parsed.errors,
            )

            # --- 模型直接给了散文：完成 —— 除非它尝试的文件改动没有发生，
            # 或者它改过的文件从未被读回。
            # 这两种情况下，这段散文描述的状态是循环无法担保的。
            if not parsed.calls and not parsed.errors:
                step.duration_s = time.perf_counter() - step_started

                if failed_writes and write_retries < self.max_write_retries:
                    write_retries += 1
                    step.notes.append("answer rejected: an attempted file change did not happen")
                    steps.append(step)
                    self._emit(step)
                    messages.append(Message(Role.ASSISTANT, reply))
                    messages.append(Message(Role.USER, self._write_incomplete_hint(failed_writes)))
                    continue

                if dirty and readback_retries < self.max_readback_retries:
                    readback_retries += 1
                    step.notes.append("answer rejected: a changed file was never read back")
                    steps.append(step)
                    self._emit(step)
                    messages.append(Message(Role.ASSISTANT, reply))
                    messages.append(Message(Role.USER, self._readback_hint(list(dirty))))
                    continue

                steps.append(step)
                self._emit(step)
                if failed_writes:
                    reason = StopReason.WRITE_INCOMPLETE
                elif dirty:
                    reason = StopReason.UNVERIFIED
                else:
                    reason = StopReason.FINAL_ANSWER
                return result(parsed.text.strip() or reply.strip(), reason)

            # --- 无法使用的工具调用：把错误交还回去，让它重试
            if not parsed.calls:
                parse_errors += len(parsed.errors)
                step.duration_s = time.perf_counter() - step_started
                messages.append(Message(Role.ASSISTANT, reply))
                messages.append(
                    Message(Role.USER, "Problem with your tool call:\n" + "\n".join(parsed.errors))
                )
                messages.append(Message(Role.USER, _REPAIR_HINT))
                steps.append(step)
                self._emit(step)
                if parse_errors > self.max_parse_errors:
                    return result(
                        "the model kept producing unparseable tool calls", StopReason.TOO_MANY_ERRORS
                    )
                continue

            # --- 执行每一个调用，回灌每一条结果
            messages.append(Message(Role.ASSISTANT, parsed.text, tool_calls=parsed.calls))
            mutations = 0
            for call in parsed.calls:
                tool = self.tools.get(call.name)
                if tool is not None and tool.mutates:
                    mutations += 1
                    if mutations > self.max_mutating_calls:
                        # 模型在看到上一条结果之前就规划了这次写入，
                        # 所以现在执行它是对陈旧状态的盲写。
                        skipped = ToolResult(
                            call_id=call.call_id,
                            name=call.name,
                            ok=False,
                            content=(
                                f"SKIPPED: {call.name} changes files and only "
                                f"{self.max_mutating_calls} such call(s) run per turn. "
                                "Do it in the next turn, after reading the result above."
                            ),
                            error="skipped: too many modifying calls in one turn",
                        )
                        step.tool_results.append(skipped)
                        messages.append(Message(Role.TOOL, skipped.to_observation(), name=skipped.name))
                        continue

                tool_result = self.tools.execute(call)
                step.tool_results.append(tool_result)
                messages.append(Message(Role.TOOL, tool_result.to_observation(), name=tool_result.name))
                # 重新观察不算循环：再读一次同一个文件，正是「改后必读」所要求的。
                # 只有重复的「动作」才计数。
                if tool is None or tool.reads is None:
                    seen[call.key()] += 1

            # 下一轮不允许被糊弄过去的事实
            failed_writes = [r for r in step.tool_results if not r.ok and self._mutates(r.name)]
            for tool_result in step.tool_results:
                for path in tool_result.reads:
                    # 读失败也算看过：「文件已经没了」是一个事实
                    dirty.pop(path, None)
                if tool_result.ok:
                    for path in tool_result.affects:
                        dirty[path] = None

            step.duration_s = time.perf_counter() - step_started
            steps.append(step)
            self._emit(step)

            repeated = [k for k, v in seen.items() if v > self.max_identical_calls]
            if repeated:
                detail = repeated[0].split(":", 1)[0]
                return result(
                    f"stopped: the model kept calling {detail} with identical arguments",
                    StopReason.LOOP_DETECTED,
                )

        return result(
            f"stopped after {self.max_steps} steps without a final answer",
            StopReason.MAX_STEPS,
        )
