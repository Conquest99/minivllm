"""The agent loop.

An "agent" is usually sold as a framework, but the control flow is this::

    messages = [system, user]
    loop:
        reply  = model(messages, tools)      # text, possibly containing tool calls
        calls  = parse(reply)                # text -> structured calls
        if no calls:  return reply           # the model decided it is done
        append assistant(reply, calls)       # keep what the model said
        for call in calls:
            result = tools.execute(call)     # never raises
            append tool(result)              # the observation
        if something looks wrong: stop       # guards, see below

Everything else in the file is a guard, and those guards are the difference
between a demo and something you would let near a user:

* **max_steps** -- a model can loop forever calling tools; bound it.
* **repeat detection** -- the same tool with the same arguments twice means the
  observation did not help, so a third attempt will not either.
* **parse-error budget** -- malformed output is recoverable (feed it back), but
  only a couple of times.
* **tool errors are observations, not crashes** -- a stack trace teaches the model
  nothing; ``ERROR: missing required argument(s) ['a']`` teaches it a lot.
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
        on_step: Callable[[Step], None] | None = None,
    ) -> None:
        self.backend = backend
        self.tools = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.max_parse_errors = max_parse_errors
        self.max_identical_calls = max_identical_calls
        self.on_step = on_step
        self.trace: list[Step] = []

    # ------------------------------------------------------------------ helpers
    @property
    def tool_schemas(self) -> list[dict]:
        return self.tools.schemas()

    def reset(self) -> None:
        self.trace.clear()

    def _emit(self, step: Step) -> None:
        self.trace.append(step)
        if self.on_step is not None:
            self.on_step(step)

    def _finish(self, **kwargs) -> AgentResult:
        return AgentResult(**kwargs)

    # --------------------------------------------------------------------- run
    def run(self, question: str) -> AgentResult:
        started = time.perf_counter()
        messages: list[Message] = [
            Message(Role.SYSTEM, self.system_prompt),
            Message(Role.USER, question),
        ]
        steps: list[Step] = []
        seen: Counter[str] = Counter()
        parse_errors = 0

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

            # --- the model answered in prose: we are done
            if not parsed.calls and not parsed.errors:
                step.duration_s = time.perf_counter() - step_started
                steps.append(step)
                self._emit(step)
                return result(parsed.text.strip() or reply.strip(), StopReason.FINAL_ANSWER)

            # --- unusable tool call: hand the error back and let it retry
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

            # --- execute every call, feed every result back
            messages.append(Message(Role.ASSISTANT, parsed.text, tool_calls=parsed.calls))
            for call in parsed.calls:
                tool_result = self.tools.execute(call)
                step.tool_results.append(tool_result)
                messages.append(Message(Role.TOOL, tool_result.to_observation(), name=tool_result.name))
                seen[call.key()] += 1

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
