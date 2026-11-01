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
* **one modifying call per turn** -- two writes in one reply are two *blind* writes:
  the model committed to both before seeing either result.  Extra writes come back
  as a ``SKIPPED`` observation so the next turn can act on real state.
* **a failed write cannot be talked away** -- if a file change did not happen, a
  prose answer claiming it did is rejected, and the model is told the file is
  unchanged.  Otherwise a 0.5B model will happily report success it never had.
* **write, then read back** -- a file the run changed must be read before a final
  answer is accepted: the answer has to be grounded in an observation of the disk,
  not in the model's intention.  That is the only defence against a confident
  description of a file that is not what the model thinks it is.
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

            # --- the model answered in prose: done -- unless a file change it
            # attempted never happened, or a file it changed was never read back.
            # In both cases the prose describes state the loop cannot vouch for.
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
            mutations = 0
            for call in parsed.calls:
                tool = self.tools.get(call.name)
                if tool is not None and tool.mutates:
                    mutations += 1
                    if mutations > self.max_mutating_calls:
                        # The model planned this write before seeing the previous
                        # result, so running it now is a blind write on stale state.
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
                # Re-observing is not a loop: reading the same file again is exactly
                # what the read-back rule asks for.  Only repeated *actions* count.
                if tool is None or tool.reads is None:
                    seen[call.key()] += 1

            # what the next turn must not be allowed to talk away
            failed_writes = [r for r in step.tool_results if not r.ok and self._mutates(r.name)]
            for tool_result in step.tool_results:
                for path in tool_result.reads:
                    # a failed read discharges the duty too: "it is gone" is a fact
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
