"""Run the agent loop and watch every step.

Default is ``--backend scripted``: a canned model that asks for two calculator
calls, so the loop's shape is visible instantly and deterministically.

    python scripts/agent_demo.py                       # no GPU, no model
    python scripts/agent_demo.py --backend local       # the local 0.5B checkpoint
    python scripts/agent_demo.py --backend api --model deepseek-chat

The scripted run is the one to read first: it shows what the loop does without
any model quality getting in the way.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from miniagent import Agent, StopReason, parse  # noqa: E402
from miniagent.backends import CallableBackend, ScriptedBackend  # noqa: E402
from miniagent.builtin_tools import default_tools  # noqa: E402
from miniagent.types import Role, Step  # noqa: E402

QUESTION = "What is (23 * 7 + 11) / 4? Use the calculator."


def tool_call(name: str, **arguments) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"


def scripted_backend() -> ScriptedBackend:
    """A fake model that needs two calculator calls, then answers."""
    return ScriptedBackend(
        [
            tool_call("calculator", expression="23 * 7"),
            tool_call("calculator", expression="(161 + 11) / 4"),
            "The answer is 43.",
        ]
    )


def scripted_backend_with_error() -> CallableBackend:
    """A fake model that gets an argument name wrong, then fixes it.

    This is the interesting one: it shows a tool error becoming an observation
    rather than an exception.
    """

    def decide(messages):
        observations = [m.content for m in messages if m.role is Role.TOOL]
        if not observations:
            return tool_call("calculator", expr="23 * 7")  # wrong: should be `expression`

        last = observations[-1]
        if "unexpected argument" in last:
            return tool_call("calculator", expression="23 * 7")

        good = [o for o in observations if "ERROR" not in o]
        if len(good) == 1:
            value = good[0].split()[-1]
            return tool_call("calculator", expression=f"({value} + 11) / 4")
        return "The answer is 43."

    return CallableBackend(decide)


def describe_backend(kind: str, args) -> object:
    if kind == "scripted":
        return scripted_backend()
    if kind == "scripted-error":
        return scripted_backend_with_error()
    if kind == "local":
        from miniagent.backends import MiniVLLMBackend

        backend = MiniVLLMBackend(device=args.device, dtype=args.dtype)
        print(backend.describe())
        return backend
    if kind == "api":
        from miniagent.backends import OpenAICompatBackend

        return OpenAICompatBackend(base_url=args.base_url, model=args.model)
    raise SystemExit(f"unknown backend {kind!r}")


def print_step(step: Step) -> None:
    print(f"\n--- step {step.index} ({step.duration_s * 1000:.0f} ms) ---")
    if step.assistant_text:
        print(f"model: {step.assistant_text.strip()}")
    for call, result in zip(step.tool_calls, step.tool_results, strict=False):
        arrow = "->" if result.ok else "!!"
        print(f"  call   {call.name}({json.dumps(call.arguments, ensure_ascii=False)})")
        print(f"  {arrow} result {result.content}")
    for err in step.parse_errors:
        print(f"  parse error: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="scripted",
        choices=["scripted", "scripted-error", "local", "api"],
        help="scripted = deterministic fake model (no torch)",
    )
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--model", default="deepseek-chat", help="API model name")
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--verbose", action="store_true", help="dump the final message list")
    args = parser.parse_args()

    backend = describe_backend(args.backend, args)
    agent = Agent(
        backend,  # type: ignore[arg-type]
        default_tools(Path(__file__).resolve().parents[1] / "agent_workspace"),
        max_steps=args.max_steps,
        on_step=print_step,
    )

    print(f"backend: {args.backend}   tools: {agent.tools.names}")
    print(f"question: {args.question}")

    started = time.perf_counter()
    result = agent.run(args.question)
    wall = time.perf_counter() - started

    print("\n" + "=" * 70)
    print(f"answer        : {result.answer}")
    print(f"stopped       : {result.stopped_reason.value}")
    print(f"steps         : {len(result.steps)}")
    print(f"tool calls    : {result.num_tool_calls} ({result.num_tool_errors} failed)")
    print(f"wall time     : {wall:.2f}s")

    if args.verbose:
        print("\nfinal conversation:")
        for message in result.messages:
            print(f"  {message!r}")

    if result.stopped_reason is not StopReason.FINAL_ANSWER:
        sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = ["parse"]
