"""Watch the agent create, read and edit a file on disk.

    python scripts/agent_files_demo.py                      # scripted: no GPU, no model
    python scripts/agent_files_demo.py --backend local      # the local 0.5B checkpoint
    python scripts/agent_files_demo.py --backend scripted-error

The scripted run is a canned model: it proves the tools work and shows the loop's
shape.  The local run is the real experiment -- a 0.5B model emitting JSON that
contains quoted text is exactly where small models fail, and the failure shows up
in the trace (a parse error, or an argument-name error the model has to fix).

The tail of every run compares the file on disk against ``EXPECTED_LINES``.  That
comparison is the whole reason this demo exists: a 0.5B model that edits the wrong
line still produces a fluent, confident final answer, so "it said it worked" is
worth nothing.  The verdict is deliberately lenient -- it does not change the exit
code -- but it makes the difference impossible to miss.

Every run gets its own ``agent_workspace/run-<timestamp>`` directory unless you
pass ``--workspace DIR``.  An agent has no memory between runs, so a file left
behind by the previous experiment is a trap: the model will happily append to it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = ROOT / "agent_workspace"
sys.path.insert(0, str(ROOT))

from miniagent import DEFAULT_SYSTEM_PROMPT, Agent, StopReason  # noqa: E402
from miniagent.backends import CallableBackend, ScriptedBackend  # noqa: E402
from miniagent.builtin_tools import Sandbox, make_file_tools, make_write_tools  # noqa: E402
from miniagent.types import Role, Step  # noqa: E402

QUESTION = (
    "Create todo.txt with three lines: 'buy milk', 'walk the dog', 'write report'. "
    "Then read it back, change the second line to 'walk the dog at 7am', "
    "and tell me what the file says."
)

# Ground truth for the built-in question. Without this the demo can only print the
# file and leave the verdict to the reader -- and a model that confidently describes
# the wrong content looks identical to one that got it right.
EXPECTED_LINES = ["buy milk", "walk the dog at 7am", "write report"]


def tool_call(tool: str, **arguments) -> str:
    return "<tool_call>" + json.dumps({"name": tool, "arguments": arguments}) + "</tool_call>"


def scripted_backend() -> ScriptedBackend:
    """A fake model that writes, reads, edits, and reads back again.

    The second read is not decoration: the loop refuses a final answer about a file
    this run changed until the file has been read *after* the change.
    """
    return ScriptedBackend(
        [
            tool_call(
                "write_lines",
                path="todo.txt",
                lines=["buy milk", "walk the dog", "write report"],
            ),
            tool_call("read_file", path="todo.txt"),
            tool_call("edit_file", path="todo.txt", old="walk the dog", new="walk the dog at 7am"),
            tool_call("read_file", path="todo.txt"),
            "todo.txt now reads: buy milk / walk the dog at 7am / write report.",
        ]
    )


def scripted_backend_with_error() -> CallableBackend:
    """A fake model that uses write_file's parameter name on write_lines, then fixes it.

    This is the write-tools version of "工具错误是观察，不是异常": the mistake is
    caught by ``Tool.validate`` and comes back as text the model can act on.
    """

    def decide(messages):
        observations = [m.content for m in messages if m.role is Role.TOOL]
        if not observations:
            return tool_call("write_lines", path="todo.txt", content="buy milk")  # wrong key
        if "unexpected argument" in observations[-1]:
            return tool_call("write_lines", path="todo.txt", lines=["buy milk"])
        if not any(text.startswith("[read_file]") for text in observations):
            return tool_call("read_file", path="todo.txt")
        return "I wrote todo.txt with one line."

    return CallableBackend(decide)


def print_step(step: Step) -> None:
    print(f"\n--- step {step.index} ({step.duration_s * 1000:.0f} ms) ---")
    if step.assistant_text:
        print(f"model: {step.assistant_text.strip()}")
    for call, result in zip(step.tool_calls, step.tool_results, strict=False):
        arrow = "->" if result.ok else "!!"
        print(f"  call   {call.name}({json.dumps(call.arguments, ensure_ascii=False)})")
        body = result.content.replace("\n", "\n           ").rstrip()
        print(f"  {arrow} result {body}")
    for err in step.parse_errors:
        print(f"  parse error: {err}")
    for note in step.notes:
        print(f"  note: {note}")


def report_verification(target: Path, expected: list[str]) -> bool:
    """Compare the file on disk with what the question actually asked for.

    Deliberately *lenient*: the verdict is printed loudly but does not change the
    exit code, which still follows ``stopped_reason``.  The point is to make a
    wrong answer visible, not to turn this demo into a test harness.
    """
    actual = target.read_text(encoding="utf-8").splitlines() if target.is_file() else []

    print("\nverification (expected vs disk):")
    if not target.is_file():
        print(f"  !! {target.name} does not exist")
        return False

    matched = True
    for i in range(max(len(expected), len(actual))):
        want = expected[i] if i < len(expected) else "<missing>"
        got = actual[i] if i < len(actual) else "<missing>"
        same = want == got
        matched = matched and same
        print(f"  {'  ' if same else '!!'} line {i + 1}: want {want!r:34} got {got!r}")

    print(f"  -> {'MATCH' if matched else 'MISMATCH'}")
    if not matched:
        print("  note: lenient mode -- exit code still follows stopped_reason")
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="scripted",
        choices=["scripted", "scripted-error", "local"],
        help="scripted = deterministic fake model (no torch)",
    )
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument(
        "--workspace",
        default=None,
        help="reuse a fixed directory; the default is a fresh agent_workspace/run-<timestamp>",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--max-mutations",
        type=int,
        default=1,
        help="modifying calls allowed per turn (1 is the safe default; 4 reproduces blind writes)",
    )
    parser.add_argument("--verbose", action="store_true", help="dump the final message list")
    parser.add_argument(
        "--expect-lines",
        nargs="+",
        default=EXPECTED_LINES,
        help="ground truth for the default question; override when you change --question",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the expected-vs-disk comparison (useful for a custom --question)",
    )
    args = parser.parse_args()

    if args.backend == "scripted":
        backend = scripted_backend()
    elif args.backend == "scripted-error":
        backend = scripted_backend_with_error()
    else:
        from miniagent.backends import MiniVLLMBackend

        backend = MiniVLLMBackend(device=args.device, dtype=args.dtype)
        print(backend.describe())

    if args.workspace:
        workspace = Path(args.workspace)
    else:
        # A fresh directory per run: the agent has no memory between runs, so a
        # leftover file from the last experiment is a trap, not a feature.
        workspace = DEFAULT_WORKSPACE / f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    box = Sandbox(workspace)
    agent = Agent(
        backend,
        [*make_file_tools(box), *make_write_tools(box)],
        system_prompt=(
            DEFAULT_SYSTEM_PROMPT + " When you change a file, read it first and make one change per turn."
        ),
        max_steps=args.max_steps,
        max_mutating_calls=args.max_mutations,
        on_step=print_step,
    )

    print(f"workspace: {box.root}")
    print(f"backend  : {args.backend}   writes/turn: {args.max_mutations}")
    print(f"tools    : {agent.tools.names}")
    print(f"question : {args.question}")

    result = agent.run(args.question)

    print("\n" + "=" * 70)
    print(f"answer     : {result.answer}")
    print(f"stopped    : {result.stopped_reason.value}")
    print(f"steps      : {len(result.steps)}")
    print(f"tool calls : {result.num_tool_calls} ({result.num_tool_errors} failed)")

    target = box.root / "todo.txt"
    if target.is_file():
        print("\ntodo.txt, straight off the disk:")
        print("-" * 70)
        print(target.read_text(encoding="utf-8").rstrip("\n"))
        print("-" * 70)

    if not args.no_verify:
        report_verification(target, args.expect_lines)

    if args.verbose:
        print("\nfinal conversation:")
        for message in result.messages:
            print(f"  {message!r}")

    if result.stopped_reason is not StopReason.FINAL_ANSWER:
        sys.exit(1)


if __name__ == "__main__":
    main()
