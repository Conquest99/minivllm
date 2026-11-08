"""看 agent 在磁盘上创建、读取、修改一个文件。

    python scripts/agent_files_demo.py                      # scripted：不用 GPU、不用模型
    python scripts/agent_files_demo.py --backend local      # 本地 0.5B 权重
    python scripts/agent_files_demo.py --backend scripted-error

scripted 那版是一个写死的模型：它证明工具能用、并展示循环的形状。local 那版才是真
实验 —— 让 0.5B 输出含引号文本的 JSON，正是小模型翻车的地方，而失败会在 trace 里
显形（一个解析错误，或一个它必须自己修掉的参数名错误）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# python scripts\agent_files_demo.py --backend local --max-mutations 4
# bug 1 同一轮里进行了4个操作，导致最后把文件覆盖了，值得注意的是，
# 模型在生成工具调用时没有看到前面操作的结果，所以它的计划是盲写的

ROOT = Path(__file__).resolve().parents[1]
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


def tool_call(name: str, **arguments) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"


def scripted_backend() -> ScriptedBackend:
    """一个假模型：写、读、再改同一个文件。"""
    return ScriptedBackend(
        [
            tool_call(
                "write_lines",
                path="todo.txt",
                lines=["buy milk", "walk the dog", "write report"],
            ),
            tool_call("read_file", path="todo.txt"),
            tool_call("edit_file", path="todo.txt", old="walk the dog", new="walk the dog at 7am"),
            "todo.txt now reads: buy milk / walk the dog at 7am / write report.",
        ]
    )


def scripted_backend_with_error() -> CallableBackend:
    """一个假模型：在 write_lines 上用了 write_file 的参数名，然后改对。

    这是「工具错误是观察，不是异常」的写入版：错误被 ``Tool.validate`` 截住，回来的是
    模型能据此行动的文字。
    """

    def decide(messages):
        observations = [m.content for m in messages if m.role is Role.TOOL]
        if not observations:
            return tool_call("write_lines", path="todo.txt", content="buy milk")  # 键名写错
        if "unexpected argument" in observations[-1]:
            return tool_call("write_lines", path="todo.txt", lines=["buy milk"])
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="scripted",
        choices=["scripted", "scripted-error", "local", "echo"],
        help="scripted = deterministic fake model (no torch); echo = reply with the question",
    )
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--workspace", default=str(ROOT / "agent_workspace"))
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
    args = parser.parse_args()

    if args.backend == "scripted":
        backend = scripted_backend()
    elif args.backend == "scripted-error":
        backend = scripted_backend_with_error()
    elif args.backend == "echo":
        from miniagent.backends import EchoBackend

        backend = EchoBackend()
    else:
        from miniagent.backends import MiniVLLMBackend

        backend = MiniVLLMBackend(device=args.device, dtype=args.dtype)
        print(backend.describe())

    box = Sandbox(args.workspace)
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

    if args.verbose:
        print("\nfinal conversation:")
        for message in result.messages:
            print(f"  {message!r}")

    if result.stopped_reason is not StopReason.FINAL_ANSWER:
        sys.exit(1)


if __name__ == "__main__":
    main()
