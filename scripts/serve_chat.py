"""把 agent 跑成一个能一直聊的 HTTP 服务。

    python scripts\\serve_chat.py                          # 本地 0.5B + SQL 工具
    python scripts\\serve_chat.py --tools default           # 计算器 + 文件沙箱
    python scripts\\serve_chat.py --backend scripted        # 假模型，不用 GPU
    python scripts\\serve_chat.py --host 0.0.0.0 --port 8000

打开 http://127.0.0.1:8000 就是一个聊天界面。上下文在服务端，浏览器只拿一个
session_id —— 刷新页面会接着上一轮继续，「新对话」才清空。

这个脚本只负责**装配**：哪个后端、哪些工具、什么提示词。HTTP 那部分在
``serve/app.py``，会话状态在 ``serve/sessions.py``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from miniagent.builtin_tools import default_tools  # noqa: E402
from miniagent.sql_tools import (  # noqa: E402
    SQLPolicy,
    SQLSandbox,
    make_sql_tools,
    sqlite_connector,
)
from serve.app import create_app  # noqa: E402

WORKSPACE = ROOT / "agent_workspace"

SQL_PROMPT = """\
You can inspect a database with read-only tools. List the tables, read the columns,
then call query_table with structured filters instead of writing SQL. Never modify
anything: every tool is read-only by design.

领域知识（中文油田月报库，必须使用这些约定）:
- 油井月报的主体表叫 `dba04`，按"厂"分表存储；"厂"是后缀：四厂 -> `_cy4`。
  所以"四厂油井月报表"是 `dba04_cy4`，不是裸的 `dba04`，也不是 `dba04_cy1`。
- 含水字段是 `HS`（百分数），井号是 `JH`，年月是 `NY`（形如 198206）。
- `HS` 为 NULL 表示当月没有测量数据，范围条件会自动排除这些行。
- 表有 53 列，所以 `query_table` 要写清 `columns`，例如 ["JH", "NY", "HS"]。

调用形状（照抄结构，不要照抄数值）:

    query_table(table="dba04_cy4", columns=["JH", "NY", "HS"],
                where=["HS > 25", "HS < 28"], order_by="NY desc", limit=20)

`where` 是列表，一个元素一个条件，不要写成 ["HS > 25 AND HS < 28"]；
`order_by` 是列名（可加 desc），不要只写 "ASC"。
"""

FILE_PROMPT = """\
You are a careful assistant with read-only access to a workspace and a calculator.
Use the tools when they help you get a factual answer; otherwise answer directly.
Never claim a tool result you did not receive.
"""


def build_tools(kind: str) -> tuple[list, str]:
    if kind == "sql":
        sys.path.insert(0, str(ROOT / "scripts"))
        from agent_sql_demo import build_sample_db  # noqa: PLC0415

        path = build_sample_db(WORKSPACE / "sqldemo.db")
        sandbox = SQLSandbox(
            sqlite_connector(path),
            SQLPolicy(denied_tables=frozenset({"mysql", "information_schema", "performance_schema"})),
        )
        return make_sql_tools(sandbox), SQL_PROMPT
    if kind == "default":
        return default_tools(), FILE_PROMPT
    raise SystemExit(f"unknown --tools {kind!r}; choose 'sql' or 'default'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--backend",
        default="local",
        choices=["local", "scripted", "api", "echo"],
        help="local = the 0.5B checkpoint; scripted/echo = canned models (no GPU)",
    )
    parser.add_argument(
        "--tools",
        default="sql",
        choices=["sql", "default"],
        help="sql = read-only database tools; default = calculator + file sandbox",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-turns", type=int, default=6, help="how many turns of history to keep")
    parser.add_argument("--max-steps", type=int, default=6, help="tool-call rounds per turn")
    parser.add_argument("--model", default="", help="override the checkpoint path")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "uvicorn is not installed. Run:\n"
            r"  uv pip install --python .venv\Scripts\python.exe -e .[serve]"
        ) from None

    if args.backend == "local":
        from miniagent.backends import MiniVLLMBackend

        backend = MiniVLLMBackend(model=args.model or None, device=args.device, dtype=args.dtype)
        print(backend.describe())
    elif args.backend == "scripted":
        from miniagent.backends import CallableBackend
        from miniagent.parser import format_tool_call

        def decide(messages):
            from miniagent.types import Role

            observations = [m.content for m in messages if m.role is Role.TOOL]
            if not observations:
                return format_tool_call(
                    "list_tables" if args.tools == "sql" else "calculator",
                    {} if args.tools == "sql" else {"expression": "6*7"},
                )
            return "（scripted 后端）拿到工具结果了，这一轮到此结束。"

        backend = CallableBackend(decide)
    elif args.backend == "echo":
        # 最适合验证服务层本身：不用 GPU、不产生工具调用、但走完整的 HTTP + SSE + 会话
        # 路径。想问「流式和上下文对不对」的时候，它比真模型可靠得多。
        from miniagent.backends import EchoBackend

        backend = EchoBackend()
    else:
        from miniagent.backends import OpenAICompatBackend

        backend = OpenAICompatBackend(model=args.model or "deepseek-chat")

    tools, prompt = build_tools(args.tools)
    app = create_app(
        backend,
        tools,
        system_prompt=prompt,
        max_turns=args.max_turns,
        max_steps=args.max_steps,
    )

    print(f"tools    : {[t.name for t in tools]}")
    print(f"chat     : http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
