"""把只读 SQL 工具当作 MCP 服务器挂出去（stdio）。

    python scripts\\serve_mcp.py                       # 只读 SQL（离线副本）
    python scripts\\serve_mcp.py --tools default        # 计算器 + 文件沙箱

这个进程**不是**给浏览器用的，是给 MCP 宿主用的（Claude Desktop、Cursor、任何
MCP 客户端）。宿主会自己起这个进程，然后用 JSON-RPC 说话：

    {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}
    {"jsonrpc":"2.0","id":2,"method":"tools/list"}
    {"jsonrpc":"2.0","id":3,"method":"tools/call",
     "params":{"name":"query_table","arguments":{"table":"dba04_cy4","where":["HS < 25"]}}}

手工试一行：

    '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python scripts\\serve_mcp.py

**stdout 只能有 JSON-RPC。** 任何调试输出都会毁掉整条流，所以这里把 ``sys.stdout``
临时指到 stderr，真实协议写进真正的那份 stdout。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORKSPACE = ROOT / "agent_workspace"


def build_registry(kind: str):
    from miniagent.tools import ToolRegistry

    if kind == "sql":
        sys.path.insert(0, str(ROOT / "scripts"))
        from agent_sql_demo import build_sample_db

        from miniagent.sql_tools import (
            SQLPolicy,
            SQLSandbox,
            make_sql_tools,
            sqlite_connector,
        )

        path = build_sample_db(WORKSPACE / "sqldemo.db")
        sandbox = SQLSandbox(
            sqlite_connector(path),
            SQLPolicy(denied_tables=frozenset({"mysql", "information_schema", "performance_schema"})),
        )
        return ToolRegistry(make_sql_tools(sandbox))

    if kind == "default":
        from miniagent.builtin_tools import default_tools

        return ToolRegistry(default_tools())

    raise SystemExit(f"unknown --tools {kind!r}; choose 'sql' or 'default'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--tools", default="sql", choices=["sql", "default"])
    parser.add_argument("--name", default="miniagent-sql")
    parser.add_argument(
        "--no-resources",
        action="store_true",
        help="do not advertise db://<table>/schema resources",
    )
    args = parser.parse_args()

    from serve.mcp_server import MCPServer

    registry = build_registry(args.tools)
    server = MCPServer(registry, name=args.name, expose_resources=not args.no_resources)

    # 真实协议流先抓住，再把 stdout 换成 stderr —— 之后任何 print 都只会进日志。
    protocol = sys.stdout
    sys.stdout = sys.stderr
    try:
        server.serve(sys.stdin, protocol)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout = protocol


if __name__ == "__main__":
    main()
