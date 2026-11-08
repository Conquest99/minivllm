"""MCP 层的协议测试：JSON-RPC 2.0 形状、错误分层、``isError`` 的语义。

跑真实 sqlite，不需要网络也不需要模型 —— 因为这一层和模型无关，它只负责
「宿主进程怎么和工具服务器说话」。
"""

from __future__ import annotations

import io
import json
import sqlite3

import pytest

from miniagent.sql_tools import SQLPolicy, SQLSandbox, make_sql_tools, sqlite_connector
from miniagent.tools import ToolRegistry
from serve.mcp_server import (
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    SUPPORTED_VERSIONS,
    MCPServer,
    to_mcp_tool,
)


@pytest.fixture
def registry(tmp_path):
    path = tmp_path / "shop.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, status TEXT);
        INSERT INTO users (name, status) VALUES ('alice', 'active'), ('bob', 'paused');
        """
    )
    conn.commit()
    conn.close()
    with SQLSandbox(sqlite_connector(path), SQLPolicy(max_rows=10)) as box:
        yield ToolRegistry(make_sql_tools(box))


@pytest.fixture
def server(registry):
    return MCPServer(registry)


def rpc(server, method, params=None, request_id=1):
    message = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        message["params"] = params
    if request_id is not None:
        message["id"] = request_id
    return server.handle(message)


# ------------------------------------------------------------- 握手


def test_initialize_reports_the_version_the_client_asked_for(server):
    result = rpc(server, "initialize", {"protocolVersion": "2024-11-05"})["result"]
    assert result["protocolVersion"] == "2024-11-05"
    assert result["serverInfo"]["name"] == "miniagent-sql"


def test_initialize_falls_back_to_the_newest_known_version(server):
    result = rpc(server, "initialize", {"protocolVersion": "1999-01-01"})["result"]
    assert result["protocolVersion"] == SUPPORTED_VERSIONS[-1]


def test_capabilities_only_advertise_what_is_implemented(server):
    """虚报 capabilities 会让客户端调用后拿到 -32601，比不声明更糟。"""
    caps = rpc(server, "initialize", {})["result"]["capabilities"]
    assert set(caps) == {"tools", "resources"}


def test_resources_can_be_turned_off(registry):
    caps = MCPServer(registry, expose_resources=False).handle(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )["result"]["capabilities"]
    assert "resources" not in caps


# ------------------------------------------------------------ tools


def test_tools_list_uses_mcp_shape(server):
    tools = rpc(server, "tools/list")["result"]["tools"]
    names = [t["name"] for t in tools]
    assert names == ["list_tables", "describe_table", "query_table", "count_rows"]
    for tool in tools:
        assert set(tool) == {"name", "description", "inputSchema"}
        assert tool["inputSchema"]["type"] == "object"


def test_the_schema_is_the_same_json_schema_as_the_openai_one(registry):
    """两套生态能互相喂，就是因为 inputSchema 和 parameters 是同一个东西。"""
    tool = registry.get("query_table")
    openai = tool.to_schema()["function"]
    mcp = to_mcp_tool(tool)
    assert mcp["inputSchema"] == openai["parameters"]
    assert mcp["description"] == openai["description"]


def test_tools_call_executes_and_wraps_text_content(server):
    result = rpc(
        server,
        "tools/call",
        {"name": "query_table", "arguments": {"table": "users", "where": ["status = 'active'"]}},
    )["result"]
    assert result["isError"] is False
    assert result["content"][0]["type"] == "text"
    assert "alice" in result["content"][0]["text"]


def test_a_tool_failure_is_isError_not_a_jsonrpc_error(server):
    """工具失败必须放在 ``result`` 里。

    JSON-RPC 的 ``error`` 字段大多数客户端不会转给模型，于是「缺少参数 a」这句
    最有用的提示就丢了 —— 模型永远学不会怎么改。
    """
    response = rpc(server, "tools/call", {"name": "query_table", "arguments": {}})
    assert "result" in response and "error" not in response
    assert response["result"]["isError"] is True
    assert "table" in response["result"]["content"][0]["text"]


def test_a_policy_refusal_also_comes_back_as_an_observation(server):
    result = rpc(
        server,
        "tools/call",
        {"name": "query_table", "arguments": {"table": "users", "where": ["1=1"]}},
    )["result"]
    assert result["isError"] is True
    assert "cannot read the condition" in result["content"][0]["text"]


def test_an_unknown_tool_is_a_protocol_error(server):
    """名字错了是宿主的编程问题，不是模型的观察 —— 这条走 JSON-RPC error。"""
    response = rpc(server, "tools/call", {"name": "drop_table", "arguments": {}})
    assert response["error"]["code"] == INVALID_PARAMS


# --------------------------------------------------------- resources


def test_resources_list_every_table_schema(server):
    uris = [r["uri"] for r in rpc(server, "resources/list")["result"]["resources"]]
    assert "db://users/schema" in uris


def test_resources_read_returns_the_describe_table_output(server):
    text = rpc(server, "resources/read", {"uri": "db://users/schema"})["result"]["contents"][0]["text"]
    assert "name" in text and "TEXT" in text


def test_a_malformed_resource_uri_is_rejected(server):
    response = rpc(server, "resources/read", {"uri": "http://example.com/x"})
    assert response["error"]["code"] == INVALID_PARAMS


# ------------------------------------------------------ 协议健壮性


def test_a_notification_gets_no_response(server):
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_an_unknown_method_is_method_not_found(server):
    assert rpc(server, "tools/frobnicate")["error"]["code"] == METHOD_NOT_FOUND


def test_a_non_jsonrpc_message_is_refused(server):
    assert server.handle({"method": "tools/list", "id": 1})["error"]["code"] == INVALID_REQUEST


def test_invalid_json_on_the_wire_is_a_parse_error(server):
    response = server.handle_line("{not json")
    assert response["error"]["code"] == PARSE_ERROR


def test_blank_lines_are_skipped(server):
    out = io.StringIO()
    server.serve(["", "   ", '{"jsonrpc":"2.0","id":7,"method":"ping"}'], out)
    lines = [line for line in out.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == 7


def test_every_response_carries_the_same_id(server):
    for request_id in (1, "abc", 99):
        assert rpc(server, "ping", request_id=request_id)["id"] == request_id


def test_a_server_bug_does_not_kill_the_stream(registry):
    class Boom(ToolRegistry):
        def execute(self, call):
            raise RuntimeError("boom")

    server = MCPServer(Boom(list(registry)))
    response = rpc(server, "tools/call", {"name": "list_tables", "arguments": {}})
    assert response["error"]["code"] == -32603
    assert "boom" in response["error"]["message"]
