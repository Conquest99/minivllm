"""把工具按 MCP（Model Context Protocol）协议挂出去，走 stdio。

这一层要解决的问题和 ``serve/app.py`` 完全不同：

* ``serve/app.py`` 说的是「模型怎么看见工具」——把 schema 渲染进 prompt，再解析
  ``<tool_call>``。那是 **function calling** 层。
* 这个文件说的是「工具从哪来」——宿主进程怎么和工具服务器通信。那是 **MCP** 层。

两者不在同一层，所以接上 MCP 之后 ``miniagent/agent.py`` 一行都不用改。

MCP 的线格式是 **JSON-RPC 2.0，一行一条消息**（不是 LSP 那种 Content-Length 分帧）。
三个要注意的地方：

1. **stdout 只能有 JSON-RPC。** 一句调试用的 ``print`` 就会毁掉整条流 —— 宿主解析
   失败后通常只报「server sent invalid JSON」，指向的是协议而不是那行 print。
   ``scripts/serve_mcp.py`` 因此把 ``sys.stdout`` 临时指向 stderr。
2. **协议错误和工具错误的区别不是形式问题。** 未知工具名是 JSON-RPC error（host 处理）；
   工具自己失败要放在 ``result`` 里并带 ``isError: true``，因为**模型必须看得见它**。
   JSON-RPC 的 error 字段大多数客户端不会转给模型。
3. **``inputSchema`` 和 OpenAI 的 ``parameters`` 是同一个东西**（JSON Schema 子集），
   所以 ``Tool.to_schema()`` 换个形状就能直接用。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from miniagent.tools import ToolRegistry
from miniagent.types import ToolCall

# 我们认识的协议版本。客户端报的版本在这个集合里就照它回，否则回最新的。
# 核心方法（initialize / tools/list / tools/call）在这几个修订版之间是稳定的。
SUPPORTED_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
LATEST_VERSION = SUPPORTED_VERSIONS[-1]

# JSON-RPC 2.0 标准错误码
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class JsonRpcError(Exception):
    """协议级错误：host 处理，不交给模型。"""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class MCPServer:
    """一个只有 tools + resources 的 MCP 服务器。

    刻意不实现 ``prompts``、``sampling``、``elicitation``：能力列表里声明什么就得真能
    做什么，虚报 ``capabilities`` 会让客户端调用后拿到 -32601。
    """

    tools: ToolRegistry
    name: str = "miniagent-sql"
    version: str = "0.1.0"
    expose_resources: bool = True
    """把每张表的结构作为 ``db://<table>/schema`` 资源公开。

    资源是 MCP 比 function calling 多出来的原语：它是**用户/宿主主动拉取的上下文**，
    不经过模型决定。这里挂 schema 正好——「这个库有什么表」该由人先看一眼，而不是让
    模型去猜着调 ``list_tables``。
    """

    # ------------------------------------------------------------ 协议入口

    def handle(self, message: Any) -> dict[str, Any] | None:
        """处理一条消息，返回响应；通知（无 ``id``）返回 None。"""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, INVALID_REQUEST, "expected a JSON-RPC 2.0 object")

        method = message.get("method")
        if not isinstance(method, str):
            return _error(message.get("id"), INVALID_REQUEST, "missing 'method'")

        is_notification = "id" not in message
        request_id = message.get("id")
        params = message.get("params") or {}
        if not isinstance(params, dict):
            return _error(request_id, INVALID_PARAMS, "'params' must be an object")

        try:
            result = self._dispatch(method, params)
        except JsonRpcError as exc:
            if is_notification:
                return None
            return _error(request_id, exc.code, exc.message)
        except Exception as exc:  # 服务器自身的 bug 不能把整条流带下去
            if is_notification:
                return None
            return _error(request_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            return self._initialize(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            return self._tools_list()
        if method == "tools/call":
            return self._tools_call(params)
        if method == "resources/list":
            if not self.expose_resources:
                raise JsonRpcError(METHOD_NOT_FOUND, "resources are not enabled")
            return self._resources_list()
        if method == "resources/read":
            if not self.expose_resources:
                raise JsonRpcError(METHOD_NOT_FOUND, "resources are not enabled")
            return self._resources_read(params)
        if method.startswith("notifications/"):
            return {}  # initialized / cancelled / 各种 list_changed：收下即可
        raise JsonRpcError(METHOD_NOT_FOUND, f"unknown method {method!r}")

    # ------------------------------------------------------------- 各方法

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        asked = params.get("protocolVersion")
        version = asked if asked in SUPPORTED_VERSIONS else LATEST_VERSION
        capabilities: dict[str, Any] = {"tools": {"listChanged": False}}
        if self.expose_resources:
            capabilities["resources"] = {"listChanged": False, "subscribe": False}
        return {
            "protocolVersion": version,
            "capabilities": capabilities,
            "serverInfo": {"name": self.name, "version": self.version},
        }

    def _tools_list(self) -> dict[str, Any]:
        return {"tools": [to_mcp_tool(tool) for tool in self.tools]}

    def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise JsonRpcError(INVALID_PARAMS, "'name' is required")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise JsonRpcError(INVALID_PARAMS, "'arguments' must be an object")
        if self.tools.get(name) is None:
            raise JsonRpcError(INVALID_PARAMS, f"unknown tool {name!r}")

        result = self.tools.execute(ToolCall(name=name, arguments=arguments))
        # 工具失败走 result + isError，**不是** JSON-RPC error：模型必须看见原始观察
        # 才能自己改正，「缺少参数 a」这种话交给 host 就白写了。
        return {
            "content": [{"type": "text", "text": result.content}],
            "isError": not result.ok,
        }

    def _resources_list(self) -> dict[str, Any]:
        return {"resources": [{"uri": uri, "name": name, "mimeType": "text/plain"} for uri, name in self._tables()]}

    def _resources_read(self, params: dict[str, Any]) -> dict[str, Any]:
        uri = params.get("uri")
        if not isinstance(uri, str):
            raise JsonRpcError(INVALID_PARAMS, "'uri' is required")
        table = _table_from_uri(uri)
        describe = self.tools.get("describe_table")
        if describe is None:
            raise JsonRpcError(METHOD_NOT_FOUND, "this server has no describe_table tool")
        try:
            text = describe.func(table=table)
        except Exception as exc:
            raise JsonRpcError(INVALID_PARAMS, f"{uri}: {type(exc).__name__}: {exc}") from exc
        return {"contents": [{"uri": uri, "mimeType": "text/plain", "text": text}]}

    def _tables(self) -> list[tuple[str, str]]:
        list_tables = self.tools.get("list_tables")
        if list_tables is None:
            return []
        try:
            listing = list_tables.func()
        except Exception:
            return []
        # list_tables 的返回是 "3 table(s): a, b, c"
        _, _, rest = listing.partition(":")
        names = [part.strip() for part in rest.split(",") if part.strip()]
        return [(f"db://{name}/schema", f"{name} schema") for name in names]

    # ------------------------------------------------------------ stdio 循环

    def serve(self, lines: Iterable[str], out: Iterator[str] | Any) -> None:
        """从 ``lines`` 读 JSON-RPC，往 ``out`` 写 JSON-RPC。

        ``out`` 只需要 ``write`` + ``flush``。测试里传一个 StringIO 就能跑，
        所以这套协议可以在纯 CPU 上验证。
        """
        for raw in lines:
            text = raw.strip()
            if not text:
                continue
            response = self.handle_line(text)
            if response is None:
                continue
            out.write(json.dumps(response, ensure_ascii=False) + "\n")
            out.flush()

    def handle_line(self, text: str) -> dict[str, Any] | None:
        try:
            message = json.loads(text)
        except json.JSONDecodeError as exc:
            return _error(None, PARSE_ERROR, f"invalid JSON: {exc.msg}")
        return self.handle(message)


def to_mcp_tool(tool: Any) -> dict[str, Any]:
    """``Tool.to_schema()`` 的 OpenAI 形状 → MCP 的 ``tools/list`` 条目。

    唯一的变化是名字：``{"type":"function","function":{...}}`` 摊平，且 ``parameters``
    改名 ``inputSchema``。**JSON Schema 本身一模一样**——这正是两套生态能互相喂的原因。
    """
    function = tool.to_schema()["function"]
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "inputSchema": function.get("parameters") or {"type": "object", "properties": {}},
    }


def _table_from_uri(uri: str) -> str:
    prefix, _, path = uri.partition("://")
    if prefix != "db" or not path.endswith("/schema"):
        raise JsonRpcError(INVALID_PARAMS, f"cannot read resource uri {uri!r}; expected db://<table>/schema")
    return path[: -len("/schema")]


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


__all__ = [
    "JsonRpcError",
    "MCPServer",
    "SUPPORTED_VERSIONS",
    "to_mcp_tool",
]
