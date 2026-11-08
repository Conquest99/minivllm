# MCP：同一个工具集，换一种协议挂出去

## 一句话定位

`miniagent` 用的是 **OpenAI function calling**，`serve/mcp_server.py` 用的是 **MCP**。
两者**不在同一层**，所以这个适配器存在之后，`miniagent/` 里一行代码都没改。

```
MCP   ↗ 解决：工具从哪来（跨进程、JSON-RPC 2.0）
      │
      │   ← miniagent 完全不碰这一层
      │
FC    ↘ 解决：模型怎么看见工具、怎么说出调用（prompt 渲染 + 文本解析）
```

`parser.py` 是最好的证据：它解析的是**模型吐出来的文本**。而 MCP 里**根本不存在这一层**
—— 服务器只把结果交给宿主，宿主自己决定怎么呈现给模型。所以任何 MCP 服务器里都不会有
「解析 `<tool_call>`」这种代码。

---

## 1. 逐项对照

| | miniagent（function calling） | MCP |
|---|---|---|
| 通信 | **进程内** Python 调用（闭包捕获 `sandbox`） | JSON-RPC 2.0，跨进程（stdio / Streamable HTTP） |
| 工具发现 | 启动时 `make_sql_tools(sandbox)` 构造 | `tools/list` 请求，可动态变化 + `list_changed` 通知 |
| schema 形状 | `{"type":"function","function":{name,description,parameters}}` | `{name, description, inputSchema}` |
| **JSON Schema 本体** | `parameters` | `inputSchema` | 
| 调用 | `ToolRegistry.execute(ToolCall)` | `tools/call`，`{name, arguments}` |
| 结果 | `ToolResult.content: str` | `content: [{type:"text", text}]`，可以是多块 |
| 失败 | `ToolResult.ok = False` | `result.isError = true` |
| 协议级错误 | 无（同进程，直接抛异常） | JSON-RPC `error: {code, message}` |
| 握手 | 无 | `initialize` 协商 `protocolVersion` + `capabilities` |
| 资源 / 提示模板 | 无 | `resources/*`、`prompts/*` 两套额外原语 |
| 人机确认 | 无（靠 `max_mutating_calls` 事后拦） | `sampling/*`、`elicitation`（服务器向用户提问） |

**唯一几乎重合的是「JSON Schema 本体」。** 这不是巧合：两套生态都在描述
「一个函数要哪些参数、各是什么类型」，标准化之后自然是同一个东西。

```python
def to_mcp_tool(tool):
    function = tool.to_schema()["function"]
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "inputSchema": function["parameters"],      # ← 只是改了个名字
    }
```

---

## 2. 三个设计要点

### 2.1 协议错误 vs 工具错误：**这不是形式问题**

```python
def _tools_call(self, params):
    if self.tools.get(name) is None:
        raise JsonRpcError(INVALID_PARAMS, f"unknown tool {name!r}")   # → JSON-RPC error

    result = self.tools.execute(ToolCall(name=name, arguments=arguments))
    return {"content": [{"type": "text", "text": result.content}],
            "isError": not result.ok}                                   # → result.isError
```

分界线是「**这句话是给谁看的**」：

| 情况 | 走哪条 | 为什么 |
|---|---|---|
| 工具名不存在 | JSON-RPC `error` | 宿主的编程问题，模型无从改起 |
| 参数缺了 / 条件写错 / 策略拒绝 | `result.isError = true` | **模型必须看见**才能自己改 |

漏了这条会白费整个 `SQLPolicy` 的设计：`condition 'HS > 25 AND HS < 28' packs more than
one condition` 这句话是写给模型读的，而大多数 MCP 客户端**不会把 JSON-RPC 的 `error`
字段转给模型**。放错位置，那句话就只进了日志。

实测（`scripts/serve_mcp.py`，stdio）：

```
> tools/call  query_table {"table":"dba04_cy4","where":["HS > 25 AND HS < 28"]}
< {"result": {"content": [{"type":"text","text":"ERROR: SQLPolicyError: condition 'HS > 25 AND HS < 28' packs more than one condition ..."}], "isError": true}}
```

### 2.2 `capabilities` 里只能声明真能做的事

```python
capabilities = {"tools": {"listChanged": False}}
if self.expose_resources:
    capabilities["resources"] = {"listChanged": False, "subscribe": False}
```

虚报的后果是客户端调用后拿到 `-32601`，比不声明更糟。所以这里**没有** `prompts`、
`sampling`、`elicitation` —— 一个都不实现，就一个都不声明。

### 2.3 stdout 只能有 JSON-RPC

```python
protocol = sys.stdout
sys.stdout = sys.stderr      # 之后任何 print 都进日志
server.serve(sys.stdin, protocol)
```

一句调试用的 `print` 就会毁掉整条流。宿主通常只报「server sent invalid JSON」，
**指向的是协议而不是那行 print** —— 属于最难查的一类问题。

---

## 3. 为什么这个设计在 MCP 下才「圆满」

现在这套只读隔离靠三层：策略、标识符验证、会话只读。但**这三层都在客户端**——如果
模型跑在你自己的进程里，它理论上能绕过去（拿到 `sandbox` 就有 `_conn`）。

做成 MCP 服务器之后，边界换了位置：

```
现在：  浏览器 → FastAPI → Agent →(闭包)→ SQLSandbox → sqlite
        （所有约束和模型同进程）

MCP：   任何宿主 → JSON-RPC → [这个进程] → SQLSandbox → sqlite
        （约束在服务器里，宿主只是个嘴巴）
```

**同一套工具可以挂给 Claude Desktop、Cursor、任何 MCP 客户端，而它们一个都不知道
`SQLPolicy` 长什么样。** 这就是「能力在服务端，客户端只负责传话」的正确形态。

---

## 4. 跑起来

```powershell
# 自己手敲一行 JSON-RPC
'{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python scripts\serve_mcp.py

# 完整的握手序列
@'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}}}
{"jsonrpc":"2.0","method":"notifications/initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"count_rows","arguments":{"table":"dba04_cy4","where":["HS > 25","HS < 28"]}}}
'@ | python scripts\serve_mcp.py
```

挂到 Claude Desktop（看官方文档确认字段名），大致是：

```json
{
  "mcpServers": {
    "minivllm-sql": {
      "command": "D:\\python-workplace\\Image_identify\\mini_vllm\\.venv\\Scripts\\python.exe",
      "args": ["D:\\python-workplace\\Image_identify\\mini_vllm\\scripts\\serve_mcp.py"]
    }
  }
}
```

---

## 5. 没做的事（以及为什么）

| 缺什么 | 影响 | 补起来要多少 |
|---|---|---|
| Streamable HTTP 传输 | 只能被本地宿主用（stdio） | 一个 `StreamingResponse` 包一层 |
| `notifications/cancelled` 真取消 | 长查询无法中断 | 要工具支持超时（见 sql_tools 的 TODO） |
| 工具列表动态变化 | 加表要重启进程 | `listChanged: true` + 主动通知 |
| `prompts` / `sampling` / `elicitation` | 用不上，但也不虚报 | — |
| 鉴权 | stdio 本来就没有网络面 | HTTP 传输时才需要 |

**资源为什么实现了：** `resources` 是 MCP 比 function calling 多出来的原语 ——
「用户或宿主**主动拉取**的上下文」，不经过模型决定。这里挂每张表的结构正好：`这个库有
什么表` 该由人先看一眼，而不是让模型去猜着调 `list_tables`。

```
> resources/read  db://dba04_cy4/schema
< {"contents":[{"uri":"db://dba04_cy4/schema","mimeType":"text/plain",
                "text":"dba04_cy4:\nJH  TEXT\nNY  TEXT\n..."}]}
```

---

## 6. 面试可以怎么讲

> 我的项目用的是 OpenAI function calling，没用 MCP。两者不在同一层：function calling
> 解决「模型怎么说」，MCP 解决「工具从哪来」。我这套的 `Tool.to_schema()` 输出就是 JSON
> Schema，和 MCP 的 `inputSchema` 几乎一一对应，所以包成 MCP server 只加了 250 行
> JSON-RPC 外壳，`miniagent` 一行没改。
>
> 真正需要重新想的是两个地方：一是**错误分层**——工具失败必须放在 `result.isError`
> 而不是 JSON-RPC `error`，否则模型看不见那句最有用的改正提示；二是**进程边界**——
> 现在工具和 agent 同进程、靠闭包共享状态，做成服务器就得把状态搬进去，`tools/list`
> 也变成了可以动态变化的。

---

## 动手练习

1. 把 `_tools_call` 里未知工具名也改成 `isError: true`，然后想：宿主拿不到
   `-32602` 之后，怎么区分「我写错了工具名」和「模型编了个工具名」？
2. 给 `MCPServer` 加 `notifications/tools/list_changed`：运行期间给 `ToolRegistry`
   加一个工具，主动推一条通知出去。（提示：`serve()` 现在只被动响应。）
3. 把 `resources/*` 关掉（`--no-resources`），再跑一遍 `initialize`，对比
   `capabilities` 的差异 —— 这就是「声明的能力必须真能实现」的实践。
4. 用 `tests/test_mcp.py` 里的 `server.serve(lines, out)` 接口写一个「回放整段会话」的
   工具：喂一个 JSONL 文件进去，把输出另存一份，用来做协议回归。
