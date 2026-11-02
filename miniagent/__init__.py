"""miniagent —— 手写 ReAct 循环，约 200 行控制流。

这里刻意不用任何框架。所谓 agent，就是「在一个消息列表上跑的 while 循环，外加几道
护栏」；用库把它包起来，恰好藏掉了面试和事故最关心的那部分。

    from miniagent import Agent, ScriptedBackend, ToolRegistry
    from miniagent.builtin_tools import calculator

    agent = Agent(ScriptedBackend([...]), [calculator])
    result = agent.run("What is (23*7+11)/4?")
"""

from miniagent.agent import DEFAULT_SYSTEM_PROMPT, Agent
from miniagent.backends import (
    CallableBackend,
    LLMBackend,
    MiniVLLMBackend,
    OpenAICompatBackend,
    ScriptedBackend,
    build_backend,
)
from miniagent.builtin_tools import (
    Sandbox,
    calculator,
    default_tools,
    make_file_tools,
    make_write_tools,
    text_stats,
)
from miniagent.parser import ParseResult, format_tool_call, parse
from miniagent.sql_tools import (
    SQLPolicy,
    SQLSandbox,
    make_sql_tools,
    mysql_connector,
    sqlite_connector,
)
from miniagent.tools import Tool, ToolRegistry, tool
from miniagent.types import (
    AgentResult,
    Message,
    Role,
    Step,
    StopReason,
    ToolCall,
    ToolResult,
)

__all__ = [
    "Agent",
    "DEFAULT_SYSTEM_PROMPT",
    "AgentResult",
    "CallableBackend",
    "LLMBackend",
    "Message",
    "MiniVLLMBackend",
    "OpenAICompatBackend",
    "ParseResult",
    "Role",
    "Sandbox",
    "ScriptedBackend",
    "SQLPolicy",
    "SQLSandbox",
    "Step",
    "StopReason",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "build_backend",
    "calculator",
    "default_tools",
    "format_tool_call",
    "make_file_tools",
    "make_sql_tools",
    "make_write_tools",
    "mysql_connector",
    "parse",
    "sqlite_connector",
    "text_stats",
    "tool",
]
