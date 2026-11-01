"""miniagent -- a hand-written ReAct loop, in about 200 lines of control flow.

There is no framework here on purpose.  An agent is a ``while`` loop over a message
list with a handful of guards; using a library hides exactly the part that
interviews and incidents are about.

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
    "make_write_tools",
    "parse",
    "text_stats",
    "tool",
]
