"""token 从哪来。

循环不关心模型是本地 0.5B 还是一个托管 API，唯一的契约是::

    complete(messages, tools=..., max_tokens=...) -> str

这一个方法换来三样东西：

* **确定性测试。**  :class:`ScriptedBackend` 回放写好的台词，
  :class:`CallableBackend` 让测试能对对话作出反应，于是 agent 循环不需要 GPU、
  也不受模型输出波动的影响就能验证。
* **本地后端**，跑在 :class:`minivllm.LLMEngine` 上 —— 这也是这个仓库为什么会带
  一个推理引擎。
* **HTTP 后端**，任何说 OpenAI chat-completions 协议的都能接（DeepSeek、vLLM
  服务器、Ollama、LM Studio）。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from miniagent.parser import format_tool_call
from miniagent.types import Message


class LLMBackend(Protocol):
    """agent 需要从模型那里得到的全部东西。"""

    name: str

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str: ...


"""``on_token`` 是可选的增量回调。

为什么要把它放在 ``complete`` 上而不是另开一个 ``stream()``：agent 需要的永远是
「这一轮的完整文本」（要拿去 ``parse``），增量只是过程中的副产品。多一个方法就得
在每个调用点写两套分支；多一个参数，不关心的后端直接忽略即可。

不实现流式的后端也能正确工作：它们要么忽略 ``on_token``，要么把整段文本作为一块发出去
（``ScriptedBackend`` 就是这么做的，所以流式路径可以在没有 GPU 的测试里跑）。
"""


# ------------------------------------------------------------------ 测试替身


class ScriptedBackend:
    """按顺序回放一份写死的回复列表。

    这是测试的主力：需要验证的是循环的控制流，不是模型的文笔。
    """

    name = "scripted"

    def __init__(self, replies: Iterable[str]) -> None:
        self._replies = deque(replies)
        self.requests: list[list[Message]] = []
        """Every conversation the agent sent, so tests can assert on prompting."""

    def complete(self, messages, *, tools=None, max_tokens=None, on_token=None) -> str:
        self.requests.append(list(messages))
        if not self._replies:
            raise RuntimeError(
                f"ScriptedBackend ran out of replies after {len(self.requests) - 1} turn(s); "
                "the agent asked for one more"
            )
        reply = self._replies.popleft()
        if on_token is not None and reply:
            on_token(reply)  # 一块发完：流式路径因此也能在纯 CPU 上测
        return reply


class CallableBackend:
    """根据到此为止的对话决定每一条回复。

    比固定列表更有表达力：它能查看 agent 追加的工具结果并据此作答，这就是「一个能从
    错误中恢复的模型」被模拟出来的方式。
    """

    name = "callable"

    def __init__(self, fn: Callable[[list[Message]], str]) -> None:
        self._fn = fn
        self.requests: list[list[Message]] = []

    def complete(self, messages, *, tools=None, max_tokens=None, on_token=None) -> str:
        self.requests.append(list(messages))
        reply = self._fn(list(messages))
        if on_token is not None and reply:
            on_token(reply)
        return reply


# --------------------------------------------------------------- 本地 minivllm


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B-Instruct"


class MiniVLLMBackend:
    """用本地的 :class:`minivllm.LLMEngine` 生成。

    ``transformers`` 直到 ``__init__`` 里才延迟导入，所以导入 ``miniagent``（以及跑
    parser / 工具测试）完全不需要 torch。

    Qwen2.5 的 chat template 认识 ``tools=`` 参数，会渲染官方的 ``<tool_call>`` 格式，
    所以模型看到的就是它训练时的 prompt 布局 —— 全仓库没有一处手写的 prompt 字符串。
    """

    name = "minivllm"

    def __init__(
        self,
        model: str | Path | None = None,
        *,
        dtype: str = "float16",
        device: str = "cuda",
        temperature: float = 0.0,
        max_num_seqs: int = 4,
        max_num_batched_tokens: int = 2048,
        max_model_len: int = 4096,
    ) -> None:
        from minivllm.config import EngineConfig
        from minivllm.llm_engine import LLMEngine

        cfg = EngineConfig(
            model=str(model or DEFAULT_MODEL),
            dtype=dtype,
            device=device,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            # 由后端自己套 chat template，因为只有它能
            # 给 apply_chat_template 传 tools=。
            use_chat_template=False,
        )
        self.engine = LLMEngine(cfg)
        self.temperature = temperature

    def complete(self, messages, *, tools=None, max_tokens=None, on_token=None) -> str:
        from minivllm.sampling import SamplingParams

        tokenizer = self.engine.tokenizer
        prompt = tokenizer.apply_chat_template(
            [m.to_hf() for m in messages],
            tools=tools or None,
            add_generation_prompt=True,
            tokenize=False,
        )
        params = SamplingParams(
            temperature=self.temperature,
            max_tokens=max_tokens or 512,
            ignore_eos=False,
        )
        if on_token is None:
            return self.engine.generate([prompt], params)[0].text.strip()

        chunks: list[str] = []
        for delta in self.engine.generate_stream(prompt, params):
            chunks.append(delta)
            on_token(delta)
        # 返回值取引擎的权威输出，而不是拼起来的增量：增量是给人看的，完整文本是给
        # parse 用的，两者不该有不同的真相来源。
        out = self.engine.last_output
        return out.text.strip() if out is not None else "".join(chunks).strip()

    def describe(self) -> str:
        return self.engine.model_runner.describe()


# ------------------------------------------------------------ OpenAI 兼容接口


class OpenAICompatBackend:
    """任何 ``/chat/completions`` 端点：DeepSeek、vLLM 服务器、Ollama、LM Studio。

    响应里原生的 ``tool_calls`` 会被转回 Qwen 的文本形式，好让 agent 其余部分永远只面对
    一种表示。
    """

    name = "openai-compatible"

    def __init__(
        self,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        api_key: str | None = None,
        *,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("AGENT_API_KEY", "")
        self.temperature = temperature
        self.timeout_s = timeout_s

    def _payload(self, messages, tools, max_tokens) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_hf() for m in messages],
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = tools
        if max_tokens:
            body["max_tokens"] = max_tokens
        return body

    def _request(self, body: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": "Bearer " + self.api_key} if self.api_key else {}),
            },
            method="POST",
        )

    def _render(self, text: str, calls: list[dict[str, Any]]) -> str:
        """把回复拼起来：散文在前，每个调用一个 ``<tool_call>`` 块。"""
        if not calls:
            return text.strip()
        rendered = [text.strip()]
        for call in calls:
            function = call.get("function", {})
            raw_args = function.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            rendered.append(format_tool_call(function.get("name", ""), args))
        return "\n".join(filter(None, rendered))

    def _stream(self, body: dict[str, Any], on_token: Callable[[str], None]) -> str:
        """读 ``stream: true`` 的 SSE 响应，边到边转发文本。

        工具调用也是增量来的：每个 delta 带一个 ``index`` 和 ``function.arguments``
        的一个*片段*，必须先按 index 拼接才能解析 JSON。漏了这一步，``{"a": 1}`` 会
        变成一截解析不了的半截字符串，然后被 parser 报成「工具调用写坏了」。
        """
        content: list[str] = []
        slots: dict[int, dict[str, str]] = {}
        with self._request(body) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue  # 空行分隔、注释、event: 行都不关心
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        content.append(piece)
                        on_token(piece)
                    for call in delta.get("tool_calls") or []:
                        slot = slots.setdefault(int(call.get("index", 0)), {"name": "", "arguments": ""})
                        function = call.get("function") or {}
                        if function.get("name"):
                            slot["name"] = function["name"]
                        if function.get("arguments"):
                            slot["arguments"] += function["arguments"]
        calls = [
            {"function": {"name": s["name"], "arguments": s["arguments"]}} for _, s in sorted(slots.items())
        ]
        return self._render("".join(content), calls)

    def complete(self, messages, *, tools=None, max_tokens=None, on_token=None) -> str:
        body = self._payload(messages, tools, max_tokens)
        try:
            if on_token is not None:
                return self._stream({**body, "stream": True}, on_token)
            with self._request(body) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"HTTP {exc.code} from {self.base_url}: {detail}") from exc

        message = data["choices"][0]["message"]
        return self._render(message.get("content") or "", message.get("tool_calls") or [])


def build_backend(kind: str, **kwargs: Any) -> LLMBackend:
    """小工厂，好让脚本能在命令行上选 ``--backend``。"""
    if kind == "local":
        return MiniVLLMBackend(**kwargs)
    if kind in {"api", "openai"}:
        return OpenAICompatBackend(**kwargs)
    raise ValueError(f"unknown backend {kind!r}; choose 'local' or 'api'")


__all__ = [
    "LLMBackend",
    "MiniVLLMBackend",
    "OpenAICompatBackend",
    "ScriptedBackend",
    "CallableBackend",
    "build_backend",
]
