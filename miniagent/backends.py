"""Where the tokens come from.

The loop does not care whether the model is a 0.5B local checkpoint or a hosted
API, so the only contract is::

    complete(messages, tools=..., max_tokens=...) -> str

That single method buys three things:

* **Deterministic tests.**  :class:`ScriptedBackend` replays canned replies and
  :class:`CallableBackend` lets a test react to the conversation, so the agent
  loop is verified without a GPU and without flaky model output.
* **A local backend** over :class:`minivllm.LLMEngine`, which is why this repo has
  a serving engine in the first place.
* **An HTTP backend** for anything that speaks the OpenAI chat-completions
  schema (DeepSeek, a vLLM server, Ollama, LM Studio).
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
    """Everything the agent needs from a model."""

    name: str

    def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> str: ...


# --------------------------------------------------------------------- test doubles


class ScriptedBackend:
    """Replay a fixed list of replies, in order.

    This is the workhorse for tests: the loop's control flow is what needs
    verifying, not the model's prose.
    """

    name = "scripted"

    def __init__(self, replies: Iterable[str]) -> None:
        self._replies = deque(replies)
        self.requests: list[list[Message]] = []
        """Every conversation the agent sent, so tests can assert on prompting."""

    def complete(self, messages, *, tools=None, max_tokens=None) -> str:
        self.requests.append(list(messages))
        if not self._replies:
            raise RuntimeError(
                f"ScriptedBackend ran out of replies after {len(self.requests) - 1} turn(s); "
                "the agent asked for one more"
            )
        return self._replies.popleft()


class CallableBackend:
    """Decide each reply from the conversation so far.

    More expressive than a fixed list: it can inspect the tool results the agent
    appended and answer accordingly, which is how a "model that recovers from an
    error" is simulated.
    """

    name = "callable"

    def __init__(self, fn: Callable[[list[Message]], str]) -> None:
        self._fn = fn
        self.requests: list[list[Message]] = []

    def complete(self, messages, *, tools=None, max_tokens=None) -> str:
        self.requests.append(list(messages))
        return self._fn(list(messages))


# ------------------------------------------------------------------- local minivllm


DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B-Instruct"


class MiniVLLMBackend:
    """Generate with the local :class:`minivllm.LLMEngine`.

    ``transformers`` is imported lazily inside ``__init__`` so that importing
    ``miniagent`` (and running the parser/tool tests) needs no torch at all.

    Qwen2.5's chat template understands the ``tools=`` argument and renders the
    official ``<tool_call>`` format, so the model sees exactly the prompt layout it
    was trained on -- no hand-rolled prompt string anywhere.
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
            # The backend applies the chat template itself, because only it can
            # pass tools= to apply_chat_template.
            use_chat_template=False,
        )
        self.engine = LLMEngine(cfg)
        self.temperature = temperature

    def complete(self, messages, *, tools=None, max_tokens=None) -> str:
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
        out = self.engine.generate([prompt], params)[0]
        return out.text.strip()

    def describe(self) -> str:
        return self.engine.model_runner.describe()


# ------------------------------------------------------------------ OpenAI-compatible


class OpenAICompatBackend:
    """Any ``/chat/completions`` endpoint: DeepSeek, vLLM server, Ollama, LM Studio.

    Native ``tool_calls`` in the response are converted back into Qwen's text form
    so the rest of the agent only ever deals with one representation.
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

    def complete(self, messages, *, tools=None, max_tokens=None) -> str:
        body = json.dumps(self._payload(messages, tools, max_tokens)).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"HTTP {exc.code} from {self.base_url}: {detail}") from exc

        message = data["choices"][0]["message"]
        text = (message.get("content") or "").strip()
        calls = message.get("tool_calls") or []
        if not calls:
            return text

        rendered = []
        for call in calls:
            function = call.get("function", {})
            raw_args = function.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            rendered.append(format_tool_call(function.get("name", ""), args))
        return "\n".join(filter(None, [text, *rendered]))


def build_backend(kind: str, **kwargs: Any) -> LLMBackend:
    """Small factory so scripts can take ``--backend`` on the command line."""
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
