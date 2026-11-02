"""Tool definition, JSON-schema generation, and the registry that runs them.

The interesting part is :func:`tool`: it turns a plain Python function into the
schema an LLM needs, using nothing but ``inspect`` and type hints.  That is exactly
how the ``tools=[...]`` payload of a function-calling API gets built -- there is no
magic, just introspection plus a docstring.

Three rules this module enforces, because each one is a real failure mode:

1. **Validate before calling.** An unknown argument name should come back as a
   readable message, not as ``TypeError: unexpected keyword argument``.
2. **Never let a tool kill the loop.** Exceptions become ``ToolResult(ok=False)``,
   which the agent feeds back to the model so it can correct itself.
3. **Keep the schema and the signature in sync.** Both come from the same
   function object, so they cannot drift.
"""

from __future__ import annotations

import inspect
import time
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, get_args, get_origin, get_type_hints

from miniagent.types import ToolCall, ToolResult

_PY_TO_JSON: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}

_DOC_SECTIONS = {
    "args",
    "arguments",
    "parameters",
    "returns",
    "return",
    "raises",
    "yields",
    "note",
    "notes",
    "example",
    "examples",
}


def _is_union(origin: Any) -> bool:
    return origin is typing.Union or origin is types.UnionType


def _json_schema(tp: Any) -> dict[str, Any]:
    """Map a Python annotation to the subset of JSON Schema that tools need."""
    if tp is inspect.Parameter.empty:
        return {"type": "string"}

    origin = get_origin(tp)

    if _is_union(origin):
        members = [a for a in get_args(tp) if a is not type(None)]
        if len(members) > 1:
            return {"anyOf": [_json_schema(m) for m in members]}
        return _json_schema(members[0]) if members else {"type": "string"}

    if origin in (list, tuple, set, frozenset):
        args = get_args(tp)
        item = _json_schema(args[0]) if args else {}
        return {"type": "array", "items": item}

    if origin is dict:
        return {"type": "object"}

    if isinstance(tp, type) and tp in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[tp]}

    return {"type": "string"}


def _docstring_summary(fn: Callable[..., Any]) -> str:
    """First paragraph of the docstring, with ``Args:`` and later sections dropped."""
    doc = inspect.getdoc(fn) or ""
    lines: list[str] = []
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.endswith(":") and stripped.rstrip(":").lower() in _DOC_SECTIONS:
            break
        lines.append(line)
    return " ".join(" ".join(lines).split())


def _param_docs(fn: Callable[..., Any]) -> dict[str, str]:
    """Parse ``name: description`` lines out of the docstring's Args section."""
    doc = inspect.getdoc(fn) or ""
    out: dict[str, str] = {}
    current: str | None = None
    in_args = False

    for line in doc.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.rstrip(":").lower() in {"args", "arguments", "parameters"}:
            in_args = True
            continue
        if not in_args:
            continue
        if stripped.endswith(":") and " " not in stripped.rstrip(":"):
            break  # next section header
        name, sep, desc = stripped.partition(":")
        bare = name.strip()
        if sep and bare and " " not in bare:
            current = bare
            out[current] = desc.strip()
        elif current:
            out[current] = f"{out[current]} {stripped}".strip()
    return out


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    param_docs: dict[str, str] = field(default_factory=dict)
    mutates: bool = False
    """True when calling the tool changes state outside the conversation.

    The agent uses this to serialise writes: a model that plans two file
    modifications in one reply planned both *before* seeing either result, so the
    second one is a blind write.  Read-only tools are never serialised.
    """
    affects: str | None = None
    """Argument name holding the path this tool changes (``"path"`` for file tools).

    The registry copies it into ``ToolResult.affects`` so the agent can demand that
    a changed file be read back before it accepts a final answer.
    """
    reads: str | None = None
    """Argument name holding the path this tool observes; that read clears the duty."""

    @property
    def required(self) -> list[str]:
        return list(self.parameters.get("required", []))

    def to_schema(self) -> dict[str, Any]:
        """The object handed to the model (OpenAI ``tools`` shape)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate(self, arguments: dict[str, Any]) -> str | None:
        """Return an error message, or None when the arguments are acceptable."""
        props = self.parameters.get("properties", {})
        unknown = [k for k in arguments if k not in props]
        if unknown:
            allowed = ", ".join(props) or "(none)"
            return f"unexpected argument(s) {unknown} for {self.name}; allowed: {allowed}"
        missing = [k for k in self.required if k not in arguments]
        if missing:
            return f"missing required argument(s) {missing} for {self.name}"
        return None

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    mutates: bool = False,
    affects: str | None = None,
    reads: str | None = None,
):
    """Turn a function into a :class:`Tool` by introspecting it.

    ``mutates=True`` marks a tool that changes state outside the conversation; the
    agent then runs at most one such call per turn, so the model has to see the
    result before it commits to the next change.  ``affects="path"`` additionally
    says *which* argument names what was changed, which lets the agent require a
    read-back before it believes the answer.

    >>> @tool
    ... def add(a: int, b: int) -> int:
    ...     '''Add two integers.'''
    ...     return a + b
    >>> add.name, add.parameters["required"]
    ('add', ['a', 'b'])
    """

    def wrap(func: Callable[..., Any]) -> Tool:
        hints = get_type_hints(func)
        signature = inspect.signature(func)
        docs = _param_docs(func)

        properties: dict[str, Any] = {}
        required: list[str] = []
        for pname, param in signature.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                raise TypeError(f"{func.__name__}: *args/**kwargs are not supported in tool signatures")
            schema = _json_schema(hints.get(pname, param.annotation))
            schema["description"] = docs.get(pname) or pname.replace("_", " ")
            properties[pname] = schema
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        return Tool(
            name=name or func.__name__,
            description=_docstring_summary(func) or f"Call {func.__name__}.",
            parameters={"type": "object", "properties": properties, "required": required},
            func=func,
            param_docs=docs,
            mutates=mutates,
            affects=affects,
            reads=reads,
        )

    return wrap(fn) if fn is not None else wrap


class ToolRegistry:
    """Name -> Tool lookup plus the execution boundary."""

    def __init__(self, tools: typing.Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or ():
            self.register(t)

    def register(self, t: Tool) -> Tool:
        if t.name in self._tools:
            raise ValueError(f"duplicate tool name: {t.name}")
        self._tools[t.name] = t
        return t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    @staticmethod
    def _declared_paths(tool: Tool | None, call: ToolCall, attribute: str) -> list[str]:
        """The path a call names in the argument that ``tool.<attribute>`` declares."""
        key = getattr(tool, attribute, None) if tool is not None else None
        value = call.arguments.get(key) if isinstance(key, str) else None
        return [value] if isinstance(value, str) and value else []

    def execute(self, call: ToolCall) -> ToolResult:
        """Run one call.  Never raises: every failure becomes ``ok=False``."""
        started = time.perf_counter()
        t = self.get(call.name)

        def fail(message: str) -> ToolResult:
            # The nudge matters for small models: an error on its own often makes a
            # 0.5B model give up and apologise instead of retrying with better
            # arguments.  Nothing happened, so retrying is always safe here.
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                content=(
                    f"ERROR: {message}\nNothing was changed. Call {call.name} again with corrected arguments."
                ),
                error=message,
                duration_s=time.perf_counter() - started,
                # a read that failed still counts as having looked: "it is gone" is
                # an observation, and it must be able to discharge a pending read-back
                reads=self._declared_paths(t, call, "reads"),
            )

        if t is None:
            return fail(f"unknown tool {call.name!r}; available: {', '.join(self.names) or '(none)'}")

        problem = t.validate(call.arguments)
        if problem:
            return fail(problem)

        try:
            value = t.func(**call.arguments)
        except Exception as exc:  # a tool bug must not break the agent loop
            return fail(f"{type(exc).__name__}: {exc}")

        return ToolResult(
            call_id=call.call_id,
            name=call.name,
            ok=True,
            content=str(value),
            duration_s=time.perf_counter() - started,
            affects=self._declared_paths(t, call, "affects"),
            reads=self._declared_paths(t, call, "reads"),
        )
