"""The agent loop's control flow, verified without a model.

Every test here uses ``ScriptedBackend`` or ``CallableBackend``, so it runs on CPU
in milliseconds and never depends on how well a 0.5B model can follow a format.
That is the whole point of keeping the loop free of model specifics.
"""

from __future__ import annotations

from miniagent.agent import Agent
from miniagent.backends import CallableBackend, ScriptedBackend
from miniagent.builtin_tools import calculator, text_stats
from miniagent.tools import tool
from miniagent.types import Role, StopReason


def tool_call(tool: str, **arguments) -> str:
    """构造一个 Qwen 风格的调用。第一个参数叫 ``tool`` 而不是 ``name``，这样即使某个
    工具本身有个叫 ``name`` 的参数也不会撞车。
    """
    import json

    payload = json.dumps({"name": tool, "arguments": arguments})
    return f"<tool_call>{payload}</tool_call>"


def test_single_tool_call_then_answer():
    backend = ScriptedBackend(
        [
            tool_call("calculator", expression="(23*7+11)/4"),
            "The answer is 43.",
        ]
    )
    agent = Agent(backend, [calculator])
    result = agent.run("What is (23*7+11)/4?")

    assert result.succeeded
    assert result.stopped_reason is StopReason.FINAL_ANSWER
    assert result.answer == "The answer is 43."
    assert result.num_tool_calls == 1
    assert result.num_tool_errors == 0
    assert len(result.steps) == 2

    # 第二轮时，观察确实送到了模型手上
    second_turn = backend.requests[1]
    assert any(m.role is Role.TOOL and "43" in m.content for m in second_turn)


def test_no_tool_needed_returns_immediately():
    backend = ScriptedBackend(["Paris."])
    agent = Agent(backend, [calculator])
    result = agent.run("Capital of France?")

    assert result.answer == "Paris."
    assert result.num_tool_calls == 0
    assert len(backend.requests) == 1


def test_multiple_calls_in_one_step_are_all_executed():
    backend = ScriptedBackend(
        [
            tool_call("calculator", expression="2+2") + tool_call("calculator", expression="3+3"),
            "done",
        ]
    )
    agent = Agent(backend, [calculator, text_stats])
    result = agent.run("add things")

    assert result.steps[0].tool_calls and len(result.steps[0].tool_calls) == 2
    contents = [r.content for r in result.steps[0].tool_results]
    assert contents == ["4", "6"]


def test_two_step_chain():
    def decide(messages):
        tool_messages = [m for m in messages if m.role is Role.TOOL]
        if not tool_messages:
            return tool_call("calculator", expression="23*7")
        if len(tool_messages) == 1:
            first = tool_messages[0].content.split()[-1]
            return tool_call("calculator", expression=f"({first}+11)/4")
        return "The result is 43."

    agent = Agent(CallableBackend(decide), [calculator])
    result = agent.run("compute (23*7+11)/4")

    assert result.answer == "The result is 43."
    assert result.num_tool_calls == 2
    assert [r.content for s in result.steps for r in s.tool_results] == ["161", "43"]


def test_tool_error_is_fed_back_so_the_model_can_recover():
    def decide(messages):
        observations = [m.content for m in messages if m.role is Role.TOOL]
        if not observations:
            return tool_call("calculator", expr="2+2")  # 参数名写错
        assert "unexpected argument" in observations[0]
        return tool_call("calculator", expression="2+2")

    agent = Agent(CallableBackend(decide), [calculator])
    result = agent.run("2+2")

    assert result.num_tool_errors == 1
    assert result.steps[0].tool_results[0].ok is False
    assert result.steps[-1].tool_results[0].content == "4"


def test_unknown_tool_is_reported_to_the_model():
    backend = ScriptedBackend([tool_call("does_not_exist", x=1), "sorry"])
    agent = Agent(backend, [calculator])
    result = agent.run("try something")

    assert result.steps[0].tool_results[0].ok is False
    assert "unknown tool" in result.steps[0].tool_results[0].content


def test_max_steps_is_enforced():
    replies = [tool_call("calculator", expression=f"{i}+1") for i in range(10)]
    agent = Agent(ScriptedBackend(replies), [calculator], max_steps=3)
    result = agent.run("loop forever")

    assert result.stopped_reason is StopReason.MAX_STEPS
    assert len(result.steps) == 3
    assert "without a final answer" in result.answer


def test_identical_calls_trip_the_loop_guard():
    same = tool_call("calculator", expression="1+1")
    agent = Agent(ScriptedBackend([same, same, same, same]), [calculator], max_identical_calls=2)
    result = agent.run("stuck")

    assert result.stopped_reason is StopReason.LOOP_DETECTED
    assert "identical arguments" in result.answer
    assert len(result.steps) == 3  # 第三次完全相同的调用才会触发它


def test_different_arguments_do_not_trip_the_loop_guard():
    replies = [
        tool_call("calculator", expression="1+1"),
        tool_call("calculator", expression="2+2"),
        tool_call("calculator", expression="3+3"),
        "done",
    ]
    agent = Agent(ScriptedBackend(replies), [calculator])
    result = agent.run("count up")
    assert result.stopped_reason is StopReason.FINAL_ANSWER
    assert result.num_tool_calls == 3


def test_parse_error_is_retried_then_gives_up():
    broken = "<tool_call>not json</tool_call>"
    agent = Agent(
        ScriptedBackend([broken, broken, broken, broken]),
        [calculator],
        max_parse_errors=2,
    )
    result = agent.run("bad format")

    assert result.stopped_reason is StopReason.TOO_MANY_ERRORS
    assert "unparseable" in result.answer
    assert all(step.parse_errors for step in result.steps)
    assert result.num_tool_calls == 0


def test_bad_tool_call_block_is_a_parse_error():
    agent = Agent(ScriptedBackend(["<tool_call>broken</tool_call>", "ok"]), [calculator])
    result = agent.run("x")

    assert result.steps[0].parse_errors
    assert "not a JSON object" in result.steps[0].parse_errors[0]
    assert result.answer == "ok"


def test_backend_failure_stops_the_run_cleanly():
    def boom(_messages):
        raise RuntimeError("model is down")

    result = Agent(CallableBackend(boom), [calculator]).run("hi")

    assert result.stopped_reason is StopReason.BACKEND_ERROR
    assert "model is down" in result.answer


def test_tools_are_advertised_to_the_backend():
    seen: dict = {}

    def record(messages):
        return "done"

    backend = CallableBackend(record)
    original = backend.complete

    def spy(messages, *, tools=None, max_tokens=None):
        seen["tools"] = tools
        return original(messages, tools=tools, max_tokens=max_tokens)

    backend.complete = spy  # type: ignore[method-assign]
    Agent(backend, [calculator, text_stats]).run("hi")

    names = [t["function"]["name"] for t in (seen["tools"] or [])]
    assert names == ["calculator", "text_stats"]


def test_on_step_callback_streams_the_trace():
    seen = []
    backend = ScriptedBackend([tool_call("calculator", expression="1+1"), "2"])
    agent = Agent(backend, [calculator], on_step=seen.append)
    agent.run("1+1")

    assert len(seen) == 2
    assert seen[0].has_tool_calls and not seen[1].has_tool_calls
    assert agent.trace == seen
    assert all(isinstance(s.summary(), str) for s in seen)


# ------------------------------------------------------- 串行化写入调用


def test_only_one_modifying_call_runs_per_turn():
    """一次回复里规划的两个写，就是两次盲写：第二个被推迟。

    循环随后会拒绝一个最终答案，直到那个被推迟的改动真的发生，于是写死的模型只好回来、
    晚一轮把它做掉。
    """
    executed: list[str] = []

    @tool(mutates=True)
    def save(path: str) -> str:
        """Pretend to write a file."""
        executed.append(path)
        return f"saved {path}"

    backend = ScriptedBackend(
        [
            tool_call("save", path="a.txt") + tool_call("save", path="b.txt"),
            tool_call("save", path="b.txt"),  # 被推迟的写入，晚一轮执行
            "done",
        ]
    )
    result = Agent(backend, [save]).run("write two files")

    assert result.steps[0].tool_results[0].ok is True, "the first write runs"
    assert result.steps[0].tool_results[1].ok is False, "the second one is deferred"
    assert "SKIPPED" in result.steps[0].tool_results[1].content
    assert executed == ["a.txt", "b.txt"]
    assert result.stopped_reason is StopReason.FINAL_ANSWER


def test_read_only_calls_are_not_serialised():
    executed: list[str] = []

    @tool
    def peek(name: str) -> str:
        """Read without changing anything."""
        executed.append(name)
        return name

    backend = ScriptedBackend([tool_call("peek", name="a") + tool_call("peek", name="b"), "done"])
    Agent(backend, [peek]).run("peek twice")

    assert executed == ["a", "b"]


def test_max_mutating_calls_can_be_raised_to_reproduce_the_bug():
    executed: list[str] = []

    @tool(mutates=True)
    def save(path: str) -> str:
        """Pretend to write a file."""
        executed.append(path)
        return f"saved {path}"

    backend = ScriptedBackend([tool_call("save", path="a.txt") + tool_call("save", path="b.txt"), "done"])
    Agent(backend, [save], max_mutating_calls=2).run("write two files")

    assert executed == ["a.txt", "b.txt"]


# ------------------------------------------------ 答案必须与磁盘状态一致


def test_an_answer_that_claims_a_failed_write_is_rejected():
    calls: list[str] = []

    @tool(mutates=True)
    def save(path: str) -> str:
        """Pretend to write a file."""
        calls.append(path)
        if len(calls) == 1:
            raise RuntimeError("disk is full")
        return f"saved {path}"

    backend = ScriptedBackend(
        [
            tool_call("save", path="a.txt"),  # 失败
            "I saved it.",  # 一个循环知道是假的说法
            tool_call("save", path="a.txt"),  # 重试，成功
            "Saved.",
        ]
    )
    result = Agent(backend, [save]).run("save it")

    assert result.stopped_reason is StopReason.FINAL_ANSWER
    assert calls == ["a.txt", "a.txt"]
    assert result.steps[1].notes, "the rejected answer should be annotated"
    nudge = [m.content for m in backend.requests[2] if m.role is Role.USER]
    assert any("did not happen" in text for text in nudge)


def test_a_write_that_is_never_retried_is_not_reported_as_success():
    @tool(mutates=True)
    def save(path: str) -> str:
        """Pretend to write a file."""
        raise RuntimeError("disk is full")

    backend = ScriptedBackend([tool_call("save", path="a.txt"), "I saved it.", "I saved it.", "I saved it."])
    result = Agent(backend, [save], max_write_retries=1, max_steps=6).run("save it")

    assert result.stopped_reason is StopReason.WRITE_INCOMPLETE
    assert not result.succeeded


# ------------------------------------------------ 写完，然后读回（接地）


def answering_model():
    """两个假工具：一个改文件，一个读文件。"""

    @tool(mutates=True, affects="path")
    def save(path: str) -> str:
        """Pretend to write a file."""
        return f"saved {path}"

    @tool(reads="path")
    def load(path: str) -> str:
        """Pretend to read a file."""
        return "buy milk\nwalk the dog\nwrite report\n"

    return save, load


def test_an_answer_about_a_changed_file_must_be_preceded_by_a_read_back():
    save, load = answering_model()
    backend = ScriptedBackend(
        [
            tool_call("save", path="todo.txt"),
            "The second line is now 'walk the dog at 7am'.",  # 从来没看过这个文件
            tool_call("load", path="todo.txt"),
            "The file says: buy milk / walk the dog / write report.",
        ]
    )
    result = Agent(backend, [save, load]).run("update todo.txt")

    assert result.stopped_reason is StopReason.FINAL_ANSWER
    assert result.steps[1].notes, "the ungrounded answer should be annotated"
    nudge = [m.content for m in backend.requests[2] if m.role is Role.USER]
    assert any("never read" in text for text in nudge)


def test_a_write_never_read_back_is_unverified_not_success():
    save, _ = answering_model()
    backend = ScriptedBackend([tool_call("save", path="a.txt"), "done", "done", "done"])

    result = Agent(backend, [save], max_readback_retries=1, max_steps=6).run("save it")

    assert result.stopped_reason is StopReason.UNVERIFIED
    assert not result.succeeded


def test_reading_the_same_file_again_is_not_a_loop():
    """「改后必读」要求重复读，所以重复读绝不能触发循环护栏。"""

    @tool(reads="path")
    def load(path: str) -> str:
        """Pretend to read a file."""
        return "content"

    replies = [tool_call("load", path="a.txt") for _ in range(5)] + ["done"]
    result = Agent(ScriptedBackend(replies), [load], max_steps=8).run("read it")

    assert result.stopped_reason is StopReason.FINAL_ANSWER
    assert result.num_tool_calls == 5
