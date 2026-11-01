"""The agent loop's control flow, verified without a model.

Every test here uses ``ScriptedBackend`` or ``CallableBackend``, so it runs on CPU
in milliseconds and never depends on how well a 0.5B model can follow a format.
That is the whole point of keeping the loop free of model specifics.
"""

from __future__ import annotations

from miniagent.agent import Agent
from miniagent.backends import CallableBackend, ScriptedBackend
from miniagent.builtin_tools import calculator, text_stats
from miniagent.types import Role, StopReason


def tool_call(name: str, **arguments) -> str:
    import json

    payload = json.dumps({"name": name, "arguments": arguments})
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

    # the observation really reached the model on the second turn
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
            return tool_call("calculator", expr="2+2")  # wrong argument name
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
    assert len(result.steps) == 3  # third identical call is what trips it


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
