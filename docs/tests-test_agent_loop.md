# `tests/test_agent_loop.py` 详解

## 这个文件干什么

**验证 agent 的控制流，完全不依赖模型。**

16 个用例，全部用 `ScriptedBackend` 或 `CallableBackend`，CPU 上 0.2 秒跑完。

这是整个 agent 部分最重要的一组测试。理由很简单：**循环逻辑不需要模型来判断对错**。用真模型测循环，你分不清失败是「循环写错了」还是「0.5B 又犯傻了」。

---

## 0. `tool_call` 辅助函数

```python
def tool_call(name: str, **arguments) -> str:
    import json

    payload = json.dumps({"name": name, "arguments": arguments})
    return f"<tool_call>{payload}</tool_call>"
```

和 `scripts/agent_demo.py` 里的同名函数重复。**有意的重复**：测试不该 import 脚本（脚本会调 `argparse`、有 `main()`、还可能触发模型加载）。

`import json` 在函数体内——因为这是测试文件唯一的用途，放外面也行，但这样更聚焦。

---

## 1. 主路径

### 单次调用 → 回答

```python
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
```

**最后那个断言是这条测试的灵魂。**

前面几条只证明「结果对了」，最后一条证明「**机制真的运转了**」——工具的输出确实作为 `role=tool` 的消息进入了第二轮。

没有它，一个「跳过工具直接返回固定答案」的错误实现也能通过。

`backend.requests[1]` 之所以存在，是因为 `ScriptedBackend` 记录了每次发送的对话快照。

### 不需要工具

```python
def test_no_tool_needed_returns_immediately():
    backend = ScriptedBackend(["Paris."])
    result = Agent(backend, [calculator]).run("Capital of France?")

    assert result.answer == "Paris."
    assert result.num_tool_calls == 0
    assert len(backend.requests) == 1
```

`len(backend.requests) == 1` 证明**没有多问一轮**。如果实现里「没调用工具」时还 `continue` 一次，这里会变成 2（而 `ScriptedBackend` 会抛「台词用完」）。

**这条测试顺便依赖了 `ScriptedBackend` 台词用完抛异常的行为**——如果它返回空串，多问的那一轮会被当成「空答案」而静默通过。

### 一轮多个调用

```python
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
```

`contents == ["4", "6"]` 验证**顺序保持**（并行调用不该乱序）。

---

## 2. 多步链（`CallableBackend` 的主场）

```python
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
```

`first = tool_messages[0].content.split()[-1]` 是在**解析观察字符串**：`"[calculator] 161"` → `"161"`。

这看起来很脏，但它模拟的正是真实模型的处境——**只能看到文本**。所以这个写法其实是忠实的。

`[r.content for s in result.steps for r in s.tool_results]` 是嵌套推导，等价于双重 `for` 循环，把「所有步骤的所有结果」拍平。

---

## 3. 错误处理

### 参数错了 → 模型自己改

```python
def test_tool_error_is_fed_back_so_the_model_can_recover():
    def decide(messages):
        observations = [m.content for m in messages if m.role is Role.TOOL]
        if not observations:
            return tool_call("calculator", expr="2+2")      # 故意写错参数名
        assert "unexpected argument" in observations[0]
        return tool_call("calculator", expression="2+2")

    agent = Agent(CallableBackend(decide), [calculator])
    result = agent.run("2+2")

    assert result.num_tool_errors == 1
    assert result.steps[0].tool_results[0].ok is False
    assert result.steps[-1].tool_results[0].content == "4"
```

三个断言分别验证：

1. 错误被记录（不是被吞掉）
2. 第一次真的失败了
3. **最后还是成功了**——错误没有让循环终止

`assert "unexpected argument" in observations[0]` 在 `decide` 内部——**这是「测试的测试」**：如果错误消息不包含这个关键词，测试会在这里失败，失败信息直接指出问题。

### 未知工具

```python
def test_unknown_tool_is_reported_to_the_model():
    backend = ScriptedBackend([tool_call("does_not_exist", x=1), "sorry"])
    agent = Agent(backend, [calculator])
    result = agent.run("try something")

    assert result.steps[0].tool_results[0].ok is False
    assert "unknown tool" in result.steps[0].tool_results[0].content
```

---

## 4. 四条护栏

### `max_steps`

```python
def test_max_steps_is_enforced():
    replies = [tool_call("calculator", expression=f"{i}+1") for i in range(10)]
    agent = Agent(ScriptedBackend(replies), [calculator], max_steps=3)
    result = agent.run("loop forever")

    assert result.stopped_reason is StopReason.MAX_STEPS
    assert len(result.steps) == 3
    assert "without a final answer" in result.answer
```

`expression=f"{i}+1"` 让每次参数不同，**避开重复检测**，从而单独测步数上限。这是「一次只测一条护栏」的写法。

### 重复检测

```python
def test_identical_calls_trip_the_loop_guard():
    same = tool_call("calculator", expression="1+1")
    agent = Agent(ScriptedBackend([same, same, same, same]), [calculator], max_identical_calls=2)
    result = agent.run("stuck")

    assert result.stopped_reason is StopReason.LOOP_DETECTED
    assert "identical arguments" in result.answer
    assert len(result.steps) == 3   # 第 3 次重复才触发
```

`max_identical_calls=2` → 第 3 次（计数 3 > 2）才触发。所以 `steps == 3`。

**这个 `== 3` 的精确断言很重要**：它把「阈值语义」钉死了。如果哪天改成 `>=`，这里立刻失败。

### 不同参数不算重复

```python
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
```

**这条和上一条是一对**：证明判据是「工具名 + 参数」而不是「工具名」。缺了这条，把 `key()` 改成只返回名字的实现也能通过所有测试。

### 解析错误预算

```python
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
```

`all(step.parse_errors for step in result.steps)` 验证**每一步都记录了错误**（可观测性）。

> 这条测试我写错过两次：
>
> **第一次**用 `"nonsense"` 当坏输出——但纯散文会被当成「模型完成了」，返回 `FINAL_ANSWER`。测试失败，我才意识到「没调用工具」和「调用格式坏了」是两条完全不同的路径。
>
> **第二次**保留了 `assert result.steps[0].parse_errors == []`（从错误版本抄过来的），跟新的 `all(...)` 断言矛盾。改了两个断言才自洽。
>
> 这类「测试自己写错」的情况比想象中多。**看到红色先怀疑测试。**

### 后端崩溃

```python
def test_backend_failure_stops_the_run_cleanly():
    def boom(_messages):
        raise RuntimeError("model is down")

    result = Agent(CallableBackend(boom), [calculator]).run("hi")

    assert result.stopped_reason is StopReason.BACKEND_ERROR
    assert "model is down" in result.answer
```

验证「后端异常不会让调用方崩」，而是转成一个可展示的结果。

---

## 5. 可观测性

### 工具真的被广告给模型了

```python
def test_tools_are_advertised_to_the_backend():
    seen: dict = {}
    backend = CallableBackend(lambda messages: "done")
    original = backend.complete

    def spy(messages, *, tools=None, max_tokens=None):
        seen["tools"] = tools
        return original(messages, tools=tools, max_tokens=max_tokens)

    backend.complete = spy  # type: ignore[method-assign]
    Agent(backend, [calculator, text_stats]).run("hi")

    names = [t["function"]["name"] for t in (seen["tools"] or [])]
    assert names == ["calculator", "text_stats"]
```

用手写 spy 替换 `backend.complete`（因为 `CallableBackend` 本身不看 `tools`）。

`names == [...]` 精确断言顺序——因为 `ToolRegistry` 用 dict 保序。如果要允许乱序，应该改成 `set(names) == {...}`。

> `# type: ignore[method-assign]` 是因为给实例属性赋方法不符合类型检查。这是测试里值得接受的妥协。

### `on_step` 回调

```python
def test_on_step_callback_streams_the_trace():
    seen = []
    backend = ScriptedBackend([tool_call("calculator", expression="1+1"), "2"])
    agent = Agent(backend, [calculator], on_step=seen.append)
    agent.run("1+1")

    assert len(seen) == 2
    assert seen[0].has_tool_calls and not seen[1].has_tool_calls
    assert agent.trace == seen
    assert all(isinstance(s.summary(), str) for s in seen)
```

`on_step=seen.append` —— 直接把 list 的 `append` 当回调传。这是 Python 的实用技巧：**任何可调用对象都能当回调**。

`agent.trace == seen` 验证两条路径（实时回调 + 事后 trace）内容一致。

`all(isinstance(s.summary(), str) for s in seen)` 看起来是废话断言，但它保证 `summary()` 对每种 step 都不会抛异常（比如 `zip` 长度不匹配、空列表索引）。

---

## 覆盖矩阵

| 控制流分支 | 测试 |
|---|---|
| 正常调用 → 回答 | 1 |
| 不需要工具 | 1 |
| 一轮多调用 | 1 |
| 多步链 | 1 |
| 工具错误恢复 | 1 |
| 未知工具 | 1 |
| `max_steps` | 1 |
| 重复检测（正例） | 1 |
| 重复检测（反例） | 1 |
| 解析重试 | 1 |
| 坏块不重试 | 1 |
| 后端崩溃 | 1 |
| 工具广告 | 1 |
| `on_step` | 1 |
| **合计** | **16** |

覆盖了 `run()` 里所有 `return` 点和所有 `continue` 分支。

---

## 用到的 pytest / Python 特性

| 特性 | 说明 |
|---|---|
| `CallableBackend` + 闭包 | 按对话内容决定下一步 |
| `seen.append` 作为回调 | 可调用对象即回调 |
| `backend.complete = spy` | monkey-patch 实例方法 |
| `zip(..., strict=False)` | 调用与结果配对 |
| 嵌套列表推导 | 拍平 steps → results |
| `assert` 里带中文消息 | 失败时说明设计意图 |

---

## 设计取舍

**为什么不用 mock 库？**

手写 `spy` / `CallableBackend` 更直观，也不引入 `unittest.mock` 的隐式行为。**测试代码的可读性比省几行更重要**——毕竟它是给人看的文档。

**为什么大量用 `ScriptedBackend` 而不是 `CallableBackend`？**

`ScriptedBackend` 强迫你**写出确切的对话轮次**。如果 agent 多问了一轮，它会抛异常——这是很好的副作用。`CallableBackend` 则可能「宽容地」返回某个兜底答案，掩盖问题。

**为什么每条护栏单独一条测试，而不是组合？**

组合测试失败时你分不清是哪条护栏触发的。而且护栏之间有交互（重复检测会掩盖步数上限），必须隔离。

---

## 常见误解

* 「`max_steps=3` 表示最多调 3 次工具」——是**模型轮次**。一轮可调多个。
* 「`max_identical_calls=2` 表示重复 2 次就停」——是**超过 2 次**（即第 3 次）才停。
* 「纯散文会触发解析错误」——不会，那是 `FINAL_ANSWER`。
* 「`result.num_tool_errors` 统计的是步数」——是**工具调用**次数。
* 「`ScriptedBackend` 台词用完会静默返回空」——会抛异常，这是刻意设计的。

---

## 动手练习

1. 把 `max_identical_calls` 的判据从 `>` 改成 `>=`，跑 `test_identical_calls_trip_the_loop_guard`，看 `len(steps)` 从 3 变成 2。
2. 把 `ToolCall.key()` 改成只返回 `self.name`，跑 `test_different_arguments_do_not_trip_the_loop_guard`，看它如何失败——这是「判据必须含参数」的直接证明。
3. 给 `test_max_steps_is_enforced` 加一条断言 `result.num_tool_calls == 3`，确认步数和调用次数相等（因为每步只调一个）。
4. 新增测试：让模型在一轮里调两个工具，第二个失败，验证 `step.tool_results` 里 `ok` 分别是 `True`/`False` 而循环继续。
