# `miniagent/agent.py` 详解 ★核心

## 这个文件干什么

**Agent 循环本身**，加上让它在真实环境里不会失控的四条护栏。

215 行，其中约 100 行是护栏。**护栏才是 demo 和能上线的东西之间的差别。**

---

## 1. 循环的形状

```
messages = [system, user]
loop:
    reply  = model(messages, tools)      # 文本，可能含工具调用
    calls  = parse(reply)                # 文本 → 结构化
    if 没有调用:  return reply           # 模型认为做完了
    messages += assistant(reply, calls)  # 保留模型说的话
    for call in calls:
        result = tools.execute(call)     # 永不抛异常
        messages += tool(result)         # 观察
    if 情况不对: break                   # 护栏
```

对应到 `run()` 里的四个分支：

```python
if not parsed.calls and not parsed.errors:      # ① 散文 → 结束
    return result(...)

if not parsed.calls:                            # ② 解析失败 → 回灌重试
    ...
    continue

messages.append(Message(Role.ASSISTANT, parsed.text, tool_calls=parsed.calls))  # ③ 执行
for call in parsed.calls:
    ...
```

---

## 2. 消息顺序为什么是这样

```python
messages.append(Message(Role.ASSISTANT, parsed.text, tool_calls=parsed.calls))
for call in parsed.calls:
    tool_result = self.tools.execute(call)
    step.tool_results.append(tool_result)
    messages.append(Message(Role.TOOL, tool_result.to_observation(), name=tool_result.name))
```

产生的对话：

```
system:    You are a careful assistant...
user:      What is (23*7+11)/4?
assistant: <tool_call>{"name": "calculator", ...}</tool_call>
tool:      [calculator] 161
assistant: <tool_call>{"name": "calculator", ...}</tool_call>
tool:      [calculator] 43
assistant: The answer is 43.
```

**assistant 消息必须带上 `tool_calls`**，不能只 append 文本。因为 Qwen 的模板会检查 `message.tool_calls`：

```jinja
{%- if (message.role == "assistant" and not message.tool_calls) %}
    ... 普通 assistant ...
{%- elif message.role == "assistant" %}
    ... 渲染 tool_calls 成 <tool_call> ... 
```

如果不传 `tool_calls`，模型看到的历史就是「我说了一串 JSON 但没有真的调用工具」，然后工具结果突然出现在一个 user 轮里。**上下文就错乱了**——模型可能重复调用，也可能不再调用。

---

## 3. 四条护栏

### ① `max_steps`：步数上限

```python
for index in range(self.max_steps):
    ...
return result(f"stopped after {self.max_steps} steps without a final answer", StopReason.MAX_STEPS)
```

`for` 循环自然结束 = 用完了步数。不需要额外计数变量。

**没有这个，模型可以永远调工具。** 有些模型在收到错误观察后会一直重试同样的调用，永远不会自己停。

### ② `max_identical_calls`：重复调用检测

```python
seen: Counter[str] = Counter()
...
for call in parsed.calls:
    ...
    seen[call.key()] += 1

repeated = [k for k, v in seen.items() if v > self.max_identical_calls]
if repeated:
    detail = repeated[0].split(":", 1)[0]
    return result(
        f"stopped: the model kept calling {detail} with identical arguments",
        StopReason.LOOP_DETECTED,
    )
```

`ToolCall.key()` 是 `f"{name}:{json.dumps(arguments, sort_keys=True)}"`。所以判据是**工具名 + 参数**完全相同。

**为什么不能只判工具名？** 因为连续调用同一个工具是正常的：

```
calculator(23 * 7)      → 161
calculator((161+11)/4)  → 43
```

同一个工具、不同参数 → 正常。
同一个工具、**同样参数** → 观察没帮上忙，再来一次也一样。

`Counter` 而不是 `set`：需要计数，因为「调用 1 次」和「调用 3 次」要区分。`max_identical_calls=2` 表示「第 3 次重复才判定死循环」。

实测：

```
step 0  call  calculator({"expr": "23 * 7"})   !! ERROR: unexpected argument...
step 1  call  calculator({"expression": "23 * 7"})  -> 161
step 2  call  calculator({"expression": "23 * 7"})  -> 161     ← 开始重复
step 3  call  calculator({"expression": "23 * 7"})  -> 161     ← 第 3 次 → 判定死循环
```

`detail = repeated[0].split(":", 1)[0]` 从 key 里取出工具名用于报错。`split(":", 1)` 只切第一个冒号，因为参数 JSON 里可能有冒号。

### ③ `max_parse_errors`：解析重试预算

```python
if not parsed.calls:
    parse_errors += len(parsed.errors)
    messages.append(Message(Role.ASSISTANT, reply))
    messages.append(Message(Role.USER, "Problem with your tool call:\n" + "\n".join(parsed.errors)))
    messages.append(Message(Role.USER, _REPAIR_HINT))
    ...
    if parse_errors > self.max_parse_errors:
        return result("the model kept producing unparseable tool calls", StopReason.TOO_MANY_ERRORS)
    continue
```

回灌两条消息：具体错误 + 格式提示。

```python
_REPAIR_HINT = (
    "Your previous reply did not contain a usable tool call. Reply with exactly one "
    'block of the form <tool_call>{{"name": "<tool>", "arguments": {{...}}}}</tool_call>, '
    "or answer the user directly without calling a tool."
)
```

注意它给了**两条出路**：「按格式重发」或「直接回答」。只给第一条的话，模型可能在被要求「必须调工具」时硬凑一个假调用。

`continue` 而不是 `break`：这一轮没干活，但还有补救机会。

**注意 `if not parsed.calls`**：这个分支只在「有 errors 但没有 calls」时进。如果既有 calls 又有 errors（比如两个块一好一坏），走的是执行分支——好的执行，坏的错误记录在 `step.parse_errors` 里但不再重试。这是刻意的：**能干活就先干活。**

### ④ 工具异常 → 观察（在 `tools.py` 里）

```python
except Exception as exc:
    return fail(f"{type(exc).__name__}: {exc}")
```

agent 这边看不到异常，只看到 `ToolResult(ok=False)`，然后照常 append 成观察。**护栏在下一层，但效果在这里体现。**

---

## 4. `on_step` 回调：可观测性

```python
def _emit(self, step: Step) -> None:
    self.trace.append(step)
    if self.on_step is not None:
        self.on_step(step)
```

每步结束时调用。`agent_demo.py` 用它实时打印：

```python
def print_step(step: Step) -> None:
    print(f"\n--- step {step.index} ({step.duration_s * 1000:.0f} ms) ---")
    if step.assistant_text:
        print(f"model: {step.assistant_text.strip()}")
    for call, result in zip(step.tool_calls, step.tool_results, strict=False):
        arrow = "->" if result.ok else "!!"
        print(f"  call   {call.name}({json.dumps(call.arguments, ensure_ascii=False)})")
        print(f"  {arrow} result {result.content}")
    for err in step.parse_errors:
        print(f"  parse error: {err}")
```

`self.trace` 和 `on_step` 都保留：前者供事后分析，后者供实时观察。

> 生产环境里 `on_step` 会接 OpenTelemetry span 或者日志，而不是 `print`。**同样的钩子，不同的消费者。**

---

## 5. 内部小工具：`result()` 闭包

```python
def result(answer: str, reason: StopReason) -> AgentResult:
    return self._finish(
        question=question, answer=answer, steps=steps,
        stopped_reason=reason, messages=messages,
        duration_s=time.perf_counter() - started,
    )
```

四个返回点（FINAL_ANSWER / BACKEND_ERROR / TOO_MANY_ERRORS / LOOP_DETECTED / MAX_STEPS）都要组装同样的结构。闭包捕获了 `question` / `steps` / `messages` / `started`，省得每处重复写。

**为什么 `messages` 直接传引用而不是复制？**

因为调用方想看到完整对话。返回后 agent 不会再改它（`run` 已结束），所以没有别名问题。

---

## 6. 写 `demo` 脚本要用的两个模式

### 模式 A：固定台词

```python
ScriptedBackend([
    tool_call("calculator", expression="23 * 7"),
    tool_call("calculator", expression="(161 + 11) / 4"),
    "The answer is 43.",
])
```

适合「一次性演示」，不适合「需要读观察结果」的场景。

### 模式 B：按观察结果决定

```python
def decide(messages):
    observations = [m.content for m in messages if m.role is Role.TOOL]
    if not observations:
        return tool_call("calculator", expr="23 * 7")     # 故意传错参数
    last = observations[-1]
    if "unexpected argument" in last:
        return tool_call("calculator", expression="23 * 7")
    good = [o for o in observations if "ERROR" not in o]
    if len(good) == 1:
        return tool_call("calculator", expression=f"({good[0].split()[-1]} + 11) / 4")
    return "The answer is 43."
```

这个函数本身就是「一个理想的模型应该怎么想」。**写它的过程就是在验证 agent 的机制够不够用**——如果写不出来，说明某个观察没传达到。

> 我曾在这里写错一次：用 `observations[0]` 判断错误，但消息列表是**累积**的，`[0]` 永远是第一条。改成 `observations[-1]` 才对。**这类下标错误在 agent 里特别常见，因为上下文是只增不减的。**

---

## 7. 设计稿：第七道护栏——证据闭合（尚未实现）

### 触发它的那次实测

`--max-rows 1` 跑「含水小于25」，库里真实匹配 **2 行**：

```text
--- step 0 ---
  call   query_table({"where": ["HS < 25"], "limit": 20})
  -> result 1 row(s) from dba04_cy4 (limit 1; MORE ROWS EXIST -- this is a partial
     result, call count_rows or narrow the filters)
           N6-D3-P2126 | 198207 | 24.0

--- step 1 ---
model: 表格复述（只有那一行）
stopped : final_answer        ← 系统放行了
```

**信号已经给出去了，模型没接。** 现有六道护栏全都只管「动作」，没有一道管「结论有没有证据」。于是 1 行被当成了全部答案。

### 为什么不能用字符串嗅探

最省事的写法是：

```python
if any("MORE ROWS EXIST" in r.content for r in step.tool_results):   # ← 不要这样
```

这和整份文档反复强调的毛病是同一个：**让上层去猜下层的文本，等于又造了一次「静默约定」**。文案改一个字（哪怕只是把 `MORE` 改成 `MORE ROWS`），护栏就永久失效，而且没有任何测试会红。

**下层要表态，就给它一个字段，而不是给它一句话。**

### 设计

**① `types.py`：两个新东西**

```python
@dataclass
class ToolReply:
    """A tool's result plus facts about the result the text cannot carry."""
    text: str
    incomplete: bool = False


@dataclass
class ToolResult:
    ...
    incomplete: bool = False
    """The result is a partial view: more rows exist than were returned."""
```

**② `tools.py`：`execute` 识别包装类型**

```python
value = t.func(**call.arguments)
if isinstance(value, ToolReply):
    content, incomplete = value.text, value.incomplete
else:
    content, incomplete = str(value), False       # 兼容所有现有工具
```

不改任何已有工具——它们返回 `str`，行为完全不变。

**③ `sql_tools.py`：截断时表态**

```python
if truncated:
    return ToolReply(head + _format_table(names, rows, box.policy), incomplete=True)
return head + _format_table(names, rows, box.policy)
```

**④ `agent.py`：形状和现有的 `dirty` 完全一致**

```python
pending_partial: dict[str, None] = {}    # key = 未消解的不完整结果
partial_retries = 0

# 每步结束后
for tool_result in step.tool_results:
    if tool_result.incomplete:
        pending_partial[f"{call.name}:{table}"] = None
    elif clears_partial(call, tool_result):      # 同一 (table, where) 上 count 成功
        pending_partial.clear()

# 模型想收工时
if pending_partial and partial_retries < self.max_partial_retries:
    partial_retries += 1
    step.notes.append("answer rejected: a truncated result was never resolved")
    messages.append(Message(Role.ASSISTANT, reply))
    messages.append(Message(Role.USER, self._partial_hint(list(pending_partial))))
    continue
```

**⑤ 三种结局，都比现在好**

| 模型行为 | 结果 |
|---|---|
| 被驳回后去调 `count_rows` | 拿到 2 → 正确作答 |
| 被驳回后仍然不查 | 重试用尽 → `stopped: unverified`，`verdict` 明说「答案不完整」 |
| 完全不理会（复读同一句） | `max_identical_calls` 先触发 → `loop_detected` |

**没有一种结局是「自信地给出 1 行」。**

### 要补的测试（约 8 个）

- 截断时 `ToolResult.incomplete is True`
- 未截断时是 `False`（不能狼来了）
- `ToolReply` 之外的返回值仍然 `str()`（兼容性）
- 模型无视截断就想收工 → 被驳回，且 `step.notes` 记录了原因
- 驳回后模型调用 `count_rows` → 允许收工
- 重试用尽 → `stopped_reason == UNVERIFIED`
- 文案改了（模拟「把 MORE ROWS 改成别的」）**不影响**护栏（证明没有依赖字符串）

最后那条是核心：**它锁住的不是行为，而是「行为不依赖文案」这件事。**

### 这一层的意义

前六道护栏管的是**动作**（别循环、别盲写、改完要读）。这一道管的是**结论**：

> 你要说的这句话，有没有证据支撑？没有就不许说。

**提示词是建议，护栏是制度。** 制度的效果不依赖对方是否配合——这正是弱模型场景下唯一能指望的东西。

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `collections.Counter` | 统计重复键 |
| 闭包捕获局部状态 | `result()` 省掉重复组装 |
| `@dataclass` 的 `field(default_factory=...)` | 每步独立的列表 |
| `zip(..., strict=False)` | 工具调用与结果一一对应 |
| `time.perf_counter` | 单调时钟，测每步耗时 |
| `for ... range(n)` + 之后的 return | 天然的步数上限 |

---

## 设计取舍

**为什么不用 `while True` + 计数器？**

`for index in range(self.max_steps)` 更安全：**忘记递增计数器就会死循环**，而 `for` 不可能忘。而且 `index` 顺便就用于 `Step.index`。

**为什么把 `steps` 和 `messages` 都返回？**

* `messages` 是**模型看到的**（调试上下文用）
* `steps` 是**结构化的执行记录**（统计、断言、UI 渲染用）

从 `messages` 反推执行过程是可能的但很别扭（要配对 assistant/tool）。两者都暴露最省事。

**为什么 `Agent` 持有 `trace` 状态（而不是无状态）？**

为了能 `agent.run(...)` 之后再 `agent.trace` 查看。代价是同一个实例不能并发 `run`——对这个用途无所谓，但如果要做异步网关，应该改成每次 `run` 返回自己的 trace（现在的 `AgentResult.steps` 其实就是）。

---

## 常见误解

* 「`max_steps` 是工具调用次数上限」——是**模型轮次**上限。一轮可以调多个工具。
* 「重复检测会误伤连续算数」——不会，判据包含参数。
* 「解析失败会直接报错退出」——会重试 `max_parse_errors` 次。
* 「assistant 消息只存文本」——必须带 `tool_calls`，否则模板渲染出的上下文是错的。
* 「`FINAL_ANSWER` 就代表成功」——还要 `answer` 非空，见 `AgentResult.succeeded`。

---

## 动手练习

1. 把 `messages.append(Message(Role.ASSISTANT, parsed.text, tool_calls=parsed.calls))` 里的 `tool_calls` 去掉，用 `--backend local` 跑，观察模型如何重复调用。
2. 把 `max_identical_calls` 改成 5，跑 `--backend scripted-error`，数数走了几步才停。
3. 在 `_emit` 里加一行 `print(len(messages))`，观察消息数是**只增不减**的——这解释了为什么上下文会越滚越长。
4. 加一条护栏 `max_total_tokens`：累计 `len(message.content)` 超过阈值就停。想想它和 `max_steps` 各防的是什么。
