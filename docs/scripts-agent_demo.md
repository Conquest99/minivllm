# `scripts/agent_demo.py` 详解

## 这个文件干什么

**把 agent 的每一步打印出来。** 三种后端，从确定性到真实模型。

```powershell
python scripts\agent_demo.py                            # 脚本后端，0.2 秒看完整循环
python scripts\agent_demo.py --backend scripted-error   # 看工具错误如何被回灌
python scripts\agent_demo.py --backend local             # 真实 0.5B 模型
python scripts\agent_demo.py --backend api --model deepseek-chat
```

---

## 1. 为什么默认是 `scripted`

因为**读者第一次要理解的是循环的形状，不是模型的能力**。

用 0.5B 跑 demo，你会看到：

```
step 0 (1268 ms)  call  calculator({"expression": "23 * 7"}) -> 161
step 1 (587 ms)   model: The result of 23 * 7 is 161.
```

两秒，两次 GPU 调用，而且**你无法确定「只调了一次工具」是设计如此还是模型偷懒**。

用脚本后端，每一步都是设计出来的：

```
step 0  call   calculator({"expression": "23 * 7"})         -> 161
step 1  call   calculator({"expression": "(161 + 11) / 4"}) -> 43
step 2  model: The answer is 43.
```

**零耗时、完全确定、能对着屏幕读伪代码。**

---

## 2. `tool_call` 辅助函数

```python
def tool_call(name: str, **arguments) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"
```

`**arguments` 让调用处写起来像 Python 调用：

```python
tool_call("calculator", expression="23 * 7")
```

而不是：

```python
'<tool_call>{"name": "calculator", "arguments": {"expression": "23 * 7"}}</tool_call>'
```

**测试文件里也有同名函数**（`tests/test_agent_loop.py`）。重复是有意的：demo 和测试不该互相 import——否则改 demo 的展示逻辑会连带影响测试。

---

## 3. 三个后端各自的用途

### `scripted`：展示「正常路径」

```python
def scripted_backend() -> ScriptedBackend:
    return ScriptedBackend(
        [
            tool_call("calculator", expression="23 * 7"),
            tool_call("calculator", expression="(161 + 11) / 4"),
            "The answer is 43.",
        ]
    )
```

注意第二个调用**硬编码了 `161`**——它假装看到了第一次的结果。真实的脚本后端不会读观察，这正是它的局限。

### `scripted-error`：展示「错误恢复」

```python
def scripted_backend_with_error() -> CallableBackend:
    def decide(messages):
        observations = [m.content for m in messages if m.role is Role.TOOL]
        if not observations:
            return tool_call("calculator", expr="23 * 7")   # 故意传错参数名

        last = observations[-1]
        if "unexpected argument" in last:
            return tool_call("calculator", expression="23 * 7")

        good = [o for o in observations if "ERROR" not in o]
        if len(good) == 1:
            value = good[0].split()[-1]
            return tool_call("calculator", expression=f"({value} + 11) / 4")
        return "The answer is 43."

    return CallableBackend(decide)
```

**这个函数是整个 demo 里最值得读的部分。** 它模拟了一个「会从错误中学习的模型」，逻辑分四段：

1. 还没有观察 → 故意写错参数名 `expr`
2. 上一条观察是参数错误 → 改对
3. 只有一条成功结果 → 用它算下一步
4. 两条都有 → 收尾回答

跑出来的 trace：

```
step 0  call   calculator({"expr": "23 * 7"})
        !! result ERROR: unexpected argument(s) ['expr'] for calculator; allowed: expression
step 1  call   calculator({"expression": "23 * 7"})   -> 161
step 2  call   calculator({"expression": "(161 + 11) / 4"})   -> 43
step 3  model: The answer is 43.
```

**注意 `!!` 和 `->` 的区别**——就是我用来区分失败/成功的视觉标记。

### 写这段时踩的坑

第一版用 `observations[0]` 判断是否有参数错误。结果：

```
step 1  call  calculator({"expression": "23 * 7"})  -> 161
step 2  call  calculator({"expression": "23 * 7"})  -> 161
step 3  call  calculator({"expression": "23 * 7"})  -> 161
        stopped: the model kept calling calculator with identical arguments
```

因为**消息列表是累积的**，`observations[0]` 永远是第一步那条错误信息，条件永远成立，于是无限重复。

修法是看 `observations[-1]`（最新一条）。

> **这是 agent 编程的典型 bug**：上下文只增不减，所有「第一条」「前一条」的判断都要格外小心。而且这个错误**不会报异常**，只会表现为「模型莫名其妙地重复」——非常难查。

---

## 4. `print_step`：trace 的渲染

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

五个信息层次，每个都可能为空：

| 层次 | 什么时候有 |
|---|---|
| 步号 + 耗时 | 总是有 |
| `model:` 散文 | 模型说了话（可能同时有工具调用） |
| `call` | 有工具调用 |
| `-> / !! result` | 有工具调用 |
| `parse error` | 解析失败 |

排版上有个细节：`call` 和 `result` 用**两行**而不是一行，因为参数 JSON 可能很长，挤在一行会折断。

`json.dumps(call.arguments, ensure_ascii=False)` 而不是 `str(...)`：这是**合法的 JSON**，可以直接复制到别的工具里验证。

---

## 5. 参数设计

```python
parser.add_argument("--backend", default="scripted", choices=["scripted", "scripted-error", "local", "api"])
parser.add_argument("--question", default=QUESTION)
parser.add_argument("--model", default="deepseek-chat")
parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
parser.add_argument("--device", default="cuda")
parser.add_argument("--dtype", default="float16")
parser.add_argument("--max-steps", type=int, default=6)
parser.add_argument("--verbose", action="store_true")
```

* **`choices=[...]`**：拼错后端名立刻报错，还能在 `--help` 里列出选项。
* **`--max-steps 6`** 而不是默认的 8：demo 想快速看到「步数不够会怎样」，6 更容易触发。
* **`--verbose`**：打印最终消息列表。想知道「模型到底看到了什么」时用它。

---

## 6. 沙箱位置

```python
default_tools(Path(__file__).resolve().parents[1] / "agent_workspace")
```

从脚本位置推 `mini_vllm/agent_workspace`（`parents[1]` = 仓库根），和 `conftest.py`、`Test.py` 里用的是同一个技巧。这样在**任何目录**运行都指向同一个沙箱，不会因为在 `scripts/` 里跑就把工作区建到 `scripts/agent_workspace`。

---

## 7. 退出码

```python
if result.stopped_reason is not StopReason.FINAL_ANSWER:
    sys.exit(1)
```

非零退出码 = 可以被 CI / shell 脚本判断。`scripted-error` 正常结束返回 0，`max_steps` 或 `loop_detected` 返回 1。

---

## 8. 真实模型跑出来的两种结果

### 能干的

```
问题: Calculate 23 * 7 with the calculator, then tell me the answer.

step 0 (1268 ms)  call  calculator({"expression": "23 * 7"})  -> 161
step 1 (587 ms)   model: The result of 23 * 7 is 161.

stopped: final_answer   tool calls: 1 (0 failed)   wall: 1.86s
```

0.5B 正确发出了 `<tool_call>`，并使用了观察结果。

### 干不了的

```
问题: What is (23 * 7 + 11) / 4?

step 0 (1552 ms)  call  calculator({"expression": "23 * 7 + 11 / 4"})  -> 163.75
step 1 (1168 ms)  model: The result of the expression (23 * 7 + 11) / 4 is 163.75.
```

**括号丢了。** 模型把 `(23 * 7 + 11) / 4` 抄成了 `23 * 7 + 11 / 4`，工具忠实执行，模型自信地报出错误答案。

这个结果非常有教学价值：

* 循环对、工具对、格式对
* **错在模型的算术/复制能力**
* 并且它**不会自己发现矛盾**（同一个问题在 prompt 里写着括号，它抄的时候丢了）

这就是「0.5B 只能验机制，不能干活」的具体证据。要上生产就换 API 或 ≥7B。

> 面试讲这段比讲「我写了个 agent」有价值得多：它能证明你真的跑过、真的观察过边界。

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `argparse` 的 `choices` | 后端名枚举校验 |
| `**arguments` 收集关键字参数 | 让 `tool_call` 调用处像 Python 函数 |
| `Path(__file__).resolve().parents[1]` | 从脚本位置推仓库根 |
| `json.dumps(..., ensure_ascii=False)` | 中文参数不转义 |
| `zip(..., strict=False)` | 调用与结果配对 |
| `sys.exit(1)` | 非零退出码给 CI 用 |
| 函数内 import（`local`/`api` 分支） | 只有需要时才加载 torch / 网络栈 |

---

## 设计取舍

**为什么不做成 `pytest` 而是一个脚本？**

测试断言「行为正确」，脚本展示「过程长什么样」。**两种目的不该混在一个入口**。测试里你不会想看 trace，demo 里你不想写断言。

**为什么 `scripted-error` 用 `CallableBackend` 而不是 `ScriptedBackend`？**

因为它必须读观察结果才能决定下一步。固定台词的版本写不出「如果参数错了就改对」这种条件逻辑（除非把两套都写死，那就不演示恢复了）。

**为什么 `print_step` 直接 `print` 而不是用 `logging`？**

demo 的输出就是给人看的。`logging` 需要配置 formatter 才能达到同样的可读性，而且默认会带时间戳噪音。真要做可观测性，应该换成 OTel span（见 `on_step` 的说明）。

---

## 常见误解

* 「`scripted` 后端在假装模型」——它是**假装一个理想模型**，用来验证机制。
* 「`--backend local` 会下载模型」——不会，用本地 `models/` 目录，找不到才联网。
* 「`agent_workspace` 是临时目录」——是仓库根下的固定目录，会被 `.gitignore` 吗？没有，目前它是空的所以不影响。
* 「exit code 1 表示程序崩了」——表示 agent 没给出最终答案。

---

## 动手练习

1. 把 `scripted_backend` 的第三步改成一个新的工具调用，观察 `max_steps` 如何触发。
2. 在 `print_step` 里加一行打印 `len(step.tool_results)`，拿它和 `len(step.tool_calls)` 对比。
3. 用 `--backend local --question "现在几点？"`（没有 `now` 工具），观察模型会不会硬编一个工具名——这就是 `unknown tool` 护栏的场景。
4. 加一个 `--dry-run` 参数，只打印将要发送给模型的 prompt 而不真正调用（提示：用 `CallableBackend` 把 `messages` 打印出来然后返回 `"stop"`）。
