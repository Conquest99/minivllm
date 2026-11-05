# `serve/` 详解：把 agent 变成一个能一直聊的服务

## 这个文件干什么

前面所有 demo 都是「起一次、问一句、退出」。真实使用不是这样：你在浏览器里打字，它
流式地回，你还想接着问上一句里的东西。这一层就补这三件事：

| 问题 | 答案在哪 | 关键决定 |
|---|---|---|
| 上下文放哪 | `serve/sessions.py` | 放服务端，浏览器只拿一个 `session_id` |
| 怎么流式 | `serve/app.py` + `minivllm/llm_engine.py` | 引擎吐 token 增量，Agent 原样转发 |
| 模型谁独占 | `serve/app.py` 的 `model_lock` | 一次一模型调用，不是一整轮独占 |

```
scripts/serve_chat.py         装配：哪个后端、哪些工具、什么提示词
        ↓
serve/app.py                  POST /chat → SSE；事件翻译；全局模型锁
serve/sessions.py             会话表 + 按轮裁剪
serve/static/index.html       一个能用的聊天界面（fetch + 手写 SSE 解析）
        ↓
miniagent.Agent.run(question, history=...)     ← 新增的 history 参数
        ↓
MiniVLLMBackend.complete(..., on_token=...)    ← 新增的可选回调
        ↓
LLMEngine.generate_stream(prompt)              ← 新增的生成器
```

---

## 1. 上下文：没有任何新机制，只是「把上一轮的 messages 传回去」

`Agent.run` 本来就把整段对话记在 `result.messages` 里。所以多轮对话所需的一切就是：

```python
result = agent.run(question, history=session.messages)     # 接着上一轮写
session.messages = result.messages                          # 存回去
```

`run` 里对应的三行（`miniagent/agent.py`）：

```python
messages: list[Message] = list(history) if history else []
if not messages or messages[0].role is not Role.SYSTEM:
    messages.insert(0, Message(Role.SYSTEM, self.system_prompt))
messages.append(Message(Role.USER, question))
```

用 `list(history)` 复制一份，是为了避免调用方以为「我传进去的列表会被就地改写」。

### ⚠️ 这里发现并修掉了一个真 bug

第一次跑多轮就暴露了：**模型的最终答案从来没有被写进 `messages`。**

`run()` 在「模型直接给了散文」那个分支里是这么走的：

```python
steps.append(step)
self._emit(step)
return result(parsed.text.strip() or reply.strip(), reason)     # ← 就这样返回了
```

前两条 `continue` 分支（写失败重试、读回重试）都会 `messages.append(Message(Role.ASSISTANT, reply))`，
只有最终返回那条没有。后果是：**下一轮模型只看得见工具结果，看不见自己上一轮回答了什么。**

单轮 demo 永远发现不了，因为没有任何东西会去看那份记录。多轮一跑，问「我刚才问了什么」，
模型只能靠工具结果去猜。

修法就是补上那一条 append。这是这个项目里**最容易漏掉、也最难在单轮里发现**的一类 bug。

### 裁剪必须按「轮」，不能按「条」

`trim_history` 只保留最近 N 轮，而且**整轮进出**：

```python
def split_turns(messages):
    turns = []
    for m in messages:
        if m.role is Role.USER or not turns:
            turns.append([m])
        else:
            turns[-1].append(m)
    return turns
```

为什么不能按条数砍：一条 `tool` 消息只有在它前面那条带 `tool_calls` 的 `assistant` 存在时
才有意义。按条裁剪会把它们拆散，而 **Qwen 的模板遇到孤立的 `tool` 消息不会报错，只会静默
降级成一条 `user` 消息** —— 模型于是看到「用户发来一段 `<tool_response>`」，却完全不知道
自己调用过什么。

> 又是「静默降级」这个家族：模板不报错，只是悄悄改变了语义。测试
> `test_trim_never_leaves_an_orphan_tool_message` 锁的就是这条。

---

## 2. 流式：三层穿透，每层只加一个参数

### 2.1 引擎层：`generate_stream`

`generate()` 是阻塞到底的。流式要的是「每生成一个 token 就吐出来」，所以在 `step()` 之间
把新文本交出去：

```python
def generate_stream(self, prompt, params=None) -> Iterator[str]:
    self.add_request(prompt, params)
    pending: list[str] = []

    def sink(_seq_id: int, delta: str) -> None:
        if delta:
            pending.append(delta)

    self._token_sink = sink
    while self.scheduler.has_unfinished():
        for finished in self.step():
            self.last_output = finished        # step() 会清空输出表，必须当场接住
        if pending:
            yield "".join(pending)
            pending.clear()
```

`_postprocess` 里每 append 一个 token 就调一次 sink。增量怎么算：

```python
def _new_text(self, seq):
    text = self._decoded_output(seq)          # 解码「到目前为止的全部输出」
    if text.endswith("\ufffd"):               # 解码边界落在字符中间：先扣住不发
        text = text[:-1]
    sent = self._sent_chars.get(seq.seq_id, 0)
    if len(text) <= sent:
        return ""
    self._sent_chars[seq.seq_id] = len(text)
    return text[sent:]
```

**两个坑，都是实测踩出来的：**

1. **不能逐 token 解码。** 一个常用汉字常被切成几个 byte-level token，单独解码会得到
   `U+FFFD`（替换符）。所以必须解码「完整前缀」，再做差。
2. **差值必须按字符算，而且末尾的 `U+FFFD` 要扣住。** 真字和替换符都是「一个字符」，
   如果按长度差下发，替换符先被发出去，下一个 token 补齐后真字**长度没变**，就永远发不
   出去了。实测症状：`函数` 变成 `\ufffd数`。

测试 `test_streaming_matches_the_blocking_output` 锁死这条不变量：
**增量拼接 == 非流式结果 == `engine.last_output.text`**。

### 2.2 后端层：`complete(..., on_token=None)`

为什么不另开一个 `stream()` 方法：agent 需要的永远是「这一轮的完整文本」（要拿去
`parse`），增量只是副产品。多一个方法就得在每个调用点写两套分支；多一个参数，不关心的
后端直接忽略。

| 后端 | `on_token` 给了之后 |
|---|---|
| `MiniVLLMBackend` | 走 `generate_stream`，真流式 |
| `ScriptedBackend` / `CallableBackend` | 整段作为一块发出去（所以流式路径能在纯 CPU 上测） |
| `OpenAICompatBackend` | 请求里加 `stream: true`，解析 SSE，边收边转发 |

最后一个值得单独说：**OpenAI 流式响应里的工具调用是分片的。** 每个 delta 带一个 `index`
和 `function.arguments` 的一个*片段*，必须先按 index 拼起来才能解析 JSON：

```python
for call in delta.get("tool_calls") or []:
    slot = slots.setdefault(int(call.get("index", 0)), {"name": "", "arguments": ""})
    if function.get("name"):
        slot["name"] = function["name"]
    if function.get("arguments"):
        slot["arguments"] += function["arguments"]
```

漏了拼接，`{"a": 1}` 会变成半截字符串，然后被 parser 报成「工具调用写坏了」—— 错误指向
parser，真正的罪犯在网络层。

### 2.3 服务层：SSE + 一个工作线程

`agent.run` 是阻塞的，而生成器每 `yield` 一次就要把控制权还给 Starlette。所以：

```python
events: queue.Queue = queue.Queue()

def work():                       # 工作线程里跑 agent
    ... emit(...) ...

yield sse("start", ...)
threading.Thread(target=work, daemon=True).start()
while True:
    item = events.get()           # 阻塞在这里，但阻塞的是这个工作线程池里的线程
    if item is None:
        break
    yield sse(*item)
```

不开这个线程，所有事件只能在 `run` 返回后一次性吐出，**流式就名存实亡**。

帧格式只有一个坑，但很致命：

```python
def sse(event, payload):
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
```

必须 `json.dumps`。手拼的话，多行内容（工具结果就是多行的，表格输出）里的裸换行会让
`data:` 提前结束这一帧，客户端只拿到第一行。`test_sse_never_emits_a_bare_newline_inside_a_frame`
锁这条。

---

## 3. 事件协议

| 事件 | 什么时候 | 载荷 |
|---|---|---|
| `start` | 立刻 | `session_id`, `turn` |
| `token` | 模型每生成一段文本 | `text` |
| `step` | 一轮模型调用结束 | `index`, `duration_s` |
| `call` | 解析出一个工具调用 | `name`, `arguments` |
| `result` | 工具执行完 | `name`, `ok`, `content`, `duration_s` |
| `parse_error` / `note` | 解析失败 / 循环层的判定 | `message` |
| `done` | 正常收工 | `answer`, `stopped`, `steps`, `tool_calls`, `duration_s` |
| `error` | 这一轮没跑成 | `type`, `message` |

两条约定：

* **流一定以 `done` 或 `error` 结束**，客户端不必处理「连接正常关了但没结论」。
* **后端的崩溃是 `error` 事件，不是断流。** `Agent` 内部把后端异常收成了一条
  `StopReason.BACKEND_ERROR` 结果，服务层再把它翻译成 `error`——否则客户端会看到
  `done` 配一句读起来像答案的 "backend error: ..."。

---

## 4. 并发：一个进程一份权重

```python
model_lock = threading.Lock()

def serialized_complete(*, on_token, **kwargs):
    with model_lock:
        return backend.complete(on_token=on_token, **kwargs)
```

锁的粒度是**一次模型调用**，不是一整轮对话。所以两个会话可以交错：A 等工具执行时，
B 能拿到模型。整轮独占会让第二个用户等到第一个人把 agent 跑完（含所有工具调用）。

### 这里最该被替换的地方

`LLMEngine` 的 `step()` 本来就是批处理的入口 —— 它能一次跑多条序列。但这个服务没法利用
它，因为 **Agent 循环是同步请求/响应**：`complete()` 进去，等一句完整回复出来。

要把并发真正用起来，得把服务端拆成两层：

```
     HTTP 请求  ──►  agent 循环（每个会话一个协程，在等工具 I/O 时让出）
                            │  提交 token 请求
                            ▼
                     推理调度器（收集当前所有在等的请求 → 一次 step() 批量跑）
                            │  增量
                            ▼
                     分发给各自的流
```

也就是说：**连续批处理要的「多个请求同时在飞」，和 agent 循环的「一次一问一答」是矛盾
的**，中间需要一层异步调度。这正是 vLLM 的 `AsyncLLMEngine` 在做的事，也是这个教学服务
刻意没做的那一层——它现在只能串行，吞吐由模型延迟直接决定。

---

## 5. 跑起来

```powershell
# 装依赖（一次性）
uv pip install --python .venv\Scripts\python.exe -e ".[serve]"

# 本地 0.5B + 只读 SQL 工具
python scripts\serve_chat.py
# 打开 http://127.0.0.1:8000

python scripts\serve_chat.py --tools default        # 计算器 + 文件沙箱
python scripts\serve_chat.py --backend scripted     # 假模型，不用 GPU
python scripts\serve_chat.py --max-turns 3          # 只留 3 轮上下文
```

`--max-turns` 调小的实际用途：**上下文超长时**。引擎会在 `add_request` 里抛
`prompt + max_tokens exceeds max_model_len`，服务层把它变成一条 `error` 事件，界面上
会提示「点新对话，或把 `--max-turns` 调小」。

---

## 用到的 Python / 特性

| 特性 | 说明 |
|---|---|
| `threading.Thread` + `queue.Queue` | 把阻塞的 `agent.run` 和异步流式接起来 |
| `StreamingResponse` + 同步生成器 | Starlette 会把同步生成器丢进线程池迭代，所以生成器里可以阻塞 |
| `dataclass` 的 `field(default_factory=...)` | `lock`/`messages` 不能做类属性默认值（可变/共享） |
| `OrderedDict.move_to_end` | 会话表的 LRU |
| SSE（`text/event-stream`） | 比 WebSocket 简单：单向、纯文本、任何代理都认 |
| `X-Accel-Buffering: no` | 让 nginx 之类不要攒着不发，否则 SSE 会被攒成一坨 |

---

## 设计取舍

**为什么上下文放服务端，不放客户端？**
放客户端意味着每轮把整段对话重发一遍（token 成本 ×2），而且历史可被篡改——tool 结果是
信任链的一环，不能由客户端提供。

**为什么用 SSE 不用 WebSocket？**
这一层的通信是单向的（服务器 → 浏览器）。WebSocket 要做双向、要处理心跳和重连协议；
SSE 就是一个不结束的 HTTP 响应，`fetch` 就能读。

**为什么会话表只在内存里？**
单进程教学服务，重启全丢是合理取舍。要持久化就换 Redis，接口不用改（`get/drop/create`）。

**为什么 `on_token` 挂在 `complete` 上，而不是搞一个 `StreamingBackend` 协议？**
`Protocol` 一旦分叉，Agent 里就要写「有 stream 就用 stream，否则退化成 complete」的
类型嗅探。一个可选参数把这件事变成了「传了就流，没传就整段」，调用方一行不用改。

---

## 常见误解

* 「流式 = 边生成边返回**最终答案**」——不，agent 的流里 90% 的时间在工具调用和等待。
  UI 必须能显示「正在调 `query_table`」，否则用户看着空白等 8 秒。
* 「多轮上下文需要 memory 模块」——不需要。`messages` 列表就是 memory，`run(history=...)`
  就是全部机制。
* 「流式输出和完整输出可能有细微差别」——不该有。增量拼接必须严格等于非流式结果，
  这一条有测试锁。
* 「按条数裁剪历史就行」——会把 `tool` 消息和它的 `assistant` 拆散，模板静默降级。
  必须按轮裁。
* 「有了 `model_lock` 就不能并发了」——能并发的是**会话**，不能并发的是**模型调用**。
  锁的粒度决定了这一点。

---

## 动手练习

1. 把 `model_lock` 去掉，开两个浏览器标签同时发问，看 PyTorch 报什么错。然后想：
   这个错为什么是「段错误 / 显存不够」而不是「数据竞争」的清晰提示？
2. 把 `_stream_turn` 里的工作线程去掉，直接在生成器里 `agent.run(...)`，用浏览器看：
   事件是「一个一个到」还是「一次性刷出来」？
3. 把 `trim_history` 改成按条数裁剪（`messages[-8:]`），构造一个多轮对话，然后在
   `scripts/serve_chat.py` 里打印每次发给模型的 messages，找那条孤立的 `tool` 消息。
4. 在 `_new_text` 里去掉 `if text.endswith("\ufffd")` 那两行，用中文提示词跑一次流式，
   找输出里的 `\ufffd`。
5. 给 `/chat` 加一个 `?include_tokens=false` 参数，让不需要流式的客户端走非流式路径
   （`on_token=None`），对比首字节时间和总耗时。
