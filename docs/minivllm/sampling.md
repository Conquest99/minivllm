# `minivllm/sampling.py` 详解

## 这个文件干什么

把模型输出的 **logits（词表长度 151936 个分数）** 变成 **1 个 token id**。

它是整个引擎里唯一与模型无关的计算部件：输入只有一个一维向量，所以可以完全在 CPU 上单测。

```python
token_id = sample(logits[row], params, generator)
# logits: [vocab] → int
```

---

## 逐块讲解

### `SamplingParams`：一次采样的全部配置

```python
@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 64
    ignore_eos: bool = False
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
```

| 字段 | 含义 | 默认值的用意 |
|---|---|---|
| `temperature` | 0 = 贪心（永远取最大）；>0 越高越随机 | 默认 0，**保证可复现**，测试才好写 |
| `top_p` | nucleus 采样：只保留累计概率前 p 的候选 | 1.0 = 不过滤 |
| `top_k` | 只保留概率最高的 k 个 | **-1 表示关闭**（用 -1 而不是 0，因为 0 是合法的「一个都不要」，虽然没意义） |
| `max_tokens` | 最多生成多少个 | 64，够 demo 又不会跑太久 |
| `ignore_eos` | 不把 EOS 当停止符 | **压测必须开**，否则模型提前结束，测出的吞吐不可比 |
| `stop` | 遇到这些字符串就停 | 在**解码后的文本**上匹配，不是 token id |
| `seed` | 单条请求的种子 | 目前引擎统一用 `torch.Generator`，这个字段暂未使用 |

```python
    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
```

校验的是**会造成无意义或崩溃**的取值：

* `temperature = 0` 是合法且常用的（贪心），但负数无意义（负温度会翻转概率排序）。
* `top_p = 0` 会让所有候选被筛掉，`softmax` 全是 `-inf` → NaN。所以必须 `> 0`。
* `max_tokens = 0` 会让调度器认为它一开始就结束。

```python
    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0
```

用**精确等于 0** 判定，不是 `< 0.01`。理由：贪心和采样是两条完全不同的代码路径，容差判断会让「0.001 温度」变得不可预期。想贪心就明确写 0。

### `sample()`：主函数

```python
@torch.inference_mode()
def sample(logits: torch.Tensor, params: SamplingParams, generator=None) -> int:
```

* `@torch.inference_mode()`：不建计算图。推理不需要梯度，这是白给的加速。
* 返回 Python `int` 而不是 0 维张量。这个选择有代价（会触发一次 GPU→CPU 同步），但换来调用侧极简：`seq.data.append(token_id)` 直接可用。每条序列每步只同步一次，这是可接受的。

#### 贪心分支

```python
if params.greedy:
    return int(torch.argmax(logits).item())
```

`argmax` 返回最大值的**下标**（即 token id）。`int(...item())` 转成 Python int。

#### 温度缩放

```python
logits = logits.float() / params.temperature
```

两个动作：

1. **`.float()` 升到 fp32**。模型输出可能是 fp16，而 fp16 在 `-inf`、极小概率这些边界上会溢出/下溢。采样这点计算量用 fp32 完全可以承受。
2. **除以温度**。温度 < 1 → logits 差异被放大 → 分布更尖锐（更确定）；> 1 → 差异被压平 → 更随机。

> 数学上这就是把 `softmax(z)` 换成 `softmax(z/T)`。

#### top-k

```python
if params.top_k > 0:
    k = min(params.top_k, logits.numel())
    kth = torch.topk(logits, k).values[-1]
    logits = logits.masked_fill(logits < kth, float("-inf"))
```

* `torch.topk(x, k)` 返回 `(values, indices)`；`.values[-1]` 是**第 k 大**的那个值（阈值）。
* 把所有**严格小于**阈值的打成 `-inf`。
* 用 `<` 而不是 `<=`：如果第 k 和第 k+1 名分数恰好相等，用 `<=` 会把两者都删掉，只剩 k-1 个。用 `<` 则至少保留 k 个（可能多保留并列项）。

#### top-p（nucleus）

```python
if params.top_p < 1.0:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    cutoff = cumulative - probs > params.top_p
    sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
    logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)
```

这一步逻辑最密，拆开看：

1. **排序**：`sort(descending=True)` 同时给出排序后的值和原始下标。必须在**排序后**做累计，因为 top-p 的定义是「从最可能的开始累加」。
2. **转概率**：`softmax` 沿最后一维（词表维）。
3. **累计和**：`cumsum` 给出「前 i 个（含自己）的概率和」。
4. **判据** `cumulative - probs > top_p`：这是「**不含自己**的累计概率」是否已经超过 p。
   * 含义：加入这个候选**之前**就已经够了 → 这个候选以及后面全部丢掉。
   * 这样保证**至少保留 1 个** token。若写成 `cumulative > top_p`，当 p 很小时连第一个都会被删掉，概率全变 `-inf` → NaN。
5. **还原顺序**：`sort` 打乱了位置，所以要用 `scatter(0, sorted_idx, sorted_logits)` 把过滤后的值放回原下标。
   * `torch.full_like(logits, -inf)` 先造一个全 `-inf` 的模板（**默认全屏蔽**），再把保留的写回去。这个顺序很重要：反过来（先复制再屏蔽）容易漏掉位置。

#### 最终采样

```python
probs = torch.softmax(logits, dim=-1)
return int(torch.multinomial(probs, num_samples=1, generator=generator).item())
```

* `softmax` 再归一化一次：前面把一些位置打成 `-inf`，剩下的概率和不再是 1。
* `multinomial(p, num_samples=1)` 按概率抽 1 个，返回下标（= token id）。
* `generator` 传进来是为了可复现：`torch.Generator` 自己维护随机状态，不受全局 `torch.manual_seed` 干扰。

---

## 关于 `torch.multinomial` 的两个坑

1. **它要求输入非负且和为 1（或至少非负有限）**。如果 `softmax` 之前全是 `-inf`（所有候选都被 top-p 干掉），`softmax` 会输出 NaN，`multinomial` 报错或给出垃圾。上面 `cumulative - probs` 的写法正是为了避免这种情况。
2. **`num_samples=1` 时它的性能其实一般**。真正的高性能实现会按 batch 一次性采样（一次 kernel 处理 B 条序列）。本项目为了可读性一条一条采——在 24 层前向面前，这点开销可以忽略。

---

## 用到的 PyTorch 特性

| API | 作用 | 注意 |
|---|---|---|
| `torch.argmax(x)` | 最大值的下标 | 返回张量，需 `.item()` |
| `x.float()` | 升 fp32 | 采样精度需要 |
| `x / scalar` | 逐元素除 | 温度缩放 |
| `torch.topk(x, k)` | 前 k 大（值+下标） | `.values[-1]` 是阈值 |
| `x.masked_fill(cond, val)` | 条件填常数 | 非原地，返回新张量 |
| `torch.sort(x, descending=True)` | 排序 + 原下标 | 下标用于还原 |
| `torch.softmax(x, dim=-1)` | 归一化 | 必须指定 dim |
| `torch.cumsum(x, dim=-1)` | 累计和 | top-p 用 |
| `torch.full_like(x, v)` | 同形状全 v | 造 `-inf` 模板 |
| `x.scatter(dim, idx, src)` | 按下标写回 | 还原排序 |
| `torch.multinomial(p, n, generator=)` | 按概率抽样 | 需要合法概率分布 |
| `torch.inference_mode()` | 不建图 | 推理必备 |

---

## 设计取舍

**为什么贪心单独一条路径，而不是「温度设成极小」？**

极低温下 `logits / 0.0001` 会溢出成 `inf`，`softmax` 出 NaN。而贪心是精确、免费、无溢出风险的。**能特判就别靠数值极限。**

**为什么 top-k 和 top-p 都实现，而不是只留一个？**

它们的形状不同：top-k 固定候选数量，top-p 按概率质量动态调整数量。生产引擎两个都支持（可以同时用：先 top-k 再 top-p）。本项目保留两者，是因为在面试里「top-p 和 top-k 的区别」是个高频问题，有代码好解释。

**为什么 `sample()` 只处理单条序列？**

为了可测试、可读。真实引擎的采样是 batch 化的（一次 `multinomial` 处理 B 个分布，甚至融合进 kernel）。这里的取舍和 gather 一样：**牺牲吞吐换可读性**。

---

## 常见误解

* 「temperature=0 就是让 logits 不变」——不，是直接 `argmax`，根本不走 softmax。
* 「top-p=1 等于不过滤」——数学上是的（所有 token 都保留），但代码里 `if params.top_p < 1.0` 会直接跳过整段计算，所以还省了排序开销。
* 「`stop` 是按 token 匹配的」——不，是按**解码后的字符串**匹配。因为一个词可能被切成多个 token，按 token 匹配会漏。
* 「`ignore_eos` 只是测试用」——实际压测、吞吐基准都必须开，否则长度不一致的对比毫无意义。

---

## 动手练习

1. 构造 `logits = torch.tensor([1., 1., 1., 1.])`，`top_p=0.5`，手动推演一遍 `sample()`，验证最终只保留了 1 个候选（每个概率 0.25，累计到第 2 个就是 0.5，`cumulative - probs > 0.5` 在第 3 个才成立）。
2. 把 `cumulative - probs > params.top_p` 改成 `cumulative > params.top_p`，用 `top_p=0.1` 跑一次，观察是否退化成「全 `-inf`」。
3. 把 `masked_fill(..., float("-inf"))` 改成 `masked_fill(..., 0.0)`，看采样分布怎么变（会变成「所有被过滤的 token 权重相同」）。
