# `tests/conftest.py` 详解

## 这个文件干什么

pytest 的**共享夹具（fixture）与路径设置**。文件名 `conftest.py` 是 pytest 的约定：它会被自动加载，里面的 fixture 对整个目录下的测试可见，不需要 import。

本文件只有 45 行，但决定了「测试跑在哪、用多大的池子、什么时候跳过」。

---

## 1. 把仓库根塞进 `sys.path`

```python
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
```

和 [scripts/_workload.py](scripts-_workload.md) 里那两行同理：让 `import minivllm` 在没 `pip install -e .` 时也能工作。

`parents[1]` 是因为本文件在 `tests/` 下：

```
.../mini_vllm/tests/conftest.py
      parents[0] = tests
      parents[1] = mini_vllm      ← 仓库根
```

> 注意：`conftest.py` 是在**收集阶段**最早被加载的文件之一，所以在这里插路径能保证所有测试都能 import 到包。

---

## 2. 两个辅助函数

```python
def model_path() -> str:
    if (LOCAL_MODEL / "config.json").exists():
        return str(LOCAL_MODEL)
    return os.environ.get("MINIVLLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def require_local_model() -> str:
    if not (LOCAL_MODEL / "config.json").exists():
        pytest.skip(f"no local checkpoint at {LOCAL_MODEL}")
    return str(LOCAL_MODEL)
```

* **`model_path()`**：有本地就本地，没有就尝试 hub id（会联网）。
* **`require_local_model()`**：没有本地就 **skip**，而不是失败。

### 为什么区分「skip」和「fail」

因为「没下载模型」不是代码 bug。CI 或新克隆的仓库里没跑 `download_model.py` 时，让 12 个测试报红是噪音；skip 则会明确告诉你「这些用例没执行」。

**但 skip 有个风险**：容易变成「测试一直没跑还以为通过了」。所以本项目把不需要模型的测试单独组织（`test_block_manager.py`、`test_scheduler.py`、`test_paged_attention.py`），那 25 个永远会跑。

检查 `config.json` 而不是 `model.safetensors`，因为 `config.json` 是最小的必需文件，存在性检查最快。

---

## 3. 最重要的设计：`runner` fixture

```python
@pytest.fixture(scope="session")
def runner():
    import torch

    from minivllm.config import EngineConfig
    from minivllm.model_runner import ModelRunner

    device = os.environ.get("MINIVLLM_TEST_DEVICE", "cpu")
    dtype = "float32" if device == "cpu" else "float16"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = EngineConfig(model=require_local_model(), dtype=dtype, device=device, num_gpu_blocks=64)
    return ModelRunner(cfg)
```

### `scope="session"`

**整个测试会话只创建一次**。这很关键：加载模型要 3-5 秒，如果每个测试都重建，10 个测试就是 50 秒。

代价是**测试之间可能相互污染**（比如某个测试改了 `runner.k_cache`）。所以这里用了一个约束：

```python
num_gpu_blocks=64
```

固定 **64 块**，而不是按显存自动算。原因：

1. 可复现（两次跑结果一样）
2. 小池子更容易暴露越界 bug
3. 测试里每个用例自己建 `BlockManager`，不共享页表

### 为什么默认 CPU + float32

```python
device = os.environ.get("MINIVLLM_TEST_DEVICE", "cpu")
dtype = "float32" if device == "cpu" else "float16"
```

三个理由：

| | CPU/fp32 | GPU/fp16 |
|---|---|---|
| 显存 | 不占（可以和别的程序共存） | 要 1 GB+ |
| 数值稳定 | fp32，可以直接用 `atol=1e-4` | fp16 误差更大，阈值得放宽 |
| 速度 | 0.5B 模型 CPU 上一次前向约 1-2 秒 | 约 50 ms |

**测试的价值在于「确定性」，不在于「快」。** fp32 让「分页实现 vs HF 实现」的差异能干净地归结为逻辑错误，而不是浮点噪声。

要验证 GPU 路径时：

```powershell
$env:MINIVLLM_TEST_DEVICE="cuda"
python -m pytest tests/ -q
```

同一套断言换设备跑，如果只有 GPU 挂，那问题就在 kernel/dtype 相关的路径上（比如 `enable_gqa`、fp16 溢出）。

### 为什么 import 写在函数体里

```python
def runner():
    import torch
    from minivllm.config import EngineConfig
```

因为 `conftest.py` 会被 pytest **无条件加载**。如果顶层 `import torch`，那么「只想跑调度器测试」的人也必须装 torch。

放函数里 → 只有真正请求 `runner` fixture 的测试才会触发 import。这是**延迟导入**的实用场景：让测试依赖按需加载。

### `EngineConfig(model=..., num_gpu_blocks=64)` 里没写的参数

用默认值：`block_size=16`、`max_model_len=2048`、`enable_prefix_caching=True`……

对 `ModelRunner` 来说只有 `block_size`、`num_gpu_blocks`、`dtype`、`device` 有意义（它不管调度）。

---

## 4. `LOCAL_MODEL` 常量

```python
LOCAL_MODEL = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
```

用 `Path` 的 `/` 运算符拼接（`Path` 重载了 `__truediv__`）。跨平台安全，比 `os.path.join` 干净。

---

## 用到的 pytest 特性

| 特性 | 说明 |
|---|---|
| `conftest.py` | 自动加载的共享配置 |
| `@pytest.fixture(scope="session")` | 全会话共享一个实例 |
| `pytest.skip(reason)` | 跳过并给出原因 |
| fixture 作为参数名 | 测试函数声明 `runner` 就自动注入 |
| `pytest.mark.parametrize` | 在测试文件里用（见其它文档） |

---

## 设计取舍

**为什么 `runner` 是 session 级，而 `engine`（在 `test_engine_e2e.py` 里）是 module 级？**

`ModelRunner` 只依赖权重，是**无状态**的（每次 `execute_model` 都是独立的，只要调用方自己管好 `BlockManager`）。而 `LLMEngine` 有内部状态（`scheduler.running`、`metrics`），不同测试之间必须隔离，所以作用域更小。

**为什么不用 `pytest-cov`、`pytest-xdist`？**

本项目规模小（41 个测试、几十秒），加这些工具只会增加配置复杂度。**工具应该和问题规模匹配。**

---

## 常见误解

* 「skip 表示测试通过」——不是。它表示「没执行」。CI 里应该有机制检查 skip 数量。
* 「fixture 每次调用都重建」——只有 `scope="function"`（默认）才那样。
* 「conftest.py 里的 import 不影响其他文件」——它会影响，因为整个目录的测试共享这个模块。

---

## 动手练习

1. 把 `scope="session"` 改成 `"function"`，计时跑 `tests/test_model_runner.py`，看总时间从几秒变成几十秒。
2. 把默认 device 改成 `cuda`，跑 `tests/`，对比数值阈值是否需要放宽（fp16 vs fp32）。
3. 删掉 `sys.path.insert` 那两行（先 `pip install -e .`），验证装包之后确实不需要它。
4. 在 `runner` fixture 里打印 `runner.num_gpu_blocks` 和 `runner.k_cache.shape`，确认池子确实是 64 块。
