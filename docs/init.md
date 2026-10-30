# `minivllm/__init__.py` 详解

## 这个文件干什么

包的**门面（facade）**。把最常用的 3 个名字提到顶层，让使用者只写一行 import：

```python
from minivllm import EngineConfig, LLMEngine, SamplingParams

engine = LLMEngine(EngineConfig(model="..."))
```

完整代码（14 行）：

```python
from minivllm.config import EngineConfig
from minivllm.llm_engine import LLMEngine
from minivllm.sampling import SamplingParams

__all__ = ["EngineConfig", "LLMEngine", "SamplingParams", "__version__"]
__version__ = "0.1.0"
```

---

## 逐块讲解

### `from minivllm.xxx import yyy`

这两个 import 不是「用到了才导」——是为了**让顶层名字可用**。Python 的 `minivllm.config` 是子模块，`EngineConfig` 是里面的类；只有显式 import 之后，`minivllm.EngineConfig` 才存在。

### `__all__`

```python
__all__ = ["EngineConfig", "LLMEngine", "SamplingParams", "__version__"]
```

一个**约定**，有两层作用：

1. `from minivllm import *` 时只导出这里列的名字，不会把 `torch`、`time` 之类顺手带出去。
2. 告诉读者「这就是公开 API，其它都是内部实现」。ruff 的 `F401`（导入了但没用）规则也会认它，所以这些 import 不会报警告。

### `__version__ = "0.1.0"`

和 [pyproject.toml](../pyproject.toml) 里的 `version` 保持手动同步。

> 为什么 import 顺序是 config → llm_engine → sampling？因为 `llm_engine` 依赖前两者。放后面能避免循环 import 的潜在问题（虽然这里其实不构成循环）。

---

## 用到的 Python 特性

| 特性 | 说明 |
|---|---|
| `__all__` | 控制 `import *` 的导出列表 |
| `__version__` | 社区约定，没有语言层面的强制 |
| 包级 re-export | 把深层模块的名字提到顶层，是库的常规做法（`torch.nn`、`transformers` 都这么干） |

---

## 设计取舍

**为什么不做 `from minivllm.llm_engine import *`？** 那样会把 `LLMEngine` 依赖的 `torch`、`Scheduler`、`Sequence` 全部泄露到顶层命名空间，副作用难追踪。显式列名字更好。

**为什么不在 `__init__` 里就实例化什么东西？** import 一个包不应该有副作用（不加载模型、不占显存）。本项目所有重量级初始化都在 `LLMEngine(cfg)` 里。

---

## 常见误解

* 「`__init__.py` 是空的也行」——可以，但这样 `from minivllm import LLMEngine` 会失败，必须写全路径。门面文件的唯一职责就是缩短 import 路径。
* 「`__all__` 影响可见性」——不影响。`minivllm.Torch` 依然能访问，只是 `import *` 时不带出来。

---

## 动手练习

1. 在仓库根目录跑 `python -c "import minivllm; print(minivllm.EngineConfig)"`，然后把 `__init__.py` 里那行 import 注释掉，再跑一次，看报什么错。
2. 改成 `__all__ = []`，跑 `python -c "from minivllm import *; print(EngineConfig)"`，观察 `NameError`。
