"""minivllm —— PagedAttention + 连续批处理的最小可读实现。

这个包的意义不是快，而是*可读*：生产引擎（vLLM / SGLang / TensorRT-LLM）藏在融合
CUDA kernel 或五千行调度器里的每个机制，这里都用平铺直叙的 PyTorch 写出来，好让
数据流能被检查、被单测、被在一台笔记本 GPU 上压测。
"""

from minivllm.config import EngineConfig
from minivllm.llm_engine import LLMEngine
from minivllm.sampling import SamplingParams

__all__ = ["EngineConfig", "LLMEngine", "SamplingParams", "__version__"]
__version__ = "0.1.0"
