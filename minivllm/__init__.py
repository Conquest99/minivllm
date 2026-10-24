"""minivllm -- a minimal, readable re-implementation of PagedAttention + continuous batching.

The point of this package is not to be fast.  It is to be *legible*: every
mechanism that a production engine (vLLM/SGLang/TensorRT-LLM) hides inside a
fused CUDA kernel or a 5k-line scheduler is written out in plain PyTorch so the
data flow can be inspected, unit-tested and benchmarked on a laptop GPU.
"""

from minivllm.config import EngineConfig
from minivllm.llm_engine import LLMEngine
from minivllm.sampling import SamplingParams

__all__ = ["EngineConfig", "LLMEngine", "SamplingParams", "__version__"]
__version__ = "0.1.0"
