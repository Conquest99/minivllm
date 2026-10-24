"""Smoke test / demo: load the engine, generate a few completions, print the stats.

python scripts/demo_generate.py --prompts "What is paged attention?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minivllm.config import EngineConfig  # noqa: E402
from minivllm.llm_engine import LLMEngine  # noqa: E402
from minivllm.sampling import SamplingParams  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "What is PagedAttention and why does it matter?",
            "Explain continuous batching in two sentences.",
            "Write a haiku about KV caches.",
        ],
    )
    parser.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-prefix-cache", action="store_true")
    args = parser.parse_args()

    cfg = EngineConfig(
        model=args.model,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=not args.no_prefix_cache,
    )
    engine = LLMEngine(cfg)
    print(engine.model_runner.describe())
    print(f"kv pool: {engine.block_manager.stats()}\n")

    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
    outputs = engine.generate(args.prompts, params)

    for out in outputs:
        print("=" * 80)
        print(f"[seq {out.seq_id}] {out.prompt}")
        print("-" * 80)
        print(out.text)
        print(
            f"-- finish={out.finish_reason} tokens={out.num_output_tokens} "
            f"ttft={out.ttft:.3f}s latency={out.latency:.3f}s preemptions={out.num_preemptions}"
        )

    print("\n" + "=" * 80)
    for key, value in engine.stats().items():
        print(f"{key:>22}: {value:.4f}" if isinstance(value, float) else f"{key:>22}: {value}")


if __name__ == "__main__":
    main()
