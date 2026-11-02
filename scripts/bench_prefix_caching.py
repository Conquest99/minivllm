"""前缀缓存到底买到了什么？

设置：每个请求都带着同一段很长的 system prompt，然后问一个不同的问题。没有前缀
缓存时，每个请求都要把这些 token 从零 prefill 一遍；有了它，第一个请求填满这些块，
其余请求按引用共享，几乎立刻开始解码。

请求是**一个一个**跑的，这是故意的。前缀缓存只能帮到「在共享前缀已经算完并登记之后
才到达」的请求，所以一次性全塞进去（也就是突发流量的样子）什么都测不出来。顺序跑还
让 TTFT 里没有排队延迟，于是这个数就是它字面上的意思：到第一个 token 要多久。

    python scripts/bench_prefix_caching.py --requests 6 --max-tokens 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _workload import summarise  # noqa: E402

SYSTEM_PROMPT = (
    "You are a meticulous distributed systems tutor. Always answer with concrete "
    "numbers, name the mechanism you are describing, and never pad the answer. "
)


def build_prompts(n: int, repeats: int) -> list[str]:
    shared = SYSTEM_PROMPT * repeats
    return [f"{shared}\n\nQuestion {i}: how does page {i} relate to the block table?" for i in range(n)]


def run_sequential(engine, prompts: list[str], max_tokens: int) -> dict:
    from minivllm.sampling import SamplingParams

    engine.scheduler.reset_cache()
    ttfts, latencies, wall = [], [], 0.0
    for prompt in prompts:
        t0 = time.perf_counter()
        out = engine.generate(
            [prompt], SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
        )[0]
        wall += time.perf_counter() - t0
        ttfts.append(out.ttft)
        latencies.append(out.latency)
    stats = engine.stats()
    return {
        "ttft": summarise(ttfts),
        "ttft_warm": summarise(ttfts[1:]),
        "ttft_per_request": ttfts,
        "latency": summarise(latencies),
        "wall_time": wall,
        "prefix_hit_blocks": stats["prefix_hit_blocks"],
        "peak_used_blocks": stats["peak_used_blocks"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prefix-repeats", type=int, default=120, help="how long the shared prompt is")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dump", default="benchmarks/results/prefix_caching.json")
    args = parser.parse_args()

    from minivllm.config import EngineConfig
    from minivllm.llm_engine import EngineMetrics, LLMEngine
    from minivllm.sampling import SamplingParams

    prompts = build_prompts(args.requests, args.prefix_repeats)
    results: dict[str, dict] = {}

    for enabled in (False, True):
        cfg = EngineConfig(
            model=args.model,
            dtype=args.dtype,
            block_size=args.block_size,
            max_num_seqs=1,  # 一次只放一个请求：隔离出 prefill，而不是排队效应
            max_model_len=args.max_model_len,
            max_num_batched_tokens=2048,
            enable_prefix_caching=enabled,
        )
        engine = LLMEngine(cfg)
        if not results:
            n_tokens = len(engine._encode_prompt(prompts[0]))
            print(f"shared prefix: {n_tokens} tokens x {args.requests} requests")
            print(engine.model_runner.describe() + "\n")

        engine.generate(["warmup"], SamplingParams(max_tokens=8, ignore_eos=True))
        engine.metrics = EngineMetrics()

        key = "on" if enabled else "off"
        results[key] = run_sequential(engine, prompts, args.max_tokens)
        del engine

    off, on = results["off"], results["on"]
    print(f"{'':<26}{'prefix off':>14}{'prefix on':>14}")
    print(f"{'wall time (s)':<26}{off['wall_time']:>14.3f}{on['wall_time']:>14.3f}")
    print(f"{'mean TTFT (s)':<26}{off['ttft']['mean']:>14.3f}{on['ttft']['mean']:>14.3f}")
    print(f"{'warm TTFT (s)':<26}{off['ttft_warm']['mean']:>14.3f}{on['ttft_warm']['mean']:>14.3f}")
    print(f"{'warm p90 TTFT (s)':<26}{off['ttft_warm']['p90']:>14.2f}{on['ttft_warm']['p90']:>14.3f}")
    print(f"{'shared blocks served':<26}{off['prefix_hit_blocks']:>14}{on['prefix_hit_blocks']:>14}")
    print(
        f"\nTTFT per request (off): {[round(t, 3) for t in off['ttft_per_request']]}"
        f"\nTTFT per request (on) : {[round(t, 3) for t in on['ttft_per_request']]}"
    )

    gain = off["ttft"]["mean"] / on["ttft"]["mean"] if on["ttft"]["mean"] else 0.0
    warm_gain = off["ttft_warm"]["mean"] / on["ttft_warm"]["mean"] if on["ttft_warm"]["mean"] else 0.0
    print(f"\nmean TTFT gain: {gain:.2f}x   |   warm-request TTFT gain: {warm_gain:.2f}x")

    if args.dump:
        out = Path(args.dump)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"config": vars(args), "results": results, "ttft_gain": gain, "warm_ttft_gain": warm_gain},
                indent=2,
            )
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
