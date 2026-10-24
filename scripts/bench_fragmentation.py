"""How much KV memory does *not* paging waste?

Two engines are compared against the same memory pool and the same requests:

* **reserve** -- the pre-paged-attention approach: every admitted request gets
  one contiguous buffer big enough for the worst case (``max_model_len``),
  because the engine cannot move KV around later.
* **paged**   -- blocks are handed out as tokens are actually computed.

Run it on CPU; it only exercises the allocator, no model.

    python scripts/bench_fragmentation.py --kv-pool-tokens 4096 --requests 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _workload import build_workload  # noqa: E402

from minivllm.block_manager import AllocStatus, BlockManager  # noqa: E402
from minivllm.sampling import SamplingParams  # noqa: E402
from minivllm.sequence import Sequence  # noqa: E402


def fabricate(seq_id: int, num_tokens: int) -> Sequence:
    """A sequence of ``num_tokens`` tokens without needing a tokenizer."""
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=[seq_id * 10000 + i for i in range(num_tokens)],
        sampling_params=SamplingParams(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-pool-tokens", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--min-output", type=int, default=16)
    parser.add_argument("--max-output", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump", default=None, help="write the numbers as JSON here")
    args = parser.parse_args()

    bs = args.block_size
    pool_blocks = args.kv_pool_tokens // bs
    workload = build_workload(
        args.requests, seed=args.seed, min_output=args.min_output, max_output=args.max_output
    )
    lengths = [args.prompt_tokens + w["max_tokens"] for w in workload]

    # ---------------------------------------------------------------- baseline
    blocks_per_reservation = (args.max_model_len + bs - 1) // bs
    reserve_concurrency = pool_blocks // blocks_per_reservation
    reserve_used_tokens = reserve_concurrency * args.max_model_len

    # ------------------------------------------------------------------- paged
    bm = BlockManager(pool_blocks, bs, enable_prefix_caching=False)
    admitted = 0
    used_tokens = 0
    for i, n in enumerate(lengths):
        seq = fabricate(i, n)
        if bm.can_allocate(seq, n) != AllocStatus.OK:
            break
        bm.allocate(seq, n)
        admitted += 1
        used_tokens += n
    # internal fragmentation: the tail of each request's last block is unusable
    slot_fragmentation = sum((n + bs - 1) // bs * bs - n for n in lengths[:admitted])

    report = {
        "pool_tokens": args.kv_pool_tokens,
        "pool_blocks": pool_blocks,
        "block_size": bs,
        "max_model_len": args.max_model_len,
        "requests_offered": len(lengths),
        "request_tokens": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)},
        "reserve": {
            "blocks_per_request": blocks_per_reservation,
            "max_concurrent_requests": reserve_concurrency,
            "tokens_held": reserve_used_tokens,
            "tokens_actually_needed": sum(lengths[:reserve_concurrency]),
        },
        "paged": {
            "max_concurrent_requests": admitted,
            "tokens_held": used_tokens,
            "internal_fragmentation_tokens": slot_fragmentation,
        },
    }
    report["concurrency_speedup"] = admitted / max(reserve_concurrency, 1)
    report["reserve_waste_pct"] = 100 * (
        1 - report["reserve"]["tokens_actually_needed"] / max(report["reserve"]["tokens_held"], 1)
    )
    report["paged_waste_pct"] = 100 * slot_fragmentation / max(used_tokens, 1)

    print(f"KV pool        : {args.kv_pool_tokens} tokens ({pool_blocks} blocks x {bs} tokens)\n")
    print(f"{'':<28}{'reserve':>14}{'paged':>14}")
    print(f"{'max concurrent requests':<28}{reserve_concurrency:>14}{admitted:>14}")
    print(f"{'tokens held':<28}{reserve_used_tokens:>14}{used_tokens:>14}")
    print(
        f"{'wasted tokens (%)':<28}{report['reserve_waste_pct']:>13.1f}%{report['paged_waste_pct']:>13.1f}%"
    )
    print(f"\nconcurrency gained: {report['concurrency_speedup']:.2f}x")

    if args.dump:
        out = Path(args.dump)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
