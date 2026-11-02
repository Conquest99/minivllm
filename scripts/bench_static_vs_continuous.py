"""静态批 vs 连续批，同一个模型、同一条代码路径。

变的只有*调度策略*，所以差异只能归因于调度，别的都不算：

* ``--mode static``       —— 一批被接纳后一直跑到最长的那个成员结束，整批排空之前
  没有新请求能加入。批里每个成员实际上都被拖成最长成员的 token 预算。
* ``--mode continuous``   —— 每一步都重新决定：结束的序列立刻释放槽位和页，排队的
  请求下一步就能加入。

请求的输出长度刻意差得很大（16..256 token）。这正是静态批吃亏的负载，也是真实对话
服务器看到的样子。

``--with-hf`` 会额外对同一负载计一遍纯 ``model.generate`` 的时间。那一行不是严格
同口径（不同代码、不同 KV 策略），但它回答了一个必然的追问：*在这台机器上，一个调
优过的实现每个 token 要花多少？*  如果我们这个连续引擎在少优化 10 倍的情况下还能
赢过 HF 的静态批，那调度的收益就是真的。

    python scripts/bench_static_vs_continuous.py --num-requests 24 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from _workload import build_workload, summarise  # noqa: E402

from minivllm.config import EngineConfig  # noqa: E402
from minivllm.llm_engine import EngineMetrics, LLMEngine  # noqa: E402
from minivllm.sampling import SamplingParams  # noqa: E402


def run_engine(engine: LLMEngine, workload: list[dict], static: bool) -> dict:
    engine.cfg.static_batching = static
    engine.scheduler.reset_cache()
    engine.metrics = EngineMetrics()
    if engine.scheduler.running or engine.scheduler.waiting:
        raise RuntimeError("engine must be idle before a run")

    t0 = time.perf_counter()
    ids = [
        engine.add_request(
            item["prompt"],
            SamplingParams(max_tokens=item["max_tokens"], temperature=0.0, ignore_eos=True),
        )
        for item in workload
    ]
    done = {}
    while engine.scheduler.has_unfinished():
        for out in engine.step():
            done[out.seq_id] = out
    wall = time.perf_counter() - t0

    outputs = [done[i] for i in ids]
    useful = sum(item["max_tokens"] for item in workload)
    generated = sum(o.num_output_tokens for o in outputs)
    return {
        "wall_time": wall,
        "useful_tokens": useful,
        "generated_tokens": generated,
        "latencies": [o.latency for o in outputs],
        "ttfts": [o.ttft for o in outputs],
        "num_steps": engine.metrics.num_steps,
        "tokens_per_step": engine.metrics.tokens_per_step,
        "preemptions": engine.scheduler.num_preemptions,
    }


def run_hf(engine: LLMEngine, workload: list[dict], batch_size: int) -> dict:
    tokenizer = engine.tokenizer
    model = engine.model_runner.hf_model
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    generated = 0
    wall = 0.0
    for offset in range(0, len(workload), batch_size):
        chunk = workload[offset : offset + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": c["prompt"]}], add_generation_prompt=True, tokenize=False
            )
            for c in chunk
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        t0 = time.perf_counter()
        out = model.generate(
            **enc,
            max_new_tokens=max(c["max_tokens"] for c in chunk),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=None,
        )
        wall += time.perf_counter() - t0
        generated += int((out[:, enc["input_ids"].shape[1] :] != tokenizer.pad_token_id).sum())

    return {
        "wall_time": wall,
        "useful_tokens": sum(c["max_tokens"] for c in workload),
        "generated_tokens": generated,
        # HF 不流式，所以整批跑完之前调用方什么都看不到
        "latencies": [wall],
        "ttfts": [wall],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--num-requests", type=int, default=24)
    parser.add_argument("--min-output", type=int, default=16)
    parser.add_argument("--max-output", type=int, default=256)
    parser.add_argument(
        "--distribution",
        choices=["uniform", "heavy_tail"],
        default="uniform",
        help="heavy_tail = most requests short, a few very long (realistic chat mix)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="max_num_seqs for both modes")
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--with-hf", action="store_true", help="also time plain model.generate")
    parser.add_argument("--dump", default="benchmarks/results/static_vs_continuous.json")
    args = parser.parse_args()

    workload = build_workload(
        args.num_requests,
        seed=args.seed,
        min_output=args.min_output,
        max_output=args.max_output,
        distribution=args.distribution,
    )
    requested = sum(w["max_tokens"] for w in workload)
    lengths = sorted(w["max_tokens"] for w in workload)
    print(
        f"workload: {len(workload)} requests, {requested} requested tokens, "
        f"min={lengths[0]} median={lengths[len(lengths) // 2]} max={lengths[-1]}\n"
    )

    cfg = EngineConfig(
        model=args.model,
        dtype=args.dtype,
        block_size=args.block_size,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=2048,
    )
    engine = LLMEngine(cfg)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(engine.model_runner.describe())
    print(f"device: {gpu}\n")

    for _ in range(args.warmup):
        engine.generate(["warmup"], SamplingParams(max_tokens=8, ignore_eos=True))

    rows: dict[str, dict] = {}
    for mode in ("static", "continuous"):
        result = run_engine(engine, workload, static=(mode == "static"))
        rows[mode] = {
            "throughput_tok_s": result["useful_tokens"] / result["wall_time"],
            "generated_tok_s": result["generated_tokens"] / result["wall_time"],
            "latency": summarise(result["latencies"]),
            "ttft": summarise(result["ttfts"]),
            "wall_time": result["wall_time"],
            "generated_tokens": result["generated_tokens"],
            "useful_tokens": result["useful_tokens"],
            "num_steps": result["num_steps"],
            "tokens_per_step": result["tokens_per_step"],
            "preemptions": result["preemptions"],
        }

    if args.with_hf:
        hf = run_hf(engine, workload, args.batch_size)
        rows["hf_generate"] = {
            "throughput_tok_s": hf["useful_tokens"] / hf["wall_time"],
            "generated_tok_s": hf["generated_tokens"] / hf["wall_time"],
            "latency": {},
            "ttft": {},
            "wall_time": hf["wall_time"],
            "generated_tokens": hf["generated_tokens"],
            "useful_tokens": hf["useful_tokens"],
        }

    names = list(rows)
    col = 26
    header = f"{'':<{col}}" + "".join(f"{n:>16}" for n in names)
    print(header)
    print("-" * len(header))

    def line(label: str, getter) -> None:
        print(f"{label:<{col}}" + "".join(f"{getter(rows[n]):>16.3f}" for n in names))

    line("wall time (s)", lambda r: r["wall_time"])
    line("useful tokens/s", lambda r: r["throughput_tok_s"])
    line("generated tokens/s", lambda r: r["generated_tok_s"])
    line("tokens emitted / step", lambda r: r.get("tokens_per_step", 0.0))
    line("mean latency (s)", lambda r: r["latency"].get("mean", 0.0))
    line("p99 latency (s)", lambda r: r["latency"].get("p99", 0.0))
    line("mean TTFT (s)", lambda r: r["ttft"].get("mean", 0.0))

    static, cont = rows["static"], rows["continuous"]
    speedup = cont["throughput_tok_s"] / static["throughput_tok_s"]
    waste = 1 - static["useful_tokens"] / static["generated_tokens"]
    print(f"\nstatic batching wasted {100 * waste:.0f}% of the tokens it generated")
    print(f"continuous batching throughput gain: {speedup:.2f}x")

    if args.dump:
        out = Path(args.dump)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "config": vars(args),
                    "requested_tokens": requested,
                    "results": rows,
                    "throughput_gain": speedup,
                    "static_wasted_fraction": waste,
                },
                indent=2,
            )
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
