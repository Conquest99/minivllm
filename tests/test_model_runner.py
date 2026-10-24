"""The correctness test that matters: paged KV cache vs a plain HF forward pass.

The oracle is deliberately dumb -- re-run the whole prefix through
``transformers`` with no cache at all.  If the paged path disagrees with that, the
bookkeeping is wrong, no matter how plausible the generated text looks.
"""

from __future__ import annotations

import pytest
import torch

from minivllm.block_manager import BlockManager
from minivllm.sampling import SamplingParams
from minivllm.sequence import Sequence


def make(cache: BlockManager, seq_id: int, ids: list[int]) -> Sequence:
    seq = Sequence(seq_id=seq_id, prompt_token_ids=ids, sampling_params=SamplingParams())
    cache.allocate(seq, len(seq.data))
    return seq


def reference_logits(runner, ids: list[int]) -> torch.Tensor:
    with torch.no_grad():
        out = runner.hf_model(torch.tensor([ids], device=runner.device))
    return out.logits[0, -1].float()


def assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str, atol: float) -> None:
    diff = (actual - expected).abs().max().item()
    assert torch.allclose(actual, expected, atol=atol), f"{label}: max |diff| = {diff:.3e}"


def test_prefill_matches_transformers(runner):
    cache = BlockManager(runner.num_gpu_blocks, runner.cfg.block_size, enable_prefix_caching=False)
    ids = runner.tokenize("The capital of France is")
    seq = make(cache, 0, ids)

    logits = runner.execute_model([seq], [len(ids)])

    assert_close(logits[0], reference_logits(runner, ids), "prefill", atol=1e-4)


def test_incremental_decode_matches_transformers(runner):
    cache = BlockManager(runner.num_gpu_blocks, runner.cfg.block_size, enable_prefix_caching=False)
    ids = runner.tokenize("List the first three prime numbers:")
    seq = make(cache, 0, ids)

    logits = runner.execute_model([seq], [len(ids)])
    seq.num_computed_tokens = len(ids)

    for step in range(8):
        token = int(logits[0].argmax())
        seq.append_token(token)
        cache.ensure_capacity(seq, len(seq.data))
        logits = runner.execute_model([seq], [1])
        seq.num_computed_tokens += 1

        expected = reference_logits(runner, seq.data)
        assert_close(logits[0], expected, f"decode step {step}", atol=2e-3)


def test_chunked_prefill_matches_single_shot(runner):
    """Splitting a prompt across steps must not change the logits."""
    ids = runner.tokenize("Explain in one sentence why paged attention avoids memory fragmentation.")

    cache = BlockManager(runner.num_gpu_blocks, runner.cfg.block_size, enable_prefix_caching=False)
    seq = make(cache, 0, ids)
    logits_one_shot = runner.execute_model([seq], [len(ids)])[0]
    assert_close(logits_one_shot, reference_logits(runner, ids), "one-shot prefill", atol=1e-4)

    cache2 = BlockManager(runner.num_gpu_blocks, runner.cfg.block_size, enable_prefix_caching=False)
    seq2 = make(cache2, 0, ids)
    chunk = runner.cfg.block_size * 2
    while seq2.num_computed_tokens < len(ids):
        n = min(chunk, len(ids) - seq2.num_computed_tokens)
        runner.execute_model([seq2], [n])
        seq2.num_computed_tokens += n

    seq2.append_token(int(logits_one_shot.argmax()))
    cache2.ensure_capacity(seq2, len(seq2.data))
    logits_chunked = runner.execute_model([seq2], [1])[0]

    assert_close(logits_chunked, reference_logits(runner, seq2.data), "chunked prefill", atol=1e-4)


def test_ragged_batch_matches_individual_runs(runner):
    """A batch of sequences with different lengths must equal running them alone."""
    cache = BlockManager(runner.num_gpu_blocks, runner.cfg.block_size, enable_prefix_caching=False)
    prompts = [
        "Hello",
        "What is the difference between paging and swapping in an operating system?",
        "1 2 3",
    ]
    seqs = [make(cache, i, runner.tokenize(p)) for i, p in enumerate(prompts)]

    batched = runner.execute_model(seqs, [len(s.data) for s in seqs])

    for i, seq in enumerate(seqs):
        assert_close(batched[i], reference_logits(runner, seq.data), f"batch row {i}", atol=1e-4)


def test_prefill_and_decode_can_share_a_batch(runner):
    """The mixed batch that static batching cannot express."""
    cache = BlockManager(runner.num_gpu_blocks, runner.cfg.block_size, enable_prefix_caching=False)

    warm = make(cache, 0, runner.tokenize("Once upon a time"))
    runner.execute_model([warm], [len(warm.data)])
    warm.num_computed_tokens = len(warm.data)
    warm.append_token(int(reference_logits(runner, warm.data).argmax()))
    cache.ensure_capacity(warm, len(warm.data))

    fresh = make(cache, 1, runner.tokenize("Attention is all you need"))

    out = runner.execute_model([warm, fresh], [1, len(fresh.data)])
    assert out.shape[0] == 2

    assert_close(out[0], reference_logits(runner, warm.data), "decode row", atol=2e-3)
    assert_close(out[1], reference_logits(runner, fresh.data), "prefill row", atol=1e-4)


def test_prefix_cache_reuses_kv_and_keeps_logits_identical(runner):
    cache = BlockManager(runner.num_gpu_blocks, runner.cfg.block_size, enable_prefix_caching=True)
    ids = runner.tokenize(
        "You are a helpful assistant that explains memory management in operating systems. "
        "Summarise paged attention in detail, and explain why it removes external fragmentation "
        "from the key-value cache of a large language model serving engine."
    )
    assert len(ids) > 2 * runner.cfg.block_size, "prompt must span at least two full blocks"

    first = make(cache, 0, ids)
    reference = runner.execute_model([first], [len(ids)])[0]
    first.num_computed_tokens = len(ids)
    cache.register_computed_blocks(first)
    cache.free(first)

    second = Sequence(seq_id=1, prompt_token_ids=list(ids), sampling_params=SamplingParams())
    hits = cache.allocate(second, len(second.data))
    assert hits > 0, "expected the prefix cache to serve part of this prompt"

    logits = runner.execute_model([second], [len(ids)])[0]
    assert_close(logits, reference, "prefix-cache replay", atol=1e-4)


@pytest.mark.parametrize("block_size", [1, 8, 32])
def test_block_size_does_not_change_results(runner, block_size):
    from minivllm.config import EngineConfig

    cfg = EngineConfig(
        model=runner.cfg.model,
        dtype=runner.cfg.dtype,
        device=runner.cfg.device,
        block_size=block_size,
        num_gpu_blocks=256,
    )
    from minivllm.model_runner import ModelRunner

    other = ModelRunner(cfg)
    ids = other.tokenize("Does block size change the math?")
    cache = BlockManager(other.num_gpu_blocks, block_size, enable_prefix_caching=False)
    seq = make(cache, 0, ids)

    logits = other.execute_model([seq], [len(ids)])[0]
    assert_close(logits, reference_logits(other, ids), f"block_size={block_size}", atol=1e-4)
