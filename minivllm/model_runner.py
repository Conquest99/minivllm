"""Runs the transformer forward pass against a paged KV cache.

The model weights are reused straight from ``transformers`` -- we only replace
the attention *bookkeeping*: where K/V go (``slot_mapping`` into a block pool) and
where they are read from (``block_table``), instead of an ever-growing contiguous
``past_key_values`` tensor per sequence.

Consequences worth understanding:

* the cache is allocated **once**, at startup, at a fixed size.  It never grows,
  never gets reallocated and never gets copied by ``torch.cat``.
* prefill and decode can sit in the same batch, because both are just "N tokens
  that need to write their KV somewhere and read a prefix".
* the logits are produced only for the last token of each sequence
  (``last_indices``), so a mixed batch costs one vocab-sized matmul per sequence
  rather than per token.
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from minivllm.config import EngineConfig
from minivllm.paged_attention import (
    copy_blocks,
    paged_attention_decode,
    paged_attention_prefill,
    write_kv_cache,
)
from minivllm.sequence import Sequence


class ModelRunner:
    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = getattr(torch, cfg.dtype)

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        self.hf_model = (
            AutoModelForCausalLM.from_pretrained(cfg.model, dtype=self.dtype, low_cpu_mem_usage=True)
            .to(self.device)
            .eval()
        )
        self.model = self.hf_model.model  # embedding + decoder layers + final norm
        self.lm_head = self.hf_model.lm_head

        hf_cfg = self.hf_model.config
        self.num_layers = hf_cfg.num_hidden_layers
        self.num_heads = hf_cfg.num_attention_heads
        self.num_kv_heads = hf_cfg.num_key_value_heads
        self.hidden_size = hf_cfg.hidden_size
        self.head_dim = getattr(hf_cfg, "head_dim", None) or self.hidden_size // self.num_heads
        self.vocab_size = hf_cfg.vocab_size
        self.scale = self.head_dim**-0.5
        self.load_time = time.perf_counter() - t0

        self.eos_token_ids = self._resolve_eos_ids()

        self.num_gpu_blocks = cfg.num_gpu_blocks or self._profile_num_blocks()
        self._allocate_kv_cache()

    # --------------------------------------------------------------- set-up
    def _resolve_eos_ids(self) -> set[int]:
        ids = set()
        eos = self.hf_model.generation_config.eos_token_id
        if eos is not None:
            ids.update(eos if isinstance(eos, (list, tuple)) else [eos])
        tok_eos = self.tokenizer.eos_token_id
        if tok_eos is not None:
            ids.add(tok_eos)
        return ids

    @property
    def per_block_bytes(self) -> int:
        element = torch.tensor([], dtype=self.dtype).element_size()
        return 2 * self.num_layers * self.cfg.block_size * self.num_kv_heads * self.head_dim * element

    def _profile_num_blocks(self) -> int:
        """How many pages to allocate.

        Bounded twice: by the VRAM left after the weights are loaded, and by the
        number of pages the scheduler could ever address
        (``max_num_seqs * max_model_len / block_size``).  Allocating more than that
        would waste the desktop's memory for addresses no request can reference.
        Set ``num_gpu_blocks`` to override.
        """
        useful = max(1, self.cfg.max_num_seqs * self.cfg.max_blocks_per_seq)
        if self.device.type == "cpu":
            return useful
        free, _total = torch.cuda.mem_get_info(self.device)
        budget = int(free * self.cfg.gpu_memory_utilization)
        return max(1, min(useful, budget // self.per_block_bytes))

    def _allocate_kv_cache(self) -> None:
        bs, nkv, hd = self.cfg.block_size, self.num_kv_heads, self.head_dim
        # The pool is one dense tensor per K and V:
        #   [layers, blocks, block_size, kv_heads, head_dim]
        # The layer dimension is essential -- every layer owns its own slice of a
        # block; sharing one slice across layers would mean layer N reads layer
        # N+1's keys.
        # +1 extra block at the end is a permanently-zero dummy page.  Padding
        # slots in a ragged decode batch point at it so the gather index stays
        # rectangular and the mask can drop them.
        shape = (self.num_layers, self.num_gpu_blocks + 1, bs, nkv, hd)
        self.k_cache = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.v_cache = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.dummy_slot = self.num_gpu_blocks * bs

    @property
    def kv_cache_bytes(self) -> int:
        return self.k_cache.numel() * self.k_cache.element_size() * 2

    def describe(self) -> str:
        return (
            f"model={self.cfg.model} dtype={self.cfg.dtype} layers={self.num_layers} "
            f"heads={self.num_heads}/{self.num_kv_heads}kv head_dim={self.head_dim} | "
            f"kv_blocks={self.num_gpu_blocks} block_size={self.cfg.block_size} "
            f"kv_cache={self.kv_cache_bytes / 2**30:.2f}GiB"
        )

    # ------------------------------------------------------------- interface
    def apply_cow(self, mapping: list[tuple[int, int]]) -> None:
        copy_blocks(self.k_cache, self.v_cache, mapping, self.cfg.block_size)

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text).input_ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ---------------------------------------------------------------- inputs
    def _build_inputs(self, seqs: list[Sequence], num_new_tokens: list[int]) -> dict[str, torch.Tensor]:
        bs = self.cfg.block_size
        device = self.device

        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        query_lens: list[int] = []
        kv_lens: list[int] = []
        last_indices: list[int] = []
        table_rows: list[list[int]] = []
        max_blocks = 1

        for seq, n in zip(seqs, num_new_tokens, strict=False):
            start = seq.num_computed_tokens
            input_ids.extend(seq.data[start : start + n])
            positions.extend(range(start, start + n))
            for t in range(start, start + n):
                slots.append(seq.block_table[t // bs] * bs + (t % bs))
            query_lens.append(n)
            kv_lens.append(start + n)
            last_indices.append(len(input_ids) - 1)
            n_blocks = (start + n + bs - 1) // bs
            table_rows.append(seq.block_table[:n_blocks])
            max_blocks = max(max_blocks, n_blocks)

        padded_tables = [[*row, *([0] * (max_blocks - len(row)))] for row in table_rows]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "positions": torch.tensor(positions, dtype=torch.long, device=device),
            "slot_mapping": torch.tensor(slots, dtype=torch.long, device=device),
            "block_tables": torch.tensor(padded_tables, dtype=torch.long, device=device),
            "query_lens": torch.tensor(query_lens, dtype=torch.long, device=device),
            "kv_lens": torch.tensor(kv_lens, dtype=torch.long, device=device),
            "last_indices": torch.tensor(last_indices, dtype=torch.long, device=device),
            # hoisted out of the layer loop: every .item()/.tolist() on a CUDA
            # tensor is a pipeline stall, and the layer loop runs 24 times
            "query_lens_list": query_lens,
            "all_decode": all(n == 1 for n in query_lens),
        }

    # --------------------------------------------------------------- forward
    @torch.inference_mode()
    def execute_model(self, seqs: list[Sequence], num_new_tokens: list[int]) -> torch.Tensor:
        """Run one step.  Returns ``[batch, vocab]`` logits, one row per sequence."""
        inp = self._build_inputs(seqs, num_new_tokens)

        hidden = self.model.embed_tokens(inp["input_ids"])  # [num_tokens, hidden]
        cos, sin = self.model.rotary_emb(hidden.unsqueeze(0), inp["positions"].unsqueeze(0))
        cos, sin = cos[0], sin[0]  # [num_tokens, head_dim]

        for layer_idx, layer in enumerate(self.model.layers):
            hidden = self._forward_layer(layer_idx, layer, hidden, cos, sin, inp)

        hidden = self.model.norm(hidden)
        return self.lm_head(hidden[inp["last_indices"]]).float()

    def _forward_layer(self, layer_idx, layer, hidden, cos, sin, inp) -> torch.Tensor:
        attn = layer.self_attn
        num_tokens = hidden.shape[0]

        residual = hidden
        x = layer.input_layernorm(hidden)

        q = attn.q_proj(x).view(num_tokens, self.num_heads, self.head_dim)
        k = attn.k_proj(x).view(num_tokens, self.num_kv_heads, self.head_dim)
        v = attn.v_proj(x).view(num_tokens, self.num_kv_heads, self.head_dim)

        # [num_tokens, heads, head_dim] layout -> broadcast cos/sin along dim 1
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        write_kv_cache(self.k_cache[layer_idx], self.v_cache[layer_idx], inp["slot_mapping"], k, v)
        out = self._attend(q, inp, layer_idx)

        hidden = residual + attn.o_proj(out)
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return hidden

    def _attend(self, q: torch.Tensor, inp: dict, layer_idx: int) -> torch.Tensor:
        k_cache = self.k_cache[layer_idx]
        v_cache = self.v_cache[layer_idx]
        if inp["all_decode"]:
            return paged_attention_decode(
                q,
                k_cache,
                v_cache,
                inp["block_tables"],
                inp["kv_lens"],
                self.scale,
                self.dummy_slot,
            )
        return paged_attention_prefill(
            q,
            k_cache,
            v_cache,
            inp["block_tables"],
            inp["query_lens_list"],
            inp["kv_lens"],
            self.scale,
        )
