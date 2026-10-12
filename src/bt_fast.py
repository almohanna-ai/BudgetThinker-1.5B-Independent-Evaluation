"""Exact speed-ups for long-context decoding with transformers 4.51.3 on a T4.

1. GQA decode attention. transformers 4.51.3's `sdpa_attention_forward` calls `repeat_kv` on every
   step, materialising the KV cache 6x (12 query heads over 2 KV heads) per layer. For one-token
   decode steps we instead view the 12 query heads as 2 groups x 6 and attend each group to its KV
   head directly. Prefill (q_len > 1) still uses the original SDPA function. The mask semantics stay
   those of `_attn_implementation == "sdpa"` because we replace the function under that key.
2. Preallocated KV cache. `DynamicCache.update` concatenates the whole cache every step; this cache
   writes into a buffer allocated once per batch and returns views.

Both change only how the same quantities are computed (verified against the stock path in
tests/test_fast.py); decoding semantics, sampling and the injection rule are untouched.
"""
from __future__ import annotations

import torch
from transformers import DynamicCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward as _stock_sdpa
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def gqa_sdpa_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
                     is_causal=None, **kwargs):
    groups = getattr(module, "num_key_value_groups", 1)
    if query.shape[2] != 1 or groups == 1 or dropout:
        return _stock_sdpa(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                           is_causal=is_causal, **kwargs)
    b, h, _, d = query.shape
    kvh, length = key.shape[1], key.shape[2]
    scale = scaling if scaling is not None else d ** -0.5
    # Scale q before the fp16 matmul (as SDPA does): Qwen2's large q/k activations overflow fp16
    # (max 65504) if q.k is formed first and scaled afterwards.
    q = (query * scale).reshape(b, kvh, groups, d)            # head index = kv * groups + g (as repeat_kv)
    scores = torch.matmul(q, key.transpose(-1, -2)).float()   # [b, kvh, groups, L]
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, :length].float()      # [b, 1, 1, L] broadcasts
    probs = torch.softmax(scores, dim=-1).to(value.dtype)
    out = torch.matmul(probs, value)                          # [b, kvh, groups, d]
    return out.reshape(b, h, 1, d).transpose(1, 2).contiguous(), None


def enable_gqa_decode():
    ALL_ATTENTION_FUNCTIONS["sdpa"] = gqa_sdpa_forward


def disable_gqa_decode():
    if "sdpa" in ALL_ATTENTION_FUNCTIONS._local_mapping:
        del ALL_ATTENTION_FUNCTIONS["sdpa"]


class PreallocCache(DynamicCache):
    """DynamicCache that writes into per-layer buffers of `max_len` positions instead of torch.cat."""

    def __init__(self, max_len: int):
        super().__init__()
        self.max_len = max_len
        self._lens: list[int] = []

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        n = key_states.shape[-2]
        if layer_idx == 0:
            self._seen_tokens += n
        if len(self.key_cache) <= layer_idx:
            b, h, _, d = key_states.shape
            kb = key_states.new_empty((b, h, self.max_len, d))
            vb = value_states.new_empty((b, h, self.max_len, d))
            kb[:, :, :n] = key_states
            vb[:, :, :n] = value_states
            self.key_cache.append(kb)
            self.value_cache.append(vb)
            self._lens.append(n)
        else:
            s = self._lens[layer_idx]
            if s + n > self.max_len:
                raise RuntimeError(f"PreallocCache overflow: {s + n} > {self.max_len}")
            self.key_cache[layer_idx][:, :, s:s + n] = key_states
            self.value_cache[layer_idx][:, :, s:s + n] = value_states
            self._lens[layer_idx] = s + n
        cur = self._lens[layer_idx]
        return self.key_cache[layer_idx][:, :, :cur], self.value_cache[layer_idx][:, :, :cur]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._lens[layer_idx] if layer_idx < len(self._lens) else 0

    def batch_select_indices(self, indices: torch.Tensor):
        for i in range(len(self.key_cache)):
            self.key_cache[i] = self.key_cache[i].index_select(0, indices)
            self.value_cache[i] = self.value_cache[i].index_select(0, indices)
