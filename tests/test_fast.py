"""bt_fast must reproduce the stock transformers path. Run: python -I tests/test_fast.py <pylib>

Tiny Qwen2 with the real head layout (12 query heads over 2 KV heads), fp32 on CPU:
  1. per-step logits of the stock path (DynamicCache + stock SDPA) vs. GQA decode + PreallocCache,
     on a left-padded batch with row pruning, agree to float tolerance;
  2. full engine runs (greedy and sampled) give identical token ids and provenance.
"""
import os
import sys

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

import bt_engine as E  # noqa: E402
import bt_fast as F  # noqa: E402

torch.manual_seed(0)
m = Qwen2ForCausalLM(Qwen2Config(vocab_size=151672, hidden_size=96, intermediate_size=192, num_hidden_layers=2,
                                 num_attention_heads=12, num_key_value_heads=2, max_position_embeddings=4096,
                                 tie_word_embeddings=False, sliding_window=None,
                                 attn_implementation="sdpa")).eval()
with torch.no_grad():
    m.lm_head.weight *= 40
    m.lm_head.weight[E.THINK_CLOSE] *= 1.5
    m.lm_head.weight[E.EOS] *= 1.4
assert m.config._attn_implementation == "sdpa"


@torch.no_grad()
def logits_trace(prompts, steps, fast):
    (F.enable_gqa_decode if fast else F.disable_gqa_decode)()
    w = max(map(len, prompts))
    ids = torch.full((len(prompts), w), E.PAD)
    attn = torch.zeros((len(prompts), w), dtype=torch.long)
    for r, p in enumerate(prompts):
        ids[r, w - len(p):] = torch.tensor(p)
        attn[r, w - len(p):] = 1
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    cache = F.PreallocCache(w + steps + 2) if fast else DynamicCache()
    out = m(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=cache, use_cache=True,
            logits_to_keep=1)
    cur = out.logits[:, -1].clone()
    trace, nxt, rows = [cur], pos[:, -1:] + 1, list(range(len(prompts)))
    for t in range(steps):
        tok = cur.argmax(-1, keepdim=True)
        if t == steps // 2:  # prune row 0 halfway, like a finished generation
            keep = torch.tensor([1, 2])
            cache.batch_select_indices(keep)
            attn, nxt, tok, rows = attn[keep], nxt[keep], tok[keep], rows[1:]
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], 1)
        out = m(input_ids=tok, attention_mask=attn, position_ids=nxt, past_key_values=cache, use_cache=True,
                logits_to_keep=1)
        nxt = nxt + 1
        cur = out.logits[:, -1].clone()
        full = torch.full((3, out.logits.shape[-1]), float("nan"))
        full[rows] = cur
        trace.append(full)
    F.disable_gqa_decode()
    return trace


P = [[151646, 151644, 10, 11, 12, 151645], [151646, 151644, 20, 151645], [151646, 151644, 30, 31, 32, 33, 151645]]
stock, fast = logits_trace(P, 40, False), logits_trace(P, 40, True)
diff = max(float(torch.nan_to_num((a - b).abs(), nan=0.0).max()) for a, b in zip(stock, fast))
print(f"max |logit diff| stock vs fast over 41 steps (padding + pruning): {diff:.2e}")
assert diff < 1e-4

specs = [E.RowSpec("a", P[0], 120, True, 1), E.RowSpec("b", P[1], 64, False, 2), E.RowSpec("c", P[2], 200, True, 3),
         E.RowSpec("d", P[0], 120, False, 1)]
for cfg in (E.GenConfig(temperature=0.0, natural_answer_cap=40, forced_answer_cap=20),
            E.GenConfig(temperature=0.6, top_p=0.95, natural_answer_cap=40, forced_answer_cap=20)):
    F.disable_gqa_decode()
    ref = E.run_rows(m, specs, cfg)
    F.enable_gqa_decode()
    new = E.run_rows(m, specs, cfg, prealloc=True)
    F.disable_gqa_decode()
    for r0, r1 in zip(ref, new):
        assert r0["output_ids"] == r1["output_ids"] and r0["provenance"] == r1["provenance"], r0["gen_id"]
    print(f"engine T={cfg.temperature}: identical ids/provenance for {len(ref)} rows; "
          f"lengths {[len(r['output_ids']) for r in ref]}; categories {[r['category'] for r in ref]}")
print("FAST PATH TEST PASSED")


# fp16 overflow regression (GPU bug found in Colab validation): large-norm q/k in half precision.
class _M:
    num_key_value_groups = 6


g = torch.Generator().manual_seed(0)
q = (torch.randn(3, 12, 1, 128, generator=g) * 40).half()
k = (torch.randn(3, 2, 300, 128, generator=g) * 40).half()
v = torch.randn(3, 2, 300, 128, generator=g).half()
mask = torch.zeros(3, 1, 1, 300)
mask[0, :, :, :17] = torch.finfo(torch.float16).min  # left padding on row 0
ref = torch.nn.functional.scaled_dot_product_attention(
    q.float(), k.float().repeat_interleave(6, 1), v.float().repeat_interleave(6, 1), attn_mask=mask)
out, _ = F.gqa_sdpa_forward(_M(), q, k, v, mask.half())
err = float((out.float().transpose(1, 2) - ref).abs().max())
print(f"fp16 large-norm GQA decode vs fp32 SDPA: max |diff| {err:.2e}, finite={bool(torch.isfinite(out).all())}")
assert torch.isfinite(out).all() and err < 2e-2
print("FP16 OVERFLOW TEST PASSED")
