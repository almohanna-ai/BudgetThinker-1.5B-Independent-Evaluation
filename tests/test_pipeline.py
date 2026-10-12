"""End-to-end CPU test: real tokenizer + tiny random Qwen2 through runner, grader, metrics, plots.

Run: python -I tests/test_pipeline.py <pylib> <tokenizer_dir> <official_eval_dir> <grader_env> <math500.jsonl> <work>
"""
import os
import shutil
import sys

PYLIB, TOK, EVAL_DIR, GRADER_ENV, MATH500, WORK = sys.argv[1:7]
sys.path.insert(0, PYLIB)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

import bt_engine as E  # noqa: E402
import bt_metrics as M  # noqa: E402
import bt_runner as R  # noqa: E402

shutil.rmtree(WORK, ignore_errors=True)
tok = AutoTokenizer.from_pretrained(TOK)
torch.manual_seed(0)
m = Qwen2ForCausalLM(Qwen2Config(vocab_size=151672, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                 num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                 tie_word_embeddings=False, sliding_window=None)).eval()
with torch.no_grad():
    m.lm_head.weight *= 40
    m.lm_head.weight[E.THINK_CLOSE] *= 1.6
    m.lm_head.weight[E.EOS] *= 1.3

rows = R.load_math500(MATH500)
qs = R.select_questions(rows, 100, seed=20251011)
assert len(qs) == 100 and len({q["unique_id"] for q in qs}) == 100
assert sorted(sum(1 for q in qs if int(q["level"]) == L) for L in range(1, 6)) == [20] * 5
assert [q["unique_id"] for q in R.select_questions(rows, 100, seed=20251011)] == [q["unique_id"] for q in qs]
pilot_qs = qs[:3]
cfg = E.GenConfig(temperature=0.6, top_p=0.95, natural_answer_cap=40, forced_answer_cap=20)
meta = {"model_id": "tiny-random", "model_revision": "none", "dataset_id": "HuggingFaceH4/MATH-500",
        "dataset_revision": "test"}
units = R.make_units(pilot_qs, [64, 128], ["A", "B"], 2, "pipe", 0.6, 0.95, tok, meta)
log = R.make_logger(os.path.join(WORK, "logs", "pipe.log"))
raw = os.path.join(WORK, "raw_generations")
st = R.run_units(m, tok, units, cfg, raw, "pipe", {64: 8, 128: 4}, log, max_units=4)
assert len(st["batches"]) >= 2
st2 = R.run_units(m, tok, units, cfg, raw, "pipe", {64: 8, 128: 4}, log)  # resume: only the rest
assert sum(b["units"] for b in st2["batches"]) == len(units) - 4
st3 = R.run_units(m, tok, units, cfg, raw, "pipe", {64: 8, 128: 4}, log)
assert st3["batches"] == []
recs = R.read_units(raw, "pipe")
assert len(recs) == 3 * 2 * 2 * 2 and len({r["gen_id"] for r in recs}) == len(recs)
# paired seeds: A and B of the same sample share the seed and agree before the first injection
by = {(r["unit_id"], r["sample_idx"], r["condition"]): r for r in recs}
for (u, s, c), r in by.items():
    if c == "B":
        a = by[(u, s, "A")]
        assert a["seed"] == r["seed"]
        k = r["provenance"].find("i")
        if k > 0:
            assert a["output_ids"][:k] == r["output_ids"][:k]

sol = {q["unique_id"]: q["solution"] for q in rows}
gold = {q["unique_id"]: q["answer"] for q in rows}
off = M.grade_official(recs, tok, sol, EVAL_DIR, GRADER_ENV, os.path.join(WORK, "metrics", "grading"))
assert len(off) == len(recs)
df = M.flatten(recs, off, M.grade_math_verify(recs, tok, gold))
summ = M.summarize(df, reps=200)
pab = M.paired_ab(df, reps=200)
ctrl = M.ctrl_positions(recs)
paths = M.make_plots(summ, df, ctrl, os.path.join(WORK, "reports", "figures"), "pipe")
assert all(os.path.exists(p) for p in paths)
print(summ[["budget", "condition", "n_gen", "BFR", "FTR", "BUR", "Accuracy", "MeanReasoning"]].to_string())
print(pab[["metric", "budget", "A", "B", "diff_B_minus_A", "p_signflip", "p_holm"]].head(6).to_string())
print(df["category"].value_counts().to_dict(), "| ctrl rows", len(ctrl))
print("PIPELINE TEST PASSED")
