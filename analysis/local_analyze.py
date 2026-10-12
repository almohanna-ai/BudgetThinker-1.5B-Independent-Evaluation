"""Grade and analyse a downloaded results package locally (same functions as notebook cells 29 and 34).

usage: python -I tools/local_analyze.py <results_dir> <tag> <run>[,<run>...] <pylib> <mvlib> <tokenizer_dir>
                                       <official_eval_dir> <grader_env> <math500.jsonl>
Writes metrics/<tag>_*.csv and reports/figures/<run>_*.png inside <results_dir>.

Math-Verify enforces its timeouts with signals/multiprocessing, which fails on Windows; locally they are
disabled (parsing_timeout=None, timeout_seconds=None). The official grader keeps its 3 s pebble timeout.
"""
import os
import sys

RES, TAG, RUNS, PYLIB, MVLIB, TOK, EVAL_DIR, GRADER_ENV, MATH500 = sys.argv[1:10]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [PYLIB, MVLIB, os.path.join(HERE, "..", "src")]


def main():
    import math_verify
    from transformers import AutoTokenizer

    import bt_metrics as M
    import bt_runner as R

    _parse, _verify = math_verify.parse, math_verify.verify
    math_verify.parse = lambda *a, **k: _parse(*a, **{**k, "parsing_timeout": None})
    math_verify.verify = lambda *a, **k: _verify(*a, **{**k, "timeout_seconds": None})

    tok = AutoTokenizer.from_pretrained(TOK)
    math = R.load_math500(MATH500)
    sol = {q["unique_id"]: q["solution"] for q in math}
    gold = {q["unique_id"]: q["answer"] for q in math}
    raw = os.path.join(RES, "raw_generations")
    recs = [r for rn in RUNS.split(",") for r in R.read_units(raw, rn)]
    print(f"{len(recs)} generations from {RUNS}", flush=True)
    off = M.grade_official(recs, tok, sol, EVAL_DIR, GRADER_ENV, os.path.join(RES, "metrics", "grading", TAG))
    mv = M.grade_math_verify(recs, tok, gold)
    df = M.flatten(recs, off, mv)
    summ, pab, ctrl = M.summarize(df), M.paired_ab(df), M.ctrl_positions(recs)
    out = os.path.join(RES, "metrics")
    df.to_csv(os.path.join(out, f"{TAG}_generations.csv"), index=False)
    summ.to_csv(os.path.join(out, f"{TAG}_summary.csv"), index=False)
    pab.to_csv(os.path.join(out, f"{TAG}_paired_AB.csv"), index=False)
    ctrl.to_csv(os.path.join(out, f"{TAG}_ctrl_tokens.csv"), index=False)
    M.expected_prob_table(recs).to_csv(os.path.join(out, f"{TAG}_expected_ctrl_probs.csv"), index=False)
    figs = os.path.join(RES, "reports", "figures")
    os.makedirs(figs, exist_ok=True)
    for rn in RUNS.split(","):
        M.make_plots(summ, df, ctrl, figs, rn)
    print(f"done: {len(df)} rows, accuracy {df.correct.mean():.3f}, grader disagreements {int(df.grader_disagree.sum())}")


if __name__ == "__main__":  # Windows spawn (pebble/multiprocessing) re-imports this module
    main()
