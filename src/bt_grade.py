"""Grade generations with the official BudgetThinker evaluation code (Qwen2.5-Math parser/grader).

Runs as a subprocess inside an isolated package dir (sympy==1.12, antlr4-python3-runtime==4.11.1,
latex2sympy2 from the repo), because those pins conflict with the torch environment.

Usage:
    python bt_grade.py <official_eval_dir> <in.jsonl> <out.jsonl>
Input lines:  {"gen_id", "text", "solution"}   (text = generated output, EOS removed)
Output lines: {"gen_id", "pred", "gt", "correct", "timeout", "pred_from_boxed"}

Mirrors evaluation/remaining_eval_multi_process.py + evaluate.py at commit 481da90:
    pred = strip_string(extract_answer(code, "math500"))         # parser.run_execute
    gt   = extract_answer(solution.strip(), "math500")           # evaluate() -> parse_ground_truth
    correct = math_equal(pred, gt) with a 3 s timeout (pebble), timeout -> False
"""
import json
import sys

EVAL_DIR, IN_PATH, OUT_PATH = sys.argv[1:4]
sys.path.insert(0, EVAL_DIR)

from concurrent.futures import TimeoutError  # noqa: E402

from pebble import ProcessPool  # noqa: E402

from grader import math_equal_process  # noqa: E402  (official)
from parser import extract_answer, run_execute  # noqa: E402  (official)


def main():
    rows = [json.loads(l) for l in open(IN_PATH, encoding="utf-8")]
    preds, gts = [], []
    for r in rows:
        pred, _ = run_execute(None, r["text"], "deepseek3", "math500")
        preds.append(pred)
        gts.append(extract_answer(r["solution"].strip(), "math500"))
    params = [(i, p, g) for i, (p, g) in enumerate(zip(preds, gts))]
    scores, timeouts = [], []
    with ProcessPool(max_workers=1) as pool:
        it = pool.map(math_equal_process, params, timeout=3).result()
        while True:
            try:
                scores.append(bool(next(it)))
                timeouts.append(False)
            except StopIteration:
                break
            except TimeoutError:
                scores.append(False)
                timeouts.append(True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r, p, g, s, t in zip(rows, preds, gts, scores, timeouts):
            f.write(json.dumps({"gen_id": r["gen_id"], "pred": p, "gt": g, "correct": s, "timeout": t,
                                "pred_from_boxed": "boxed" in r["text"]}, ensure_ascii=False) + "\n")
    print(f"graded {len(rows)}: correct {sum(scores)}, timeouts {sum(timeouts)}")


if __name__ == "__main__":
    main()
