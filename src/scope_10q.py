# Approved scope (user, 2026-10-11): 10 questions instead of 100 "to get a picture".
# The 10 = first 2 questions of each MATH level in the fixed 100-question order
# (select_questions(seed=20251011)); budgets, samples, conditions and decoding are unchanged.
import json
import time

QS_100 = list(QS)
QS = [q for L in range(1, 6) for q in [x for x in QS_100 if int(x["level"]) == L][:2]]
RPB = {500: 48, 1000: 48, 2000: 48, 4000: 32, 6000: 32, 10000: 16, "default": 16}
APPROVED_FULL_RUN = True
FULL_RUNS = ["greedy_ref", "sampled"]
scope = {
    "approved": "user message 2026-10-11: run 10 questions instead of 100",
    "n_questions": len(QS),
    "rule": "first 2 per MATH level in the fixed 100-question order (seed 20251011)",
    "questions": [{"unique_id": q["unique_id"], "level": int(q["level"]), "answer": q["answer"]} for q in QS],
    "budgets": CFG["budgets"],
    "n_samples": CFG["sampling"]["n_samples"],
    "conditions": ["A", "B"],
    "runs": FULL_RUNS,
    "rows_per_batch": {str(k): v for k, v in RPB.items()},
    "engine": "fast" if FAST else "stock",
    "expected_generations": len(QS) * len(CFG["budgets"]) * 2 * (1 + CFG["sampling"]["n_samples"]),
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
json.dump(scope, open("configs/run_scope_10q.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in scope.items() if k != "questions"}, indent=1))
for q in scope["questions"]:
    print(q["level"], q["unique_id"], q["answer"])
