"""Check that the official vLLM patch can parse the budget from our prompt token ids.

The real to_replace/vllm/worker/model_runner.py (lines 1845-1884 @481da90) does not read the budget
from an env var (as the README snippet suggests): it searches the prompt ids for two fixed token
sequences A (before the digits) and B (after the digits) and decodes digit tokens (id - 15).
If A/B are not found, it prints "budget is None" and injects nothing.

Run: python -I tests/test_budget_parse.py <pylib> <tokenizer_dir> <math500.jsonl>
"""
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from transformers import AutoTokenizer  # noqa: E402

import bt_engine as E  # noqa: E402

A = [7, 12548, 7274, 2878, 220]
B = [11211, 476, 16745, 11, 220, 22, 3281, 11211, 320, 220, 151671]


def official_parse(prompt_ids):
    """Literal port of the budget-parsing loop (first sequence group only)."""
    a_indices = b_indices = None
    lst = list(prompt_ids)
    for i in range(len(lst)):
        if lst[i:i + len(A)] == A:
            a_indices = i + len(A)
        if lst[i:i + len(B)] == B:
            b_indices = i
    if a_indices is None or b_indices is None:
        return None
    between = lst[a_indices:b_indices]
    budget = 0
    for i in range(len(between)):
        budget += (between[i] - 15) * (10 ** (len(between) - i - 1))
    return budget


tok = AutoTokenizer.from_pretrained(sys.argv[2])
print("A =", repr(tok.decode(A)), "| B =", repr(tok.decode(B)))
rows = [json.loads(l) for l in open(sys.argv[3], encoding="utf-8")]
fails = 0
for q in rows:
    for b in (500, 1000, 2000, 4000, 6000, 10000):
        got = official_parse(tok(E.official_prompt(q["problem"], b)).input_ids)
        if got != b:
            fails += 1
            if fails <= 5:
                print("MISMATCH", q["unique_id"], b, got)
print(f"official budget parse == B for {6 * len(rows) - fails}/{6 * len(rows)} (question, budget) prompts")
assert fails == 0
