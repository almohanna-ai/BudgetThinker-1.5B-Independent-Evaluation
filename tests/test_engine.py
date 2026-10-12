"""CPU tests for bt_engine. Run: python -I tests/test_engine.py <pylib_dir> <official_eval_dir>

1. Prompt string == official _8ratio_prompt(deepseek3 template).
2. Incremental injection rule == literal port of the vLLM patch, on random sequences that
   include model-generated control tokens.
3. State machine on a scripted stub model: natural / forced / eos-in-think paths, provenance,
   counts, suppressed </think>, self-generated control tokens shifting the schedule.
4. Tiny random Qwen2 (fp32): batch invariance under left padding + pruning; A/B rows with the
   same seed are identical up to the first injection.
"""
import os
import random
import sys

sys.path.insert(0, sys.argv[1])
OFFICIAL_EVAL = sys.argv[2]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch  # noqa: E402

import bt_engine as E  # noqa: E402


def test_prompt():
    sys.path.insert(0, OFFICIAL_EVAL)
    import eval_tools  # official
    q = E.PROMPT_PREFIX + "If $f(x) = 2x$, find $f(3)$."
    for b in (500, 2000, 10000):
        official = eval_tools._8ratio_prompt(["<｜User｜>{input}<｜Assistant｜>".format(input=q)], b)[0]
        mine = E.official_prompt("If $f(x) = 2x$, find $f(3)$.", b)
        assert official == mine, (official, mine)
    sys.path.remove(OFFICIAL_EVAL)
    print("ok prompt == official _8ratio_prompt")


def test_rule_parity():
    rng = random.Random(0)
    checked = 0
    for trial in range(400):
        budget = rng.choice([100, 500, 1000, 2000, 4000])
        spec = E.RowSpec("t", [1], budget, True, 0)
        st = E.RowState(spec, E.GenConfig())
        seq = []
        p_self = rng.choice([0.0, 0.002, 0.02])
        for _ in range(rng.randint(50, budget + budget // 4)):
            ref = E.official_rule_reference(seq, budget)
            fast = st._injection()
            assert ref == fast, (trial, len(seq), ref, fast)
            tok = ref if ref is not None else (rng.choice(E.CTRL_IDS) if rng.random() < p_self
                                               else rng.randint(0, 1000))
            st._append(tok, "i" if ref is not None else "m")
            seq.append(tok)
            checked += 1
    print(f"ok incremental rule == literal official port ({checked} decisions)")


def test_schedule():
    for b in (500, 1000, 2000, 4000, 6000, 10000):
        seq, got = [], []
        while len(seq) < b + 200:
            f = E.official_rule_reference(seq, b)
            if f is not None:
                got.append((len(seq), f))
            seq.append(f if f is not None else 5)
        assert got == E.expected_schedule(b), (b, got[:3])
        assert [t for _, t in got] == list(range(151671, 151664, -1))
        assert got[-1][0] < b, (b, got[-1])  # all 7 injections fall inside the budget
    print("ok schedule: 7 tokens 7/8..1/8 at delta + (k-1)(delta+1), delta = B//8 + 1")


class StubModel(torch.nn.Module):
    """Emits script[gen_len] (or token 5) with ~prob 1; same script for every row."""

    def __init__(self, script, prompt_len, ctrl_mass_at=None):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.script, self.prompt_len = script, prompt_len
        self.ctrl_mass_at = ctrl_mass_at or {}

    def forward(self, input_ids, attention_mask, position_ids, past_key_values, use_cache, logits_to_keep):
        b = input_ids.shape[0]
        gen_len = int(position_ids[0, -1]) + 1 - self.prompt_len
        logits = torch.full((b, 1, 151672), -30.0)
        logits[:, 0, self.script.get(gen_len, 5)] = 30.0
        if gen_len in self.ctrl_mass_at:  # give a control token a visible probability
            tok, val = self.ctrl_mass_at[gen_len]
            logits[:, 0, tok] = val
        return type("O", (), {"logits": logits})()


def _run_stub(script, budget, inject, cfg=None, prompt_len=4, ctrl_mass_at=None):
    m = StubModel(script, prompt_len, ctrl_mass_at)
    spec = E.RowSpec("g", list(range(prompt_len)), budget, inject, 1)
    return E.run_rows(m, [spec], cfg or E.GenConfig(temperature=0.0))[0]


def test_state_machine():
    b = 500  # cap 510, delta 63
    sched = E.expected_schedule(b)
    # (a) model never stops -> forced at cap; B gets 7 injections; suffix; forced answer cap 50
    rb = _run_stub({0: E.THINK_OPEN}, b, True)
    ra = _run_stub({0: E.THINK_OPEN}, b, False)
    for r in (ra, rb):
        assert r["termination"] == "forced" and r["category"] == "forced", r["category"]
        assert r["counts"]["reasoning_tokens"] == 510
        assert r["output_ids"][510:519] == list(E.FORCED_SUFFIX_IDS)
        assert r["provenance"][510:519] == "s" * 9
        assert r["counts"]["answer_tokens"] == 50 and r["stop_reason"] == "forced_answer_cap"
    assert [(e["pos"], e["token"]) for e in rb["ctrl_events"]] == sched
    assert all(e["source"] == "injected" for e in rb["ctrl_events"])
    assert rb["provenance"].count("i") == 7 and ra["provenance"].count("i") == 0
    assert ra["counts"]["ctrl_self_total"] == 0
    assert all(x == y for x, y, p in zip(ra["output_ids"], rb["output_ids"], rb["provenance"]) if p != "i")
    # (b) natural </think> at 300 then EOS at 310 -> natural_within_budget
    r = _run_stub({0: E.THINK_OPEN, 300: E.THINK_CLOSE, 310: E.EOS}, b, True)
    assert r["category"] == "natural_within_budget" and r["counts"]["reasoning_tokens"] == 300
    assert r["stop_reason"] == "natural_eos" and r["counts"]["answer_tokens"] == 10
    assert r["counts"]["ctrl_injected_think"] == 4  # positions 63,127,191,255 < 300
    # (c) </think> at 505 (between B and cap) -> natural_over_budget
    r = _run_stub({0: E.THINK_OPEN, 505: E.THINK_CLOSE, 507: E.EOS}, b, False)
    assert r["category"] == "natural_over_budget"
    # (d) model wants </think> exactly at an injection slot -> overwritten (suppressed)
    r = _run_stub({0: E.THINK_OPEN, 63: E.THINK_CLOSE, 64: E.THINK_CLOSE, 70: E.EOS}, b, True)
    assert r["counts"]["suppressed_think_close"] == 1 and r["think_end_pos"] == 64
    assert r["ctrl_events"][0]["overwrote"] == E.THINK_CLOSE
    # (e) self-generated control token: model emits 6/8 at pos 40 -> schedule restarts from it
    r = _run_stub({0: E.THINK_OPEN, 40: 151670}, b, True)
    ev = [(e["pos"], e["token"], e["source"]) for e in r["ctrl_events"]]
    assert ev[0] == (40, 151670, "model") and ev[1] == (104, 151669, "injected"), ev[:3]
    assert r["counts"]["ctrl_self_think"] == 1 and r["counts"]["ctrl_injected_think"] == 5
    # (f) EOS without </think>
    r = _run_stub({0: E.THINK_OPEN, 20: E.EOS}, b, False)
    assert r["category"] == "eos_in_think" and r["stop_reason"] == "eos_in_think"
    # (g) natural answer longer than its cap
    cfg = E.GenConfig(temperature=0.0, natural_answer_cap=30)
    r = _run_stub({0: E.THINK_OPEN, 100: E.THINK_CLOSE}, b, False, cfg)
    assert r["stop_reason"] == "natural_answer_cap" and r["counts"]["answer_tokens"] == 30
    # (h) control-token probability trace and expected-position probabilities
    r = _run_stub({0: E.THINK_OPEN}, b, False, ctrl_mass_at={63: (151671, 29.0)})
    assert any(t[0] == 63 and t[2] == 151671 for t in r["ctrl_trace"])
    e63 = [e for e in r["expected_ctrl_probs"] if e["pos"] == 63][0]
    assert 0.2 < e63["p"] < 0.3, e63  # softmax of (30, 29) ~ (0.73, 0.27)
    print("ok state machine: forced / natural / over-budget / suppressed </think> / self ctrl / eos / caps / traces")


def _tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=151672, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=False, use_cache=True, sliding_window=None)
    m = Qwen2ForCausalLM(cfg).eval()
    with torch.no_grad():
        # A random 64-dim model is almost uniform over 151k tokens, where 1e-7 float noise
        # flips samples; scale the head so distributions are peaked like a real LM's.
        m.lm_head.weight *= 40
        m.lm_head.weight[E.THINK_CLOSE] *= 1.5  # make </think>/EOS reachable (natural paths)
        m.lm_head.weight[E.EOS] *= 1.5
    return m


def test_tiny_model_batching():
    m = _tiny_qwen()
    cfg = E.GenConfig(temperature=0.6, top_p=0.95, natural_answer_cap=40, forced_answer_cap=20)
    a = E.RowSpec("a", [151646, 151644, 10, 11, 12, 151645], 120, True, 11)
    others = [E.RowSpec("b", [151646, 151644, 20, 151645], 64, False, 12),
              E.RowSpec("c", [151646, 151644, 30, 31, 32, 33, 34, 35, 36, 151645], 200, True, 13),
              E.RowSpec("d", [151646, 151644, 40, 41, 151645], 120, False, 11)]
    alone = E.run_rows(m, [a], cfg)[0]
    batched = E.run_rows(m, [others[0], a, *others[1:]], cfg)
    got = batched[1]
    assert got["output_ids"] == alone["output_ids"], "row changed when batched with padding/pruning"
    assert got["provenance"] == alone["provenance"]
    lens = sorted(len(r["output_ids"]) for r in batched)
    assert lens[0] < lens[-1], "expected rows to finish at different steps (pruning exercised)"
    # greedy too
    g = E.GenConfig(temperature=0.0, natural_answer_cap=40, forced_answer_cap=20)
    assert E.run_rows(m, [a], g)[0]["output_ids"] == E.run_rows(m, [others[2], a], g)[1]["output_ids"]
    # A/B pairing: same prompt, same seed -> identical until the first injection
    for seed in range(100, 140):  # find a seed whose generation outlives the first slot (delta=5)
        pa = E.RowSpec("pa", a.prompt_ids, 32, False, seed)
        pb = E.RowSpec("pb", a.prompt_ids, 32, True, seed)
        ra, rb = E.run_rows(m, [pa, pb], cfg)
        first_inj = rb["provenance"].find("i")
        if first_inj >= 0:
            break
    assert first_inj == E.injection_delta(32), first_inj
    assert ra["output_ids"][:first_inj] == rb["output_ids"][:first_inj]
    assert rb["output_ids"][first_inj] == E.FIRST_CTRL
    cats = {r["category"] for r in batched + [ra, rb]}
    print(f"ok tiny Qwen2: batch-invariant with padding+pruning; A/B identical before first "
          f"injection (pos {first_inj}); categories seen {sorted(cats)}")


if __name__ == "__main__":
    test_prompt()
    test_rule_parity()
    test_schedule()
    test_state_machine()
    test_tiny_model_batching()
    print("ALL TESTS PASSED")
