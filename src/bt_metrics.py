"""BudgetThinker independent evaluation: grading, metrics, statistics, plots.

Uncertainty: questions are the sampling unit. All 95% CIs are cluster (question-level) percentile
bootstraps; A-vs-B comparisons are paired at the question level (same questions, same seeds) with a
sign-flip permutation test on per-question differences and Holm correction across budgets.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import bt_engine as E

# ----------------------------------------------------------------------------- grading
def grading_text(rec: dict, tokenizer) -> str:
    ids = rec["output_ids"]
    if ids and ids[-1] == E.EOS:
        ids = ids[:-1]
    return tokenizer.decode(ids, skip_special_tokens=False)


def grade_official(records, tokenizer, solutions: dict, eval_dir: str, grader_env: str, work_dir: str):
    """Run the official parser/grader in its isolated environment; returns {gen_id: result}."""
    os.makedirs(work_dir, exist_ok=True)
    inp, out = os.path.join(work_dir, "grade_in.jsonl"), os.path.join(work_dir, "grade_out.jsonl")
    with open(inp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"gen_id": r["gen_id"], "text": grading_text(r, tokenizer),
                                "solution": solutions[r["q_uid"]]}, ensure_ascii=False) + "\n")
    env = {**os.environ, "PYTHONPATH": grader_env}
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bt_grade.py")
    p = subprocess.run([sys.executable, script, eval_dir, inp, out], env=env, cwd=work_dir,
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-3000:])
    print(p.stdout.strip().splitlines()[-1])
    return {d["gen_id"]: d for d in map(json.loads, open(out, encoding="utf-8"))}


def answer_segment(rec: dict) -> str:
    """Text after </think>. For forced stops the engine suffix ends in '\\boxed', which is not part
    of answer_text (that starts after the suffix), so it is restored here."""
    seg = rec.get("answer_text") or ""
    return "**Final Answer**\\boxed" + seg if rec.get("termination") == "forced" else seg


def grade_math_verify(records, tokenizer, gold: dict) -> dict:
    """Secondary grader (HF Math-Verify) on the answer segment; None if unavailable."""
    try:
        from math_verify import parse, verify
    except ImportError:
        return {}
    res = {}
    for r in records:
        seg = answer_segment(r)
        try:
            g = parse("$" + gold[r["q_uid"]] + "$")
            res[r["gen_id"]] = bool(verify(g, parse(seg))) if seg else False
        except Exception:
            res[r["gen_id"]] = None
    return res


def has_balanced_boxed(s: str) -> bool:
    i = s.rfind("boxed")
    if i < 0 or i + 5 >= len(s) or s[i + 5] != "{":
        return False
    depth = 0
    for ch in s[i + 5:]:
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            return True
    return False


# ----------------------------------------------------------------------------- flatten
def flatten(records, official: dict, mv: dict) -> pd.DataFrame:
    rows = []
    for r in records:
        c, b = r["counts"], r["budget"]
        g = official.get(r["gen_id"], {})
        ans = answer_segment(r)
        rows.append({
            "gen_id": r["gen_id"], "run": r["run"], "q_uid": r["q_uid"], "q_level": r["q_level"],
            "budget": b, "condition": r["condition"], "sample_idx": r["sample_idx"], "seed": r["seed"],
            "category": r["category"], "termination": r["termination"], "stop_reason": r["stop_reason"],
            "natural_within": r["category"] == "natural_within_budget",
            "natural_any": r["termination"] == "natural",
            "forced": r["termination"] == "forced",
            "eos_in_think": r["termination"] == "eos_in_think",
            "reasoning_tokens": c["reasoning_tokens"], "reasoning_ratio": c["reasoning_tokens"] / b,
            "reasoning_excl_ctrl": c["reasoning_tokens_excl_ctrl"],
            "answer_tokens": c["answer_tokens"], "generated_total": c["generated_total"],
            "total_ratio": (c["reasoning_tokens"] + c["answer_tokens"]) / b,
            "ctrl_injected": c["ctrl_injected_total"], "ctrl_self": c["ctrl_self_total"],
            "ctrl_self_think": c["ctrl_self_think"], "ctrl_injected_think": c["ctrl_injected_think"],
            "suppressed_think_close": c["suppressed_think_close"], "suppressed_eos": c["suppressed_eos"],
            "think_open_generated": E.THINK_OPEN in r["output_ids"][:3],  # BOS may come first
            "bos_generated": bool(r["output_ids"]) and r["output_ids"][0] == E.BOS,
            "pred": g.get("pred"), "gt": g.get("gt"), "correct": bool(g.get("correct", False)),
            "grader_timeout": g.get("timeout"), "pred_from_boxed": g.get("pred_from_boxed"),
            "correct_mv": mv.get(r["gen_id"]),
            "answer_complete": has_balanced_boxed(ans) and r["stop_reason"] in ("natural_eos", "forced_eos"),
            "ctrl_in_answer": sum(1 for e in r["ctrl_events"] if e["phase"] == "answer"),
            "wall_s": r["timing"]["batch_wall_s"], "gpu": r["timing"].get("gpu"),
        })
    df = pd.DataFrame(rows)
    df["grader_disagree"] = df["correct_mv"].notna() & (df["correct_mv"] != df["correct"])
    return df


# ----------------------------------------------------------------------------- statistics
INDICATORS = {"BFR": "natural_within", "BFR_slack": "natural_any", "FTR": "forced",
              "EOS_in_think": "eos_in_think", "Accuracy": "correct", "Answer_complete": "answer_complete"}


def _per_question(d: pd.DataFrame, value: str, where: str | None = None):
    x = d if where is None else d[d[where]]
    g = x.groupby("q_uid")[value].agg(["sum", "count"])
    return g.reindex(sorted(d["q_uid"].unique())).fillna(0.0)


def cluster_boot(num: np.ndarray, den: np.ndarray, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(reps, len(num)))
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = num[idx].sum(1) / den[idx].sum(1)
    est = num.sum() / den.sum() if den.sum() > 0 else np.nan
    lo, hi = np.nanpercentile(stats, [2.5, 97.5]) if np.isfinite(stats).any() else (np.nan, np.nan)
    return est, lo, hi


def icc_binary(d: pd.DataFrame, col: str) -> float:
    """One-way ANOVA ICC for a 0/1 outcome clustered by question."""
    g = d.groupby("q_uid")[col]
    k = g.size().mean()
    means, grand = g.mean(), d[col].mean()
    n_q = g.ngroups
    if n_q < 2 or k < 2:
        return np.nan
    msb = k * ((means - grand) ** 2).sum() / (n_q - 1)
    msw = g.var(ddof=1).sum() / n_q
    return float((msb - msw) / (msb + (k - 1) * msw)) if (msb + (k - 1) * msw) > 0 else np.nan


def summarize(df: pd.DataFrame, reps=2000) -> pd.DataFrame:
    out = []
    for (run, b, c), d in df.groupby(["run", "budget", "condition"]):
        row = {"run": run, "budget": b, "condition": c, "n_gen": len(d), "n_q": d["q_uid"].nunique()}
        for name, col in INDICATORS.items():
            pq = _per_question(d.assign(_v=d[col].astype(float)), "_v")
            row[name], row[name + "_lo"], row[name + "_hi"] = cluster_boot(pq["sum"].values, pq["count"].values, reps)
        # BUR: reasoning tokens / B among natural-within-budget generations (paper Fig. 3b)
        for name, col in (("BUR", "reasoning_ratio"), ("BUR_total", "total_ratio")):
            pq = _per_question(d, col, "natural_within")
            row[name], row[name + "_lo"], row[name + "_hi"] = cluster_boot(pq["sum"].values, pq["count"].values, reps)
        pq = _per_question(d, "reasoning_tokens")
        row["MeanReasoning"], row["MeanReasoning_lo"], row["MeanReasoning_hi"] = cluster_boot(
            pq["sum"].values, pq["count"].values, reps)
        row["MedianReasoning"] = float(d["reasoning_tokens"].median())
        row["Acc_natural"] = d.loc[d["natural_any"], "correct"].mean()
        row["Acc_forced"] = d.loc[d["forced"], "correct"].mean()
        row["mean_ctrl_self"] = d["ctrl_self"].mean()
        row["frac_gen_with_self_ctrl"] = (d["ctrl_self"] > 0).mean()
        row["mean_ctrl_injected"] = d["ctrl_injected"].mean()
        row["suppressed_think_close_rate"] = (d["suppressed_think_close"] > 0).mean()
        row["ctrl_in_answer_rate"] = (d["ctrl_in_answer"] > 0).mean()
        row["pred_not_boxed"] = (d["pred_from_boxed"] == False).mean()  # noqa: E712
        row["grader_disagree"] = d["grader_disagree"].mean()
        row["ICC_correct"] = icc_binary(d.assign(_c=d["correct"].astype(float)), "_c")
        row["ICC_BFR"] = icc_binary(d.assign(_c=d["natural_within"].astype(float)), "_c")
        out.append(row)
    return pd.DataFrame(out)


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def sign_flip_p(dq: np.ndarray, reps=20000, seed=0) -> float:
    dq = dq[np.isfinite(dq)]
    if len(dq) == 0 or np.allclose(dq, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(dq.mean())
    flips = rng.choice([-1.0, 1.0], size=(reps, len(dq)))
    return float((np.abs((flips * dq).mean(1)) >= obs - 1e-12).mean())


def paired_ab(df: pd.DataFrame, reps=2000) -> pd.DataFrame:
    """B - A per budget, paired by question; cluster bootstrap CI + sign-flip p (Holm over budgets)."""
    metrics = {"BFR": ("natural_within", None), "FTR": ("forced", None), "Accuracy": ("correct", None),
               "Answer_complete": ("answer_complete", None), "BUR": ("reasoning_ratio", "natural_within"),
               "MeanReasoning": ("reasoning_tokens", None)}
    out = []
    for run, dr in df.groupby("run"):
        for name, (col, where) in metrics.items():
            rows = []
            for b, d in dr.groupby("budget"):
                qs = sorted(d["q_uid"].unique())
                num, den = {}, {}
                for c in ("A", "B"):
                    x = d[d["condition"] == c].assign(_v=lambda t: t[col].astype(float))
                    pq = _per_question(x, "_v", where).reindex(qs).fillna(0.0)
                    num[c], den[c] = pq["sum"].values, pq["count"].values
                rng = np.random.default_rng(1)
                idx = rng.integers(0, len(qs), size=(reps, len(qs)))
                with np.errstate(invalid="ignore", divide="ignore"):
                    boot = (num["B"][idx].sum(1) / den["B"][idx].sum(1)) - (num["A"][idx].sum(1) / den["A"][idx].sum(1))
                    est = num["B"].sum() / den["B"].sum() - num["A"].sum() / den["A"].sum()
                    dq = num["B"] / den["B"] - num["A"] / den["A"]
                lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                rows.append({"run": run, "metric": name, "budget": b, "A": num["A"].sum() / max(den["A"].sum(), 1e-9),
                             "B": num["B"].sum() / max(den["B"].sum(), 1e-9), "diff_B_minus_A": est,
                             "ci_lo": lo, "ci_hi": hi, "n_q_paired": int(np.isfinite(dq).sum()),
                             "p_signflip": sign_flip_p(dq)})
            adj = holm([r["p_signflip"] for r in rows])
            for r, a in zip(rows, adj):
                r["p_holm"] = a
            out += rows
    return pd.DataFrame(out)


def ctrl_positions(records) -> pd.DataFrame:
    """One row per control token occurrence (both sources) with alignment to the ideal schedule."""
    rows = []
    for r in records:
        b = r["budget"]
        exp = {tok: pos for pos, tok in E.expected_schedule(b)}
        self_seq = [e["token"] for e in r["ctrl_events"] if e["source"] == "model"]
        ordered = all(x > y for x, y in zip(self_seq, self_seq[1:]))
        for e in r["ctrl_events"]:
            rows.append({"run": r["run"], "gen_id": r["gen_id"], "budget": b, "condition": r["condition"],
                         "source": e["source"], "label": e["label"], "token": e["token"], "pos": e["pos"],
                         "rel_pos": e["pos"] / b, "expected_pos": exp[e["token"]],
                         "dev_tokens": e["pos"] - exp[e["token"]], "dev_rel": (e["pos"] - exp[e["token"]]) / b,
                         "phase": e["phase"], "p_model": e["p_model"], "self_seq_ordered": ordered})
    return pd.DataFrame(rows)


def expected_prob_table(records) -> pd.DataFrame:
    rows = [{"run": r["run"], "budget": r["budget"], "condition": r["condition"], "pos": e["pos"],
             "label": E.CTRL_LABEL[e["token"]], "p": e["p"]}
            for r in records for e in r["expected_ctrl_probs"]]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- plots
COL = {"A": "#C2410C", "B": "#1D4ED8"}
NAME = {"A": "A: no external control tokens", "B": "B: official injection (K=8)"}


def _line(ax, s, metric, ylabel, pct=True):
    for c in ("A", "B"):
        d = s[s["condition"] == c].sort_values("budget")
        k = 100 if pct else 1
        ax.errorbar(d["budget"], d[metric] * k, yerr=[(d[metric] - d[metric + "_lo"]) * k, (d[metric + "_hi"] - d[metric]) * k],
                    marker="o", ms=5, lw=1.8, capsize=3, color=COL[c], label=NAME[c])
    ax.set_xscale("log")
    ax.set_xticks(sorted(s["budget"].unique()))
    ax.set_xticklabels([str(x) for x in sorted(s["budget"].unique())])
    ax.set_xlabel("Thinking budget B (tokens)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)


def make_plots(summary: pd.DataFrame, df: pd.DataFrame, ctrl: pd.DataFrame, out_dir: str, run: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    s = summary[summary["run"] == run]
    paths = []
    for metric, ylabel, fn, pct in (("BFR", "Budget Following Ratio (%)", "bfr", True),
                                    ("BUR", "Budget Utilization Ratio (%)", "bur", True),
                                    ("Accuracy", "Accuracy (%)", "accuracy", True),
                                    ("FTR", "Forced Termination Rate (%)", "ftr", True)):
        fig, ax = plt.subplots(figsize=(6, 4))
        _line(ax, s, metric, ylabel, pct)
        ax.set_title(f"{ylabel.split(' (')[0]} vs budget  [{run}]", fontsize=10)
        fig.tight_layout()
        p = os.path.join(out_dir, f"{run}_{fn}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    d = df[df["run"] == run]
    budgets = sorted(d["budget"].unique())
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.2 * len(budgets), 3.4), squeeze=False)
    for ax, b in zip(axes[0], budgets):
        for c in ("A", "B"):
            x = d[(d["budget"] == b) & (d["condition"] == c)]["reasoning_ratio"]
            ax.hist(x, bins=np.linspace(0, 1.05, 22), alpha=0.55, color=COL[c], label=c)
        ax.axvline(1.0, color="k", lw=0.8, ls="--")
        ax.set_title(f"B={b}", fontsize=9)
        ax.set_xlabel("reasoning tokens / B")
    axes[0][0].set_ylabel("generations")
    axes[0][0].legend(frameon=False)
    fig.suptitle(f"Reasoning length distribution [{run}]", fontsize=10)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{run}_length_dist.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    cs = ctrl[(ctrl["run"] == run) & (ctrl["source"] == "model")] if len(ctrl) else ctrl
    fig, ax = plt.subplots(figsize=(7, 3.6))
    if len(cs):
        for lab in sorted(cs["label"].unique(), reverse=True):
            x = cs[cs["label"] == lab]["rel_pos"]
            ax.hist(x, bins=np.linspace(0, 1.1, 45), alpha=0.6, label=f"self {lab}")
        for k in range(1, 8):
            ax.axvline(k / 8, color="grey", lw=0.6, ls=":")
        ax.legend(frameon=False, fontsize=7, ncol=4)
    else:
        ax.text(0.5, 0.5, "No self-generated control tokens observed", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("position / B  (dotted: ideal k/8 marks)")
    ax.set_ylabel("count")
    ax.set_title(f"Positions of self-generated control tokens [{run}]", fontsize=10)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{run}_self_ctrl_positions.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths
