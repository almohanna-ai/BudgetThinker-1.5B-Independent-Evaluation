"""Extra figures and numbers for the Word report, computed from the extracted results package.

usage: python -I tools/report_figures.py <bt_results dir> <out dir>
"""
import glob
import gzip
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
COL = {"A": "#C2410C", "B": "#1D4ED8"}
NAME = {"A": "A: no external control tokens", "B": "B: official injection (K=8)"}
BUDGETS = [500, 1000, 2000, 4000, 6000, 10000]

g = pd.read_csv(os.path.join(src, "metrics", "full_generations.csv"))
s = g[g.run == "sampled"]

# 1) reasoning-length distribution, 2 x 3 grid (readable on an A4 page)
fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharey=False)
bins = np.linspace(0, 1.05, 22)
for ax, b in zip(axes.flat, BUDGETS):
    for c in ("A", "B"):
        x = s[(s.budget == b) & (s.condition == c)].reasoning_tokens / b
        ax.hist(x.clip(upper=1.04), bins=bins, color=COL[c], alpha=0.55, label=NAME[c])
    ax.axvline(1.0, ls="--", lw=1, color="k")
    ax.set_title(f"B = {b}", fontsize=10)
    ax.set_xlabel("reasoning tokens / B", fontsize=9)
    ax.tick_params(labelsize=8)
axes[0, 0].set_ylabel("generations", fontsize=9)
axes[1, 0].set_ylabel("generations", fontsize=9)
axes[0, 0].legend(fontsize=8, loc="upper left")
n_cell = int(s.groupby(["budget", "condition"]).size().max())
fig.suptitle(f"Reasoning length relative to budget (sampled; {n_cell} generations per condition and budget)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(out, "length_dist_grid.png"), dpi=200)
plt.close(fig)

# 2) RQ1: model probability of the expected control token at schedule positions + trace check
e = pd.read_csv(os.path.join(src, "metrics", "full_expected_ctrl_probs.csv"))
fig, ax = plt.subplots(figsize=(7.5, 3.8))
lb = np.linspace(-40, 0, 41)
for c in ("A", "B"):
    v = np.log10(e[e.condition == c].p.clip(lower=1e-40))
    ax.hist(v, bins=lb, color=COL[c], alpha=0.55,
            label=f"{NAME[c]}  (n={len(v)}, max={10 ** v.max():.1e})")
ax.axvline(-3, ls="--", color="k", lw=1)
ax.text(-3.3, ax.get_ylim()[1] * 0.9, "p = 1e-3", ha="right", fontsize=8)
ax.set_xlabel("log10 P(expected control token) at its scheduled position", fontsize=9)
ax.set_ylabel("positions", fontsize=9)
ax.set_title("Probability the model assigns to the scheduled control token", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out, "rq1_expected_prob.png"), dpi=200)
plt.close(fig)

# 3) numbers quoted in the report
n_gen = n_trace = n_self = tok = 0
for p in glob.glob(os.path.join(src, "raw_generations", "**", "*.jsonl.gz"), recursive=True):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n_gen += 1
            n_trace += len(r["ctrl_trace"])
            n_self += r["counts"]["ctrl_self_total"]
            tok += r["counts"]["generated_total"]
smoke_path = os.path.join(src, "raw_generations", "smoke.json")
smoke = json.load(open(smoke_path)) if os.path.exists(smoke_path) else {"records": []}
for r in smoke["records"]:
    n_gen += 1
    n_trace += len(r["ctrl_trace"])
    n_self += r["counts"]["ctrl_self_total"]
    tok += r["counts"]["generated_total"]
print(f"generations={n_gen} generated_tokens={tok} self_ctrl={n_self} steps_ctrl_mass_gt_1e-3={n_trace}")
for c in ("A", "B"):
    v = e[e.condition == c].p
    print(f"expected-token prob {c}: n={len(v)} max={v.max():.3e} median={v.median():.3e}")
