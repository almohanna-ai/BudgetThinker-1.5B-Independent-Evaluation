"""BudgetThinker independent evaluation: question selection, job units, resumable execution.

A *unit* is one (question, budget) pair with all its rows: n samples x conditions (A, B).
Conditions A and B of the same sample index share one seed (common random numbers), so the two
generations are token-identical until the first injected control token.
Each finished unit is written atomically to raw_generations/<run>/B<budget>/<unit_id>.jsonl.gz;
units with an existing file are skipped on restart.
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import platform
import random
import time

import torch

import bt_engine as E


def load_math500(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def select_questions(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Stratified by MATH level (equal share per level), deterministic given seed."""
    rng = random.Random(seed)
    by_level: dict[int, list[dict]] = {}
    for r in rows:
        by_level.setdefault(int(r["level"]), []).append(r)
    levels = sorted(by_level)
    per = [n // len(levels) + (1 if i < n % len(levels) else 0) for i in range(len(levels))]
    chosen = []
    for lvl, k in zip(levels, per):
        pool = sorted(by_level[lvl], key=lambda r: r["unique_id"])
        chosen += rng.sample(pool, k)
    order = rng.sample(range(len(chosen)), len(chosen))  # interleave levels for early-stop balance
    return [chosen[i] for i in order]


def make_units(questions, budgets, conditions, n_samples, run, temperature, top_p, tokenizer, meta):
    units = []
    for q in questions:
        for b in budgets:
            prompt = E.official_prompt(q["problem"], b)
            ids = tokenizer(prompt).input_ids  # adds BOS, as vLLM does for a string prompt
            uid = f"{q['unique_id'].replace('/', '_').replace('.json', '')}_B{b}"
            rows = []
            for s in range(n_samples):
                seed = E.derive_seed(run, q["unique_id"], b, s)
                for c in conditions:
                    rows.append(E.RowSpec(
                        gen_id=f"{run}|{uid}|s{s}|{c}", prompt_ids=ids, budget=b, inject=(c == "B"),
                        seed=seed,
                        meta={**meta, "run": run, "unit_id": uid, "q_uid": q["unique_id"],
                              "q_level": int(q["level"]), "q_subject": q["subject"],
                              "gt_answer": q["answer"], "sample_idx": s, "prompt_text": prompt,
                              "temperature": temperature, "top_p": top_p if temperature > 0 else 1.0}))
            units.append({"unit_id": uid, "budget": b, "rows": rows, "q": q})
    return units


def unit_path(out_dir: str, run: str, unit: dict) -> str:
    return os.path.join(out_dir, run, f"B{unit['budget']}", unit["unit_id"] + ".jsonl.gz")


def write_unit(path: str, records: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_units(out_dir: str, run: str) -> list[dict]:
    recs = []
    root = os.path.join(out_dir, run)
    for dp, _, fs in os.walk(root):
        for fn in sorted(fs):
            if fn.endswith(".jsonl.gz"):
                with gzip.open(os.path.join(dp, fn), "rt", encoding="utf-8") as f:
                    recs += [json.loads(l) for l in f]
    return recs


def env_info() -> dict:
    import transformers
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_mem_gb=round(p.total_memory / 2**30, 2),
                    capability=f"{p.major}.{p.minor}")
    return info


def run_units(model, tokenizer, units, cfg: E.GenConfig, out_dir: str, run: str,
              rows_per_batch: dict, log: logging.Logger, max_units: int | None = None,
              prealloc: bool = False) -> dict:
    """Run pending units grouped by budget; returns timing stats for this call."""
    pending = [u for u in units if not os.path.exists(unit_path(out_dir, run, u))]
    n_done = len(units) - len(pending)
    if max_units is not None:
        pending = pending[:max_units]
    log.info(f"[{run}] {len(units)} units, {n_done} already done, {len(pending)} to run now")
    stats = {"batches": [], "gen_tokens": 0, "wall_s": 0.0}
    by_budget: dict[int, list] = {}
    for u in pending:
        by_budget.setdefault(u["budget"], []).append(u)
    for b in sorted(by_budget):
        queue = by_budget[b]
        cap = rows_per_batch.get(b, rows_per_batch.get("default", 16))
        while queue:
            group, n_rows = [], 0
            while queue and (not group or n_rows + len(queue[0]["rows"]) <= cap):
                n_rows += len(queue[0]["rows"])
                group.append(queue.pop(0))
            specs = [r for u in group for r in u["rows"]]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                recs = E.run_rows(model, specs, cfg, tokenizer, prealloc=prealloc)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(group) == 1 and len(specs) > 1:
                    log.warning(f"OOM at B={b} with {len(specs)} rows; splitting the unit in halves")
                    half = len(specs) // 2
                    recs = E.run_rows(model, specs[:half], cfg, tokenizer, prealloc=prealloc)
                    torch.cuda.empty_cache()
                    recs += E.run_rows(model, specs[half:], cfg, tokenizer, prealloc=prealloc)
                else:
                    log.warning(f"OOM at B={b} with {len(group)} units; requeueing one unit per batch")
                    for u in reversed(group):
                        queue.insert(0, u)
                    rows_per_batch[b] = max(len(group[0]["rows"]), cap // 2)
                    cap = rows_per_batch[b]
                    continue
            wall = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            ntok = sum(len(r["output_ids"]) for r in recs)
            gpu = env_info().get("gpu", "cpu")
            by_id = {r["gen_id"]: r for r in recs}
            for u in group:
                urecs = [by_id[s.gen_id] for s in u["rows"]]
                for r in urecs:
                    r["timing"]["gpu"] = gpu
                    r["timing"]["peak_mem_gb"] = round(peak, 2)
                write_unit(unit_path(out_dir, run, u), urecs)
            row = {"budget": b, "units": len(group), "rows": len(specs), "wall_s": round(wall, 2),
                   "gen_tokens": ntok, "tok_per_s": round(ntok / wall, 1), "peak_mem_gb": round(peak, 2),
                   "max_len": max(len(r["output_ids"]) for r in recs)}
            stats["batches"].append(row)
            stats["gen_tokens"] += ntok
            stats["wall_s"] += wall
            log.info(f"[{run}] B={b} units={len(group)} rows={len(specs)} wall={wall:.1f}s "
                     f"tokens={ntok} ({ntok / wall:.0f} tok/s) peak={peak:.2f}GB")
    return stats


def make_logger(path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log = logging.getLogger("bt")
    log.setLevel(logging.INFO)
    log.propagate = False  # Colab's root handler would print every line twice
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(path)
               for h in log.handlers):
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
    if not any(type(h) is logging.StreamHandler for h in log.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        log.addHandler(sh)
    return log
