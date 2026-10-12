"""BudgetThinker independent evaluation: token-level generation engine (HF transformers).

Re-implements the official BudgetThinker inference control at the token level so that
every generated token carries an unambiguous provenance tag:

    m  sampled by the model
    i  control token injected by the engine (it overwrote the token the model sampled)
    s  engine-appended forced-termination suffix

Official references (MobileLLM/BudgetThinker @ 481da9045f761ba34899271db846d659110b7065):
  * Injection rule: to_replace/vllm/worker/model_runner.py, branch remaining="8ratio"
    (patch for vLLM 0.7.3). Ported literally in `official_rule_reference` and incrementally
    in `RowState._injection`.
  * Prompt: evaluation/eval_tools.py::_8ratio_prompt + PROMPT_TEMPLATES["deepseek3"].
  * Stage-1 length cap: evaluation/remaining_eval_multi_process.py, max_tokens = B + B//50.
  * Forced-termination suffix: same file, stage 2: "\n</think>\n\n**Final Answer**\\boxed".
"""
from __future__ import annotations

import dataclasses
import hashlib
import time
from typing import Optional

import torch

# --- Token ids of Xin-Rui/BudgetThinker-1.5b (verified against its tokenizer) -------------
CTRL_IDS = tuple(range(151665, 151672))  # "\n<remaining>1/8</remaining>\n" .. 7/8
CTRL_SET = frozenset(CTRL_IDS)
CTRL_LABEL = {151665 + i: f"{i + 1}/8" for i in range(7)}
FIRST_CTRL, LAST_CTRL = 151671, 151665  # 7/8 is injected first, 1/8 last
THINK_OPEN, THINK_CLOSE = 151648, 151649
EOS = 151643  # <｜end▁of▁sentence｜>
BOS = 151646  # <｜begin▁of▁sentence｜>
PAD = 151643
FORCED_SUFFIX_TEXT = "\n</think>\n\n**Final Answer**\\boxed"
FORCED_SUFFIX_IDS = (198, 151649, 271, 334, 19357, 21806, 334, 59, 79075)
PROMPT_PREFIX = "Return your final response within \\boxed{}. "


def official_prompt(problem: str, budget: int) -> str:
    """Prompt string exactly as built by the official evaluation for tip=8ratio, deepseek3."""
    ctrl = " , ".join(f"\n<remaining>{k}/8</remaining>\n" for k in range(7, 0, -1))
    tip = (f"\n(Complete thinking within {budget} tokens or fewer, 7 special tokens ( {ctrl} ) "
           "will split the thinking process into 8 parts.)")
    return "<｜User｜>" + PROMPT_PREFIX + problem + tip + "<｜Assistant｜>"


def stage1_cap(budget: int) -> int:
    """Official stage-1 max_tokens: B + B//50."""
    return budget + budget // 50


def injection_delta(budget: int, k: int = 8) -> int:
    return budget // k + 1


def expected_schedule(budget: int, k: int = 8) -> list[tuple[int, int]]:
    """(position, token) pairs the official rule produces when the model emits no control token."""
    d = injection_delta(budget, k)
    out, pos = [], d
    for tok in range(FIRST_CTRL, LAST_CTRL - 1, -1):
        out.append((pos, tok))
        pos += d + 1
    return out


def official_rule_reference(output_token_ids, budget: int, n_parts: int = 8) -> Optional[int]:
    """Literal port of the vLLM patch (remaining='8ratio'). Returns the id that overwrites the
    token sampled at index len(output_token_ids), or None. Used to test the fast version."""
    delta = budget // n_parts + 1
    special_tokens = [151665 + i for i in range(n_parts - 1)]
    last_special_token_idx, last_special_token = None, None
    for idx in range(len(output_token_ids) - 1, -1, -1):
        token_id = output_token_ids[idx]
        if token_id in special_tokens:
            last_special_token_idx = idx
            last_special_token = token_id
            break
    if last_special_token == 151665:
        return None
    if last_special_token_idx is not None:
        distance_to_last_special_token = len(output_token_ids) - last_special_token_idx - 1
        if distance_to_last_special_token == delta:
            return last_special_token - 1
        return None
    last_special_token = 151671 + 1
    if len(output_token_ids) == delta:
        return last_special_token - 1
    return None


def derive_seed(*parts) -> int:
    """Stable 63-bit seed from arbitrary parts (independent of Python hash randomization)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


@dataclasses.dataclass
class GenConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    n_parts: int = 8
    natural_answer_cap: int = 512   # tokens allowed after a natural </think>
    forced_answer_cap: int = 50     # tokens allowed after the forced suffix (paper, Sec. 3.1)
    ctrl_trace_threshold: float = 1e-3
    inject_in_answer: bool = True   # official rule keeps running after </think>


@dataclasses.dataclass
class RowSpec:
    gen_id: str
    prompt_ids: list
    budget: int
    inject: bool  # True = Condition B (official injection), False = Condition A
    seed: int
    meta: dict = dataclasses.field(default_factory=dict)


class RowState:
    """Per-generation state machine: think -> (answer | suffix -> forced_answer) -> done."""

    def __init__(self, spec: RowSpec, cfg: GenConfig):
        self.spec, self.cfg = spec, cfg
        self.cap = stage1_cap(spec.budget)
        self.delta = injection_delta(spec.budget, cfg.n_parts)
        self.expected = dict(expected_schedule(spec.budget, cfg.n_parts))
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(spec.seed)
        self.out: list[int] = []
        self.prov: list[str] = []
        self.phase = "think"
        self.done = False
        self.last_ctrl_idx: Optional[int] = None
        self.last_ctrl_tok: Optional[int] = None
        self.ctrl_events: list[dict] = []
        self.ctrl_trace: list[list] = []      # [pos, total ctrl mass, argmax ctrl id, its prob]
        self.expected_probs: list[dict] = []  # model prob of the scheduled token at scheduled pos
        self.overwritten: list[int] = []
        self.think_end: Optional[int] = None
        self.answer_start: Optional[int] = None
        self.termination: Optional[str] = None
        self.stop_reason: Optional[str] = None
        self.suffix_i = 0
        self.t_finish: Optional[float] = None

    # Official rule, O(1) per step; equivalent to official_rule_reference (see tests).
    def _injection(self) -> Optional[int]:
        n = len(self.out)
        if self.last_ctrl_tok == LAST_CTRL:
            return None
        if self.last_ctrl_idx is not None:
            return self.last_ctrl_tok - 1 if n - self.last_ctrl_idx - 1 == self.delta else None
        return FIRST_CTRL if n == self.delta else None

    def _append(self, tok: int, prov: str):
        if tok in CTRL_SET:
            self.last_ctrl_idx, self.last_ctrl_tok = len(self.out), tok
        self.out.append(tok)
        self.prov.append(prov)

    def _finish(self, reason: str):
        self.done, self.stop_reason, self.phase = True, reason, "done"
        self.t_finish = time.time()

    def _log_probs(self, n: int, ctrl_p: list):
        mass = sum(ctrl_p)
        if mass > self.cfg.ctrl_trace_threshold:
            j = max(range(7), key=ctrl_p.__getitem__)
            self.ctrl_trace.append([n, round(mass, 6), CTRL_IDS[j], round(ctrl_p[j], 6)])
        if n in self.expected:
            tok = self.expected[n]
            self.expected_probs.append({"pos": n, "token": tok, "p": ctrl_p[CTRL_IDS.index(tok)],
                                        "phase": self.phase})

    def _sampled_or_injected(self, sampled: int, ctrl_p: list, allow_inject: bool) -> int:
        n = len(self.out)
        self._log_probs(n, ctrl_p)
        tok, prov = sampled, "m"
        forced = self._injection() if (self.spec.inject and allow_inject) else None
        if forced is not None:
            tok, prov = forced, "i"
            self.overwritten.append(sampled)
        if tok in CTRL_SET:
            self.ctrl_events.append({
                "pos": n, "token": tok, "label": CTRL_LABEL[tok],
                "source": "injected" if prov == "i" else "model", "phase": self.phase,
                "p_model": ctrl_p[CTRL_IDS.index(tok)],
                "overwrote": sampled if prov == "i" else None,
            })
        self._append(tok, prov)
        return tok

    def step(self, sampled: int, ctrl_p: list) -> int:
        """Consume the token the model sampled at this step; return the token actually emitted."""
        if self.phase == "think":
            tok = self._sampled_or_injected(sampled, ctrl_p, True)
            if tok == THINK_CLOSE:
                self.think_end = len(self.out) - 1
                self.answer_start = len(self.out)
                self.termination, self.phase = "natural", "answer"
            elif tok == EOS:
                self.termination = "eos_in_think"
                self._finish("eos_in_think")
            elif len(self.out) >= self.cap:
                self.termination, self.phase = "forced", "suffix"
            return tok
        if self.phase == "suffix":
            tok = FORCED_SUFFIX_IDS[self.suffix_i]
            self._append(tok, "s")
            self.suffix_i += 1
            if self.suffix_i == len(FORCED_SUFFIX_IDS):
                self.phase, self.answer_start = "forced_answer", len(self.out)
            return tok
        if self.phase in ("answer", "forced_answer"):
            # Official stage 2 is a fresh call, so the rule cannot fire within 50 forced tokens;
            # after a natural </think> the official rule keeps counting positions.
            allow = self.phase == "answer" and self.cfg.inject_in_answer
            tok = self._sampled_or_injected(sampled, ctrl_p, allow)
            cap = self.cfg.natural_answer_cap if self.phase == "answer" else self.cfg.forced_answer_cap
            if tok == EOS:
                self._finish("natural_eos" if self.phase == "answer" else "forced_eos")
            elif len(self.out) - self.answer_start >= cap:
                self._finish("natural_answer_cap" if self.phase == "answer" else "forced_answer_cap")
            return tok
        raise RuntimeError(f"step() on finished row {self.spec.gen_id}")

    def needs_sample(self) -> bool:
        return self.phase in ("think", "answer", "forced_answer")

    def record(self, tokenizer=None) -> dict:
        s, out = self.spec, self.out
        b = s.budget
        if self.termination == "natural":
            reasoning = self.think_end
        elif self.termination == "forced":
            reasoning = self.cap
        else:  # eos_in_think
            reasoning = len(out) - 1
        think_slice = range(0, reasoning)
        ctrl_inj_think = sum(1 for i in think_slice if self.prov[i] == "i")
        ctrl_self_think = sum(1 for i in think_slice if self.prov[i] == "m" and out[i] in CTRL_SET)
        ctrl_inj_all = self.prov.count("i")
        ctrl_self_all = sum(1 for t, p in zip(out, self.prov) if p == "m" and t in CTRL_SET)
        answer_tokens = len(out) - self.answer_start if self.answer_start is not None else 0
        counts = {
            "generated_total": len(out),
            "reasoning_tokens": reasoning,  # tokens before </think> (incl. <think> and control tokens)
            "reasoning_tokens_excl_ctrl": reasoning - ctrl_inj_think - ctrl_self_think,
            "reasoning_tokens_excl_injected": reasoning - ctrl_inj_think,
            # The model emits BOS before <think> (as in its SFT targets), so look past it.
            "think_open_generated": THINK_OPEN in out[:3],
            "bos_generated": bool(out and out[0] == BOS),
            "ctrl_injected_think": ctrl_inj_think,
            "ctrl_self_think": ctrl_self_think,
            "ctrl_injected_total": ctrl_inj_all,
            "ctrl_self_total": ctrl_self_all,
            "suffix_tokens": self.prov.count("s"),
            "answer_tokens": answer_tokens,
            "stage1_cap": self.cap,
            "suppressed_think_close": sum(1 for t in self.overwritten if t == THINK_CLOSE),
            "suppressed_eos": sum(1 for t in self.overwritten if t == EOS),
        }
        if self.termination == "natural":
            category = "natural_within_budget" if reasoning <= b else "natural_over_budget"
        else:
            category = self.termination
        rec = {
            "gen_id": s.gen_id, "budget": b, "condition": "B" if s.inject else "A", "seed": s.seed,
            **s.meta,
            "prompt_ids": list(s.prompt_ids), "prompt_len": len(s.prompt_ids),
            "output_ids": list(out), "provenance": "".join(self.prov),
            "termination": self.termination, "category": category, "stop_reason": self.stop_reason,
            "think_end_pos": self.think_end, "answer_start": self.answer_start,
            "counts": counts, "ctrl_events": self.ctrl_events, "ctrl_trace": self.ctrl_trace,
            "expected_ctrl_probs": self.expected_probs,
        }
        if tokenizer is not None:
            rec["text"] = tokenizer.decode(out, skip_special_tokens=False)
            rec["answer_text"] = (tokenizer.decode(out[self.answer_start:], skip_special_tokens=False)
                                  if self.answer_start is not None else "")
        return rec


def sample_tokens(logits: torch.Tensor, temperature: float, top_p: float, gens: list):
    """vLLM-style temperature + top-p sampling with one CPU generator per row (inverse CDF).

    Returns (token ids [b] long, diagnostic probs [b, V] float32). For greedy decoding the
    diagnostic probs are softmax(logits) (temperature 1).
    """
    if temperature == 0:
        return logits.argmax(dim=-1), torch.softmax(logits, dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    sp, si = probs.sort(dim=-1, descending=False)
    if top_p < 1.0:
        mask = sp.cumsum(dim=-1) <= (1.0 - top_p)
        mask[:, -1] = False
        sp = sp.masked_fill(mask, 0.0)
    cum = sp.cumsum(dim=-1)
    u = torch.tensor([[torch.rand(1, generator=g).item()] for g in gens],
                     dtype=cum.dtype, device=cum.device) * cum[:, -1:]
    k = torch.searchsorted(cum, u, right=True).clamp_(max=cum.shape[-1] - 1)
    return si.gather(1, k).squeeze(1), probs


@torch.inference_mode()
def run_rows(model, specs: list[RowSpec], cfg: GenConfig, tokenizer=None, max_steps: int = None,
             prealloc: bool = False):
    """Generate all rows in one left-padded batch; finished rows are pruned from the KV cache.

    prealloc=True uses bt_fast.PreallocCache (same values, no per-step torch.cat)."""
    from transformers import DynamicCache

    dev = next(model.parameters()).device
    n = len(specs)
    width = max(len(s.prompt_ids) for s in specs)
    ids = torch.full((n, width), PAD, dtype=torch.long)
    attn = torch.zeros((n, width), dtype=torch.long)
    for r, s in enumerate(specs):
        ids[r, width - len(s.prompt_ids):] = torch.tensor(s.prompt_ids)
        attn[r, width - len(s.prompt_ids):] = 1
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    ids, attn, pos = ids.to(dev), attn.to(dev), pos.to(dev)
    ctrl_index = torch.tensor(CTRL_IDS, device=dev)

    t0 = time.time()
    if prealloc:
        from bt_fast import PreallocCache
        longest = max(stage1_cap(s.budget) for s in specs) + max(
            cfg.natural_answer_cap, len(FORCED_SUFFIX_IDS) + cfg.forced_answer_cap)
        cache = PreallocCache(width + longest + 2)
    else:
        cache = DynamicCache()
    out = model(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=cache,
                use_cache=True, logits_to_keep=1)
    logits = out.logits[:, -1, :].float()
    next_pos = pos[:, -1:] + 1
    t_prefill = time.time() - t0

    states = [RowState(s, cfg) for s in specs]
    active = list(range(n))
    steps = 0
    while active:
        toks, probs = sample_tokens(logits, cfg.temperature, cfg.top_p, [states[j].gen for j in active])
        ctrl_p = probs.index_select(1, ctrl_index).tolist()
        toks = toks.tolist()
        emitted = [states[j].step(toks[k], ctrl_p[k]) for k, j in enumerate(active)]
        steps += 1
        keep = [k for k, j in enumerate(active) if not states[j].done]
        if not keep or (max_steps is not None and steps >= max_steps):
            break
        if len(keep) < len(active):
            sel = torch.tensor(keep, device=dev)
            cache.batch_select_indices(sel)
            attn, next_pos = attn.index_select(0, sel), next_pos.index_select(0, sel)
            active = [active[k] for k in keep]
            emitted = [emitted[k] for k in keep]
        feed = torch.tensor(emitted, device=dev).unsqueeze(1)
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=1)
        out = model(input_ids=feed, attention_mask=attn, position_ids=next_pos, past_key_values=cache,
                    use_cache=True, logits_to_keep=1)
        logits = out.logits[:, -1, :].float()
        next_pos = next_pos + 1
    wall = time.time() - t0

    recs = []
    for st in states:
        rec = st.record(tokenizer)
        rec["timing"] = {"batch_wall_s": round(wall, 3), "prefill_s": round(t_prefill, 3),
                         "batch_steps": steps, "batch_rows": n,
                         "finish_offset_s": round(st.t_finish - t0, 3) if st.t_finish else None}
        recs.append(rec)
    return recs
