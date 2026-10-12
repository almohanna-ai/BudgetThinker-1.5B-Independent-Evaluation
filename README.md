# Does a Budget-Aware Reasoning Model Track Its Own Thinking Budget?

**An independent evaluation of BudgetThinker-1.5B on 100 MATH-500 problems**

**Salam Mohammed Ghazi Almohanna**<br>
School of Software, Faculty of Electronic and Information Engineering, Xi'an Jiaotong University<br>
Department of Computer Engineering Technologies, Imam Al-Kadhim University College

This repository contains the code, configuration and summary results of an independent evaluation of the publicly
released [BudgetThinker-1.5B](https://huggingface.co/Xin-Rui/BudgetThinker-1.5b) checkpoint
([paper](https://arxiv.org/abs/2508.17196), [official code](https://github.com/MobileLLM/BudgetThinker)).
BudgetThinker controls chain-of-thought length by periodically inserting control tokens that announce the remaining
fraction of a token budget. We test whether the model tracks its budget on its own (condition **A**, no external
control tokens) or depends on the external injection (condition **B**, official injection rule, K = 8).

- Technical report (Zenodo): [10.5281/zenodo.23318433](https://doi.org/10.5281/zenodo.23318433)
- Generations, analysis outputs and original packages (Hugging Face):
  [almohanna-ai/BudgetThinker-1.5B-Independent-Evaluation](https://huggingface.co/datasets/almohanna-ai/BudgetThinker-1.5B-Independent-Evaluation)

## Research questions

| | Question |
|---|---|
| RQ1 | Does the model generate control tokens on its own? |
| RQ2 | Does the model follow the budget without injection? |
| RQ3 | How much does the official injection (K = 8) improve the budget-following ratio (BFR)? |
| RQ4 | What is the effect of injection on budget utilisation (BUR), accuracy and answer completeness? |

## Main findings

100 MATH-500 problems stratified by difficulty, budgets 500 / 1,000 / 2,000 / 4,000 / 6,000 / 10,000 tokens, 8 samples
per problem and condition (T = 0.6, top-p = 0.95) plus a greedy reference: 10,800 generations, bf16 on an NVIDIA A100.

- **RQ1:** the model never produced a control token by itself, and assigned at most about 1.0 × 10⁻⁶ probability to
  the scheduled token.
- **RQ2:** without injection it rarely finished within small budgets (BFR 23.0%, 10.6% and 7.4% at 500, 1,000 and
  2,000 tokens).
- **RQ3:** injection raised the BFR by 8.5 to 62.1 percentage points at every budget (Holm-corrected p < 0.001).
- **RQ4:** reasoning was 9.0–16.9% shorter with injection, with no statistically significant change in accuracy after
  Holm correction. Because the official rule keeps counting after the end of reasoning, it also inserted control tokens
  into final answers, corrupting 105 of 4,800 sampled answers.
- On ten shared problems, bf16 on an A100 gave 10.0 points lower accuracy than fp16 on a T4 with identical seeds; this
  configuration effect does not affect the A-versus-B comparison.

See the technical report for methods, statistics (problem-level cluster bootstrap, sign-flip permutation tests with
Holm correction) and limitations.

## Repository layout

| Path | Contents |
|---|---|
| `notebooks/BudgetThinker_Independent_Eval.ipynb` | Complete Colab notebook (38 cells) used for the run; it writes the modules in `src/` itself |
| `src/bt_engine.py` | Token-level generation engine (HF transformers) with per-token provenance: sampled, injected, forced suffix |
| `src/bt_fast.py` | Exact GQA decode attention and preallocated KV cache (fast path, validated against the standard path) |
| `src/bt_runner.py` | Data loading, fixed question selection, batching, resumable per-unit output |
| `src/bt_grade.py` | Wrapper around the official grader (run in an isolated environment) |
| `src/bt_metrics.py` | BFR, BUR, FTR, accuracy, completeness, cluster bootstrap, permutation tests, plots |
| `src/bt_export.py` | Packages all outputs with a SHA-256 manifest |
| `src/scope_10q.py` | Scope override used for the preliminary 10-problem run |
| `configs/experiment.json` | Budgets, decoding, seeds, batch sizes |
| `configs/questions_100.json` | The 100 selected problems (seed 20251011, stratified by level) |
| `configs/run_manifest.json` | Pinned model, tokenizer, dataset and code revisions; environment of the full run |
| `tests/` | CPU tests: prompt identity, injection rule vs. a literal port of the official patch, engine state machine, fast-path equivalence |
| `analysis/` | Local re-grading and analysis of a results package; extra report figures |
| `results/metrics/` | Summary and paired A-vs-B tables of the full study |
| `results/figures/` | Figures of the full study |

## Pinned versions

| Item | Value |
|---|---|
| Model | `Xin-Rui/BudgetThinker-1.5b` @ `b7304c41480eca765e2f77b5a6641133c64c1ea7` |
| Dataset | `HuggingFaceH4/MATH-500` @ `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |
| Official code | `MobileLLM/BudgetThinker` @ `481da9045f761ba34899271db846d659110b7065` |
| transformers | 4.51.3 |
| Full run | Python 3.13, torch 2.11.0+cu130, CUDA 13.0, NVIDIA A100-SXM4-40GB, bf16 |

The official grader runs in an isolated environment with its own pins (sympy 1.12, antlr4-python3-runtime 4.11.1);
the notebook installs it automatically.

## Reproducing the study

1. Open `notebooks/BudgetThinker_Independent_Eval.ipynb` in Google Colab with a GPU runtime (A100 for bf16; a T4 runs
   fp16).
2. Edit the parameters cell. The notebook clones the official repository at the pinned commit, installs the pinned
   libraries, downloads the model and verifies its hash.
3. Run the smoke test and the pilot cells. Set `APPROVED_FULL_RUN = True` to run the full study (`greedy_ref` and
   `sampled`). Progress is saved per unit, so an interrupted run resumes where it stopped.
4. The last cell packages every output with a SHA-256 manifest.

The full study took about 6 h 15 min on one A100 (about 33 Colab compute units). The notebook's markdown headings
are in Arabic; code, comments and outputs are in English.

## Relation to the official code

The notebook clones the official repository at the pinned commit; its files are not redistributed here. The
injection rule, prompt and termination policy were re-implemented from the official sources listed at the top of
`src/bt_engine.py`. The only exception is a short literal port of the official injection rule,
`official_rule_reference`, which is used only as a test oracle in `tests/test_engine.py`. At the time of writing the
official repository did not declare a license.

## Citation

If you use this code or these results, please cite the technical report (see `CITATION.cff`) and the original
BudgetThinker paper:

```bibtex
@article{wen2025budgetthinker,
  title   = {BudgetThinker: Empowering Budget-Aware LLM Reasoning with Control Tokens},
  author  = {Wen, H. and Wu, X. and Sun, Y. and Zhang, F. and Chen, L. and Wang, J. and
             Liu, Y. and Liu, Y. and Zhang, Y.-Q. and Li, Y.},
  journal = {arXiv preprint arXiv:2508.17196},
  year    = {2025}
}
```

## AI assistance

The evaluation code, the execution on Google Colab, the statistical analysis and the drafting of the report were
carried out with substantial assistance from an AI assistant, under the direction of the author,
who set the scope, approved the experimental settings and compute use, and takes responsibility for the content.

## License

Code: MIT (see `LICENSE`). Results and figures: CC BY 4.0.
