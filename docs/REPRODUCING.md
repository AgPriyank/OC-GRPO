# Reproducing the paper's experiments

This document is the complete map from the paper's tables to the runs in this
repository: every config, its data, compute footprint, checkpoint schedule,
and the evaluation protocol.

Throughout: **s1** = the default config (no explicit seeding — the config's
`trainer.seed: 0` means "do not seed", so s1 runs use default RNG state),
**s2** = seed 123 (`*_s2` configs), **s3** = seed 456 (`*_s3` configs);
`data.seed` stays 42 for all of them, so the data order is identical across
seeds. "SPE" = optimizer steps per epoch (= train rows ÷ 32). All runs train
4 epochs with checkpoints at every epoch boundary.

## 1. Shared hyperparameters (Appendix G of the paper)

Set in `verl_grpo/config/ppo_trainer_7b.yaml` (the Hydra base that **all** run
configs inherit, including 3B/1.5B which override only `model.path` and GPU
layout):

| Hyperparameter | Value |
|---|---|
| LoRA | rank 64, α 128, dropout 0.05, all linear layers; base weights frozen |
| Optimizer | AdamW, lr 1e-5, weight decay 0.01, grad clip 1.0 |
| Batch | 32 prompts/step, n = 16 rollouts/prompt, PPO mini-batch 32, 1 PPO epoch |
| Objective | GRPO advantages, clip ε = 0.2, **no KL penalty**, no entropy bonus, token-level loss aggregation |
| Rollouts | vLLM, temperature 0.7, top-p 0.95, max 1200 response tokens |
| Reward | binary: `\boxed{}` answer matches ground truth under symbolic equivalence (`baseline.py` → `scripts/answer_extraction.py`, math-verify) |
| IS correction (OC-GRPO runs) | per-token, weights clamped to [0.01, 1] (`hints.is_correction_mode=per_token`, `hints.is_correction_clamp_min=0.01`) |
| Prefix cascade | fractions {0.2, 0.4, 0.6, 0.8, 1.0} of the reference solution, character-fraction with word-boundary snapping |

## 2. Main results — Table 2 (Qwen2.5-7B-Instruct, 3 seeds)

Training data (shipped):

- Online runs: `verl_grpo/data_extremehard_595/` — 536 train / 59 val rows
  (595 MATH L3–5 problems with 0/64 correct under the base model, split 90/10 seed 42).
  SPE = 16, checkpoints at steps **16, 32, 48, 64**.
- Offline runs: `verl_grpo/data_offline_guided_prefix/` — 960 train / 105 val
  rows (each solved problem contributes its original row + one prefix-guided
  row). SPE = 30, checkpoints at steps **30, 60, 90, 120**.

All 7B runs: 4× A100-40GB, vLLM TP=2, `gpu_memory_utilization=0.6`.

| Paper row | Config (× `_s2`, `_s3`) | Data | Epoch-4 step |
|---|---|---|---|
| Vanilla GRPO | `verl_run_652_n16` | extremehard_595 | 64 |
| PrefixRL\* | `verl_run_901_prefixrl_n16` | offline_guided_prefix | 120 |
| POPE\* | `verl_run_901_n16` | offline_guided_prefix | 120 |
| OC-GRPO-Fixed | `verl_run_901_masked_IS_n16` | offline_guided_prefix | 120 |
| BREAD\* | `verl_run_852_non_masked_n16` | extremehard_595 | 64 |
| OC-GRPO-Adaptive | `verl_run_852_n16` | extremehard_595 | 64 |

Paper cells are the mean ± std over s1/s2/s3 of pass@1 / pass@16 (Section 5
below), evaluated at the fourth-epoch checkpoint.

## 3. Main results — Tables 3 & 4 (Qwen2.5-3B / 1.5B, single seed)

Same six methods, same base config, model path + GPU layout overridden.
All 3B/1.5B runs: 2× A100-40GB, TP=1, `gpu_memory_utilization=0.5`.

**Table 3 — Qwen2.5-3B-Instruct.** Data: `data_qwen3b_extremehard/`
(614/68 rows from the 682-problem ExtremeHard set; SPE 19, ckpts 19/38/57/76)
and `data_offline_guided_qwen3b_prefix/` (1119/125 rows; SPE 34, ckpts
34/68/102/136).

| Paper row | Config | Epoch-4 step |
|---|---|---|
| Vanilla GRPO | `verl_run_950_n16` | 76 |
| PrefixRL\* | `verl_run_954_prefixrl_n16` | 136 |
| POPE\* | `verl_run_954_n16` | 136 |
| OC-GRPO-Fixed | `verl_run_954_masked_IS_n16` | 136 |
| BREAD\* | `verl_run_951_non_masked_n16` | 76 |
| OC-GRPO-Adaptive | `verl_run_951_n16` | 76 |

**Table 4 — Qwen2.5-1.5B-Instruct.** Data: `data_qwen1_5b_extremehard/`
(540/60 rows from a 681-problem ExtremeHard set downsampled to 600, seed 42;
SPE 16, ckpts 16/32/48/64) and `data_offline_guided_qwen1_5b_prefix/`
(1012/117 rows; SPE 31, ckpts 31/62/93/124).

| Paper row | Config | Epoch-4 step |
|---|---|---|
| Vanilla GRPO | `verl_run_1500_n16` | 64 |
| PrefixRL\* | `verl_run_1506_prefixrl_n16` | 124 |
| POPE\* | `verl_run_1506_n16` | 124 |
| OC-GRPO-Fixed | `verl_run_1506_masked_IS_n16` | 124 |
| BREAD\* | `verl_run_1501_non_masked_n16` | 64 |
| OC-GRPO-Adaptive | `verl_run_1501_n16` | 64 |

## 4. Hint-cascade experiments — Table 5 (Qwen2.5-7B)

These runs use LLM-generated **hints** (Appendix H/I of the paper) instead of
solution prefixes. They ran in the original n=8 rollout series with 3 seeds
(configs without the `_n16` suffix; `_n16` single-seed variants are also
included for completeness). Table 5 reports, per seed, the best checkpoint by
validation pass@1.

| Paper row | Config (× `_s2`, `_s3`) | Data | GPUs | Ckpts |
|---|---|---|---|---|
| OC-GRPO-Fixed (Hints) | `verl_run_900_masked_IS` | `data_offline_guided_hint/` (953/108 rows) | 4 | 29/58/87/116 |
| OC-GRPO-Adaptive (Hints) | `verl_run_853` | `data_extremehard_595/` | 4 | 16/32/48/64 |
| OC-GRPO (Self-Correction) | `verl_run_854` | `data_extremehard_595/` | 4 | 16/32/48/64 |
| OC-GRPO-Adaptive (Frontier Hints) | `verl_run_858` | `data_extremehard_595/` | 4 | 16/32/48/64 |

Hint levels (prompt templates in `verl_grpo/hints/hint_generator.py`, quoted
in Appendix I of the paper): L1 feedback on the failed attempt (blind — no
reference solution), L2 conceptual comparison, L3 strategy, L4 detailed
guidance, L5 full-solution paraphrase. `verl_run_854` uses L1 only —
self-correction from the model's own failures. `verl_run_858` generates hints
with `claude-sonnet-4-20250514` (set `ANTHROPIC_API_KEY`; API generation makes
exact reproduction nondeterministic).

## 5. Evaluation protocol

Every paper number comes from `evaluate_trained.py` via
`scripts/run_eval.sh <RUN_ID> <STEP> <TEST_SET>`:

- **16 trajectories** per problem, temperature 0.7, top-p 0.95,
  `max_tokens=1000`, vLLM seed 42, chat template on.
- The LoRA adapter is merged into the base model (bf16, `merge_and_unload`)
  and cached under `models/merged_models/` before vLLM loads it.
- Grading: extract the last `\boxed{}` answer, compare to ground truth with
  math-verify semantic equivalence (`scripts/answer_extraction.py`).
- `results.json` stores per-problem `n_correct`/`n_total`;
  **pass@k = 1 − C(n−c, k)/C(n, k)** averaged over problems
  (pass@1 = c/n, pass@16 = 1 if c > 0 with n = 16).
- **Label caveat:** in `summary.txt` / `results.json`, the paper's pass@1
  corresponds to the **`mean_accuracy`** field (mean of `n_correct/n_total`).
  The field named `pass_at_1_accuracy` is a different, noisier statistic
  (fraction of problems whose *first* trajectory is correct) and is **not**
  what the paper reports.

Test sets used in the paper (rebuild before evaluating; sources are HF
datasets):

| TEST_SET | Problems | Source | Builder |
|---|---|---|---|
| `aime_1983_2024` | 933 | di-zhang-fdu/AIME_1983_2024 | `scripts/download_benchmarks.py` |
| `aime25` | 30 | opencompass/AIME2025 | `scripts/download_benchmarks.py` |
| `aime26` | 30 | MathArena/aime_2026 | `scripts/download_benchmarks.py` |
| `gaokao2023en` | 385 | MARIO-Math-Reasoning/Gaokao2023-Math-En | `scripts/download_benchmarks.py` |
| `omnimath1000` | 1000 | KbsdJames/Omni-MATH | `scripts/download_omnimath.py` + subsample below |
| `olympiadbench` | 674 | Hothan/OlympiadBench (OE_TO_maths_en_COMP) | `scripts/download_benchmarks.py` |
| `olymmath` | 200 | RUC-AIBOX/OlymMATH (en-easy + en-hard) | `scripts/extract_new_test_sets.py` |
| `minerva` | 272 | math-ai/minervamath | `scripts/download_benchmarks.py` |
| `amc23` | 40 | math-ai/amc23 | `scripts/download_benchmarks.py` |
| `extremehard595` / `682` / `600` | 595/682/600 | shipped in `data/` | — (training-set eval) |

The paper's AIME column (1983–2026) pools `aime_1983_2024` + `aime25` +
`aime26` (993 problems). Either evaluate the three sets separately and pool
the per-problem results, or build the combined file with
`python scripts/combine_aime_datasets.py` and evaluate it directly with
`TEST_SET=aime_1983_2026`. `omnimath1000` is a 1000-problem subsample
(seed 42) of the full 4428-problem Omni-MATH export:

```python
import json, random
d = json.load(open("data/omnimath_test_4428_problems.json"))
random.seed(42)
d["problems"] = random.sample(d["problems"], 1000)
d["metadata"]["n_problems"] = 1000
json.dump(d, open("data/omnimath_test_1000_problems.json", "w"), indent=2)
```

## 6. Data pipeline from scratch

The shipped parquets are the paper's exact training data — you only need this
section to rebuild them or to target a new model.

**Step 0 — problem pool.** `data/math_train_5586_problems.json`: MATH train
split, levels 3–5, from `EleutherAI/hendrycks_math` (rebuildable with
`extract_data.py`). Each entry: `id`, `question`, `answer` (full reference
solution), `ground_truth`, `level`, `subject`.

**Step 1 — ExtremeHard extraction (GPU).** Roll out the *base* model 64× per
problem at T=0.7/top-p 0.95/1200 tokens, sharded:

```bash
python evaluate_math_full.py --split train --levels 3 4 5 \
    --n_trajectories 64 --temperature 0.7 --top_p 0.95 --max_tokens 1200 \
    --shard_id $i --n_shards 20 ...
python scripts/merge_eval_shards.py --input_dir results/math_full_64traj/<model> --n_shards 20
python scripts/extract_hard_problems.py \
    --results results/math_full_64traj/<model>/results.json \
    --dataset math \
    --output data/<model>_extremehard_problems.json   # keeps n_correct == 0 problems
```

Results for the paper's models are shipped:
`data/math_train_extremehard_595_problems.json` (7B),
`data/qwen3b_extremehard_problems.json` (3B, 682 problems),
`data/qwen2.5_1_5b_extremehard_600_problems.json` (1.5B; the raw 681-problem
set downsampled to 600 with `random.seed(42); random.sample(problems, 600)`).

**Step 2 — base parquet.**

```bash
python scripts/prepare_data.py \
    --problems data/math_train_extremehard_595_problems.json \
    --output verl_grpo/data_extremehard_595 --val-ratio 0.1 --seed 42
```

**Step 3 — offline augmentation (GPU, stochastic).** For each hard problem,
sample 8 completions from the base model under each prefix fraction
(0.2 → 1.0) and record the *lowest* fraction with ≥1 correct answer:

```bash
for s in 0 1 2 3 4; do SHARD_ID=$s sbatch jobs/generate_offline_prefixes.sbatch; done
sbatch --dependency=afterok:<ids> jobs/merge_offline_guided.sbatch
```

The merge preserves the base train/val split, adds one guided row per solved
problem (prompt pre-baked with the prefix text + anti-repetition system
prompt for partial prefixes; the 100% level instead presents the full
reference solution with "Verify it, show your work step by step, and output
the final answer within \boxed{}" and no anti-repetition prompt), and
shuffles with seed 42. The hint-mode analog (`MODE=hint`) produced
`data_offline_guided_hint/`; there, the top level (L5) asks for a paraphrase
of the reference solution.

## 7. Determinism and known caveats

- **vLLM sampling is not bit-deterministic** across GPU types/driver versions
  even with fixed seeds; expect small variations when re-training and
  re-evaluating. The paper reports seed-averaged results for 7B. The s1 runs
  are additionally the *unseeded* default (see Section 1's seed note).
- **Anti-repetition system prompt wording:** the shipped code and baked
  parquets use the "When given a **hint**, use it to guide your reasoning …
  do not repeat, copy, or paraphrase the **hint** text" wording for both hint
  and prefix guidance. The paper's Appendix I quotes a prefix-specific
  variant ("partial reference solution"); the repo is the ground truth of
  what was run.
- **Two configs were adjusted for the release** (each carries a NOTE comment
  inline): `verl_run_853.yaml` had `total_epochs: 7` from a later extension
  of the original s1 run — reset to 4 (the paper's scope; all paper evals use
  steps ≤ 64, and `_s2`/`_s3` always had 4). `verl_run_858.yaml` carried
  resume-from-checkpoint state from the original s1 run — removed so a fresh
  clone trains from scratch. All other configs are byte-identical to the
  files that produced the paper's runs.
- **Always pass a run config.** `python -m verl_grpo.main_ppo` without
  `--config-name` falls back to the base `ppo_trainer.yaml`, whose default
  data paths (`verl_grpo/data/`) are not shipped. Use
  `scripts/run_training.sh <CONFIG_NAME>` (or pass `--config-name`
  explicitly).
- The offline augmented parquets are one stochastic realization of Step 3;
  the shipped files are the paper's realization — prefer them for faithful
  reproduction.
- `verl_run_858*` (frontier hints) calls the Anthropic API at training time;
  hint text is not reproducible bit-for-bit.
- Training must be launched from the repo root: config, data, and reward
  paths are CWD-relative; checkpoints land in `runs/<experiment_name>/`.
- Evaluation loads the merged model with vLLM `max_model_len=2048`
  (prompt + 1000 generated tokens must fit), `VLLM_USE_V1=0`.
- The two verl-fork extension modules (`verl.trainer.constants_ppo`,
  `verl.utils.dataset.sampler`) and `verl_uncommitted.patch` are mandatory;
  see README installation.
