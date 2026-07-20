# OC-GRPO: Off-Context GRPO

Reference implementation and reproduction package for the paper:

> **Off-Context GRPO: Learning to Reason on Hard Problems using Privileged Information**
> Priyank Agrawal, Ankur Samanta, Shervin Ghasemlou, Boris Vidolov, Jalaj Bhandari, Kavosh Asadi, Daniel Jiang, Aditya Modi.

Typical RLVR training receives zero learning signal on problems the model never
solves (the *learning cliff*). OC-GRPO breaks the cliff by sampling rollouts
under a **guided prompt** (a solution prefix or hint) while applying a
**per-token importance-sampling correction** so the update still targets the
original, unguided objective. This repo contains the exact trainer, configs,
and training data behind the paper's experiments, built on a patched
[veRL](https://github.com/volcengine/verl) v0.4.1.

## What is in this repo

```
verl_grpo/                  Training package (Hydra entry point, custom RayPPOTrainer,
                            dataset, binary reward, prefix/hint generation)
verl_grpo/config/           Base config + one YAML per paper run (see mapping below)
verl_grpo/data_*/           The exact train/val parquets used in the paper (~7 MB)
data/                       Raw problem JSONs (MATH L3-5 pool + per-model ExtremeHard sets)
verl/                       Git submodule: our verl fork, branch custom-patches-v0.4.1
verl_uncommitted.patch      Small additional verl patch (apply before installing verl)
scripts/                    Data pipeline, launchers, test-set builders
jobs/                       SLURM templates (adapt #SBATCH headers to your cluster)
evaluate_trained.py         Checkpoint evaluation (LoRA merge -> vLLM -> pass@k)
baseline.py                 Eval engine + answer grading used by reward and eval
```

## Quick start

```bash
# 0. clone (the --recurse-submodules flag matters: it pulls our verl fork)
git clone --recurse-submodules https://github.com/AgPriyank/OC-GRPO.git
cd OC-GRPO

# 1. environment (Linux, CUDA 12.4, A100-40GB tested)
conda create -n oc-grpo python=3.10 -y
conda activate oc-grpo
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation
pip install -r requirements.txt

# 2. patched verl (submodule is already pinned to branch custom-patches-v0.4.1)
cd verl && git apply ../verl_uncommitted.patch && pip install --no-deps -e . && cd ..

# 3. train OC-GRPO-Fixed on the paper's exact 7B training data (4x A100-40GB)
bash scripts/run_training.sh verl_run_901_masked_IS_n16
```

That's it — **the training data ships in this repo** (`verl_grpo/data_*/`, the
exact parquets behind the paper's runs), so step 3 works immediately after
install; no data preparation, no downloads beyond the base model from
HuggingFace. Checkpoints appear under
`runs/verl_grpo_run901_masked_IS_n16/global_step_<N>/actor/lora_adapter/`
at every epoch boundary (steps 30/60/90/120 for this run).

## Paper method → run config mapping

Main results (Tables 2–4). All runs: LoRA (r=64, α=128, dropout 0.05), AdamW
lr 1e-5, batch 32 prompts × 16 rollouts, 4 epochs, no KL penalty, clip 0.2,
temperature 0.7. Seeds: s1 = default config (unseeded), s2 = 123 (`*_s2`), s3 = 456 (`*_s3`).

| Paper method | Qwen2.5-7B (3 seeds) | Qwen2.5-3B | Qwen2.5-1.5B | Mechanism (config flags) |
|---|---|---|---|---|
| Vanilla GRPO | `verl_run_652_n16` | `verl_run_950_n16` | `verl_run_1500_n16` | `hints.enabled=false`, ExtremeHard data |
| PrefixRL\* | `verl_run_901_prefixrl_n16` | `verl_run_954_prefixrl_n16` | `verl_run_1506_prefixrl_n16` | offline augmented data + `data.hint_mode=prefix_continuation` |
| POPE\* | `verl_run_901_n16` | `verl_run_954_n16` | `verl_run_1506_n16` | offline augmented data, `hints.enabled=false` |
| **OC-GRPO-Fixed** | `verl_run_901_masked_IS_n16` | `verl_run_954_masked_IS_n16` | `verl_run_1506_masked_IS_n16` | offline augmented data + `hints.hint_mode=prompt_masked` + per-token IS (`clamp_min=0.01`) |
| BREAD\* | `verl_run_852_non_masked_n16` | `verl_run_951_non_masked_n16` | `verl_run_1501_non_masked_n16` | online prefix cascade, `hints.prompt_mode=hinted`, no IS |
| **OC-GRPO-Adaptive** | `verl_run_852_n16` | `verl_run_951_n16` | `verl_run_1501_n16` | online prefix cascade, `hints.prompt_mode=original` + per-token IS |

7B rows have `_s2`/`_s3` seed variants; 3B and 1.5B are single-seed, matching the paper.

Hint-cascade experiments (Table 5, 7B, 3 seeds at n = 8 rollouts):

| Paper method | Config | Mechanism |
|---|---|---|
| OC-GRPO-Fixed (Hints) | `verl_run_900_masked_IS` | offline **hint**-augmented data + `prompt_masked` + IS |
| OC-GRPO-Adaptive (Hints) | `verl_run_853` | online hierarchical hints L1–L5 (self-generated) + IS |
| OC-GRPO (Self-Correction) | `verl_run_854` | online hints, L1 only (blind feedback on failed attempts) + IS |
| OC-GRPO-Adaptive (Frontier Hints) | `verl_run_858` | online hints L1–L5 from Claude Sonnet (uses the Anthropic API) + IS |

## Installation details

Tested configuration: Linux, Python 3.10, CUDA 12.4, NVIDIA A100-40GB.
Pinned versions in `requirements.txt` are the exact ones used for the paper.

Install order matters:

1. **torch 2.6.0 (cu124)** first — `flash-attn` compiles against it.
2. **flash-attn** with `--no-build-isolation`.
3. **`requirements.txt`** — vllm 0.8.5.post1, transformers 4.57.6,
   ray 2.53.0, hydra 1.3.2, etc. (torch stays untouched: the vllm pin is
   satisfied by the cu124 build).
4. **The patched verl fork**, editable, `--no-deps`:

```bash
cd verl
git apply ../verl_uncommitted.patch   # adds enable_thinking chat-template compat
pip install --no-deps -e .
cd ..
```

If you cloned without `--recurse-submodules`, run
`git submodule update --init` first.

**About the verl patches.** The `verl/` submodule points to
[our fork](https://github.com/AgPriyank/verl) of veRL, branch
`custom-patches-v0.4.1` — upstream v0.4.1 plus four patches: LoRA seed
propagation across Ray workers, `n_override` + SFT-forward support,
`n_override` for vLLM ≥ 0.7 (`vllm_rollout_spmd.py`), and a vLLM sleep-safety
check. `verl_uncommitted.patch` additionally adds `enable_thinking`-safe chat
templating to verl's `RLHFDataset`; it is **required** — the PrefixRL runs
(`*_prefixrl*`) read the attribute it introduces. Stock verl v0.4.1 will not
work: the trainer imports `verl.trainer.constants_ppo` and
`verl.utils.dataset.sampler`, which exist only in the fork.

## Training hyperparameters

All runs share the same recipe, set in `verl_grpo/config/ppo_trainer_7b.yaml`
and inherited by every run config:

| Hyperparameter | Value |
|---|---|
| LoRA | rank 64, α 128, dropout 0.05, all linear layers (base weights frozen) |
| Optimizer | AdamW, lr 1e-5, weight decay 0.01, gradient clipping at norm 1.0 |
| Batch | 32 prompts/step × n = 16 rollouts/prompt, PPO mini-batch 32, 1 PPO epoch |
| Objective | GRPO advantages, clip ε = 0.2, no KL penalty, no entropy bonus, token-level loss aggregation |
| Rollouts | vLLM, temperature 0.7, top-p 0.95, max 1200 response tokens |
| Reward | binary: `\boxed{}` answer matches ground truth under symbolic equivalence |
| IS correction (OC-GRPO runs) | per-token, weights clamped to [0.01, 1] |
| Prefix cascade | fractions {0.2, 0.4, 0.6, 0.8, 1.0} of the reference solution |
| Schedule | 4 epochs; LoRA checkpoint at every epoch boundary → `runs/<experiment_name>/global_step_<N>/actor/lora_adapter/` |

7B runs use 4× A100-40GB (TP=2); 3B and 1.5B runs use 2× A100-40GB (TP=1).
The hint-cascade runs (`verl_run_900_masked_IS`, `verl_run_853/854/858`) use
n = 8 rollouts per prompt. Launch any run from the repo root with
`bash scripts/run_training.sh <CONFIG_NAME>` (SLURM templates in `jobs/`).

## Evaluation

Test sets are **not** shipped (they are third-party benchmarks); rebuild them
from HuggingFace first:

```bash
python scripts/download_benchmarks.py     # AIME 1983-2024/24/25/26, AMC23, Minerva, OlympiadBench, Gaokao2023-En
python scripts/download_omnimath.py       # Omni-MATH (full 4428-problem export)
python scripts/extract_new_test_sets.py   # OlymMATH
```

The paper's `omnimath1000` set is a seed-42 subsample of the full export:

```python
import json, random
d = json.load(open("data/omnimath_test_4428_problems.json"))
random.seed(42)
d["problems"] = random.sample(d["problems"], 1000)
d["metadata"]["n_problems"] = 1000
json.dump(d, open("data/omnimath_test_1000_problems.json", "w"), indent=2)
```

Then evaluate a checkpoint with the paper protocol (16 trajectories,
temperature 0.7, top-p 0.95, max 1000 tokens, seed 42):

```bash
bash scripts/run_eval.sh 901_masked_IS_n16 120 gaokao2023en
bash scripts/run_eval.sh 852_n16 64 omnimath1000
```

The first eval of a checkpoint merges the LoRA adapter into a full model
(cached under `models/merged_models/`); results land in
`results/verl_run<RUN>_step<STEP>_<TESTSET>/results.json` with per-problem
trajectory correctness, from which pass@k is computed with the unbiased
estimator 1 − C(n−c, k)/C(n, k). Epoch-4 checkpoint
steps: 7B online 64, 7B offline 120; 3B online 76, offline 136; 1.5B online
64, offline 124.
