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
docs/REPRODUCING.md         Full run matrix, data pipeline, eval protocol
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

Hint-cascade experiments (Table 5, 7B, 3 seeds at n=8 rollouts — see
[docs/REPRODUCING.md](docs/REPRODUCING.md) for details):

| Paper method | Config | Mechanism |
|---|---|---|
| OC-GRPO-Fixed (Hints) | `verl_run_900_masked_IS` | offline **hint**-augmented data + `prompt_masked` + IS |
| OC-GRPO-Adaptive (Hints) | `verl_run_853` | online hierarchical hints L1–L5 (self-generated) + IS |
| OC-GRPO (Self-Correction) | `verl_run_854` | online hints, L1 only (blind feedback on failed attempts) + IS |
| OC-GRPO-Adaptive (Frontier Hints) | `verl_run_858` | online hints L1–L5 from Claude Sonnet (needs `ANTHROPIC_API_KEY`) + IS |

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

## Training

Run from the repo root (all paths are CWD-relative). One command per paper
run — the launcher reads the config, finds its shipped training parquet,
computes the epoch length, and starts training:

```bash
bash scripts/run_training.sh <CONFIG_NAME> [extra hydra overrides...]
```

| Run | Command | GPUs |
|---|---|---|
| OC-GRPO-Fixed, 7B | `bash scripts/run_training.sh verl_run_901_masked_IS_n16` | 4× A100-40GB |
| OC-GRPO-Adaptive, 7B | `bash scripts/run_training.sh verl_run_852_n16` | 4× A100-40GB |
| Vanilla GRPO, 7B | `bash scripts/run_training.sh verl_run_652_n16` | 4× A100-40GB |
| POPE\* / PrefixRL\* / BREAD\*, 7B | `verl_run_901_n16` / `verl_run_901_prefixrl_n16` / `verl_run_852_non_masked_n16` | 4× A100-40GB |
| Any 3B run | e.g. `bash scripts/run_training.sh verl_run_951_n16` | 2× A100-40GB |
| Any 1.5B run | e.g. `bash scripts/run_training.sh verl_run_1506_masked_IS_n16` | 2× A100-40GB |
| Seed variants (7B) | append `_s2` (seed 123) or `_s3` (seed 456), e.g. `verl_run_852_n16_s2` | 4× A100-40GB |
| Hint-cascade runs (Table 5) | `verl_run_900_masked_IS`, `verl_run_853`, `verl_run_854`, `verl_run_858` | 4× A100-40GB |

Useful overrides:

```bash
# wandb logging (needs WANDB_API_KEY)
bash scripts/run_training.sh verl_run_852_n16 trainer.logger='[console,wandb]'
```

On SLURM: `CONFIG_NAME=verl_run_852_n16 sbatch jobs/train_4gpu.sbatch`
(edit the `#SBATCH` headers for your cluster; 3B/1.5B runs use
`jobs/train_2gpu.sbatch`). Wall-clock: roughly 3–13 h for 1.5B/3B runs and
up to 1–2 days for 7B runs on the listed GPUs (we used 24 h / 48 h SLURM
walltimes for 2-GPU / 4-GPU jobs).

Each run trains 4 epochs and saves a LoRA checkpoint at every epoch boundary
to `runs/<experiment_name>/global_step_<N>/actor/lora_adapter/` — see
[docs/REPRODUCING.md](docs/REPRODUCING.md) for the step numbers per run.
The parquets in `verl_grpo/data_*/` are the exact training data used in the
paper; `data/*.json` are the raw problem sets they were built from, and
`scripts/prepare_data.py` regenerates the base parquets from them if you want
to rebuild (see README section below).

## Evaluation

Test sets are **not** shipped (they are third-party benchmarks); rebuild them
from HuggingFace first:

```bash
python scripts/download_benchmarks.py     # AIME 1983-2024/24/25/26, AMC23, Minerva, OlympiadBench, Gaokao2023-En
python scripts/download_omnimath.py       # Omni-MATH (then subsample to 1000, see docs/REPRODUCING.md)
python scripts/extract_new_test_sets.py   # OlymMATH
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
estimator 1 − C(n−c, k)/C(n, k). See
[docs/REPRODUCING.md](docs/REPRODUCING.md) for checkpoint steps per run and
the full protocol.

## Rebuilding the data from scratch (optional)

The shipped parquets are the paper's exact training data. To regenerate them
end-to-end (or apply OC-GRPO to a new model/dataset), the pipeline is:

1. **Pool**: `data/math_train_5586_problems.json` — MATH levels 3–5 train split.
2. **ExtremeHard extraction**: roll out the base model 64× per problem
   (`evaluate_math_full.py`), merge shards (`scripts/merge_eval_shards.py`),
   keep problems with 0/64 correct (`scripts/extract_hard_problems.py`).
3. **Base parquet**: `python scripts/prepare_data.py --problems <extremehard.json> --output verl_grpo/data_<name> --val-ratio 0.1 --seed 42`
4. **Offline augmentation** (for POPE\*/PrefixRL\*/OC-GRPO-Fixed): generate
   minimal solving prefixes with the base model
   (`jobs/generate_offline_prefixes.sbatch`, 5 GPU shards), then merge
   (`jobs/merge_offline_guided.sbatch`). Note this step is stochastic — the
   shipped augmented parquets are the paper's realization.

Full details in [docs/REPRODUCING.md](docs/REPRODUCING.md).

## Environment variables

| Variable | When needed |
|---|---|
| `WANDB_API_KEY` | only with `trainer.logger='[console,wandb]'` |
| `ANTHROPIC_API_KEY` | only for frontier-hint runs (`verl_run_858*`) |

## License

Apache-2.0 (same as veRL). MATH problems originate from the
[Hendrycks et al. MATH dataset](https://github.com/hendrycks/math); benchmark
test sets are downloaded from their respective HuggingFace sources and keep
their original licenses.

## Citation

```bibtex
@article{agrawal2026ocgrpo,
  title   = {Off-Context GRPO: Learning to Reason on Hard Problems using Privileged Information},
  author  = {Agrawal, Priyank and Samanta, Ankur and Ghasemlou, Shervin and Vidolov, Boris and Bhandari, Jalaj and Asadi, Kavosh and Jiang, Daniel and Modi, Aditya},
  year    = {2026},
  journal = {arXiv preprint}
}
```
