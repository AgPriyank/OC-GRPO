#!/bin/bash
#SBATCH --job-name=ocgrpo_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=16:00:00
##SBATCH --account=YOUR_ACCOUNT        # <- set your allocation
##SBATCH --partition=YOUR_PARTITION    # <- set your partition

# ============================================================================
# Single eval job: one run, one checkpoint, one test set (paper protocol:
# 16 trajectories, T=0.7, top_p=0.95, max_tokens=1000, seed=42).
#
# Required env vars: RUN_ID, STEP, TEST_SET
#
# Submit:
#   RUN_ID=852_n16 STEP=64 TEST_SET=gaokao2023en sbatch jobs/eval_single.sbatch
#
# NOTE: the first eval of a checkpoint merges the LoRA adapter into a full
# model under models/merged_models/. When submitting several test sets for the
# SAME checkpoint, chain them with --dependency=afterok:<first_job_id> so two
# jobs don't race on the merge.
# ============================================================================

if [ -z "${RUN_ID}" ] || [ -z "${STEP}" ] || [ -z "${TEST_SET}" ]; then
  echo "ERROR: RUN_ID, STEP, and TEST_SET env vars required."; exit 1
fi

# --- adapt to your cluster ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate oc-grpo
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

nvidia-smi

bash scripts/run_eval.sh "${RUN_ID}" "${STEP}" "${TEST_SET}"
