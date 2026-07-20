#!/bin/bash
#SBATCH --job-name=ocgrpo_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --time=48:00:00
##SBATCH --account=YOUR_ACCOUNT        # <- set your allocation
##SBATCH --partition=YOUR_PARTITION    # <- set your partition

# ============================================================================
# 4-GPU (A100-40GB) training template — used for all 7B runs
# (TP=2 for vLLM rollouts, FSDP actor; see configs).
#
# Required env var: CONFIG_NAME (e.g. verl_run_852_n16)
#
# Submit:
#   CONFIG_NAME=verl_run_852_n16 sbatch jobs/train_4gpu.sbatch
# ============================================================================

if [ -z "${CONFIG_NAME}" ]; then
  echo "ERROR: CONFIG_NAME env var not set."; exit 1
fi

# --- adapt to your cluster ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate oc-grpo
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

nvidia-smi

# Optional: log to wandb by appending trainer.logger='[console,wandb]'
bash scripts/run_training.sh "${CONFIG_NAME}"
