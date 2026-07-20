#!/bin/bash
#SBATCH --job-name=ocgrpo_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=2
#SBATCH --mem=120G
#SBATCH --time=24:00:00
##SBATCH --account=YOUR_ACCOUNT        # <- set your allocation
##SBATCH --partition=YOUR_PARTITION    # <- set your partition

# ============================================================================
# 2-GPU (A100-40GB) training template — used for all 3B and 1.5B runs
# (TP=1 for vLLM rollouts; see configs).
#
# Required env var: CONFIG_NAME (e.g. verl_run_950_n16, verl_run_1506_masked_IS_n16)
#
# Submit:
#   CONFIG_NAME=verl_run_950_n16 sbatch jobs/train_2gpu.sbatch
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

bash scripts/run_training.sh "${CONFIG_NAME}"
