#!/bin/bash
#SBATCH --job-name=offline_merge
#SBATCH --output=logs/offline_merge_%j.out
#SBATCH --error=logs/offline_merge_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
##SBATCH --account=YOUR_ACCOUNT        # <- set your allocation
##SBATCH --partition=YOUR_PARTITION    # <- set your partition

# ============================================================================
# Offline guided-data merge (Phase 2, CPU-only): combines the 5 generation
# shards with the base parquet into the augmented training parquet.
#
# Optional env vars: MODE (prefix|hint), SHARD_DIR, BASE_PARQUET_DIR, OUTPUT_DIR
#
# Per-model defaults used in the paper:
#   7B:   SHARD_DIR=offline_guided_data          BASE_PARQUET_DIR=verl_grpo/data_extremehard_595       OUTPUT_DIR=verl_grpo/data_offline_guided_prefix
#   3B:   SHARD_DIR=offline_guided_data_qwen3b   BASE_PARQUET_DIR=verl_grpo/data_qwen3b_extremehard    OUTPUT_DIR=verl_grpo/data_offline_guided_qwen3b_prefix
#   1.5B: SHARD_DIR=offline_guided_data_qwen1_5b BASE_PARQUET_DIR=verl_grpo/data_qwen1_5b_extremehard  OUTPUT_DIR=verl_grpo/data_offline_guided_qwen1_5b_prefix
#
# Submit (after the 5 generation shards finish):
#   sbatch --dependency=afterok:<shard_job_ids> jobs/merge_offline_guided.sbatch
# ============================================================================

MODE=${MODE:-prefix}
SHARD_DIR=${SHARD_DIR:-offline_guided_data}
BASE_PARQUET_DIR=${BASE_PARQUET_DIR:-verl_grpo/data_extremehard_595}
# Default output has a _regen suffix so a regeneration run does NOT overwrite
# the shipped paper parquets (verl_grpo/data_offline_guided_*). Set OUTPUT_DIR
# explicitly if you really want to replace them.
OUTPUT_DIR=${OUTPUT_DIR:-verl_grpo/data_offline_guided_${MODE}_regen}

# --- adapt to your cluster ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate oc-grpo
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

python scripts/create_offline_parquet.py \
    --mode "${MODE}" \
    --n_shards 5 \
    --shard_dir "${SHARD_DIR}" \
    --base_parquet_dir "${BASE_PARQUET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42
