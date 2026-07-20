#!/bin/bash
#SBATCH --job-name=offline_prefix
#SBATCH --output=logs/offline_prefix_%j.out
#SBATCH --error=logs/offline_prefix_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=8:00:00
##SBATCH --account=YOUR_ACCOUNT        # <- set your allocation
##SBATCH --partition=YOUR_PARTITION    # <- set your partition

# ============================================================================
# Offline guided-data generation (Phase 1 of the offline pipeline): one shard.
# Runs the solution-prefix cascade [20% -> 40% -> 60% -> 80% -> 100%] with the
# BASE model and records, per problem, the lowest prefix fraction that yields
# at least one correct rollout (8 samples per fraction, T=0.7, top_p=0.95).
#
# NOTE: this regenerates the augmented data from scratch (stochastic). The
# exact parquets used in the paper are already shipped under
# verl_grpo/data_offline_guided_* — you only need this to rebuild from raw
# JSON or to apply the method to new problems/models.
#
# Required env var:  SHARD_ID (0-4)
# Optional env vars: MODEL_PATH, PROBLEMS_FILE, OUTPUT_DIR, MODE (prefix|hint)
#
# Per-model defaults used in the paper:
#   7B:   MODEL_PATH=Qwen/Qwen2.5-7B-Instruct   PROBLEMS_FILE=data/math_train_extremehard_595_problems.json   OUTPUT_DIR=offline_guided_data
#   3B:   MODEL_PATH=Qwen/Qwen2.5-3B-Instruct   PROBLEMS_FILE=data/qwen3b_extremehard_problems.json           OUTPUT_DIR=offline_guided_data_qwen3b
#   1.5B: MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct PROBLEMS_FILE=data/qwen2.5_1_5b_extremehard_600_problems.json OUTPUT_DIR=offline_guided_data_qwen1_5b
#
# Submit (all 5 shards):
#   for s in 0 1 2 3 4; do SHARD_ID=$s sbatch jobs/generate_offline_prefixes.sbatch; done
# ============================================================================

if [ -z "${SHARD_ID}" ]; then
  echo "ERROR: SHARD_ID not set (0-4)"; exit 1
fi

N_SHARDS=5
MODE=${MODE:-prefix}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
PROBLEMS_FILE=${PROBLEMS_FILE:-data/math_train_extremehard_595_problems.json}
OUTPUT_DIR=${OUTPUT_DIR:-offline_guided_data}

# --- adapt to your cluster ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate oc-grpo
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs "${OUTPUT_DIR}"

nvidia-smi

export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0

python scripts/generate_offline_guided_data.py \
    --mode "${MODE}" \
    --shard "${SHARD_ID}" \
    --n_shards ${N_SHARDS} \
    --model_path "${MODEL_PATH}" \
    --problems_file "${PROBLEMS_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_traj 8 \
    --temperature 0.7 \
    --top_p 0.95 \
    --max_tokens 1200 \
    --gpu_memory_utilization 0.85
