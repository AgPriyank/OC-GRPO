#!/bin/bash
# ============================================================================
# Generic training launcher for OC-GRPO runs.
#
# Usage:
#   bash scripts/run_training.sh <CONFIG_NAME> [extra hydra overrides...]
#
# Examples:
#   bash scripts/run_training.sh verl_run_852_n16                 # OC-GRPO-Adaptive, 7B
#   bash scripts/run_training.sh verl_run_901_masked_IS_n16       # OC-GRPO-Fixed, 7B
#   bash scripts/run_training.sh verl_run_1500_n16 trainer.logger='[console,wandb]'
#
# Must be run from the repository root (all config paths are CWD-relative).
# Checkpoints are written to runs/<experiment_name>/global_step_<N>/.
# Saving/validation default to epoch boundaries (save_freq = steps per epoch),
# matching the paper's checkpointing policy.
# ============================================================================
set -e

CONFIG_NAME=${1:?"Usage: bash scripts/run_training.sh <CONFIG_NAME> [hydra overrides...]"}
shift || true

if [ ! -f "verl_grpo/config/${CONFIG_NAME}.yaml" ]; then
  echo "ERROR: verl_grpo/config/${CONFIG_NAME}.yaml not found. Run from the repo root."
  exit 1
fi

PYTHON=$(command -v python || command -v python3)
if [ -z "${PYTHON}" ]; then
  echo "ERROR: no python interpreter on PATH (activate the oc-grpo conda env)."
  exit 1
fi

export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN

# Resolve train parquet + batch size from the run config (falling back to its
# hydra defaults chain), then compute steps per epoch for save/test frequency.
read -r TRAIN_PARQUET BATCH_SIZE <<< "$("${PYTHON}" - "${CONFIG_NAME}" <<'EOF'
import os, sys, yaml

cfg_dir = "verl_grpo/config"

def load(name):
    with open(os.path.join(cfg_dir, name + ".yaml")) as f:
        return yaml.safe_load(f) or {}

def lookup(cfg, *keys):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

name = sys.argv[1]
chain = [load(name)]
for d in chain[0].get("defaults", []):
    if isinstance(d, str) and d != "_self_":
        chain.append(load(d))

train_files = next((v for c in chain if (v := lookup(c, "data", "train_files"))), None)
batch = next((v for c in chain if (v := lookup(c, "data", "train_batch_size"))), 32)
if not train_files:
    sys.exit(f"ERROR: data.train_files not found for config {name}")
print(train_files, batch)
EOF
)"

if [ -z "${TRAIN_PARQUET}" ]; then
  echo "ERROR: could not resolve data.train_files for ${CONFIG_NAME} (is pyyaml installed?)."
  exit 1
fi
if [ ! -f "${TRAIN_PARQUET}" ]; then
  echo "ERROR: training data not found: ${TRAIN_PARQUET}"
  echo "Rebuild it with scripts/prepare_data.py from the raw JSONs in data/."
  exit 1
fi

N_TRAIN=$("${PYTHON}" -c "import pandas as pd; print(len(pd.read_parquet('${TRAIN_PARQUET}')))")
STEPS_PER_EPOCH=$((N_TRAIN / BATCH_SIZE))
SAVE_FREQ=${SAVE_FREQ:-${STEPS_PER_EPOCH}}

echo "Config:           ${CONFIG_NAME}"
echo "Train data:       ${TRAIN_PARQUET} (${N_TRAIN} rows)"
echo "Batch size:       ${BATCH_SIZE}"
echo "Steps per epoch:  ${STEPS_PER_EPOCH}"
echo "Save/test freq:   ${SAVE_FREQ}"

"${PYTHON}" -m verl_grpo.main_ppo \
    --config-name="${CONFIG_NAME}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${STEPS_PER_EPOCH}" \
    "$@"
