#!/bin/bash
# ============================================================================
# Generic evaluation launcher: one run, one checkpoint, one test set.
#
# Usage:
#   bash scripts/run_eval.sh <RUN_ID> <STEP> <TEST_SET>
#
# Examples (paper protocol: 16 trajectories, T=0.7, top_p=0.95,
# max_tokens=1000, seed=42):
#   bash scripts/run_eval.sh 852_n16 64 gaokao2023en
#   bash scripts/run_eval.sh 901_masked_IS_n16 120 omnimath1000
#
# RUN_ID is the config suffix (experiment_name = verl_grpo_run<RUN_ID>), so
# checkpoints are read from runs/verl_grpo_run<RUN_ID>/global_step_<STEP>/.
#
# Test-set JSONs are NOT shipped in this repo; build them first with
# scripts/download_benchmarks.py / download_omnimath.py / extract_new_test_sets.py
# (see docs/REPRODUCING.md, 'Evaluation').
# ============================================================================
set -e

RUN_ID=${1:?"Usage: bash scripts/run_eval.sh <RUN_ID> <STEP> <TEST_SET>"}
STEP=${2:?"Usage: bash scripts/run_eval.sh <RUN_ID> <STEP> <TEST_SET>"}
TEST_SET=${3:?"Usage: bash scripts/run_eval.sh <RUN_ID> <STEP> <TEST_SET>"}

LORA_PATH="runs/verl_grpo_run${RUN_ID}/global_step_${STEP}/actor/lora_adapter/"
OUTPUT_DIR="results/verl_run${RUN_ID}_step${STEP}_${TEST_SET}"

# Map TEST_SET to a problems file (paper test sets + per-model training sets).
case ${TEST_SET} in
  aime_1983_2024) PROBLEMS_FILE="data/aime_1983_2024_test_problems.json";       N_PROBLEMS=933;  ;;
  aime25)         PROBLEMS_FILE="data/aime25_test_problems.json";               N_PROBLEMS=30;   ;;
  aime26)         PROBLEMS_FILE="data/aime26_test_problems.json";               N_PROBLEMS=30;   ;;
  gaokao2023en)   PROBLEMS_FILE="data/gaokao2023en_test_problems.json";         N_PROBLEMS=385;  ;;
  omnimath1000)   PROBLEMS_FILE="data/omnimath_test_1000_problems.json";        N_PROBLEMS=1000; ;;
  olympiadbench)  PROBLEMS_FILE="data/olympiadbench_test_problems.json";        N_PROBLEMS=674;  ;;
  olymmath)       PROBLEMS_FILE="data/olymmath_test_problems.json";             N_PROBLEMS=200;  ;;
  minerva)        PROBLEMS_FILE="data/minerva_test_problems.json";              N_PROBLEMS=272;  ;;
  amc23)          PROBLEMS_FILE="data/amc23_test_problems.json";                N_PROBLEMS=40;   ;;
  aime_1983_2026) PROBLEMS_FILE="data/aime_1983_2026_combined_test_problems.json"; N_PROBLEMS=993; ;;  # paper AIME column (build with scripts/combine_aime_datasets.py)
  # Training-set evals (files shipped in this repo):
  extremehard595) PROBLEMS_FILE="data/math_train_extremehard_595_problems.json";      N_PROBLEMS=595; ;;
  extremehard682) PROBLEMS_FILE="data/qwen3b_extremehard_problems.json";              N_PROBLEMS=682; ;;
  extremehard600) PROBLEMS_FILE="data/qwen2.5_1_5b_extremehard_600_problems.json";    N_PROBLEMS=600; ;;
  *) echo "ERROR: Unknown TEST_SET=${TEST_SET}"; exit 1 ;;
esac

if [ ! -f "${PROBLEMS_FILE}" ]; then
  echo "ERROR: ${PROBLEMS_FILE} not found."
  echo "Build test sets first — see docs/REPRODUCING.md ('Evaluation')."
  exit 1
fi
if [ ! -d "${LORA_PATH}" ]; then
  echo "ERROR: checkpoint not found: ${LORA_PATH}"
  exit 1
fi

mkdir -p models/merged_models "${OUTPUT_DIR}"

python evaluate_trained.py \
  --lora_path "${LORA_PATH}" \
  --dataset math \
  --problems_file "${PROBLEMS_FILE}" \
  --n_problems ${N_PROBLEMS} \
  --use_chat_template \
  --n_trajectories 16 \
  --temperature 0.7 \
  --top_p 0.95 \
  --max_tokens 1000 \
  --seed 42 \
  --tensor_parallel_size 1 \
  --output_dir "${OUTPUT_DIR}"

echo "Results written to ${OUTPUT_DIR}/results.json (+ summary.txt)"
