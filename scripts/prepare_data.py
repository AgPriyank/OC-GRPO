"""
Convert math_train_100_problems.json (+ optional hints) to parquet for veRL.

Parquet schema (per row):
  - data_source: str ("math")
  - prompt: list[dict] (chat format, e.g. [{"role": "user", "content": ...}])
  - ability: str ("math")
  - reward_model: dict ({"ground_truth": answer_str})
  - extra_info: dict (index, problem_id, question, hints, level, subject, solution)

Usage:
  python scripts/prepare_data.py \
      --problems data/math_train_100_problems.json \
      --output verl_grpo/data \
      --val-ratio 0.1 \
      [--hints hints/qwen2.5-3b-instruct_hints_with_solution.json]
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import pandas as pd


def load_problems(path: str) -> List[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["problems"]


def load_hints(path: str) -> Dict[str, List[str]]:
    """Load hints file. Returns {problem_id: [hint1, hint2, ...]}."""
    with open(path) as f:
        data = json.load(f)
    hints = {}
    for pid, info in data["problems"].items():
        hints[pid] = info.get("hints", [])
    return hints


def build_rows(problems: List[dict], hints: Optional[Dict[str, List[str]]] = None) -> List[dict]:
    rows = []
    for idx, prob in enumerate(problems):
        pid = prob["id"]
        question = prob["question"]
        ground_truth = prob["ground_truth"]
        level = prob.get("level", "")
        subject = prob.get("subject", "")

        # Chat-format prompt (veRL applies chat_template at runtime)
        prompt = [{"role": "user", "content": question}]

        # Hints for this problem (empty list if not a cliff problem or no hints file)
        problem_hints = []
        if hints is not None and pid in hints:
            problem_hints = hints[pid]

        row = {
            "data_source": "math",
            "prompt": prompt,
            "ability": "math",
            "reward_model": {"ground_truth": ground_truth},
            "extra_info": {
                "index": idx,
                "problem_id": pid,
                "question": question,
                "hints": problem_hints,
                "level": level,
                "subject": subject,
                "solution": prob.get("answer", ""),
            },
        }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare parquet data for veRL GRPO training")
    parser.add_argument("--problems", required=True, help="Path to math_train_100_problems.json")
    parser.add_argument("--hints", default=None, help="Optional: path to hints JSON file")
    parser.add_argument("--output", required=True, help="Output directory for parquet files")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    # Load data
    problems = load_problems(args.problems)
    print(f"Loaded {len(problems)} problems from {args.problems}")

    hints = None
    if args.hints:
        hints = load_hints(args.hints)
        print(f"Loaded hints for {len(hints)} problems from {args.hints}")

    # Build rows
    rows = build_rows(problems, hints)
    df = pd.DataFrame(rows)

    # Split train/val
    n_val = max(1, int(len(df) * args.val_ratio))
    df_shuffled = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df_val = df_shuffled.iloc[:n_val]
    df_train = df_shuffled.iloc[n_val:]

    print(f"Train: {len(df_train)} problems, Val: {len(df_val)} problems")

    # Save
    os.makedirs(args.output, exist_ok=True)
    train_path = os.path.join(args.output, "train.parquet")
    val_path = os.path.join(args.output, "val.parquet")

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)

    print(f"Saved: {train_path} ({len(df_train)} rows)")
    print(f"Saved: {val_path} ({len(df_val)} rows)")

    # Verify
    df_check = pd.read_parquet(train_path)
    print(f"\nVerification - columns: {list(df_check.columns)}")
    print(f"First row prompt type: {type(df_check.iloc[0]['prompt'])}")
    print(f"First row extra_info keys: {list(df_check.iloc[0]['extra_info'].keys())}")


if __name__ == "__main__":
    main()
