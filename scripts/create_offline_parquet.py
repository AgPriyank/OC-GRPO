"""
Phase 1: Merge offline guided data shards and create augmented training parquets.

Reads all 5 shard JSONs produced by generate_offline_guided_data.py, combines
them with the existing extremehard_595 parquet, and creates two new augmented
parquets:
  - verl_grpo/data_offline_guided_hint/train.parquet
  - verl_grpo/data_offline_guided_prefix/train.parquet

Each parquet row is either:
  - An original row (prompt = raw problem text, same as existing parquet)
  - A guided row (prompt pre-baked with hint/prefix, matching 852/853 format exactly)

The train/val split from the existing parquet is preserved (same problem_id membership).

Usage:
    python scripts/create_offline_parquet.py --mode hint
    python scripts/create_offline_parquet.py --mode prefix
"""

import argparse
import json
import os
import random
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd

ANTI_REP_PROMPT = (
    "You are a math problem solver. When given a hint, use it to guide your reasoning, "
    "but write your solution independently. Do not repeat, copy, or paraphrase the hint text. "
    "Show your own step-by-step work and arrive at the answer through your own reasoning."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hint", "prefix"], required=True,
                        help="Which guided mode to build parquet for")
    parser.add_argument("--n_shards", type=int, default=5)
    parser.add_argument("--shard_dir", default="offline_guided_data",
                        help="Directory containing shard_*.json files")
    parser.add_argument("--base_parquet_dir", default="verl_grpo/data_extremehard_595",
                        help="Existing parquet directory (train.parquet + val.parquet)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: verl_grpo/data_offline_guided_{mode})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle")
    return parser.parse_args()


def load_shard_results(shard_dir: str, mode: str, n_shards: int) -> Dict[str, Dict]:
    """Load and merge all shard JSONs. Returns {problem_id: record}."""
    results: Dict[str, Dict] = {}
    for shard in range(n_shards):
        path = os.path.join(shard_dir, f"shard_{mode}_{shard}_of_{n_shards}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing shard file: {path}\n"
                f"Run generate_offline_guided_data.py --mode {mode} --shard {shard} first."
            )
        with open(path) as f:
            records = json.load(f)
        for rec in records:
            results[rec["problem_id"]] = rec
    print(f"Loaded {len(results)} problem records from {n_shards} shards")
    return results


def build_guided_messages(mode: str, extra: dict, record: dict) -> Optional[List[Dict]]:
    """Build the message list for a guided parquet row.

    Returns a list of {role, content} dicts suitable for the parquet `prompt` column
    (veRL applies the chat template at training time). Returns None if not solved.
    """
    if not record.get("is_solved"):
        return None

    question = extra["question"]

    if mode == "hint":
        hint_text = record.get("hint_text", "")
        hint_level = record.get("hint_level", 1)

        if hint_level == 5:
            # L5: full solution rephrase — NO system prompt (intentional rephrasing)
            instruction = (
                "Rephrase the solution above in your own words. "
                "Show your step-by-step work and output the final answer within \\boxed{}."
            )
            content = f"Problem: {question}\n\nFull solution: {hint_text}\n\n{instruction}"
            return [{"role": "user", "content": content}]
        else:
            # L1-L4: standard hint framing + anti-repetition system prompt
            instruction = (
                "Let's think step by step and output the final answer within \\boxed{}. "
                "You can use the hint that follows the question to help you solve the problem."
            )
            content = f"Problem: {question}\n\nHint: {hint_text}\n\n{instruction}"
            return [
                {"role": "system", "content": ANTI_REP_PROMPT},
                {"role": "user", "content": content},
            ]

    else:  # prefix
        prefix_text = record.get("prefix_text", "")
        fraction = record.get("prefix_fraction", 1.0)

        if fraction >= 1.0:
            # Full solution: NO system prompt (rephrasing is intentional)
            instruction = (
                "Here is the full reference solution. Verify it, show your work "
                "step by step, and output the final answer within \\boxed{}."
            )
            content = f"Problem: {question}\n\nReference solution: {prefix_text}\n\n{instruction}"
            return [{"role": "user", "content": content}]
        else:
            # Partial prefix: anti-repetition system prompt
            instruction = (
                "Here is a partial reference solution to the problem above. "
                "Complete the rest of the solution step by step and output the "
                "final answer within \\boxed{}."
            )
            content = (
                f"Problem: {question}\n\n"
                f"Partial reference solution: {prefix_text}\n\n"
                f"{instruction}"
            )
            return [
                {"role": "system", "content": ANTI_REP_PROMPT},
                {"role": "user", "content": content},
            ]


def build_rows(
    df: pd.DataFrame,
    shard_results: Dict[str, Dict],
    mode: str,
    split: str,
) -> List[Dict]:
    """Build augmented rows from existing parquet rows."""
    rows = []
    n_original = 0
    n_guided = 0

    for _, row in df.iterrows():
        extra = dict(row["extra_info"])
        orig_id = extra["problem_id"]

        # --- Original row (unchanged except guidance_type added) ---
        orig_extra = {**extra, "guidance_type": "original"}
        rows.append({
            "data_source": row["data_source"],
            "prompt": list(row["prompt"]),
            "ability": row["ability"],
            "reward_model": dict(row["reward_model"]),
            "extra_info": orig_extra,
        })
        n_original += 1

        # --- Guided row (if this problem was solved) ---
        record = shard_results.get(orig_id, {"is_solved": False})
        messages = build_guided_messages(mode, extra, record)
        if messages is not None:
            if mode == "hint":
                lv = record["hint_level"]
                guided_id = f"{orig_id}_guided_hint_L{lv}"
                guided_extra = {
                    **extra,
                    "guidance_type": "hint",
                    "problem_id": guided_id,
                    "hint_level": lv,
                    "hint_level_name": record.get("hint_level_name", f"L{lv}"),
                    "hint_text": record["hint_text"],
                }
            else:
                pct = int(record["prefix_fraction"] * 100)
                guided_id = f"{orig_id}_guided_prefix_{pct}pct"
                guided_extra = {
                    **extra,
                    "guidance_type": "prefix",
                    "problem_id": guided_id,
                    "prefix_fraction": record["prefix_fraction"],
                    "prefix_text": record["prefix_text"],
                }

            rows.append({
                "data_source": row["data_source"],
                "prompt": messages,
                "ability": row["ability"],
                "reward_model": dict(row["reward_model"]),
                "extra_info": guided_extra,
            })
            n_guided += 1

    n_total = n_original + n_guided
    pct = 100 * n_guided / n_original if n_original else 0
    print(f"  [{split}] {n_original} original + {n_guided} guided "
          f"({pct:.1f}% solved) = {n_total} total rows")
    return rows


def print_stats(shard_results: Dict[str, Dict], mode: str):
    n_total = len(shard_results)
    n_solved = sum(1 for r in shard_results.values() if r.get("is_solved"))
    print(f"\nOverall: {n_solved}/{n_total} problems solved ({100*n_solved/n_total:.1f}%)")

    if mode == "hint":
        level_counts: Dict[int, int] = {}
        for r in shard_results.values():
            if r.get("is_solved"):
                lv = r["hint_level"]
                level_counts[lv] = level_counts.get(lv, 0) + 1
        print("Level distribution:")
        for lv in sorted(level_counts):
            print(f"  L{lv}: {level_counts[lv]} problems")
    else:
        frac_counts: Dict[float, int] = {}
        for r in shard_results.values():
            if r.get("is_solved"):
                f = r["prefix_fraction"]
                frac_counts[f] = frac_counts.get(f, 0) + 1
        print("Fraction distribution:")
        for f in sorted(frac_counts):
            print(f"  {int(f*100)}%: {frac_counts[f]} problems")


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = f"verl_grpo/data_offline_guided_{args.mode}"

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Base parquet: {args.base_parquet_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    shard_results = load_shard_results(args.shard_dir, args.mode, args.n_shards)
    print_stats(shard_results, args.mode)

    random.seed(args.seed)

    for split in ["train", "val"]:
        parquet_path = os.path.join(args.base_parquet_dir, f"{split}.parquet")
        df = pd.read_parquet(parquet_path)
        print(f"\nBuilding {split} rows from {len(df)} base rows...")

        rows = build_rows(df, shard_results, args.mode, split)

        # Shuffle to randomly interleave original and guided rows
        random.shuffle(rows)

        out_df = pd.DataFrame(rows)
        out_path = os.path.join(args.output_dir, f"{split}.parquet")
        out_df.to_parquet(out_path, index=False)
        print(f"  Saved {len(out_df)} rows → {out_path}")

    # Verification
    print("\n--- Verification ---")
    for split in ["train", "val"]:
        out_path = os.path.join(args.output_dir, f"{split}.parquet")
        df = pd.read_parquet(out_path)
        types = [r.get("guidance_type", "?") for r in df["extra_info"]]
        counts = Counter(types)
        print(f"  {split}: {dict(counts)}  (total={len(df)})")

    print("\nDone. Sample guided row prompt:")
    df = pd.read_parquet(os.path.join(args.output_dir, "train.parquet"))
    guided_rows = df[df["extra_info"].apply(lambda x: x.get("guidance_type") != "original")]
    if len(guided_rows) > 0:
        sample = guided_rows.iloc[0]
        print(f"  problem_id: {sample['extra_info']['problem_id']}")
        for msg in sample["prompt"]:
            role = msg.get("role", "?")
            content_preview = msg.get("content", "")[:120].replace("\n", "\\n")
            print(f"  [{role}] {content_preview}...")


if __name__ == "__main__":
    main()
