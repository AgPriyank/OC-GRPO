"""
Phase 1: Offline guided data generation for POPE-style augmented GRPO training.

Generates hint-augmented or prefix-augmented trajectories for each problem
in the ExtremeHard 595 dataset. Uses the base untrained model (Qwen2.5-7B-Instruct)
to generate trajectories at each hint/prefix level. Finds the lowest level that
solved each problem and saves the hint/prefix text.

Usage:
    python scripts/generate_offline_guided_data.py \\
        --mode hint \\
        --shard 0 --n_shards 5 \\
        --model_path Qwen/Qwen2.5-7B-Instruct \\
        --problems_file data/math_train_extremehard_595_problems.json \\
        --output_dir offline_guided_data/
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# Ensure project root is on sys.path (needed when run as `python scripts/...`)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Force vLLM V0 engine (same as baseline.py)
os.environ['VLLM_USE_V1'] = '0'

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from scripts.answer_extraction import extract_answer_math, answers_match_math
from verl_grpo.utils.chat_template import apply_chat_template_compat
from verl_grpo.hints.hint_generator import (
    HintGenerator,
    HIERARCHICAL_HINT_PROMPTS,
    HIERARCHICAL2_HINT_LEVELS,
)

ANTI_REP_PROMPT = (
    "You are a math problem solver. When given a hint, use it to guide your reasoning, "
    "but write your solution independently. Do not repeat, copy, or paraphrase the hint text. "
    "Show your own step-by-step work and arrive at the answer through your own reasoning."
)

PREFIX_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hint", "prefix"], required=True,
                        help="Generation mode: hint (L1-L5 cascade) or prefix (fraction cascade)")
    parser.add_argument("--shard", type=int, required=True,
                        help="Shard index (0-based)")
    parser.add_argument("--n_shards", type=int, default=5,
                        help="Total number of shards")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model path for base model")
    parser.add_argument("--problems_file",
                        default="data/math_train_extremehard_595_problems.json",
                        help="JSON file with problems list")
    parser.add_argument("--output_dir", default="offline_guided_data",
                        help="Directory to save shard JSON outputs")
    parser.add_argument("--n_traj", type=int, default=8,
                        help="Trajectories per problem per level")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=1200)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--no_thinking", action="store_true",
                        help="Disable thinking mode for Qwen3+ models")
    return parser.parse_args()


def load_shard(problems_file: str, shard: int, n_shards: int) -> List[Dict]:
    """Load and return the shard-th chunk of problems."""
    with open(problems_file) as f:
        data = json.load(f)
    all_problems = data["problems"]
    chunk_size = (len(all_problems) + n_shards - 1) // n_shards
    start = shard * chunk_size
    end = min(start + chunk_size, len(all_problems))
    print(f"Shard {shard}/{n_shards}: problems[{start}:{end}] = {end-start} problems")
    return all_problems[start:end]


def build_vanilla_prompts(problems: List[Dict], tokenizer, enable_thinking=None) -> List[str]:
    """Standard vanilla prompts (no hints) for base trajectory generation."""
    prompts = []
    for p in problems:
        messages = [{"role": "user", "content": p["question"]}]
        prompt = apply_chat_template_compat(
            tokenizer, messages, enable_thinking=enable_thinking,
            tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    return prompts


def build_hint_traj_prompts(
    level: str,
    problems: List[Dict],
    hints: List[Optional[str]],
    tokenizer,
    enable_thinking=None,
) -> List[Optional[str]]:
    """Build chat-formatted prompts for trajectory generation at a hint level.

    L1-L4: hint text + anti-repetition system prompt.
    L5: full solution rephrase framing, no system prompt.
    Returns None for problems with no hint at this level.
    """
    prompts = []
    for prob, hint in zip(problems, hints):
        if hint is None:
            prompts.append(None)
            continue

        q = prob["question"]
        if level == "L5_solution":
            instruction = (
                "Rephrase the solution above in your own words. "
                "Show your step-by-step work and output the final answer within \\boxed{}."
            )
            prompt_text = f"Problem: {q}\n\nFull solution: {hint}\n\n{instruction}"
            messages = [{"role": "user", "content": prompt_text}]
        else:
            instruction = (
                "Let's think step by step and output the final answer within \\boxed{}. "
                "You can use the hint that follows the question to help you solve the problem."
            )
            prompt_text = f"Problem: {q}\n\nHint: {hint}\n\n{instruction}"
            messages = [
                {"role": "system", "content": ANTI_REP_PROMPT},
                {"role": "user", "content": prompt_text},
            ]

        formatted = apply_chat_template_compat(
            tokenizer, messages, enable_thinking=enable_thinking,
            tokenize=False, add_generation_prompt=True
        )
        prompts.append(formatted)
    return prompts


def build_prefix_traj_prompts(
    fraction: float,
    problems: List[Dict],
    prefixes: List[str],
    tokenizer,
    enable_thinking=None,
) -> List[str]:
    """Build chat-formatted prompts for prefix-guided trajectory generation."""
    prompts = []
    for prob, prefix in zip(problems, prefixes):
        q = prob["question"]
        if fraction >= 1.0:
            instruction = (
                "Here is the full reference solution. Verify it, show your work "
                "step by step, and output the final answer within \\boxed{}."
            )
            prompt_text = f"Problem: {q}\n\nReference solution: {prefix}\n\n{instruction}"
            messages = [{"role": "user", "content": prompt_text}]
        else:
            instruction = (
                "Here is a partial reference solution to the problem above. "
                "Complete the rest of the solution step by step and output the "
                "final answer within \\boxed{}."
            )
            prompt_text = (
                f"Problem: {q}\n\nPartial reference solution: {prefix}\n\n{instruction}"
            )
            messages = [
                {"role": "system", "content": ANTI_REP_PROMPT},
                {"role": "user", "content": prompt_text},
            ]

        formatted = apply_chat_template_compat(
            tokenizer, messages, enable_thinking=enable_thinking,
            tokenize=False, add_generation_prompt=True
        )
        prompts.append(formatted)
    return prompts


def generate_and_check(
    llm: LLM,
    prompts: List[Optional[str]],
    sampling_params: SamplingParams,
    problems: List[Dict],
) -> List[Dict]:
    """Generate trajectories for all problems, check correctness.

    Skips problems where prompt is None.
    Returns list of per-problem dicts: {correct, correct_texts, all_texts}.
    """
    valid_indices = [i for i, p in enumerate(prompts) if p is not None]
    valid_prompts = [prompts[i] for i in valid_indices]

    results = [{"correct": False, "correct_texts": [], "all_texts": []} for _ in problems]
    if not valid_prompts:
        return results

    outputs = llm.generate(valid_prompts, sampling_params)

    for prob_idx, output in zip(valid_indices, outputs):
        prob = problems[prob_idx]
        gt = prob["ground_truth"]
        texts = [o.text for o in output.outputs]
        correct_texts = [
            t for t in texts
            if answers_match_math(extract_answer_math(t), gt)
        ]
        results[prob_idx] = {
            "correct": len(correct_texts) > 0,
            "correct_texts": correct_texts,
            "all_texts": texts,
        }

    return results


def run_hint_mode(
    problems: List[Dict],
    llm: LLM,
    tokenizer,
    sampling_params: SamplingParams,
    enable_thinking=None,
) -> List[Dict]:
    """Run L1-L5 hint cascade for each problem.

    Steps:
    1. Generate base trajectories to get a failed trajectory for L1 context.
    2. Generate L1-L4 hints (batched). L5 = full solution directly.
    3. Generate n_traj trajectories at each level (batched per level).
    4. Find lowest level (L1→L5) where >=1 correct. Record result.
    """
    n = len(problems)
    hint_gen = HintGenerator(tokenizer, hint_mode="solution_aware")

    # Step 1: base trajectories for L1 failed trajectory context
    print(f"  [hint] Generating base trajectories ({n} problems)...")
    vanilla_prompts = build_vanilla_prompts(problems, tokenizer, enable_thinking=enable_thinking)
    vanilla_results = generate_and_check(llm, vanilla_prompts, sampling_params, problems)

    failed_trajs = []
    for res in vanilla_results:
        failed = [t for t in res["all_texts"] if t not in res["correct_texts"]]
        failed_trajs.append(failed[0] if failed else (res["all_texts"][0] if res["all_texts"] else ""))

    # Step 2: generate L1-L4 hints (all problems at once)
    print(f"  [hint] Generating L1-L4 hints ({n} problems)...")
    l1_l4_levels = ["L1_blind", "L2_conceptual", "L3_strategic", "L4_detailed"]
    all_hint_prompts, prompt_map = hint_gen.build_all_hierarchical_prompts(
        l1_l4_levels,
        [p["question"] for p in problems],
        [p["answer"] for p in problems],
        failed_trajs,
    )
    # Generate 1 candidate per hint prompt
    hint_sampling = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512, n=1)
    hint_outputs = llm.generate(all_hint_prompts, hint_sampling)

    hints_by_level: Dict[str, List[Optional[str]]] = {lv: [None] * n for lv in l1_l4_levels}
    for (level, pidx), output in zip(prompt_map, hint_outputs):
        text = output.outputs[0].text
        parsed = HintGenerator.parse_hierarchical_hint(text)
        hints_by_level[level][pidx] = parsed

    # L5: full reference solution (no generation needed)
    hints_by_level["L5_solution"] = [p["answer"] for p in problems]

    # Step 3: trajectory generation per level
    level_results = {}
    for level in HIERARCHICAL2_HINT_LEVELS:
        print(f"  [hint] Level {level}: generating trajectories ({n} problems)...")
        traj_prompts = build_hint_traj_prompts(level, problems, hints_by_level[level], tokenizer,
                                               enable_thinking=enable_thinking)
        level_results[level] = generate_and_check(llm, traj_prompts, sampling_params, problems)

    # Step 4: find lowest level that solved each problem
    records = []
    for pidx, prob in enumerate(problems):
        record = {"problem_id": prob["id"], "is_solved": False}
        for level_num, level in enumerate(HIERARCHICAL2_HINT_LEVELS, start=1):
            if level_results[level][pidx]["correct"]:
                correct_texts = level_results[level][pidx]["correct_texts"]
                record = {
                    "problem_id": prob["id"],
                    "is_solved": True,
                    "hint_level": level_num,
                    "hint_level_name": level,
                    "hint_text": hints_by_level[level][pidx],
                    "one_correct_trajectory": correct_texts[0] if correct_texts else "",
                }
                break
        records.append(record)

    n_solved = sum(r["is_solved"] for r in records)
    print(f"  [hint] Solved: {n_solved}/{n} ({100*n_solved/n:.1f}%)")
    level_counts: Dict[int, int] = {}
    for r in records:
        if r["is_solved"]:
            level_counts[r["hint_level"]] = level_counts.get(r["hint_level"], 0) + 1
    for lv in sorted(level_counts):
        print(f"    L{lv}: {level_counts[lv]} problems")

    return records


def run_prefix_mode(
    problems: List[Dict],
    llm: LLM,
    tokenizer,
    sampling_params: SamplingParams,
    enable_thinking=None,
) -> List[Dict]:
    """Run prefix fraction cascade for each problem.

    Extracts prefixes at [0.2, 0.4, 0.6, 0.8, 1.0] of the reference solution.
    For each fraction, generates n_traj trajectories and checks correctness.
    Finds the lowest fraction where >=1 correct trajectory.
    """
    n = len(problems)

    # Pre-extract all prefixes (deterministic, no LLM needed)
    all_prefixes = {
        fraction: [HintGenerator.get_solution_prefix(p["answer"], fraction) for p in problems]
        for fraction in PREFIX_FRACTIONS
    }

    level_results = {}
    for fraction in PREFIX_FRACTIONS:
        pct = int(fraction * 100)
        print(f"  [prefix] Fraction {pct}%: generating trajectories ({n} problems)...")
        traj_prompts = build_prefix_traj_prompts(
            fraction, problems, all_prefixes[fraction], tokenizer,
            enable_thinking=enable_thinking
        )
        level_results[fraction] = generate_and_check(llm, traj_prompts, sampling_params, problems)

    records = []
    for pidx, prob in enumerate(problems):
        record = {"problem_id": prob["id"], "is_solved": False}
        for fraction in PREFIX_FRACTIONS:
            if level_results[fraction][pidx]["correct"]:
                correct_texts = level_results[fraction][pidx]["correct_texts"]
                record = {
                    "problem_id": prob["id"],
                    "is_solved": True,
                    "prefix_fraction": fraction,
                    "prefix_text": all_prefixes[fraction][pidx],
                    "one_correct_trajectory": correct_texts[0] if correct_texts else "",
                }
                break
        records.append(record)

    n_solved = sum(r["is_solved"] for r in records)
    print(f"  [prefix] Solved: {n_solved}/{n} ({100*n_solved/n:.1f}%)")
    frac_counts: Dict[float, int] = {}
    for r in records:
        if r["is_solved"]:
            frac_counts[r["prefix_fraction"]] = frac_counts.get(r["prefix_fraction"], 0) + 1
    for f in sorted(frac_counts):
        print(f"    {int(f*100)}%: {frac_counts[f]} problems")

    return records


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(
        args.output_dir,
        f"shard_{args.mode}_{args.shard}_of_{args.n_shards}.json",
    )

    if os.path.exists(out_file):
        print(f"Output already exists: {out_file}. Delete to re-run.")
        return

    print("=" * 60)
    print(f"Mode: {args.mode}  |  Shard: {args.shard}/{args.n_shards}")
    print(f"Model: {args.model_path}")
    print(f"Output: {out_file}")
    print("=" * 60)

    problems = load_shard(args.problems_file, args.shard, args.n_shards)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=6000,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n_traj,
    )

    enable_thinking = False if args.no_thinking else None

    if args.mode == "hint":
        records = run_hint_mode(problems, llm, tokenizer, sampling_params,
                                enable_thinking=enable_thinking)
    else:
        records = run_prefix_mode(problems, llm, tokenizer, sampling_params,
                                  enable_thinking=enable_thinking)

    with open(out_file, "w") as f:
        json.dump(records, f, indent=2)

    n_solved = sum(r["is_solved"] for r in records)
    print(f"\nSaved {len(records)} records to {out_file}")
    print(f"Solved: {n_solved}/{len(records)} ({100*n_solved/len(records):.1f}%)")


if __name__ == "__main__":
    main()
