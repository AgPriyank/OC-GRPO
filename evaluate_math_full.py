"""
Evaluate a model on the full MATH dataset (Levels 3, 4, 5) with per-level and per-subject breakdowns.
Standalone script — only imports load_math_problems from baseline.py for data loading.
"""
import argparse
import json
import re
import os
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict

# Force vLLM to use V0 engine (more stable)
os.environ['VLLM_USE_V1'] = '0'

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from math_verify import parse, verify

# Import centralized answer extraction module
from scripts.answer_extraction import (
    extract_answer_math,
    extract_math_ground_truth,
    answers_match_math
)

from baseline import load_math_problems


# ============================================================================
# MATH Answer Extraction & Matching - Now imported from scripts.answer_extraction
# ============================================================================
# Functions available via import:
#   - extract_answer_math(text: str) -> Optional[str]
#   - extract_math_ground_truth(solution: str) -> str
#   - answers_match_math(predicted: Optional[str], ground_truth: str) -> bool


# ============================================================================
# Prompt Formatting (copied from baseline.py — modify as needed)
# ============================================================================

def format_prompts(problems: List[Dict], tokenizer, add_instruction: bool = True, use_chat_template: bool = True, enable_thinking: Optional[bool] = None) -> List[str]:
    """Format problems into prompts using chat template or plain text."""
    instruction = 'Please reason step by step, and put your final answer within \\boxed{}.'

    prompts = []
    for problem in problems:
        question = problem['question']
        if add_instruction:
            question = question + " " + instruction

        if use_chat_template:
            # For instruct models: use chat template
            messages = [{"role": "user", "content": question}]
            kwargs = dict(
                tokenize=False,
                add_generation_prompt=True
            )
            if enable_thinking is not None:
                kwargs['enable_thinking'] = enable_thinking
            prompt = tokenizer.apply_chat_template(messages, **kwargs)
        else:
            # For base models: just use plain text
            prompt = question

        prompts.append(prompt)

    return prompts


# ============================================================================
# Evaluation (adapted from baseline.py — MATH only)
# ============================================================================

def evaluate_model(
    model_path: str,
    problems: List[Dict],
    n_trajectories: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    tensor_parallel_size: int,
    seed: int,
    use_chat_template: bool = True,
    max_model_len: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
) -> List[Dict]:
    """Run evaluation: load model, generate, evaluate. MATH-specific."""

    # Load tokenizer
    print(f"\nLoading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Format prompts
    prompts = format_prompts(problems, tokenizer, use_chat_template=use_chat_template, enable_thinking=enable_thinking)

    # Load model with vLLM
    print(f"\nLoading model with vLLM: {model_path}")
    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype="auto",
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )
    if max_model_len is not None:
        llm_kwargs['max_model_len'] = max_model_len
    llm = LLM(**llm_kwargs)

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n_trajectories,
        seed=seed
    )

    # Generate trajectories
    print(f"\nGenerating {n_trajectories} trajectories per prompt...")
    outputs = llm.generate(prompts, sampling_params)

    # Extract trajectories
    all_trajectories = []
    for output in outputs:
        trajectories = [(out.text, len(out.token_ids), out.finish_reason) for out in output.outputs]
        all_trajectories.append(trajectories)

    # Evaluate
    print("\nEvaluating trajectories...")
    results = []
    for problem, trajectories in tqdm(zip(problems, all_trajectories), total=len(problems)):
        trajectory_results = []
        correct_count = 0

        for traj_text, n_tokens, finish_reason in trajectories:
            predicted_answer = extract_answer_math(traj_text)
            is_correct = answers_match_math(predicted_answer, problem['ground_truth'])

            if is_correct:
                correct_count += 1

            trajectory_results.append({
                'predicted_answer': predicted_answer,
                'correct': is_correct,
                'n_tokens': n_tokens,
                'truncated': finish_reason == "length",
            })

        # Compute success rate
        success_rate = correct_count / len(trajectories)

        # Categorize
        if success_rate == 0.0:
            category = 'hard_cliff'
        elif success_rate <= 0.25:
            category = 'soft_cliff'
        elif success_rate >= 0.75:
            category = 'too_easy'
        else:
            category = 'good'

        # Compute avg tokens
        traj_tokens = [t['n_tokens'] for t in trajectory_results]
        avg_tokens = sum(traj_tokens) / len(traj_tokens)

        results.append({
            'id': problem['id'],
            'level': problem.get('level', 'unknown'),
            'subject': problem.get('subject', 'unknown'),
            'ground_truth': problem['ground_truth'],
            'trajectories': trajectory_results,
            'success_rate': success_rate,
            'n_correct': correct_count,
            'n_total': len(trajectories),
            'category': category,
            'avg_tokens': avg_tokens,
        })

    return results


# ============================================================================
# Statistics (per-level and per-subject breakdowns)
# ============================================================================

def compute_math_statistics(results: List[Dict]) -> Dict:
    """Compute overall, per-level, and per-subject statistics."""
    total_problems = len(results)

    # Overall
    mean_accuracy = sum(r['success_rate'] for r in results) / total_problems

    hard_cliff = [r for r in results if r['category'] == 'hard_cliff']
    soft_cliff = [r for r in results if r['category'] == 'soft_cliff']
    good = [r for r in results if r['category'] == 'good']
    too_easy = [r for r in results if r['category'] == 'too_easy']

    mean_avg_tokens = sum(r['avg_tokens'] for r in results) / total_problems

    # Truncation stats
    all_trajs = [t for r in results for t in r['trajectories']]
    n_truncated = sum(1 for t in all_trajs if t.get('truncated', False))
    n_truncated_correct = sum(1 for t in all_trajs if t.get('truncated', False) and t.get('correct', False))
    n_truncated_incorrect = sum(1 for t in all_trajs if t.get('truncated', False) and not t.get('correct', False))

    overall = {
        'n_problems': total_problems,
        'mean_accuracy': mean_accuracy,
        'mean_avg_tokens': mean_avg_tokens,
        'n_hard_cliff': len(hard_cliff),
        'n_soft_cliff': len(soft_cliff),
        'n_good': len(good),
        'n_too_easy': len(too_easy),
        'n_truncated': n_truncated,
        'n_truncated_correct': n_truncated_correct,
        'n_truncated_incorrect': n_truncated_incorrect,
        'n_total_trajectories': len(all_trajs),
    }

    # Per-level
    per_level = defaultdict(list)
    for r in results:
        per_level[r['level']].append(r)

    per_level_stats = {}
    for level in sorted(per_level.keys()):
        items = per_level[level]
        per_level_stats[level] = {
            'n_problems': len(items),
            'mean_accuracy': sum(r['success_rate'] for r in items) / len(items),
            'mean_avg_tokens': sum(r['avg_tokens'] for r in items) / len(items),
        }

    # Per-subject
    per_subject = defaultdict(list)
    for r in results:
        per_subject[r['subject']].append(r)

    per_subject_stats = {}
    for subject in sorted(per_subject.keys()):
        items = per_subject[subject]
        per_subject_stats[subject] = {
            'n_problems': len(items),
            'mean_accuracy': sum(r['success_rate'] for r in items) / len(items),
            'mean_avg_tokens': sum(r['avg_tokens'] for r in items) / len(items),
        }

    return {
        'overall': overall,
        'per_level': per_level_stats,
        'per_subject': per_subject_stats,
    }


# ============================================================================
# Printing
# ============================================================================

def print_math_summary(stats: Dict):
    """Print summary with per-level and per-subject breakdowns."""
    overall = stats['overall']

    print("\n" + "=" * 60)
    print("MATH EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total problems:    {overall['n_problems']}")
    print(f"Mean accuracy:     {overall['mean_accuracy']:.4f}")
    print(f"Mean avg tokens:   {overall['mean_avg_tokens']:.1f}")
    print(f"\nProblem categorization:")
    print(f"  Hard cliffs (0% success):     {overall['n_hard_cliff']:3d} ({100 * overall['n_hard_cliff'] / overall['n_problems']:.1f}%)")
    print(f"  Soft cliffs (>0%, ≤25%):      {overall['n_soft_cliff']:3d} ({100 * overall['n_soft_cliff'] / overall['n_problems']:.1f}%)")
    print(f"  Good (>25%, <75%):            {overall['n_good']:3d} ({100 * overall['n_good'] / overall['n_problems']:.1f}%)")
    print(f"  Too easy (≥75%):              {overall['n_too_easy']:3d} ({100 * overall['n_too_easy'] / overall['n_problems']:.1f}%)")

    n_total_traj = overall.get('n_total_trajectories', 0)
    n_trunc = overall.get('n_truncated', 0)
    if n_total_traj > 0:
        print(f"\nTruncation stats:")
        print(f"  Total trajectories:     {n_total_traj}")
        print(f"  Truncated:              {n_trunc} ({100 * n_trunc / n_total_traj:.1f}%)")
        print(f"    Truncated + correct:  {overall.get('n_truncated_correct', 0)}")
        print(f"    Truncated + wrong:    {overall.get('n_truncated_incorrect', 0)}")

    print("\n" + "-" * 60)
    print("PER-LEVEL ACCURACY")
    print("-" * 60)
    for level, s in sorted(stats['per_level'].items()):
        print(f"  {level:10s}:  acc={s['mean_accuracy']:.4f}  avg_tok={s['mean_avg_tokens']:.1f}  (n={s['n_problems']})")

    print("\n" + "-" * 60)
    print("PER-SUBJECT ACCURACY")
    print("-" * 60)
    for subject, s in sorted(stats['per_subject'].items()):
        print(f"  {subject:30s}:  acc={s['mean_accuracy']:.4f}  avg_tok={s['mean_avg_tokens']:.1f}  (n={s['n_problems']})")

    print("=" * 60)


# ============================================================================
# Saving
# ============================================================================

def save_math_results(results: List[Dict], stats: Dict, args, output_dir: str):
    """Save results JSON (no trajectory text) and summary text."""
    os.makedirs(output_dir, exist_ok=True)

    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model': args.model,
            'levels': args.levels,
            'subjects': args.subjects,
            'split': args.split,
            'n_problems_requested': args.n_problems,
            'n_problems_actual': len(results),
            'n_trajectories': args.n_trajectories,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'max_tokens': args.max_tokens,
            'seed': args.seed,
        },
        'statistics': stats,
        'per_problem': results,
    }

    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary text
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        overall = stats['overall']
        f.write("=" * 60 + "\n")
        f.write("MATH EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Levels: {args.levels}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Timestamp: {output_data['metadata']['timestamp']}\n")
        f.write(f"\nTotal problems:    {overall['n_problems']}\n")
        f.write(f"Mean accuracy:     {overall['mean_accuracy']:.4f}\n")
        f.write(f"Mean avg tokens:   {overall['mean_avg_tokens']:.1f}\n")
        f.write(f"\nProblem categorization:\n")
        f.write(f"  Hard cliffs (0%):        {overall['n_hard_cliff']:3d} ({100 * overall['n_hard_cliff'] / overall['n_problems']:.1f}%)\n")
        f.write(f"  Soft cliffs (≤25%):      {overall['n_soft_cliff']:3d} ({100 * overall['n_soft_cliff'] / overall['n_problems']:.1f}%)\n")
        f.write(f"  Good (>25%, <75%):       {overall['n_good']:3d} ({100 * overall['n_good'] / overall['n_problems']:.1f}%)\n")
        f.write(f"  Too easy (≥75%):         {overall['n_too_easy']:3d} ({100 * overall['n_too_easy'] / overall['n_problems']:.1f}%)\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write("PER-LEVEL ACCURACY\n")
        f.write("-" * 60 + "\n")
        for level, s in sorted(stats['per_level'].items()):
            f.write(f"  {level:10s}:  acc={s['mean_accuracy']:.4f}  avg_tok={s['mean_avg_tokens']:.1f}  (n={s['n_problems']})\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write("PER-SUBJECT ACCURACY\n")
        f.write("-" * 60 + "\n")
        for subject, s in sorted(stats['per_subject'].items()):
            f.write(f"  {subject:30s}:  acc={s['mean_accuracy']:.4f}  avg_tok={s['mean_avg_tokens']:.1f}  (n={s['n_problems']})\n")

    print(f"Summary saved to {summary_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate model on MATH dataset with per-level/per-subject breakdowns')

    # Model and data
    parser.add_argument('--model', type=str, required=True,
                        help='Model path (HF name or local checkpoint)')
    parser.add_argument('--n_problems', type=int, default=999999,
                        help='Max number of problems to evaluate (default: all)')
    parser.add_argument('--levels', type=int, nargs='+', default=[3, 4, 5],
                        help='Difficulty levels to include (default: 3 4 5)')
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                        help='Subjects to include (default: all)')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'test'], help='Dataset split')
    parser.add_argument('--problems_file', type=str, default=None,
                        help='Path to local JSON problems file (overrides HF MATH loading)')
    parser.add_argument('--use_chat_template', action='store_true',
                        help='Use chat template (for instruct models)')
    parser.add_argument('--no_chat_template', dest='use_chat_template', action='store_false',
                        help='Do not use chat template (for base models)')
    parser.set_defaults(use_chat_template=True)
    parser.add_argument('--enable_thinking', action='store_true', default=None,
                        help='Enable thinking mode (for Qwen3 etc.)')
    parser.add_argument('--no_thinking', dest='enable_thinking', action='store_false',
                        help='Disable thinking mode (for Qwen3 non-think evaluation)')

    # Generation parameters (same defaults as baseline.py)
    parser.add_argument('--n_trajectories', type=int, default=8,
                        help='Number of trajectories per problem')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p sampling')
    parser.add_argument('--max_tokens', type=int, default=512,
                        help='Max generation length')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Infrastructure
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--max_model_len', type=int, default=None,
                        help='Max model context length for vLLM (default: model config)')

    # Sharding (for parallel eval across multiple jobs)
    parser.add_argument('--shard_id', type=int, default=None,
                        help='Shard index (0-based). Use with --n_shards for parallel eval.')
    parser.add_argument('--n_shards', type=int, default=1,
                        help='Total number of shards (default: 1 = no sharding)')

    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')

    # W&B logging
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='W&B project name (optional)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name (optional)')

    args = parser.parse_args()

    print("=" * 60)
    print("MATH FULL EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Levels: {args.levels}")
    print(f"Subjects: {args.subjects or 'all'}")
    print(f"Split: {args.split}")
    print(f"Max problems: {args.n_problems}")
    print(f"Trajectories per problem: {args.n_trajectories}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print(f"Output: {args.output_dir}")
    if args.shard_id is not None:
        print(f"Shard: {args.shard_id} of {args.n_shards}")
    print("=" * 60)

    # ---- Load problems ----
    if args.problems_file:
        print(f"Loading problems from local file: {args.problems_file}")
        with open(args.problems_file, 'r') as f:
            data = json.load(f)
        problems = data['problems']
        if args.n_problems < len(problems):
            problems = problems[:args.n_problems]
        print(f"Loaded {len(problems)} problems from {args.problems_file}")
    else:
        problems = load_math_problems(
            n_problems=args.n_problems,
            levels=args.levels,
            subjects=args.subjects,
            split=args.split,
            seed=args.seed,
        )

    # ---- Apply sharding ----
    if args.shard_id is not None:
        total = len(problems)
        problems = problems[args.shard_id::args.n_shards]
        print(f"Shard {args.shard_id}/{args.n_shards}: {len(problems)} problems (of {total} total)")

    # ---- Evaluate ----
    results = evaluate_model(
        model_path=args.model,
        problems=problems,
        n_trajectories=args.n_trajectories,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        use_chat_template=args.use_chat_template,
        max_model_len=args.max_model_len,
        enable_thinking=args.enable_thinking,
    )

    # ---- Compute statistics ----
    stats = compute_math_statistics(results)

    # ---- Print and save ----
    print_math_summary(stats)
    save_math_results(results, stats, args, args.output_dir)

    # ---- W&B logging ----
    if args.wandb_project:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

        overall = stats['overall']
        log_dict = {
            "mean_accuracy": overall['mean_accuracy'],
            "n_problems": overall['n_problems'],
            "n_hard_cliff": overall['n_hard_cliff'],
            "n_soft_cliff": overall['n_soft_cliff'],
            "n_good": overall['n_good'],
            "n_too_easy": overall['n_too_easy'],
        }
        for level, s in stats['per_level'].items():
            log_dict[f"accuracy/{level}"] = s['mean_accuracy']
        for subject, s in stats['per_subject'].items():
            log_dict[f"accuracy/{subject}"] = s['mean_accuracy']
        wandb.log(log_dict)

        # Problem-level table
        table = wandb.Table(columns=["id", "level", "subject", "success_rate", "n_correct", "n_total", "category"])
        for r in results:
            table.add_data(r['id'], r['level'], r['subject'], r['success_rate'], r['n_correct'], r['n_total'], r['category'])
        wandb.log({"problem_details": table})

        wandb.finish()
        print("Results logged to W&B")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()