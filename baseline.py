"""
Simple evaluation script for GSM8K, AIME, GSM-Plus, and MATH.
Loads model with vLLM, generates trajectories, evaluates correctness, computes statistics.
"""
import argparse
import json
import re
import os
from typing import List, Dict, Optional
from datetime import datetime

# Force vLLM to use V0 engine (more stable)
os.environ['VLLM_USE_V1'] = '0'

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from math_verify import parse, verify

# Import centralized answer extraction module (for MATH dataset)
from scripts.answer_extraction import (
    extract_answer_math,
    extract_math_ground_truth,
    answers_match_math
)


# ============================================================================
# Answer Extraction Functions (GSM8K/AIME/GSM-Plus)
# ============================================================================

def extract_answer(text: str, method: str = "flexible") -> Optional[str]:
    """Extract numerical answer from model output."""
    # Method 1: Look for #### pattern (GSM8K format)
    answer_pattern = re.search(r'####\s*([-+]?[\d,]*\.?\d+)', text)
    if answer_pattern:
        answer = answer_pattern.group(1).replace(',', '')
        return answer
    
    # Method 2 (flexible): Extract last number in text
    if method == "flexible":
        numbers = re.findall(r'[-+]?[\d,]*\.?\d+', text)
        if numbers:
            answer = numbers[-1].replace(',', '')
            return answer
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    
    answer = answer.replace(',', '').strip()
    
    # Normalize decimals (42.0 -> 42)
    try:
        num = float(answer)
        if num.is_integer():
            answer = str(int(num))
        else:
            answer = str(num)
    except (ValueError, AttributeError):
        pass
    
    return answer.lower()


def answers_match(predicted: Optional[str], ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    if predicted is None:
        return False
    return normalize_answer(predicted) == normalize_answer(ground_truth)


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract numerical answer from GSM8K's answer format."""
    answer_pattern = re.search(r'####\s*([-+]?[\d,]*\.?\d+)', answer_text)
    if answer_pattern:
        return answer_pattern.group(1).replace(',', '')
    return answer_text.strip()


# ============================================================================
# MATH Dataset Functions - Now imported from scripts.answer_extraction
# ============================================================================
# Functions available via import:
#   - extract_answer_math(text: str) -> Optional[str]
#   - extract_math_ground_truth(solution: str) -> str
#   - answers_match_math(predicted: Optional[str], ground_truth: str) -> bool
#
# These functions handle:
#   - Nested brace matching for LaTeX \boxed{} extraction
#   - LaTeX normalization (\dfrac, \tfrac, \cfrac variants)
#   - Semantic equivalence via math_verify
#   - Fallback to string comparison when math_verify fails


# ============================================================================
# Data Loading
# ============================================================================

def load_gsm8k_problems(n_problems: int, split: str = "test", seed: int = 42) -> List[Dict]:
    """Load GSM8K problems."""
    print(f"Loading GSM8K {split} split...")
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    
    # Sample problems
    if n_problems < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(n_problems))
    
    problems = []
    for idx, item in enumerate(dataset):
        problems.append({
            'id': f"gsm8k_{split}_{idx}",
            'question': item['question'],
            'answer': item['answer'],
            'ground_truth': extract_gsm8k_answer(item['answer'])
        })
    
    print(f"Loaded {len(problems)} GSM8K problems")
    return problems


def load_aime_problems(n_problems: int, split: str = "train", seed: int = 42) -> List[Dict]:
    """Load AIME problems from gneubig/aime-1983-2024 dataset."""
    print(f"Loading AIME problems...")
    dataset = load_dataset("gneubig/aime-1983-2024", split=split)
    
    # Sample problems
    if n_problems < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(n_problems))
    
    problems = []
    for idx, item in enumerate(dataset):
        # Dataset fields: ID, Year, Problem Number, Question, Answer, Part
        problems.append({
            'id': item['ID'],  # e.g., "1983-1"
            'question': item['Question'],
            'answer': str(item['Answer']),  # AIME answers are integers 0-999
            'ground_truth': str(item['Answer'])
        })
    
    print(f"Loaded {len(problems)} AIME problems")
    return problems


def load_gsmplus_problems(n_problems: int, split: str = "test", seed: int = 42) -> List[Dict]:
    """
    Load GSM-Plus problems from qintongli/GSM-Plus dataset.
    
    GSM-Plus is a robustness evaluation dataset with 8 perturbations per GSM8K problem:
    - numerical substitution, digit expansion, integer-decimal-fraction conversion
    - adding operation, reversing operation, problem understanding
    - distraction insertion, critical thinking
    
    Args:
        n_problems: Number of problems to load
        split: 'test' (10,552 examples) or 'testmini' (2,400 examples)
        seed: Random seed for shuffling
    
    Returns:
        List of problem dictionaries with keys:
            - id: Problem identifier
            - question: Problem text
            - answer: Full answer text (for compatibility)
            - ground_truth: Extracted numerical answer
    """
    print(f"Loading GSM-Plus {split} split...")
    dataset = load_dataset("qintongli/GSM-Plus", split=split)
    
    # Sample problems
    if n_problems < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(n_problems))
    
    problems = []
    for idx, item in enumerate(dataset):
        # GSM-Plus fields: question, solution, answer, perturbation_type, seed_question, seed_solution, seed_answer
        # We only use: question and answer
        problems.append({
            'id': f"gsmplus_{split}_{idx}",
            'question': item['question'],
            'answer': item['answer'],  # Already in simple format (e.g., "27")
            'ground_truth': item['answer']  # Answer is already clean, no extraction needed
        })
    
    print(f"Loaded {len(problems)} GSM-Plus problems")
    return problems


def load_math_problems(n_problems: int, levels: List[int] = [1, 2], subjects: Optional[List[str]] = None, split: str = "train", seed: int = 42) -> List[Dict]:
    """
    Load MATH dataset problems (Hendrycks et al.).
    
    Args:
        n_problems: Number of problems to load
        levels: List of difficulty levels to include (1-5). Default: [1, 2]
        subjects: List of subjects to include. Default: all subjects
                 Options: algebra, counting_and_probability, geometry, 
                         intermediate_algebra, number_theory, prealgebra, precalculus
        split: 'train' or 'test'
        seed: Random seed for shuffling
    
    Returns:
        List of problem dictionaries
    """
    # All available subjects
    all_subjects = [
        'algebra', 'counting_and_probability', 'geometry',
        'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'
    ]
    
    if subjects is None:
        subjects = all_subjects
    
    print(f"Loading MATH dataset (levels {levels}, subjects: {len(subjects)})...")
    
    # Load and concatenate datasets from all subjects
    all_problems = []
    for subject in subjects:
        dataset = load_dataset("EleutherAI/hendrycks_math", subject, split=split)
        
        # Filter by level
        for item in dataset:
            # Parse level - some entries have "Level ?" which we skip
            try:
                level_num = int(item['level'].split()[-1])  # "Level 3" -> 3
            except ValueError:
                # Skip entries with invalid level like "Level ?"
                continue
                
            if level_num in levels:
                all_problems.append({
                    'problem': item['problem'],
                    'solution': item['solution'],
                    'level': item['level'],
                    'subject': subject
                })
    
    # Shuffle and sample
    import random
    random.seed(seed)
    random.shuffle(all_problems)
    
    if n_problems < len(all_problems):
        all_problems = all_problems[:n_problems]
    
    # Format with IDs and ground truth
    problems = []
    for idx, item in enumerate(all_problems):
        ground_truth = extract_math_ground_truth(item['solution'])
        problems.append({
            'id': f"math_{split}_{idx}",
            'question': item['problem'],
            'answer': item['solution'],
            'ground_truth': ground_truth,
            'level': item['level'],
            'subject': item['subject']
        })
    
    print(f"Loaded {len(problems)} MATH problems")
    return problems


def format_prompts(problems: List[Dict], tokenizer, add_instruction: bool = True, use_chat_template: bool = True) -> List[str]:
    """Format problems into prompts using Qwen chat template or plain text."""
    instruction = 'Let\'s think step by step and output the final answer after "####".'
    
    prompts = []
    for problem in problems:
        question = problem['question']
        if add_instruction:
            question = question + " " + instruction
        
        if use_chat_template:
            # For instruct models: use chat template
            messages = [{"role": "user", "content": question}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # For base models: just use plain text
            prompt = question
        
        prompts.append(prompt)
    
    return prompts


# ============================================================================
# Evaluation
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
    dataset: str = "gsm8k"
) -> List[Dict]:
    """Run evaluation: load model, generate, evaluate."""
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Format prompts
    prompts = format_prompts(problems, tokenizer, use_chat_template=use_chat_template)
    
    # Load model with vLLM
    print(f"\nLoading model with vLLM: {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.85,  # Leave some memory for other processes
    )
    
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
        trajectories = [(out.text, len(out.token_ids)) for out in output.outputs]
        all_trajectories.append(trajectories)
    
    # Evaluate
    print("\nEvaluating trajectories...")
    results = []
    for problem, trajectories in tqdm(zip(problems, all_trajectories), total=len(problems)):
        trajectory_results = []
        correct_count = 0
        
        for traj_text, n_tokens in trajectories:
            # Use dataset-specific extraction and matching
            if dataset == "math":
                predicted_answer = extract_answer_math(traj_text)
                is_correct = answers_match_math(predicted_answer, problem['ground_truth'])
            else:
                predicted_answer = extract_answer(traj_text, method="flexible")
                is_correct = answers_match(predicted_answer, problem['ground_truth'])

            if is_correct:
                correct_count += 1

            trajectory_results.append({
                'text': traj_text,
                'predicted_answer': predicted_answer,
                'correct': is_correct,
                'n_tokens': n_tokens
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
        
        # Pass@1: whether the first trajectory is correct
        pass_at_1 = trajectory_results[0]['correct'] if trajectory_results else False

        results.append({
            'id': problem['id'],
            'question': problem['question'],
            'ground_truth': problem['ground_truth'],
            'trajectories': trajectory_results,
            'success_rate': success_rate,
            'pass_at_1': pass_at_1,
            'n_correct': correct_count,
            'n_total': len(trajectories),
            'category': category
        })
    
    return results


# ============================================================================
# Statistics
# ============================================================================

def compute_statistics(results: List[Dict]) -> Dict:
    """Compute overall statistics."""
    total_problems = len(results)
    
    hard_cliff = [r for r in results if r['category'] == 'hard_cliff']
    soft_cliff = [r for r in results if r['category'] == 'soft_cliff']
    good = [r for r in results if r['category'] == 'good']
    too_easy = [r for r in results if r['category'] == 'too_easy']
    
    mean_accuracy = sum(r['success_rate'] for r in results) / total_problems
    pass_at_1_accuracy = sum(1 for r in results if r.get('pass_at_1', False)) / total_problems

    return {
        'n_problems': total_problems,
        'mean_accuracy': mean_accuracy,
        'pass_at_1_accuracy': pass_at_1_accuracy,
        'n_hard_cliff': len(hard_cliff),
        'n_soft_cliff': len(soft_cliff),
        'n_good': len(good),
        'n_too_easy': len(too_easy),
        'hard_cliff_ids': [r['id'] for r in hard_cliff],
        'soft_cliff_ids': [r['id'] for r in soft_cliff],
        'good_ids': [r['id'] for r in good],
        'too_easy_ids': [r['id'] for r in too_easy]
    }


def print_summary(stats: Dict):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total problems: {stats['n_problems']}")
    print(f"Mean accuracy (Pass@N): {stats['mean_accuracy']:.3f}")
    print(f"Pass@1 accuracy:        {stats.get('pass_at_1_accuracy', 0):.3f}")
    print(f"\nProblem categorization:")
    print(f"  Hard cliffs (0% success):     {stats['n_hard_cliff']:3d} ({100*stats['n_hard_cliff']/stats['n_problems']:.1f}%)")
    print(f"  Soft cliffs (>0%, ≤25%):      {stats['n_soft_cliff']:3d} ({100*stats['n_soft_cliff']/stats['n_problems']:.1f}%)")
    print(f"  Good (>25%, <75%):            {stats['n_good']:3d} ({100*stats['n_good']/stats['n_problems']:.1f}%)")
    print(f"  Too easy (≥75%):              {stats['n_too_easy']:3d} ({100*stats['n_too_easy']/stats['n_problems']:.1f}%)")
    print("="*60)


# ============================================================================
# Saving Results
# ============================================================================

def save_results(results: List[Dict], stats: Dict, args, output_dir: str):
    """Save results to JSON."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'dataset': args.dataset,
            'model': args.model,
            'n_problems': args.n_problems,
            'n_trajectories': args.n_trajectories,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'max_tokens': args.max_tokens,
            'seed': args.seed
        },
        'statistics': stats,
        'results': results
    }
    
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Save summary
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Dataset: {args.dataset.upper()}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Timestamp: {output_data['metadata']['timestamp']}\n")
        f.write(f"\nTotal problems: {stats['n_problems']}\n")
        f.write(f"Mean accuracy (Pass@N): {stats['mean_accuracy']:.3f}\n")
        f.write(f"Pass@1 accuracy:        {stats.get('pass_at_1_accuracy', 0):.3f}\n")
        f.write(f"\nProblem categorization:\n")
        f.write(f"  Hard cliffs (0%):        {stats['n_hard_cliff']:3d} ({100*stats['n_hard_cliff']/stats['n_problems']:.1f}%)\n")
        f.write(f"  Soft cliffs (≤25%):      {stats['n_soft_cliff']:3d} ({100*stats['n_soft_cliff']/stats['n_problems']:.1f}%)\n")
        f.write(f"  Good (>25%, <75%):       {stats['n_good']:3d} ({100*stats['n_good']/stats['n_problems']:.1f}%)\n")
        f.write(f"  Too easy (≥75%):         {stats['n_too_easy']:3d} ({100*stats['n_too_easy']/stats['n_problems']:.1f}%)\n")
    
    print(f"Summary saved to {summary_path}")


# ============================================================================
# W&B Logging
# ============================================================================

def log_to_wandb(results: List[Dict], stats: Dict, args):
    """Log results to Weights & Biases."""
    if not args.wandb_project:
        return
    
    import wandb
    
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args)
    )
    
    # Log summary metrics
    wandb.log({
        "mean_accuracy": stats['mean_accuracy'],
        "n_hard_cliff": stats['n_hard_cliff'],
        "n_soft_cliff": stats['n_soft_cliff'],
        "n_good": stats['n_good'],
        "n_too_easy": stats['n_too_easy'],
        "pct_hard_cliff": 100 * stats['n_hard_cliff'] / stats['n_problems'],
        "pct_soft_cliff": 100 * stats['n_soft_cliff'] / stats['n_problems'],
        "pct_good": 100 * stats['n_good'] / stats['n_problems'],
        "pct_too_easy": 100 * stats['n_too_easy'] / stats['n_problems'],
    })
    
    # Log problem-level table
    table = wandb.Table(columns=["problem_id", "success_rate", "n_correct", "n_total", "category"])
    for r in results:
        table.add_data(r['id'], r['success_rate'], r['n_correct'], r['n_total'], r['category'])
    wandb.log({"problem_details": table})
    
    # Log results as artifact
    import os
    results_path = os.path.join(args.output_dir, 'results.json')
    artifact = wandb.Artifact('evaluation_results', type='evaluation')
    artifact.add_file(results_path)
    wandb.log_artifact(artifact)
    
    wandb.finish()
    print("Results logged to W&B")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate model on GSM8K/AIME/GSM-Plus/MATH')
    
    # Model and data
    parser.add_argument('--model', type=str, required=True,
                        help='Model path (HF name or local checkpoint)')
    parser.add_argument('--dataset', type=str, default='gsm8k',
                        choices=['gsm8k', 'aime', 'gsmplus', 'math'], help='Dataset to evaluate on')
    parser.add_argument('--problems_file', type=str, default=None,
                        help='Path to local JSON problems file (overrides --dataset loading)')
    parser.add_argument('--n_problems', type=int, default=50,
                        help='Number of problems to evaluate')
    parser.add_argument('--gsm8k_split', type=str, default='test',
                        choices=['train', 'test'], help='GSM8K split')
    parser.add_argument('--use_chat_template', action='store_true',
                        help='Use chat template (for instruct models). Omit for base models.')
    parser.add_argument('--no_chat_template', dest='use_chat_template', action='store_false',
                        help='Do not use chat template (for base models)')
    parser.set_defaults(use_chat_template=True)  # Default: use chat template
    
    # Generation parameters
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
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results')
    
    # W&B logging
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='W&B project name (optional)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name (optional)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("EVALUATION")
    print("="*60)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Model: {args.model}")
    print(f"Problems: {args.n_problems}")
    print(f"Trajectories per problem: {args.n_trajectories}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}")
    print(f"Output: {args.output_dir}")
    print("="*60)
    
    # Load problems based on dataset
    if args.problems_file:
        # Load from local JSON file
        print(f"Loading problems from local file: {args.problems_file}")
        with open(args.problems_file, 'r') as f:
            data = json.load(f)
        problems = data['problems'] if 'problems' in data else data
        if args.n_problems and args.n_problems < len(problems):
            problems = problems[:args.n_problems]
        print(f"Loaded {len(problems)} problems from {args.problems_file}")
    elif args.dataset == 'gsm8k':
        problems = load_gsm8k_problems(
            n_problems=args.n_problems,
            split=args.gsm8k_split,
            seed=args.seed
        )
    elif args.dataset == 'aime':
        problems = load_aime_problems(
            n_problems=args.n_problems,
            split='train',  # AIME only has train split
            seed=args.seed
        )
    elif args.dataset == 'gsmplus':
        problems = load_gsmplus_problems(
            n_problems=args.n_problems,
            split='test',
            seed=args.seed
        )
    elif args.dataset == 'math':
        problems = load_math_problems(
            n_problems=args.n_problems,
            levels=[3, 4, 5],  # Only Level 3, 4, and 5
            subjects=None,  # All subjects
            split='train',
            seed=args.seed
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Evaluate
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
        dataset=args.dataset
    )
    
    # Compute statistics
    stats = compute_statistics(results)
    
    # Print summary
    print_summary(stats)
    
    # Save results
    save_results(results, stats, args, args.output_dir)
    
    # Log to W&B
    log_to_wandb(results, stats, args)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()