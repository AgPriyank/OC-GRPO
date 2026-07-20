"""
Extract MATH problems for GRPO training and evaluation.

Train problems are drawn from the HF 'train' split.
Test problems are drawn from the HF 'test' split (independent).

Usage:
    # Extract 500 train (HF train) + 100 test (HF test) — default
    python extract_data.py

    # Extract the original 100-problem set (backward compatible)
    python extract_data.py --n-train 100 --n-test 0

    # Custom split sizes
    python extract_data.py --n-train 500 --n-test 100 --seed 42

    # Use HF train split for test too (old behavior)
    python extract_data.py --test-split train
"""
import json
import random
import argparse
import os
from datasets import load_dataset
from baseline import extract_math_ground_truth


ALL_SUBJECTS = [
    'algebra', 'counting_and_probability', 'geometry',
    'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'
]


def load_all_math_problems(levels=[3, 4, 5], split='train'):
    """Load all MATH problems at the given difficulty levels."""
    print(f"Loading MATH dataset (levels {levels}, split={split})...")

    all_problems = []
    for subject in ALL_SUBJECTS:
        dataset = load_dataset("EleutherAI/hendrycks_math", subject, split=split)

        for item in dataset:
            try:
                level_num = int(item['level'].split()[-1])
            except ValueError:
                continue

            if level_num in levels:
                ground_truth = extract_math_ground_truth(item['solution'])
                all_problems.append({
                    'problem': item['problem'],
                    'solution': item['solution'],
                    'level': item['level'],
                    'subject': subject,
                    'ground_truth': ground_truth
                })

    print(f"Loaded {len(all_problems)} total problems from {len(ALL_SUBJECTS)} subjects")
    return all_problems


def format_problems(raw_problems, id_prefix="math_train"):
    """Convert raw problems to the standard output format."""
    problems = []
    for idx, item in enumerate(raw_problems):
        problems.append({
            'id': f"{id_prefix}_{idx}",
            'question': item['problem'],
            'answer': item['solution'],
            'ground_truth': item['ground_truth'],
            'level': item['level'],
            'subject': item['subject']
        })
    return problems


def save_problems(problems, output_file, levels, seed, split_name):
    """Save problems to JSON file."""
    data = {
        'metadata': {
            'n_problems': len(problems),
            'levels': levels,
            'seed': seed,
            'split': split_name,
            'source': 'EleutherAI/hendrycks_math'
        },
        'problems': problems
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(problems)} problems to {output_file}")


def print_split_stats(problems, label):
    """Print subject/level distribution for a split."""
    from collections import Counter
    subjects = Counter(p['subject'] for p in problems)
    levels = Counter(p['level'] for p in problems)

    print(f"\n  {label} ({len(problems)} problems):")
    print(f"    Levels: {dict(sorted(levels.items()))}")
    print(f"    Subjects: {dict(sorted(subjects.items()))}")


def main():
    parser = argparse.ArgumentParser(description='Extract MATH problems for GRPO training')
    parser.add_argument('--n-train', type=int, default=500, help='Number of training problems')
    parser.add_argument('--n-test', type=int, default=200, help='Number of test problems')
    parser.add_argument('--levels', type=int, nargs='+', default=[3, 4, 5], help='Difficulty levels')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test-split', type=str, default='test',
                        help='HuggingFace split to use for test problems (default: test)')
    args = parser.parse_args()

    os.makedirs('data', exist_ok=True)

    # Load training problems from HF train split
    all_train_problems = load_all_math_problems(levels=args.levels, split='train')

    # Shuffle deterministically
    random.seed(args.seed)
    random.shuffle(all_train_problems)

    # Take first n_train for training (-1 means all)
    if args.n_train == -1:
        train_raw = all_train_problems
        args.n_train = len(train_raw)
    elif args.n_train > len(all_train_problems):
        raise ValueError(
            f"Requested {args.n_train} train problems "
            f"but only {len(all_train_problems)} available at levels {args.levels}"
        )
    else:
        train_raw = all_train_problems[:args.n_train]

    # Load test problems from HF test split (independent of training data)
    if args.n_test > 0:
        all_test_problems = load_all_math_problems(levels=args.levels, split=args.test_split)
        if args.n_test > len(all_test_problems):
            raise ValueError(
                f"Requested {args.n_test} test problems from '{args.test_split}' split "
                f"but only {len(all_test_problems)} available at levels {args.levels}"
            )
        random.seed(args.seed)
        random.shuffle(all_test_problems)
        test_raw = all_test_problems[:args.n_test]
    else:
        test_raw = []

    # Format with appropriate ID prefixes
    train_problems = format_problems(train_raw, id_prefix="math_train")
    test_problems = format_problems(test_raw, id_prefix="math_test")

    # Save training set
    train_file = f'data/math_train_{args.n_train}_problems.json'
    save_problems(train_problems, train_file, args.levels, args.seed, 'train')

    # Save test set (if requested)
    if args.n_test > 0:
        test_file = f'data/math_test_{args.n_test}_problems.json'
        save_problems(test_problems, test_file, args.levels, args.seed, 'test')

    # Print stats
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Seed: {args.seed}")
    print(f"  Levels: {args.levels}")
    print(f"  Train source: HF 'train' split ({len(all_train_problems)} available)")
    if args.n_test > 0:
        print(f"  Test source:  HF '{args.test_split}' split ({len(all_test_problems)} available)")
    print_split_stats(train_problems, "Train")
    if args.n_test > 0:
        print_split_stats(test_problems, "Test")
    print("=" * 60)

    # Backward compatibility: also save 100-problem file if train >= 100
    # The first 100 of the shuffled pool (seed=42) match the original file
    if args.seed == 42 and args.n_train >= 100:
        compat_problems = format_problems(all_train_problems[:100], id_prefix="math_train")
        compat_file = 'data/math_train_100_problems.json'
        save_problems(compat_problems, compat_file, args.levels, args.seed, 'train')
        print(f"\n(Backward compat) Also saved {compat_file}")


if __name__ == "__main__":
    main()