#!/usr/bin/env python3
"""
Merge sharded evaluation results into a single results.json.

Usage:
    python scripts/merge_eval_shards.py \
        --input_dir results/math_full_64traj/qwen2.5_7b \
        --n_shards 24
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def compute_statistics(results):
    """Compute overall, per-level, and per-subject statistics (no GPU deps)."""
    total = len(results)
    mean_accuracy = sum(r['success_rate'] for r in results) / total
    mean_avg_tokens = sum(r['avg_tokens'] for r in results) / total

    cats = defaultdict(int)
    for r in results:
        cats[r['category']] += 1

    all_trajs = [t for r in results for t in r['trajectories']]
    n_truncated = sum(1 for t in all_trajs if t.get('truncated', False))
    n_trunc_correct = sum(1 for t in all_trajs if t.get('truncated', False) and t.get('correct', False))
    n_trunc_incorrect = sum(1 for t in all_trajs if t.get('truncated', False) and not t.get('correct', False))

    overall = {
        'n_problems': total,
        'mean_accuracy': mean_accuracy,
        'mean_avg_tokens': mean_avg_tokens,
        'n_hard_cliff': cats.get('hard_cliff', 0),
        'n_soft_cliff': cats.get('soft_cliff', 0),
        'n_good': cats.get('good', 0),
        'n_too_easy': cats.get('too_easy', 0),
        'n_truncated': n_truncated,
        'n_truncated_correct': n_trunc_correct,
        'n_truncated_incorrect': n_trunc_incorrect,
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

    return {'overall': overall, 'per_level': per_level_stats, 'per_subject': per_subject_stats}


def print_summary(stats):
    """Print summary with per-level and per-subject breakdowns."""
    o = stats['overall']
    print(f"\n{'='*60}")
    print("MERGED EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total problems:    {o['n_problems']}")
    print(f"Mean accuracy:     {o['mean_accuracy']:.4f}")
    print(f"Mean avg tokens:   {o['mean_avg_tokens']:.1f}")
    print(f"\nProblem categorization:")
    print(f"  Hard cliffs (0% success):     {o['n_hard_cliff']:5d} ({100*o['n_hard_cliff']/o['n_problems']:.1f}%)")
    print(f"  Soft cliffs (>0%, ≤25%):      {o['n_soft_cliff']:5d} ({100*o['n_soft_cliff']/o['n_problems']:.1f}%)")
    print(f"  Good (>25%, <75%):            {o['n_good']:5d} ({100*o['n_good']/o['n_problems']:.1f}%)")
    print(f"  Too easy (≥75%):              {o['n_too_easy']:5d} ({100*o['n_too_easy']/o['n_problems']:.1f}%)")

    print(f"\n{'-'*60}")
    print("PER-LEVEL ACCURACY")
    print(f"{'-'*60}")
    for level, s in sorted(stats['per_level'].items()):
        print(f"  {level:10s}:  acc={s['mean_accuracy']:.4f}  avg_tok={s['mean_avg_tokens']:.1f}  (n={s['n_problems']})")

    print(f"\n{'-'*60}")
    print("PER-SUBJECT ACCURACY")
    print(f"{'-'*60}")
    for subject, s in sorted(stats['per_subject'].items()):
        print(f"  {subject:30s}:  acc={s['mean_accuracy']:.4f}  avg_tok={s['mean_avg_tokens']:.1f}  (n={s['n_problems']})")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Merge sharded eval results')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Base directory containing shard_0/, shard_1/, ...')
    parser.add_argument('--n_shards', type=int, required=True,
                        help='Number of shards to merge')
    args = parser.parse_args()

    base_dir = Path(args.input_dir)

    # Load and merge all shards
    all_results = []
    metadata = None
    for shard_id in range(args.n_shards):
        shard_path = base_dir / f"shard_{shard_id}" / "results.json"
        print(f"Loading {shard_path}")
        with open(shard_path, 'r') as f:
            shard_data = json.load(f)

        all_results.extend(shard_data['per_problem'])
        if metadata is None:
            metadata = shard_data['metadata'].copy()

    # Sort by problem ID for deterministic ordering
    all_results.sort(key=lambda r: r['id'])

    print(f"\nMerged {len(all_results)} problems from {args.n_shards} shards")

    # Recompute statistics on merged data
    stats = compute_statistics(all_results)

    # Update metadata
    metadata['n_problems_actual'] = len(all_results)
    metadata.pop('shard_id', None)
    metadata.pop('n_shards', None)

    merged = {
        'metadata': metadata,
        'statistics': stats,
        'per_problem': all_results,
    }

    output_path = base_dir / 'results.json'
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"Saved merged results to {output_path}")

    print_summary(stats)


if __name__ == '__main__':
    main()
