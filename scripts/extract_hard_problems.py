#!/usr/bin/env python3
"""
Extract hard problems (0 correct trajectories) from evaluation results.

Usage:
    python scripts/extract_hard_problems.py \
        --results results_polished/training_with_hints/base_instruct_hftest200/results.json \
        --dataset data/math_test_200_problems.json \
        --output data/hard_problems_identified.json
"""

import json
import argparse
from pathlib import Path


def extract_hard_problems(results_file, dataset_file, output_file, threshold=0):
    """
    Extract problems where model got <= threshold correct trajectories.

    Args:
        results_file: Path to results.json with evaluation data
        dataset_file: Path to original dataset with problem details
        output_file: Path to save extracted hard problems
        threshold: Max number of correct trajectories (default 0 for hard cliffs)
    """
    # Load results
    print(f"Loading results from {results_file}")
    with open(results_file, 'r') as f:
        results_data = json.load(f)

    # Load original dataset
    print(f"Loading dataset from {dataset_file}")
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    # Create problem lookup dict
    problem_dict = {p['id']: p for p in dataset['problems']}

    # Extract hard problems (0 correct)
    hard_problems = []
    for result in results_data.get('per_problem', results_data.get('results', [])):
        prob_id = result['id']
        n_correct = sum(1 for t in result['trajectories'] if t.get('correct', False))

        if n_correct <= threshold:
            # Get full problem data from original dataset
            if prob_id in problem_dict:
                problem_data = problem_dict[prob_id].copy()
                # Add evaluation info
                problem_data['n_correct'] = n_correct
                problem_data['n_total'] = len(result['trajectories'])
                # Add first 3 failed trajectories for analysis (if text available)
                failed_trajs = [t for t in result['trajectories'] if not t.get('correct', False)]
                problem_data['failed_trajectories'] = [
                    {
                        'text': t.get('text', ''),
                        'predicted_answer': t.get('predicted_answer', '')
                    }
                    for t in failed_trajs[:3]  # Keep first 3 for space
                ]
                hard_problems.append(problem_data)

    # Save to output
    output_data = {
        'metadata': {
            'source_results': str(results_file),
            'source_dataset': str(dataset_file),
            'n_hard_problems': len(hard_problems),
            'threshold': threshold,
            'description': f'Problems with <= {threshold} correct trajectories out of {results_data["metadata"].get("n_trajectories", "unknown")}'
        },
        'problems': hard_problems
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nExtracted {len(hard_problems)} hard problems")
    print(f"Saved to {output_file}")

    # Print summary stats
    if hard_problems:
        subjects = {}
        levels = {}
        for p in hard_problems:
            subj = p.get('subject', 'unknown')
            lev = p.get('level', 'unknown')
            subjects[subj] = subjects.get(subj, 0) + 1
            levels[lev] = levels.get(lev, 0) + 1

        print("\nBreakdown by subject:")
        for subj, count in sorted(subjects.items(), key=lambda x: x[1], reverse=True):
            print(f"  {subj}: {count}")

        print("\nBreakdown by level:")
        for lev, count in sorted(levels.items(), key=lambda x: x[1], reverse=True):
            print(f"  {lev}: {count}")

    return hard_problems


def main():
    parser = argparse.ArgumentParser(description='Extract hard problems from evaluation results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results.json file')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to original dataset JSON file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save extracted hard problems')
    parser.add_argument('--threshold', type=int, default=0,
                       help='Max number of correct trajectories (default 0)')

    args = parser.parse_args()

    extract_hard_problems(
        results_file=args.results,
        dataset_file=args.dataset,
        output_file=args.output,
        threshold=args.threshold
    )


if __name__ == '__main__':
    main()
