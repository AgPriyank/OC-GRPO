#!/usr/bin/env python3
"""
Download KbsdJames/Omni-MATH dataset from HuggingFace and convert to standard JSON format.

Usage:
    python scripts/download_omnimath.py

Output:
    data/omnimath_test_4428_problems.json
"""

import json
from collections import Counter
from datasets import load_dataset


def main():
    print("Downloading KbsdJames/Omni-MATH from HuggingFace...")
    ds = load_dataset("KbsdJames/Omni-MATH")["test"]
    print(f"Loaded {len(ds)} problems")

    # Convert to standard format
    problems = []
    difficulties = []
    domains_set = set()

    for i, ex in enumerate(ds):
        # domain is a list — take first element, strip "Mathematics -> " prefix
        domain_raw = ex["domain"][0] if ex["domain"] else "Unknown"
        # e.g. "Mathematics -> Algebra -> Other" -> "Algebra -> Other"
        subject = domain_raw.replace("Mathematics -> ", "", 1)

        difficulty = ex["difficulty"]
        difficulties.append(difficulty)
        domains_set.add(subject.split(" -> ")[0])  # top-level domain

        problems.append({
            "id": f"omnimath_{i}",
            "question": ex["problem"],
            "answer": ex["solution"],          # full step-by-step solution text
            "ground_truth": ex["answer"],       # final answer value
            "level": f"Difficulty {difficulty}",
            "subject": subject,
            "source": ex["source"],
            "difficulty": difficulty,
        })

    # Compute metadata
    diff_counter = Counter(difficulties)
    n_problems = len(problems)

    output = {
        "metadata": {
            "n_problems": n_problems,
            "source": "KbsdJames/Omni-MATH",
            "split": "test",
            "difficulty_range": [min(difficulties), max(difficulties)],
            "n_unique_difficulties": len(diff_counter),
            "top_level_domains": sorted(domains_set),
        },
        "problems": problems,
    }

    output_path = f"data/omnimath_test_{n_problems}_problems.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {n_problems} problems to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("OMNI-MATH DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total problems: {n_problems}")
    print(f"Difficulty range: {min(difficulties)} - {max(difficulties)}")
    print(f"Unique difficulty levels: {len(diff_counter)}")

    print(f"\nDifficulty distribution:")
    for d in sorted(diff_counter.keys()):
        bar = "#" * (diff_counter[d] // 10)
        print(f"  {d:5.1f}: {diff_counter[d]:4d} {bar}")

    print(f"\nTop-level domains: {sorted(domains_set)}")

    # Source distribution
    source_counter = Counter(p["source"] for p in problems)
    print(f"\nTop sources:")
    for src, count in source_counter.most_common(10):
        print(f"  {src:30s}: {count}")

    # Sample answers
    print(f"\nSample answers (first 10):")
    for p in problems[:10]:
        ans = p["ground_truth"][:80]
        print(f"  [{p['id']}] diff={p['difficulty']}  ans={ans}")


if __name__ == "__main__":
    main()
