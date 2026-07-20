#!/usr/bin/env python3
"""
Download math benchmark datasets from HuggingFace and convert to our standard format.

Output format (per file):
{
  "metadata": {"n_problems": N, "source": "hf_repo_id", "split": "...", ...},
  "problems": [{"id": "...", "question": "...", "ground_truth": "..."}, ...]
}

Usage:
    python scripts/download_benchmarks.py [--output_dir data]
"""

import json
import os
import argparse
from datasets import load_dataset


def download_aime24(output_dir: str):
    """AIME 2024 — 30 competition math problems (integer answers 0-999)."""
    ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
    problems = []
    for i, row in enumerate(ds):
        problems.append({
            "id": f"aime24_{i}",
            "question": row["problem"],
            "ground_truth": str(row["answer"]).strip(),
        })
    data = {
        "metadata": {
            "n_problems": len(problems),
            "source": "HuggingFaceH4/aime_2024",
            "split": "train",
        },
        "problems": problems,
    }
    path = os.path.join(output_dir, "aime24_test_problems.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  AIME24: {len(problems)} problems -> {path}")
    return len(problems)


def download_aime25(output_dir: str):
    """AIME 2025 — 30 competition math problems (two configs: I and II)."""
    problems = []
    for config in ["AIME2025-I", "AIME2025-II"]:
        try:
            ds = load_dataset("opencompass/AIME2025", config, split="test")
        except Exception:
            # Fallback: try without config
            ds = load_dataset("opencompass/AIME2025", split="test")
            for i, row in enumerate(ds):
                problems.append({
                    "id": f"aime25_{i}",
                    "question": row["question"],
                    "ground_truth": str(row["answer"]).strip(),
                })
            break
        for i, row in enumerate(ds):
            suffix = "I" if "I" in config and "II" not in config else "II"
            problems.append({
                "id": f"aime25_{suffix}_{i}",
                "question": row["question"],
                "ground_truth": str(row["answer"]).strip(),
            })
    data = {
        "metadata": {
            "n_problems": len(problems),
            "source": "opencompass/AIME2025",
            "split": "test",
        },
        "problems": problems,
    }
    path = os.path.join(output_dir, "aime25_test_problems.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  AIME25: {len(problems)} problems -> {path}")
    return len(problems)


def download_amc23(output_dir: str):
    """AMC 2023 — 40 multiple-choice competition math problems."""
    ds = load_dataset("math-ai/amc23", split="test")
    problems = []
    for i, row in enumerate(ds):
        problems.append({
            "id": f"amc23_{i}",
            "question": row["question"],
            "ground_truth": str(row["answer"]).strip(),
        })
    data = {
        "metadata": {
            "n_problems": len(problems),
            "source": "math-ai/amc23",
            "split": "test",
        },
        "problems": problems,
    }
    path = os.path.join(output_dir, "amc23_test_problems.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  AMC23: {len(problems)} problems -> {path}")
    return len(problems)


def download_minerva(output_dir: str):
    """Minerva Math — 272 undergraduate-level quantitative reasoning problems."""
    ds = load_dataset("math-ai/minervamath", split="test")
    problems = []
    for i, row in enumerate(ds):
        problems.append({
            "id": f"minerva_{i}",
            "question": row["question"],
            "ground_truth": str(row["answer"]).strip(),
        })
    data = {
        "metadata": {
            "n_problems": len(problems),
            "source": "math-ai/minervamath",
            "split": "test",
        },
        "problems": problems,
    }
    path = os.path.join(output_dir, "minerva_test_problems.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Minerva: {len(problems)} problems -> {path}")
    return len(problems)


def download_olympiadbench(output_dir: str):
    """OlympiadBench — Text-only, open-ended, English math competition problems."""
    # Use OE_TO_maths_en_COMP config: Open-Ended, Text-Only, maths, English, Competition
    ds = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")
    problems = []
    for i, row in enumerate(ds):
        # final_answer is a list; take first element
        answer = row["final_answer"]
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        problems.append({
            "id": f"olympiad_{i}",
            "question": row["question"],
            "ground_truth": str(answer).strip(),
        })
    data = {
        "metadata": {
            "n_problems": len(problems),
            "source": "Hothan/OlympiadBench",
            "config": "OE_TO_maths_en_COMP",
            "split": "train",
        },
        "problems": problems,
    }
    path = os.path.join(output_dir, "olympiadbench_test_problems.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  OlympiadBench: {len(problems)} problems -> {path}")
    return len(problems)


def download_gaokao2023en(output_dir: str):
    """Gaokao 2023 Math (English) — Chinese college entrance exam translated to English."""
    ds = load_dataset("MARIO-Math-Reasoning/Gaokao2023-Math-En", split="train")
    problems = []
    for i, row in enumerate(ds):
        problems.append({
            "id": f"gaokao_{i}",
            "question": row["question"],
            "ground_truth": str(row["answer"]).strip(),
        })
    data = {
        "metadata": {
            "n_problems": len(problems),
            "source": "MARIO-Math-Reasoning/Gaokao2023-Math-En",
            "split": "train",
        },
        "problems": problems,
    }
    path = os.path.join(output_dir, "gaokao2023en_test_problems.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Gaokao2023en: {len(problems)} problems -> {path}")
    return len(problems)


def download_aime_1983_2024(output_dir: str):
    """AIME 1983-2024 — 933 competition math problems with year metadata."""
    ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
    problems = []
    for row in ds:
        problems.append({
            "id": str(row["ID"]),
            "question": row["Question"],
            "ground_truth": str(row["Answer"]).strip(),
            "year": int(row["Year"]),
            "problem_number": int(row["Problem Number"]),
        })
    data = {
        "metadata": {
            "n_problems": len(problems),
            "source": "di-zhang-fdu/AIME_1983_2024",
            "split": "train",
        },
        "problems": problems,
    }
    path = os.path.join(output_dir, "aime_1983_2024_test_problems.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  AIME 1983-2024: {len(problems)} problems -> {path}")
    return len(problems)


def download_aime26(output_dir: str):
    """AIME 2026 — 30 competition math problems (Part I + Part II)."""
    ds = load_dataset("MathArena/aime_2026", split="train")
    problems = []
    for row in ds:
        idx = row["problem_idx"]
        part = "I" if idx <= 15 else "II"
        problems.append({
            "id": f"aime26_{part}_{idx}",
            "question": row["problem"],
            "ground_truth": str(row["answer"]).strip(),
        })
    data = {
        "metadata": {
            "n_problems": len(problems),
            "source": "MathArena/aime_2026",
            "split": "train",
        },
        "problems": problems,
    }
    path = os.path.join(output_dir, "aime26_test_problems.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  AIME26: {len(problems)} problems -> {path}")
    return len(problems)


def main():
    parser = argparse.ArgumentParser(description="Download math benchmark datasets")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for JSON files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Downloading math benchmark datasets from HuggingFace...\n")

    datasets = [
        ("AIME24", download_aime24),
        ("AIME25", download_aime25),
        ("AMC23", download_amc23),
        ("Minerva", download_minerva),
        ("OlympiadBench", download_olympiadbench),
        ("Gaokao2023en", download_gaokao2023en),
        ("AIME_1983_2024", download_aime_1983_2024),
        ("AIME26", download_aime26),
    ]

    total = 0
    for name, fn in datasets:
        try:
            n = fn(args.output_dir)
            total += n
        except Exception as e:
            print(f"  ERROR downloading {name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone! {total} total problems across all datasets.")


if __name__ == "__main__":
    main()
