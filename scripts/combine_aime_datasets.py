#!/usr/bin/env python3
"""Combine AIME 1983-2024, 2025, and 2026 datasets into a single JSON file."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

INPUT_FILES = [
    DATA_DIR / "aime_1983_2024_test_problems.json",
    DATA_DIR / "aime25_test_problems.json",
    DATA_DIR / "aime26_test_problems.json",
]

OUTPUT_FILE = DATA_DIR / "aime_1983_2026_combined_test_problems.json"


def main():
    all_problems = []
    for fpath in INPUT_FILES:
        with open(fpath) as f:
            data = json.load(f)
        print(f"Loaded {len(data['problems'])} problems from {fpath.name}")
        all_problems.extend(data["problems"])

    output = {
        "metadata": {
            "n_problems": len(all_problems),
            "source": "AIME 1983-2026 combined",
            "split": "test",
        },
        "problems": all_problems,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(all_problems)} problems to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
