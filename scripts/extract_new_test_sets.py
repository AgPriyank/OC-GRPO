"""
Extract and prepare new math test sets from HuggingFace for evaluation.

Downloads datasets, converts to our standard format:
  {"metadata": {...}, "problems": [{"id", "question", "ground_truth"}]}

Saves to data/ directory.

Usage:
  python scripts/extract_new_test_sets.py [--dataset NAME]

  --dataset: optionally specify a single dataset to extract (e.g., "putnam")
             default: extract all datasets
"""

import json, os, sys, argparse, random
import numpy as np

random.seed(42)
np.random.seed(42)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def save_dataset(name, problems, metadata=None):
    """Save in standard format."""
    n = len(problems)
    fname = f"{DATA_DIR}/{name}_test_problems.json"
    out = {
        "metadata": {
            "n_problems": n,
            "source": metadata.get("source", "") if metadata else "",
            **(metadata or {}),
        },
        "problems": problems,
    }
    with open(fname, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  Saved {n} problems to {fname}")


def clean_text(text):
    """Clean up whitespace in problem text."""
    if text is None:
        return ""
    return str(text).strip()


# ── 1. Putnam-AXIOM ──────────────────────────────────────────────────
def extract_putnam():
    """Putnam-AXIOM: Gated dataset, skipping."""
    print("\n=== Putnam-AXIOM: SKIPPED (gated dataset, requires access request) ===")
    print("  Visit https://huggingface.co/datasets/Putnam-AXIOM/putnam-axiom-dataset-v1 to request access")


# ── 2. MathOlympiadBench ─────────────────────────────────────────────
def extract_math_olympiad_bench():
    """MathOlympiadBench: IMO problems — Lean4 format, extract informal statements."""
    print("\n=== Extracting MathOlympiadBench ===")
    from datasets import load_dataset
    import re

    try:
        ds = load_dataset("Goedel-LM/MathOlympiadBench")
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    split = list(ds.keys())[0]
    print(f"  Split '{split}': {len(ds[split])} rows, columns: {ds[split].column_names}")

    # This dataset is in Lean4 format. The informal problem is in 'informal_prefix'
    # The problems are proof-based (no numerical answer), so not ideal for our pipeline
    # Extract the informal statements anyway for reference
    problems = []
    for i, row in enumerate(ds[split]):
        informal = row.get('informal_prefix', '')
        # Extract problem text from the /-! ... -/ comment block
        match = re.search(r'/\-!\s*\n(.*?)\n\-/', informal, re.DOTALL)
        if match:
            text = match.group(1).strip()
            # Remove the title line (e.g., "# USA Mathematical Olympiad 2005, Problem 2")
            lines = text.split('\n')
            title = lines[0].replace('#', '').strip() if lines[0].startswith('#') else ''
            body = '\n'.join(l for l in lines if not l.startswith('#')).strip()
        else:
            body = informal
            title = row.get('name', '')

        if not body:
            continue

        category = row.get('category', '')
        problems.append({
            "id": f"matholympiad_test_{i}",
            "question": clean_text(body),
            "ground_truth": "proof",  # These are proof problems, no numerical answer
            "name": row.get('name', ''),
            "category": category,
            "tags": row.get('tags', []),
        })

    print(f"  WARNING: These are proof-based problems (no numerical answers).")
    print(f"  Not suitable for standard answer-extraction evaluation.")
    print(f"  Extracted {len(problems)} problems for reference only.")

    if problems:
        save_dataset("matholympiad", problems, {
            "source": "Goedel-LM/MathOlympiadBench",
            "description": "IMO/USAMO problems (PROOF-BASED, no numerical answers)",
            "warning": "Not suitable for standard eval pipeline - proof problems",
        })


# ── 3. OlymMATH ─────────────────────────────────────────────────────
def extract_olym_math():
    """OlymMATH: Curated olympiad problems with easy/hard configs."""
    print("\n=== Extracting OlymMATH ===")
    from datasets import load_dataset

    problems = []
    for config in ['en-easy', 'en-hard']:
        try:
            ds = load_dataset("RUC-AIBOX/OlymMATH", config)
        except Exception as e:
            print(f"  ERROR loading config '{config}': {e}")
            continue

        split = list(ds.keys())[0]
        print(f"  Config '{config}', split '{split}': {len(ds[split])} rows, columns: {ds[split].column_names}")
        if len(ds[split]) > 0:
            print(f"  Sample: {dict(list(ds[split][0].items())[:5])}")

        for i, row in enumerate(ds[split]):
            question = row.get('problem', row.get('question', ''))
            answer = row.get('answer', row.get('solution', ''))
            if not question:
                continue
            difficulty = "easy" if "easy" in config else "hard"
            problems.append({
                "id": f"olymmath_test_{len(problems)}",
                "question": clean_text(question),
                "ground_truth": clean_text(answer),
                "difficulty": difficulty,
            })

    if problems:
        save_dataset("olymmath", problems, {
            "source": "RUC-AIBOX/OlymMATH (en-easy + en-hard)",
            "description": "Curated olympiad math problems with easy/hard split",
        })


# ── 4. LiveMathBench ─────────────────────────────────────────────────
def extract_livemathbench():
    """LiveMathBench: Gated dataset, skipping."""
    print("\n=== LiveMathBench: SKIPPED (gated dataset, requires access request) ===")
    print("  Visit https://huggingface.co/datasets/opencompass/LiveMathBench to request access")


# ── 5. JEE Main 2025 Math ───────────────────────────────────────────
def extract_jee():
    """JEE Main 2025 Math: Indian engineering entrance exam (both sessions)."""
    print("\n=== Extracting JEE Main 2025 Math ===")
    from datasets import load_dataset

    all_problems = []
    for config in ['jan', 'apr']:
        try:
            ds = load_dataset("PhysicsWallahAI/JEE-Main-2025-Math", config)
        except Exception as e:
            print(f"  ERROR loading config '{config}': {e}")
            continue

        split = list(ds.keys())[0]
        print(f"  Config '{config}', split '{split}': {len(ds[split])} rows, columns: {ds[split].column_names}")
        if len(ds[split]) > 0:
            sample = ds[split][0]
            print(f"  Sample keys: {list(sample.keys()) if isinstance(sample, dict) else 'N/A'}")
            # Print first few key-value pairs
            if isinstance(sample, dict):
                for k, v in list(sample.items())[:4]:
                    print(f"    {k}: {str(v)[:100]}")

        for i, row in enumerate(ds[split]):
            question = row.get('problem', row.get('question', row.get('Question', '')))
            answer = row.get('answer', row.get('solution', row.get('Answer', row.get('correct_answer', ''))))
            if not question:
                continue
            all_problems.append({
                "id": f"jee_math_test_{len(all_problems)}",
                "question": clean_text(question),
                "ground_truth": clean_text(answer),
                "session": config,
            })

    # Sample 200 if more
    if len(all_problems) > 200:
        all_problems = random.sample(all_problems, 200)
        for i, p in enumerate(all_problems):
            p['id'] = f"jee_math_test_{i}"

    if all_problems:
        save_dataset("jee_main_2025_math", all_problems, {
            "source": "PhysicsWallahAI/JEE-Main-2025-Math (jan + apr)",
            "description": "JEE Main 2025 Mathematics (Indian engineering entrance)",
            "sampled": len(all_problems) <= 200,
        })


# ── 6. OmniMath Stratified Sample ───────────────────────────────────
def extract_omnimath_stratified():
    """Sample 500 problems from full OmniMath, stratified by difficulty."""
    print("\n=== Extracting OmniMath Stratified Sample (500) ===")

    full_path = f"{DATA_DIR}/omnimath_test_4428_problems.json"
    if not os.path.exists(full_path):
        print(f"  Full OmniMath not found at {full_path}, trying HuggingFace...")
        from datasets import load_dataset
        try:
            ds = load_dataset("KbsdJames/Omni-MATH", split="test")
            print(f"  Loaded {len(ds)} problems from HuggingFace")
            all_problems = []
            for i, row in enumerate(ds):
                all_problems.append({
                    "id": f"omnimath_test_{i}",
                    "question": clean_text(row.get('problem', row.get('question', ''))),
                    "ground_truth": clean_text(row.get('answer', row.get('solution', ''))),
                    "difficulty": row.get('difficulty', None),
                    "subject": row.get('subject', ''),
                    "source": row.get('source', ''),
                })
        except Exception as e:
            print(f"  ERROR: {e}")
            return
    else:
        with open(full_path) as f:
            data = json.load(f)
        all_problems = data['problems'] if 'problems' in data else data

    print(f"  Total available: {len(all_problems)}")

    # Stratify by difficulty bins
    bins = [(0, 3), (3, 5), (5, 7), (7, 10)]
    per_bin = 500 // len(bins)  # 125 per bin
    sampled = []

    for lo, hi in bins:
        in_bin = [p for p in all_problems
                  if p.get('difficulty') is not None and lo <= float(p['difficulty']) < hi]
        n_take = min(per_bin, len(in_bin))
        chosen = random.sample(in_bin, n_take) if len(in_bin) > n_take else in_bin
        sampled.extend(chosen)
        print(f"  Difficulty [{lo}-{hi}): {len(in_bin)} available, took {len(chosen)}")

    # Fill remaining from any bin
    remaining = 500 - len(sampled)
    if remaining > 0:
        used_ids = {p['id'] for p in sampled}
        rest = [p for p in all_problems if p['id'] not in used_ids]
        extra = random.sample(rest, min(remaining, len(rest)))
        sampled.extend(extra)

    # Re-index
    for i, p in enumerate(sampled):
        p['id'] = f"omnimath_strat_test_{i}"

    save_dataset("omnimath_stratified_500", sampled, {
        "source": "KbsdJames/Omni-MATH (stratified sample)",
        "description": "500 problems stratified by difficulty from full OmniMath",
        "strategy": "125 per difficulty bin [0-3), [3-5), [5-7), [7-10)",
    })


# ── Main ─────────────────────────────────────────────────────────────
EXTRACTORS = {
    "putnam": extract_putnam,
    "matholympiad": extract_math_olympiad_bench,
    "olymmath": extract_olym_math,
    "livemathbench": extract_livemathbench,
    "jee": extract_jee,
    "omnimath_strat": extract_omnimath_stratified,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None,
                        help='Extract specific dataset (putnam, matholympiad, olymmath, livemathbench, jee, omnimath_strat)')
    args = parser.parse_args()

    if args.dataset:
        if args.dataset in EXTRACTORS:
            EXTRACTORS[args.dataset]()
        else:
            print(f"Unknown dataset: {args.dataset}. Choose from: {list(EXTRACTORS.keys())}")
    else:
        for name, func in EXTRACTORS.items():
            try:
                func()
            except Exception as e:
                print(f"  FAILED {name}: {e}")
                import traceback
                traceback.print_exc()

    print("\nDone!")
