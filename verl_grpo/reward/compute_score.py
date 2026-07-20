"""
Reward function for veRL GRPO training on MATH dataset.

Registered via YAML:
  custom_reward_function:
    path: verl_grpo/reward/compute_score.py
    name: compute_score
"""

import os
import sys

# Add project root to path so we can import baseline.py
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Compute binary reward for a MATH problem.

    Args:
        data_source: str, e.g. "math"
        solution_str: str, model's generated solution text
        ground_truth: str, the correct answer (from reward_model.ground_truth)
        extra_info: dict or None, additional info from the dataset

    Returns:
        float: 1.0 if correct, 0.0 otherwise
    """
    from baseline import extract_answer_math, answers_match_math

    # PrefixRL: when prefix is in the vLLM prompt (not the response),
    # the response only contains the continuation. Prepend the prefix
    # so the full solution is evaluated for \boxed{answer} extraction.
    if extra_info and "prefixrl_prefix_for_reward" in extra_info:
        prefix_text = extra_info["prefixrl_prefix_for_reward"]
        if prefix_text:
            solution_str = prefix_text + solution_str

    predicted = extract_answer_math(solution_str)
    if predicted is not None and answers_match_math(predicted, ground_truth):
        return 1.0
    return 0.0
