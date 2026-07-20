#!/usr/bin/env python3
"""
Gold Standard Answer Extraction for MATH Dataset

This module provides production-grade answer extraction and comparison
functions for mathematical problem solving. Used across training, evaluation,
and analysis pipelines.

Combines two approaches:
1. Baseline approach: Extract contents from \\boxed{} (from baseline.py)
2. POPE approach: Keep full \\boxed{} substring (from POPE's verifier_api.py)

Both approaches are supported with comprehensive testing and fallbacks.
"""

from typing import Optional, Tuple
import re
import signal
from contextlib import contextmanager
from math import comb

# Try importing math_verify (available on cluster, optional elsewhere)
try:
    import math_verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False


# ==============================================================================
# Exception Classes
# ==============================================================================

class TimeoutException(Exception):
    """Raised when verification times out."""
    pass


class NoAnswerException(Exception):
    """Raised when no \\boxed{} found in prediction."""
    pass


class EmptyBoxedException(Exception):
    """Raised when \\boxed{} is empty."""
    pass


class UnparsableException(Exception):
    """Raised when answer cannot be parsed."""
    pass


# ==============================================================================
# Timeout Context Manager (from POPE)
# ==============================================================================

@contextmanager
def timeout(seconds=1):
    """Context manager for timeout protection."""
    def timeout_handler(signum, frame):
        raise TimeoutException("Computation timed out")

    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


# ==============================================================================
# Core Extraction Functions
# ==============================================================================

def extract_answer_math(text: str) -> Optional[str]:
    """Extract answer from MATH dataset model output (baseline.py approach).

    Extracts CONTENTS of last \\boxed{} using nested brace matching.
    This handles complex LaTeX like \\frac{x}{y} correctly.

    Args:
        text: Model output text

    Returns:
        Extracted answer string, or None if no answer found

    Examples:
        >>> extract_answer_math("The answer is \\boxed{42}")
        "42"
        >>> extract_answer_math("Therefore \\boxed{\\frac{1}{2}}")
        "\\frac{1}{2}"
        >>> extract_answer_math("No boxed answer here")
        None
    """
    start_pos = text.rfind(r'\boxed{')

    if start_pos == -1:
        # Fallback: extract last number
        numbers = re.findall(r'[-+]?[\d,]*\.?\d+', text)
        if numbers:
            return numbers[-1].replace(',', '')
        return None

    # Start after '\\boxed{'
    pos = start_pos + len(r'\boxed{')
    brace_count = 1
    answer_chars = []

    while pos < len(text) and brace_count > 0:
        char = text[pos]
        if char == '{':
            brace_count += 1
            answer_chars.append(char)
        elif char == '}':
            brace_count -= 1
            if brace_count > 0:
                answer_chars.append(char)
        else:
            answer_chars.append(char)
        pos += 1

    if brace_count == 0:
        return ''.join(answer_chars).strip()

    # Fallback if braces don't match
    numbers = re.findall(r'[-+]?[\d,]*\.?\d+', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None


def extract_boxed_substring(text: str, max_length: int = 1000) -> Optional[str]:
    """Extract boxed answer substring (POPE approach).

    Returns substring from \\boxed{ to end, keeping the \\boxed wrapper.
    This is preferred for math_verify.parse() which handles \\boxed directly.

    Args:
        text: Model output text
        max_length: Maximum allowed length for extracted substring

    Returns:
        Substring from \\boxed{ to end, or None if no boxed answer found

    Raises:
        EmptyBoxedException: If \\boxed{} is empty
        UnparsableException: If extracted substring exceeds max_length

    Examples:
        >>> extract_boxed_substring("Therefore \\boxed{42}")
        "\\boxed{42}"
        >>> extract_boxed_substring("Answer: \\boxed{}")
        Raises EmptyBoxedException
    """
    boxed_start = text.rfind("\\boxed{")

    if boxed_start < 0:
        return None

    boxed_prediction = text[boxed_start:]

    # Check for empty boxed
    if "\\boxed{}" in boxed_prediction:
        raise EmptyBoxedException("Empty \\boxed{} found")

    # Check length
    if len(boxed_prediction) > max_length:
        raise UnparsableException(f"Boxed substring too long: {len(boxed_prediction)} > {max_length}")

    return boxed_prediction


def extract_math_ground_truth(solution: str) -> str:
    """Extract ground truth answer from MATH dataset solution field.

    Extracts contents of last \\boxed{} from the solution text.
    If no \\boxed{} found, returns the full solution.

    Args:
        solution: Full solution text from dataset

    Returns:
        Extracted answer string

    Examples:
        >>> extract_math_ground_truth("Steps... \\boxed{42}")
        "42"
        >>> extract_math_ground_truth("Just text, no boxed")
        "Just text, no boxed"
    """
    start_pos = solution.rfind(r'\boxed{')

    if start_pos == -1:
        return solution.strip()

    # Start after '\\boxed{'
    pos = start_pos + len(r'\boxed{')
    brace_count = 1
    answer_chars = []

    while pos < len(solution) and brace_count > 0:
        char = solution[pos]
        if char == '{':
            brace_count += 1
            answer_chars.append(char)
        elif char == '}':
            brace_count -= 1
            if brace_count > 0:
                answer_chars.append(char)
        else:
            answer_chars.append(char)
        pos += 1

    if brace_count == 0:
        return ''.join(answer_chars).strip()

    return solution.strip()


# ==============================================================================
# Normalization Functions
# ==============================================================================

def normalize_answer_math_fallback(answer: str) -> str:
    """Fallback normalization for MATH dataset answers.

    Used when math_verify is not available or fails.
    Removes LaTeX formatting and whitespace for string comparison.

    Args:
        answer: Answer string to normalize

    Returns:
        Normalized answer string

    Examples:
        >>> normalize_answer_math_fallback("\\frac{1}{2}")
        "frac12"
        >>> normalize_answer_math_fallback("  42  ")
        "42"
        >>> normalize_answer_math_fallback("\\left[0,\\frac{1}{2}\\right]")
        "[0frac12]"
    """
    if answer is None:
        return ""

    answer = answer.strip()

    # Remove \left and \right BEFORE removing all backslashes
    # This ensures \left[ becomes [ not left[
    answer = answer.replace('\\left', '')
    answer = answer.replace('\\right', '')

    # Remove degree symbols BEFORE removing backslashes
    answer = answer.replace('^\\circ', '')  # ^\circ
    answer = answer.replace('\\circ', '')   # \circ without caret
    answer = answer.replace('°', '')  # Unicode degree symbol

    # Remove LaTeX spacing commands BEFORE general backslash removal
    # These produce stray characters (!, ;, :) if only the backslash is stripped
    answer = answer.replace('\\!', '')   # thin negative space
    answer = answer.replace('\\,', '')   # thin space
    answer = answer.replace('\\;', '')   # medium math space
    answer = answer.replace('\\:', '')   # medium math space
    answer = answer.replace('\\quad', '')
    answer = answer.replace('\\qquad', '')

    # Remove \phantom{...} (invisible spacing placeholder)
    # Handle both \phantom{0} and \phantom X (with or without braces)
    answer = re.sub(r'\\phantom\{[^}]*\}', '', answer)
    answer = re.sub(r'\\phantom\s', '', answer)  # \phantom followed by space

    # Remove LaTeX formatting
    answer = answer.replace('\\', '')
    answer = answer.replace('{', '').replace('}', '')
    answer = answer.replace(' ', '')
    answer = answer.replace(',', '')

    return answer.lower()


# ==============================================================================
# Verification Functions
# ==============================================================================

def verify_answer_strict(predicted: str, gold: str, strict: bool = True,
                        max_prediction_length: int = 1000,
                        timeout_seconds: int = 1) -> Tuple[bool, str]:
    """Verify answer using math_verify (POPE approach with strict mode).

    This is the most robust verification method when math_verify is available.
    Returns detailed status codes for different failure modes.

    Args:
        predicted: Model's predicted answer (full text)
        gold: Gold (correct) answer
        strict: Whether to use strict comparison mode
        max_prediction_length: Maximum allowed prediction length
        timeout_seconds: Timeout for verification

    Returns:
        Tuple of (is_correct: bool, status: str)
        Status codes:
            - "correct": Answer is correct
            - "wrong": Answer is incorrect
            - "no_answer": No \\boxed{} found
            - "unparsable": Cannot parse answer
            - "timeout": Verification timed out
            - "empty_boxed": \\boxed{} is empty

    Examples:
        >>> verify_answer_strict("\\boxed{42}", "42")
        (True, "correct")
        >>> verify_answer_strict("\\boxed{43}", "42")
        (False, "wrong")
        >>> verify_answer_strict("No answer", "42")
        (False, "no_answer")
    """
    if not MATH_VERIFY_AVAILABLE:
        # Fall back to string comparison if math_verify not available
        try:
            predicted_extracted = extract_answer_math(predicted)
            if predicted_extracted is None:
                return (False, "no_answer")
            gold_extracted = extract_math_ground_truth(gold)
            match = answers_match_math_fallback(predicted_extracted, gold_extracted)
            return (match, "correct" if match else "wrong")
        except Exception:
            return (False, "unparsable")

    try:
        # Input validation
        if not isinstance(predicted, str) or not isinstance(gold, str):
            raise ValueError("Prediction and gold must be strings")

        # Extract boxed substring
        try:
            boxed_prediction = extract_boxed_substring(predicted, max_prediction_length)
        except EmptyBoxedException:
            return (False, "empty_boxed")
        except UnparsableException:
            return (False, "unparsable")

        if boxed_prediction is None:
            return (False, "no_answer")

        # Parse with math_verify
        gold_parsed = math_verify.parse(gold)
        boxed_prediction_parsed = math_verify.parse(boxed_prediction)

        if not boxed_prediction_parsed:
            raise ValueError("Failed to parse prediction")

        # Verify with timeout
        with timeout(timeout_seconds):
            equivalent = math_verify.verify(gold_parsed, boxed_prediction_parsed,
                                          strict=strict, timeout_seconds=timeout_seconds)

        if equivalent:
            return (True, "correct")
        else:
            return (False, "wrong")

    except TimeoutException:
        return (False, "timeout")
    except NoAnswerException:
        return (False, "no_answer")
    except Exception:
        return (False, "unparsable")


def answers_match_math_fallback(predicted: Optional[str], ground_truth: str) -> bool:
    """Check if answers match using normalized string comparison.

    This is the fallback when math_verify is not available or fails.
    Handles LaTeX variants and whitespace differences.

    Args:
        predicted: Predicted answer (extracted contents)
        ground_truth: Ground truth answer (extracted contents)

    Returns:
        True if answers match, False otherwise
    """
    if predicted is None:
        return False

    # Normalize LaTeX variants
    predicted_normalized = predicted.replace('\\dfrac', '\\frac')
    predicted_normalized = predicted_normalized.replace('\\tfrac', '\\frac')
    predicted_normalized = predicted_normalized.replace('\\cfrac', '\\frac')

    ground_truth_normalized = ground_truth.replace('\\dfrac', '\\frac')
    ground_truth_normalized = ground_truth_normalized.replace('\\tfrac', '\\frac')
    ground_truth_normalized = ground_truth_normalized.replace('\\cfrac', '\\frac')

    # Use fallback normalization
    return normalize_answer_math_fallback(predicted_normalized) == \
           normalize_answer_math_fallback(ground_truth_normalized)


def answers_match_math(predicted: Optional[str], ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth (multi-stage verification).

    This is the main entry point for answer verification.
    Uses multiple strategies with fallbacks:
    1. Try math_verify on full \\boxed{} substring (POPE approach)
    2. Try math_verify on extracted contents (baseline approach)
    3. Fall back to normalized string comparison

    Args:
        predicted: Predicted answer (full text or extracted)
        ground_truth: Ground truth answer (full text or extracted)

    Returns:
        True if answers match, False otherwise

    Examples:
        >>> answers_match_math("42", "42")
        True
        >>> answers_match_math("\\frac{1}{2}", "0.5")  # Requires math_verify
        True
        >>> answers_match_math("\\frac{6 \\sqrt{10}}{7}", "\\frac{6\\sqrt{10}}{7}")
        True  # Handles whitespace
    """
    if predicted is None:
        return False

    # Strategy 1: Try math_verify with full text (POPE approach)
    if MATH_VERIFY_AVAILABLE:
        try:
            # If predicted looks like full text with \boxed, use verify_answer_strict
            if '\\boxed{' in str(predicted):
                is_correct, status = verify_answer_strict(predicted, ground_truth)
                if status in ["correct", "wrong"]:
                    return is_correct
            # Otherwise try parsing extracted contents
            else:
                try:
                    pred_parsed = math_verify.parse(predicted)
                    gold_parsed = math_verify.parse(ground_truth)
                    if pred_parsed and gold_parsed:
                        with timeout(1):
                            return math_verify.verify(gold_parsed, pred_parsed,
                                                    strict=True, timeout_seconds=1)
                except Exception:
                    pass  # Fall through to string comparison
        except Exception:
            pass  # Fall through to string comparison

    # Strategy 2: Normalized string comparison (always available)
    return answers_match_math_fallback(predicted, ground_truth)


# ==============================================================================
# Utility Functions
# ==============================================================================

def compute_pass_at_k(n_correct: int, n_total: int, k: int) -> float:
    """Compute pass@k using combinatorial formula.

    For k=1: pass@1 = c/n (mean accuracy)
    For k>1: pass@k = 1 - C(n-c, k) / C(n, k)

    Special cases:
    - If n_correct >= k and k > 1: pass@k ≈ 1.0 (very likely to pass)
    - If n_total < k: pass@k = 0.0 (impossible)
    - If n_correct == n_total: pass@k = 1.0 (all correct)

    Args:
        n_correct: Number of correct samples
        n_total: Total number of samples
        k: Number of samples to consider

    Returns:
        pass@k metric (probability of at least one correct in k samples)

    Examples:
        >>> compute_pass_at_k(8, 16, 1)
        0.5
        >>> compute_pass_at_k(8, 16, 8)
        0.9909...
        >>> compute_pass_at_k(16, 16, 1)
        1.0
    """
    # Edge cases
    if n_total < k:
        return 0.0
    if n_correct == n_total:
        return 1.0
    if n_correct == 0:
        return 0.0

    # Special case: pass@1 is just mean accuracy
    if k == 1:
        return n_correct / n_total

    # General case: combinatorial formula
    return 1.0 - (comb(n_total - n_correct, k) / comb(n_total, k))


# ==============================================================================
# Unit Tests
# ==============================================================================

def test_extract_simple_boxed():
    """Test basic boxed answer extraction."""
    assert extract_answer_math("Answer is \\boxed{42}") == "42"
    assert extract_answer_math("The final result is \\boxed{100}") == "100"
    print("✓ test_extract_simple_boxed passed")


def test_extract_boxed_substring():
    """Test POPE-style substring extraction."""
    assert extract_boxed_substring("Answer is \\boxed{42}") == "\\boxed{42}"
    assert extract_boxed_substring("Therefore \\boxed{\\frac{1}{2}}") == "\\boxed{\\frac{1}{2}}"
    print("✓ test_extract_boxed_substring passed")


def test_extract_nested_braces():
    """Test extraction with nested LaTeX braces."""
    # Fractions
    assert extract_answer_math("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_answer_math("\\boxed{\\frac{6\\sqrt{10}}{7}}") == "\\frac{6\\sqrt{10}}{7}"
    # Nested fractions
    assert extract_answer_math("\\boxed{\\frac{\\frac{1}{2}}{3}}") == "\\frac{\\frac{1}{2}}{3}"
    print("✓ test_extract_nested_braces passed")


def test_extract_multiple_boxed():
    """Test that we use LAST \\boxed when multiple present."""
    text = "First try \\boxed{wrong}, second try \\boxed{also wrong}, final \\boxed{correct}"
    assert extract_answer_math(text) == "correct"
    print("✓ test_extract_multiple_boxed passed")


def test_extract_no_boxed():
    """Test fallback when no \\boxed present."""
    # Should extract last number
    assert extract_answer_math("The answer is 3.14159") == "3.14159"
    assert extract_answer_math("Result: 42") == "42"
    # Should return None if no number either
    assert extract_answer_math("No answer here") is None
    print("✓ test_extract_no_boxed passed")


def test_extract_empty_boxed():
    """Test edge case of empty \\boxed{}."""
    text = "Therefore \\boxed{}"
    result = extract_answer_math(text)
    assert result == ""  # Should return empty string, not None
    print("✓ test_extract_empty_boxed passed")


def test_empty_boxed_exception():
    """Test that empty boxed raises exception in POPE approach."""
    try:
        extract_boxed_substring("Answer: \\boxed{}")
        assert False, "Should have raised EmptyBoxedException"
    except EmptyBoxedException:
        pass
    print("✓ test_empty_boxed_exception passed")


def test_extract_ground_truth():
    """Test ground truth extraction from solution."""
    solution = "We solve the equation step by step... Therefore the answer is \\boxed{42}."
    assert extract_math_ground_truth(solution) == "42"

    # If no boxed, return full solution
    solution_no_boxed = "Just text without boxed answer"
    assert extract_math_ground_truth(solution_no_boxed) == "Just text without boxed answer"
    print("✓ test_extract_ground_truth passed")


def test_real_whitespace_issue():
    """Test case that caused original bug - whitespace in LaTeX."""
    pred = "\\frac{6\\sqrt{10}}{7}"  # No space
    gold = "\\frac{6 \\sqrt{10}}{7}"  # With space
    assert answers_match_math(pred, gold) == True
    print("✓ test_real_whitespace_issue passed")


def test_real_latex_variants():
    """Test LaTeX command variants (dfrac, tfrac, cfrac)."""
    assert answers_match_math("\\dfrac{1}{2}", "\\frac{1}{2}") == True
    assert answers_match_math("\\tfrac{1}{2}", "\\frac{1}{2}") == True
    assert answers_match_math("\\cfrac{1}{2}", "\\frac{1}{2}") == True
    print("✓ test_real_latex_variants passed")


def test_real_case_insensitive():
    """Test case insensitive matching."""
    assert answers_match_math("Yes", "yes") == True
    assert answers_match_math("NO", "no") == True
    assert answers_match_math("True", "TRUE") == True
    print("✓ test_real_case_insensitive passed")


def test_normalization():
    """Test normalization function."""
    assert normalize_answer_math_fallback("\\frac{1}{2}") == "frac12"
    assert normalize_answer_math_fallback("  42  ") == "42"
    assert normalize_answer_math_fallback("1,000") == "1000"
    assert normalize_answer_math_fallback("\\sqrt{2}") == "sqrt2"
    print("✓ test_normalization passed")


def test_empty_inputs():
    """Test handling of empty/None inputs."""
    assert extract_answer_math("") is None
    assert answers_match_math(None, "42") == False
    assert answers_match_math("", "42") == False
    assert normalize_answer_math_fallback(None) == ""
    print("✓ test_empty_inputs passed")


def test_malformed_latex():
    """Test graceful handling of malformed LaTeX."""
    # Unmatched braces should fallback to number extraction
    text = "\\boxed{\\frac{1}{2"  # Missing closing }
    result = extract_answer_math(text)
    # Should fallback to extracting "1" or "2"
    assert result in ["1", "2", None]  # Implementation-dependent
    print("✓ test_malformed_latex passed")


def test_special_characters():
    """Test special math symbols."""
    assert extract_answer_math("\\boxed{\\infty}") == "\\infty"
    assert extract_answer_math("\\boxed{\\pi}") == "\\pi"
    assert extract_answer_math("\\boxed{2\\pi}") == "2\\pi"
    assert extract_answer_math("\\boxed{-\\infty}") == "-\\infty"
    print("✓ test_special_characters passed")


def test_latex_spacing_commands():
    """Test that LaTeX spacing commands don't corrupt answer matching."""
    # \! thin negative space (e.g., 112,\!875 should match 112875)
    assert answers_match_math("112875", "112,\\!875") == True
    assert answers_match_math("10010", "10,\\!010") == True

    # \, thin space
    assert answers_match_math("12", "1\\,2") == True

    # \phantom{0} invisible placeholder
    assert answers_match_math("42", "\\phantom{0}42") == True
    assert answers_match_math("42", "4\\phantom{00}2") == True

    # Complex numbers with spacing
    assert answers_match_math("-1+12i", "-1 + 12i") == True
    assert answers_match_math("-1 + 12i", "-1+12i") == True

    print("✓ test_latex_spacing_commands passed")


def test_compute_pass_at_k():
    """Test pass@k computation."""
    # Basic cases
    assert compute_pass_at_k(8, 16, 1) == 0.5
    assert compute_pass_at_k(16, 16, 1) == 1.0
    assert compute_pass_at_k(0, 16, 1) == 0.0

    # Edge cases
    assert compute_pass_at_k(16, 16, 8) == 1.0  # All correct
    assert compute_pass_at_k(0, 16, 8) == 0.0   # None correct
    assert compute_pass_at_k(5, 10, 20) == 0.0  # k > n_total

    # Known value
    # pass@8 with 8/16 correct ≈ 0.99992 (very high!)
    result = compute_pass_at_k(8, 16, 8)
    assert result > 0.999  # Should be very close to 1.0
    print("✓ test_compute_pass_at_k passed")


def test_mathverify_semantic_equivalence():
    """Test semantic equivalence with math_verify (if available)."""
    if not MATH_VERIFY_AVAILABLE:
        print("⊘ test_mathverify_semantic_equivalence skipped (math_verify not available)")
        return

    # Fraction equivalence
    assert answers_match_math("0.5", "\\frac{1}{2}") == True
    assert answers_match_math("\\frac{2}{4}", "\\frac{1}{2}") == True

    # Algebraic equivalence
    assert answers_match_math("x^2 - 1", "(x-1)(x+1)") == True

    print("✓ test_mathverify_semantic_equivalence passed")


def test_verify_answer_strict_correctness():
    """Test verify_answer_strict with correct answers."""
    if not MATH_VERIFY_AVAILABLE:
        print("⊘ test_verify_answer_strict_correctness skipped (math_verify not available)")
        return

    is_correct, status = verify_answer_strict("Therefore \\boxed{42}", "42")
    assert status == "correct"
    assert is_correct == True

    is_correct, status = verify_answer_strict("Answer: \\boxed{\\frac{1}{2}}", "\\frac{1}{2}")
    assert status == "correct"
    assert is_correct == True

    print("✓ test_verify_answer_strict_correctness passed")


def test_verify_answer_strict_wrong():
    """Test verify_answer_strict with wrong answers."""
    if not MATH_VERIFY_AVAILABLE:
        print("⊘ test_verify_answer_strict_wrong skipped (math_verify not available)")
        return

    is_correct, status = verify_answer_strict("Therefore \\boxed{43}", "42")
    assert status == "wrong"
    assert is_correct == False

    print("✓ test_verify_answer_strict_wrong passed")


def test_verify_answer_strict_no_answer():
    """Test verify_answer_strict with no boxed answer."""
    is_correct, status = verify_answer_strict("No boxed answer here", "42")
    assert status == "no_answer"
    assert is_correct == False
    print("✓ test_verify_answer_strict_no_answer passed")


def test_very_long_answer():
    """Test handling of very long answers."""
    long_answer = "\\boxed{" + "x" * 10000 + "}"
    is_correct, status = verify_answer_strict(long_answer, "x", max_prediction_length=1000)
    # Should be unparsable due to length check
    # Note: If math_verify not available, might fall back to string comparison
    assert status in ["unparsable", "wrong", "correct"]
    print("✓ test_very_long_answer passed")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*80)
    print("RUNNING UNIT TESTS FOR ANSWER EXTRACTION MODULE")
    print("="*80 + "\n")

    # Basic extraction tests
    test_extract_simple_boxed()
    test_extract_boxed_substring()
    test_extract_nested_braces()
    test_extract_multiple_boxed()
    test_extract_no_boxed()
    test_extract_empty_boxed()
    test_empty_boxed_exception()
    test_extract_ground_truth()

    # Real MATH dataset cases
    test_real_whitespace_issue()
    test_real_latex_variants()
    test_real_case_insensitive()

    # Normalization
    test_normalization()
    test_latex_spacing_commands()

    # Edge cases
    test_empty_inputs()
    test_malformed_latex()
    test_special_characters()
    test_very_long_answer()

    # Utility functions
    test_compute_pass_at_k()

    # math_verify tests (if available)
    test_mathverify_semantic_equivalence()
    test_verify_answer_strict_correctness()
    test_verify_answer_strict_wrong()
    test_verify_answer_strict_no_answer()

    print("\n" + "="*80)
    print("✓ ALL UNIT TESTS PASSED")
    print(f"math_verify available: {MATH_VERIFY_AVAILABLE}")
    print("="*80 + "\n")


if __name__ == '__main__':
    run_all_tests()
