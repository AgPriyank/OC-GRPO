"""
Hint generation logic for adaptive hint injection in veRL GRPO training.

Generates hints for zero-reward groups using the embedded vLLM engine.
Supports solution-aware, solution-blind, and hierarchical hint modes.

Ported from train_grpo_adaptive_hints.py.
"""

import re
from typing import Dict, List, Optional, Tuple

import torch


# ============================================================================
# Hint Prompt Templates (copied from train_grpo_adaptive_hints.py)
# ============================================================================

HINT_PROMPT_BLIND = """\
You are an expert math tutor. You need to help student with difficult math problems

Original Problem for the student:
{problem}

Think about the concepts covered in the problem. Create a hint, that helps the student solve the problem. The hint could be first few steps of the problem or the main formula or concept used in the problem.

Respond in the format:
Thought: [thought]
Hint: [hint]."""

HINT_PROMPT_SOLUTION_AWARE = """\
You are an expert math tutor. You need to help student with difficult math problems

Original Problem for the student:
{problem}

Original correct solution:
{solution}

Think about the concepts covered in the problem. Create a hint, that helps the student solve the problem. The hint could be first few steps of the problem or the main formula or concept used in the problem but do not leak the solution.

Respond in the format:
Thought: [thought]
Hint: [hint]."""

HINT_PROMPT_SOLUTION_AWARE_FAILED_CONDITIONAL = """\
You are an expert math tutor. A student attempted a math problem but got it wrong.

Original Problem:
{problem}

Correct solution:
{solution}

Incorrect student answer:
{trajectory}

Compare the student's approach with the correct solution. \
Create a hint that helps the student solve the problem correctly. \
Do not leak the full solution.

Respond in the format:
Hint: [hint]."""


# ============================================================================
# Hierarchical Hint Prompt Templates
# ============================================================================
# L1/L2 from scripts/generate_hierarchical_hints.py (with {trajectory})
# L3/L4 from scripts/generate_hierarchical_hints_no_fail_con.py (NO {trajectory})

HIERARCHICAL_HINT_PROMPTS = {
    "L1_blind": """\
You are an expert math tutor. A student attempted this problem but got it wrong.

Problem:
{problem}

Student's failed attempt:
{trajectory}

Analyze the student's work and think about the concepts in the problem. Provide a guidance or hint that redirects the student towards the correct approach.

Hint:""",

    "L2_conceptual": """\
You are an expert math tutor. A student attempted this problem but got it wrong.

Problem:
{problem}

Reference solution:
{solution}

Student's failed attempt:
{trajectory}

Compare the student's approach with the correct solution and think about the key concepts that the student missed and provide guidance or hint that redirects the student to the correct solution. Do not reveal full solution.

Hint:""",

    "L3_strategic": """\
You are an expert math tutor. A student is struggling with this problem.

Problem:
{problem}

Reference solution:
{solution}

Provide a STEP-BY-STEP STRATEGY for solving this problem as a numbered list. Highlight the key approach needed. Do not reveal final answer.

Hint:""",

    "L4_detailed": """\
You are an expert math tutor. A student is struggling with this problem.

Problem:
{problem}

Reference solution:
{solution}

Provide DETAILED GUIDANCE to help the student solve this problem. Include:
- The correct approach with key intermediate calculations
- Specific values and expressions the student needs
- The critical steps that lead to the answer
You may reveal most of the solution path, but do NOT state the final boxed answer.

Hint:""",
}

HIERARCHICAL_HINT_LEVELS = ["L1_blind", "L2_conceptual", "L3_strategic", "L4_detailed"]

# hierarchical2 mode: adds L5_solution (reference solution rephrase, no LLM generation needed)
HIERARCHICAL2_HINT_LEVELS = ["L1_blind", "L2_conceptual", "L3_strategic", "L4_detailed", "L5_solution"]


class HintGenerator:
    """Generates hints for zero-reward math problems.

    Uses the same model/vLLM engine as the training model (embedded vLLM).
    Hints are generated as tutor-format prompts, parsed from "Hint: ..." responses.
    """

    def __init__(self, tokenizer, hint_mode: str = "solution_aware"):
        self.tokenizer = tokenizer
        self.hint_mode = hint_mode

    def build_hint_prompts(
        self, questions: List[str], solutions: Optional[List[str]] = None,
        failed_trajectories: Optional[List[str]] = None,
    ) -> List[str]:
        """Build chat-formatted hint prompts for a batch of questions.

        Args:
            questions: Raw math question texts.
            solutions: Full step-by-step solutions (for solution_aware mode).
            failed_trajectories: Decoded text of a failed trajectory per question
                (for solution_aware_failed_conditional mode).

        Returns:
            List of tokenizer-formatted prompt strings ready for generation.
        """
        if self.hint_mode == "solution_aware_failed_conditional" and solutions and failed_trajectories:
            raw_prompts = [
                HINT_PROMPT_SOLUTION_AWARE_FAILED_CONDITIONAL.format(
                    problem=q, solution=s, trajectory=t
                )
                for q, s, t in zip(questions, solutions, failed_trajectories)
            ]
        elif self.hint_mode == "solution_aware" and solutions:
            raw_prompts = [
                HINT_PROMPT_SOLUTION_AWARE.format(problem=q, solution=s)
                for q, s in zip(questions, solutions)
            ]
        else:
            raw_prompts = [
                HINT_PROMPT_BLIND.format(problem=q) for q in questions
            ]

        formatted = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in raw_prompts
        ]
        return formatted

    def parse_hints_from_group(
        self,
        response_tokens: torch.Tensor,
        n: int,
        tokenizer,
    ) -> Optional[str]:
        """Parse n hint candidates from response tokens, return the longest valid hint.

        Args:
            response_tokens: Tensor of shape [n, response_length] with generated hint responses.
            n: Number of candidates per problem.
            tokenizer: Tokenizer for decoding.

        Returns:
            The longest valid hint string, or None if all candidates fail to parse.
        """
        valid_hints = []
        for k in range(n):
            resp_ids = response_tokens[k]
            # Remove padding tokens
            non_pad = resp_ids != tokenizer.pad_token_id
            if non_pad.any():
                valid_ids = resp_ids[non_pad]
                resp_text = tokenizer.decode(valid_ids, skip_special_tokens=True)
                hint = self.parse_hint(resp_text)
                if hint:
                    valid_hints.append(hint)

        if not valid_hints:
            return None

        # Return the longest hint (most detailed)
        return max(valid_hints, key=len)

    def parse_hints_from_texts(
        self,
        response_texts: List[str],
    ) -> Optional[str]:
        """Parse hint candidates from decoded text strings, return the longest valid hint.

        Used with the external vLLM hint server (text-in/text-out interface).

        Args:
            response_texts: List of decoded response strings from the hint server.

        Returns:
            The longest valid hint string, or None if all candidates fail to parse.
        """
        valid_hints = []
        for text in response_texts:
            hint = self.parse_hint(text)
            if hint:
                valid_hints.append(hint)
        return max(valid_hints, key=len) if valid_hints else None

    @staticmethod
    def parse_hint(response_text: str) -> Optional[str]:
        """Extract hint text from 'Hint: ...' format in a response.

        Args:
            response_text: Full response text from the model.

        Returns:
            Extracted hint string, or None if parsing fails.
        """
        match = re.search(r"Hint:\s*(.+)", response_text, re.DOTALL)
        if match:
            hint = match.group(1).strip()
            # Take first line only (clean up multi-line hints)
            hint = hint.split("\n")[0].strip()
            return hint if hint else None
        return None

    # ================================================================
    # Hierarchical hint methods
    # ================================================================

    def _format_hierarchical_prompt(
        self, level: str, question: str, solution: str, trajectory: str,
    ) -> str:
        """Format a single hierarchical hint prompt from template.

        L1_blind: uses problem + trajectory (no solution).
        L2_conceptual: uses problem + solution + trajectory.
        L3_strategic, L4_detailed: uses problem + solution (no trajectory).
        """
        template = HIERARCHICAL_HINT_PROMPTS[level]
        if level in ("L1_blind",):
            return template.format(problem=question, trajectory=trajectory)
        elif level in ("L2_conceptual",):
            return template.format(problem=question, solution=solution, trajectory=trajectory)
        else:
            return template.format(problem=question, solution=solution)

    def build_all_hierarchical_prompts(
        self,
        levels: List[str],
        questions: List[str],
        solutions: List[str],
        failed_trajectories: List[str],
    ) -> Tuple[List[str], List[Tuple[str, int]]]:
        """Build chat-formatted prompts for ALL hierarchical levels at once.

        Returns a flat list of prompts and a mapping from prompt index to
        (level, problem_index) for parsing results.

        Args:
            levels: List of hint level names (e.g., HIERARCHICAL_HINT_LEVELS).
            questions: Raw math question texts (M problems).
            solutions: Full step-by-step solutions (M problems).
            failed_trajectories: Decoded text of a failed trajectory per problem (M).

        Returns:
            Tuple of:
              - all_prompts: Flat list of len(levels)*M formatted prompt strings.
              - prompt_map: List of (level, problem_index) tuples, same length as all_prompts.
        """
        all_prompts = []
        prompt_map = []

        for level in levels:
            if level == "L5_solution":
                continue  # L5 uses reference solution directly, no generation needed
            for pidx, (q, s, t) in enumerate(zip(questions, solutions, failed_trajectories)):
                raw = self._format_hierarchical_prompt(level, q, s, t)
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": raw}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                all_prompts.append(formatted)
                prompt_map.append((level, pidx))

        return all_prompts, prompt_map

    def build_all_hierarchical_prompts_raw(
        self,
        levels: List[str],
        questions: List[str],
        solutions: List[str],
        failed_trajectories: List[str],
    ) -> Tuple[List[str], List[Tuple[str, int]]]:
        """Build raw text prompts for ALL hierarchical levels (no chat template).

        Like build_all_hierarchical_prompts() but returns raw prompt text
        suitable for frontier API calls (Anthropic/OpenAI messages format).

        Uses the SAME HIERARCHICAL_HINT_PROMPTS templates — L1/L2 include
        {trajectory}, L3/L4 do not. Ensures fair comparison with self-hinting.

        Args:
            levels: List of hint level names (e.g., HIERARCHICAL_HINT_LEVELS).
            questions: Raw math question texts (M problems).
            solutions: Full step-by-step solutions (M problems).
            failed_trajectories: Decoded text of a failed trajectory per problem (M).

        Returns:
            Tuple of:
              - all_prompts: Flat list of len(levels)*M raw prompt strings.
              - prompt_map: List of (level, problem_index) tuples, same length.
        """
        all_prompts = []
        prompt_map = []

        for level in levels:
            if level == "L5_solution":
                continue  # L5 uses reference solution directly, no generation needed
            for pidx, (q, s, t) in enumerate(zip(questions, solutions, failed_trajectories)):
                raw = self._format_hierarchical_prompt(level, q, s, t)
                all_prompts.append(raw)
                prompt_map.append((level, pidx))

        return all_prompts, prompt_map

    @staticmethod
    def parse_hierarchical_hint(response_text: str) -> Optional[str]:
        """Parse a hierarchical hint from model response, preserving full multi-line content.

        Unlike parse_hint() which extracts first-line-only from "Hint: ..." format,
        this takes the FULL response as the hint (the prompt ends with "Hint:" so the
        model's response IS the hint).

        Args:
            response_text: Full response text from the model.

        Returns:
            Cleaned hint string, or None if empty.
        """
        text = response_text.strip()
        if not text:
            return None

        # Strip "Hint:" prefix if model echoes it
        if text.lower().startswith("hint:"):
            text = text[5:].strip()

        # Trim trailing incomplete sentences for very long hints
        if len(text) > 1500 and not text.endswith(('.', '!', '?', '}')):
            last_period = text.rfind('.')
            if last_period > len(text) * 0.7:
                text = text[:last_period + 1]

        return text if text else None

    def parse_hierarchical_hints_from_group(
        self,
        response_tokens: torch.Tensor,
        n_candidates: int,
        tokenizer,
    ) -> Optional[str]:
        """Parse n_candidates hint responses from tokens, return the longest valid hint.

        Used with embedded vLLM (tensor-based responses).

        Args:
            response_tokens: Tensor of shape [n_candidates, response_length].
            n_candidates: Number of candidates to parse.
            tokenizer: Tokenizer for decoding.

        Returns:
            The longest valid hint string, or None if all candidates fail.
        """
        valid_hints = []
        for k in range(min(n_candidates, response_tokens.shape[0])):
            resp_ids = response_tokens[k]
            non_pad = resp_ids != tokenizer.pad_token_id
            if non_pad.any():
                valid_ids = resp_ids[non_pad]
                resp_text = tokenizer.decode(valid_ids, skip_special_tokens=True)
                hint = self.parse_hierarchical_hint(resp_text)
                if hint:
                    valid_hints.append(hint)

        if not valid_hints:
            return None
        return max(valid_hints, key=len)

    def parse_hierarchical_hints_from_texts(
        self,
        response_texts: List[str],
    ) -> Optional[str]:
        """Parse hint candidates from decoded text strings, return the longest valid hint.

        Used with external vLLM hint server (text-in/text-out interface).

        Args:
            response_texts: List of decoded response strings.

        Returns:
            The longest valid hint string, or None if all candidates fail.
        """
        valid_hints = []
        for text in response_texts:
            hint = self.parse_hierarchical_hint(text)
            if hint:
                valid_hints.append(hint)
        return max(valid_hints, key=len) if valid_hints else None

    # ================================================================
    # Solution prefix methods
    # ================================================================

    @staticmethod
    def strip_asymptote(text: str) -> str:
        """Remove [asy]...[/asy] diagram blocks from solution text."""
        return re.sub(r'\[asy\].*?\[/asy\]', '', text, flags=re.DOTALL).strip()

    @staticmethod
    def get_solution_prefix(solution_text: str, fraction: float) -> str:
        """Get first `fraction` of solution by char count, snapped to word boundary.

        Args:
            solution_text: Full reference solution text.
            fraction: Float in [0.0, 1.0]. E.g., 0.2 = first 20% of characters.

        Returns:
            Prefix string (cut at nearest preceding word boundary).
        """
        text = HintGenerator.strip_asymptote(solution_text)

        if fraction >= 1.0:
            return text

        target_len = max(1, int(fraction * len(text)))

        if target_len >= len(text):
            return text

        # Snap to nearest word boundary (space or newline) — look backward
        cut_pos = target_len
        while cut_pos > 0 and text[cut_pos] not in ' \n\t':
            cut_pos -= 1

        # If no boundary found, cut at target
        if cut_pos == 0:
            cut_pos = target_len

        return text[:cut_pos].rstrip()
