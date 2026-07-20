"""
Hinted prompt construction and tokenization for veRL GRPO training.

Builds math problem prompts with embedded hints and converts them
to DataProto objects suitable for generate_sequences().
"""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl import DataProto
from verl_grpo.utils.chat_template import apply_chat_template_compat


def compute_position_id_with_mask(mask):
    """Recompute position IDs from attention mask."""
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)


class HintedPromptBuilder:
    """Build tokenized hinted prompts as DataProto for generate_sequences()."""

    def __init__(self, tokenizer, max_prompt_length: int, enable_thinking=None):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.enable_thinking = enable_thinking
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def build_hinted_prompt_text(self, question: str, hint: str) -> str:
        """Build a chat-formatted math prompt with an embedded hint.

        Ported from train_grpo_adaptive_hints.py _format_hinted_prompt().

        Args:
            question: Raw math question text.
            hint: Hint text to include.

        Returns:
            Fully formatted prompt string with chat template applied.
        """
        instruction = (
            'Let\'s think step by step and output the final answer within \\boxed{}. '
            "You can use the hint that follows the question to help you solve the problem."
        )
        prompt_text = f"Problem: {question}\n\nHint: {hint}\n\n{instruction}"
        messages = [{"role": "user", "content": prompt_text}]
        return apply_chat_template_compat(
            self.tokenizer, messages, enable_thinking=self.enable_thinking,
            tokenize=False, add_generation_prompt=True
        )

    def build_hinted_prompt_text_v2(
        self, question: str, hint: str,
        system_prompt: str = None, is_l5: bool = False
    ) -> str:
        """Build a chat-formatted math prompt with hint, optional system prompt, and L5 framing.

        Used by hierarchical2 mode. For L1-L4, optionally injects an anti-repetition
        system prompt. For L5, uses a "rephrase in your own words" framing with no
        system prompt (rephrasing is intentional at L5).

        Args:
            question: Raw math question text.
            hint: Hint text (for L1-L4) or reference solution text (for L5).
            system_prompt: Anti-repetition system prompt (applied for L1-L4 only).
            is_l5: If True, use L5 rephrase framing instead of standard hint framing.

        Returns:
            Fully formatted prompt string with chat template applied.
        """
        if is_l5:
            # L5: Full solution rephrase — NO system prompt (rephrasing is intentional)
            instruction = (
                "Rephrase the solution above in your own words. "
                "Show your step-by-step work and output the final answer within \\boxed{}."
            )
            prompt_text = f"Problem: {question}\n\nFull solution: {hint}\n\n{instruction}"
            messages = [{"role": "user", "content": prompt_text}]
        else:
            # L1-L4: Standard hint framing with optional system prompt
            instruction = (
                'Let\'s think step by step and output the final answer within \\boxed{}. '
                "You can use the hint that follows the question to help you solve the problem."
            )
            prompt_text = f"Problem: {question}\n\nHint: {hint}\n\n{instruction}"
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt_text})

        return apply_chat_template_compat(
            self.tokenizer, messages, enable_thinking=self.enable_thinking,
            tokenize=False, add_generation_prompt=True
        )

    def build_prefix_prompt_text(
        self, question: str, prefix: str, fraction: float,
        system_prompt: str = None
    ) -> str:
        """Build a chat-formatted math prompt with a solution prefix as the hint.

        Uses different framing for partial prefixes vs. the full solution:
        - Partial (fraction < 1.0): "Partial reference solution: ... Complete the rest..."
        - Full (fraction >= 1.0): "Reference solution: ... Verify it..."

        Args:
            question: Raw math question text.
            prefix: The solution prefix text (from HintGenerator.get_solution_prefix).
            fraction: The fraction of the solution this prefix represents (0.0-1.0).
            system_prompt: Optional anti-repetition system prompt. Applied for partial
                prefixes (fraction < 1.0) only; skipped for full solution (fraction >= 1.0)
                where rephrasing is intentional.

        Returns:
            Fully formatted prompt string with chat template applied.
        """
        if fraction >= 1.0:
            # Full solution: NO system prompt (rephrasing is intentional)
            instruction = (
                "Here is the full reference solution. Verify it, show your work "
                "step by step, and output the final answer within \\boxed{}."
            )
            prompt_text = (
                f"Problem: {question}\n\n"
                f"Reference solution: {prefix}\n\n"
                f"{instruction}"
            )
            messages = [{"role": "user", "content": prompt_text}]
        else:
            # Partial prefix: optional anti-repetition system prompt
            instruction = (
                "Here is a partial reference solution to the problem above. "
                "Complete the rest of the solution step by step and output the "
                "final answer within \\boxed{}."
            )
            prompt_text = (
                f"Problem: {question}\n\n"
                f"Partial reference solution: {prefix}\n\n"
                f"{instruction}"
            )
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt_text})
        return apply_chat_template_compat(
            self.tokenizer, messages, enable_thinking=self.enable_thinking,
            tokenize=False, add_generation_prompt=True
        )

    def tokenize_prompts(self, prompt_texts: List[str]) -> DataProto:
        """Tokenize prompt texts into a DataProto suitable for generate_sequences().

        Prompts are left-padded to the max length in the batch (matching veRL convention).

        Args:
            prompt_texts: List of fully formatted prompt strings.

        Returns:
            DataProto with batch={input_ids, attention_mask, position_ids}
            and non_tensor_batch={raw_prompt_ids}.
        """
        # Tokenize each prompt
        all_ids = []
        all_raw_ids = []
        for text in prompt_texts:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            # Truncate from the left if exceeding max length
            if len(ids) > self.max_prompt_length:
                ids = ids[-self.max_prompt_length:]
            all_ids.append(torch.tensor(ids, dtype=torch.long))
            all_raw_ids.append(ids)

        # Left-pad to uniform length within this batch
        max_len = max(len(ids) for ids in all_ids)

        padded_input_ids = []
        padded_attention_mask = []
        for ids in all_ids:
            pad_len = max_len - len(ids)
            padded = F.pad(ids, (pad_len, 0), value=self.pad_token_id)
            mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(len(ids), dtype=torch.long),
            ])
            padded_input_ids.append(padded)
            padded_attention_mask.append(mask)

        input_ids = torch.stack(padded_input_ids)          # [N, max_len]
        attention_mask = torch.stack(padded_attention_mask)  # [N, max_len]
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_td = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(prompt_texts),
        )

        non_tensor_batch = {
            "raw_prompt_ids": np.array(all_raw_ids, dtype=object),
        }

        return DataProto(batch=batch_td, non_tensor_batch=non_tensor_batch)

    def build_prefixrl_prompts(self, questions: List[str], prefix_texts: List[str]) -> DataProto:
        """Build PrefixRL-style prompts: chat_template(question) + prefix_text tokens.

        The prefix is placed after <|im_start|>assistant in the tokenized prompt,
        so vLLM generates continuation from the prefix. Gradients are naturally
        excluded from the prefix since it's part of the 'prompts' tensor.

        Args:
            questions: List of raw math question texts.
            prefix_texts: List of partial solution texts (one per question).

        Returns:
            DataProto with batch={input_ids, attention_mask, position_ids}
            and non_tensor_batch={raw_prompt_ids}.
        """
        all_ids = []
        all_raw_ids = []
        for question, prefix_text in zip(questions, prefix_texts):
            messages = [{"role": "user", "content": question}]
            raw_prompt = apply_chat_template_compat(
                self.tokenizer, messages, enable_thinking=self.enable_thinking,
                add_generation_prompt=True, tokenize=False
            )
            base_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
            prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            full_ids = base_ids + prefix_ids

            # Right-truncate if exceeds max length (keep question, trim prefix end)
            if len(full_ids) > self.max_prompt_length:
                full_ids = full_ids[: self.max_prompt_length]

            all_ids.append(torch.tensor(full_ids, dtype=torch.long))
            all_raw_ids.append(full_ids)

        # Left-pad to uniform length within this batch
        max_len = max(len(ids) for ids in all_ids)

        padded_input_ids = []
        padded_attention_mask = []
        for ids in all_ids:
            pad_len = max_len - len(ids)
            padded = F.pad(ids, (pad_len, 0), value=self.pad_token_id)
            mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(len(ids), dtype=torch.long),
            ])
            padded_input_ids.append(padded)
            padded_attention_mask.append(mask)

        input_ids = torch.stack(padded_input_ids)
        attention_mask = torch.stack(padded_attention_mask)
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_td = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=len(questions),
        )

        non_tensor_batch = {
            "raw_prompt_ids": np.array(all_raw_ids, dtype=object),
        }

        return DataProto(batch=batch_td, non_tensor_batch=non_tensor_batch)
