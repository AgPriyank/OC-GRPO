"""
Custom RLHFDataset that promotes extra_info fields to top-level
so they appear in batch.non_tensor_batch during training.

Fields promoted:
  - hints: list[str] (hint texts for cliff problems, [] otherwise)
  - problem_id: str (e.g. "math_train_0")
  - question: str (raw question text)
  - level: str (e.g. "Level 5")
  - subject: str (e.g. "precalculus")
  - solution: str (full step-by-step solution text)
  - guidance_type: str ("original", "hint", or "prefix" — for offline augmented data)
  - hint_level_name: str (e.g. "L2_conceptual" — for offline hint-guided rows)
  - prefix_fraction: str (e.g. "0.6" — for offline prefix-guided rows)
  - prefix_text: str (partial solution text — for offline prefix-guided rows)

PrefixRL mode (hint_mode=prefix_continuation):
  For guided prefix rows, the prompt is rebuilt as:
    chat_template(question) + tokenize(prefix_text)
  so the prefix appears after <|im_start|>assistant and vLLM continues from there.
  Gradients are naturally masked on the prefix since it's part of the 'prompts' tensor.
"""

import torch

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
from verl_grpo.utils.chat_template import apply_chat_template_compat


class HintRLHFDataset(RLHFDataset):
    """Extends RLHFDataset to expose extra_info fields as non_tensor_batch entries.

    When hint_mode='prefix_continuation' (set via data config), guided prefix rows
    have their prompts rebuilt in PrefixRL style: the prefix text is appended after
    the assistant header token instead of being embedded in the user message.
    """

    def __init__(self, data_files, tokenizer, config, processor=None):
        self.hint_mode = config.get("hint_mode", None)
        super().__init__(data_files, tokenizer, config, processor)

    def _promote_extra_info(self, row_dict):
        """Promote extra_info fields to top-level for easy access in trainer."""
        extra = row_dict.get("extra_info", {})
        if isinstance(extra, dict):
            row_dict["hints"] = extra.get("hints", [])
            row_dict["problem_id"] = extra.get("problem_id", "")
            row_dict["question"] = extra.get("question", "")
            row_dict["level"] = extra.get("level", "")
            row_dict["subject"] = extra.get("subject", "")
            row_dict["solution"] = extra.get("solution", "")
            # Offline augmented data fields
            row_dict["guidance_type"] = extra.get("guidance_type", "original")
            row_dict["hint_level_name"] = extra.get("hint_level_name", "")
            row_dict["prefix_fraction"] = str(extra.get("prefix_fraction", ""))
            row_dict["prefix_text"] = extra.get("prefix_text", "")
        return row_dict

    def _is_prefixrl_guided_row(self, extra):
        """Check if this row should use PrefixRL-style prefix injection."""
        if self.hint_mode != "prefix_continuation":
            return False
        guidance_type = extra.get("guidance_type", "original")
        if guidance_type != "prefix":
            return False
        prefix_fraction = extra.get("prefix_fraction", 0)
        try:
            frac = float(prefix_fraction)
        except (ValueError, TypeError):
            return False
        # Drop 100% prefix rows — treat as original
        return 0 < frac < 1.0

    def _build_prefixrl_prompt(self, question, prefix_text):
        """Build PrefixRL-style prompt: chat_template(question) + prefix_text tokens.

        The resulting token sequence is:
          <|im_start|>user\\n{question}<|im_end|>\\n<|im_start|>assistant\\n{prefix_text}

        vLLM will generate continuation tokens from after prefix_text.
        """
        messages = [{"role": "user", "content": question}]
        raw_prompt = apply_chat_template_compat(
            self.tokenizer, messages, enable_thinking=self.enable_thinking,
            add_generation_prompt=True, tokenize=False
        )
        # Tokenize chat template portion
        base_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        # Tokenize prefix text (no special tokens — seamless continuation)
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        # Concatenate: chat_template + prefix
        full_ids = base_ids + prefix_ids

        # Truncate if exceeds max_prompt_length (right truncation to keep question intact)
        if len(full_ids) > self.max_prompt_length:
            full_ids = full_ids[: self.max_prompt_length]

        # Create tensors and left-pad
        input_ids = torch.tensor([full_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation="error",
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        return input_ids[0], attention_mask[0], position_ids[0], full_ids

    def __getitem__(self, item):
        row_dict = super().__getitem__(item)
        row_dict = self._promote_extra_info(row_dict)

        # PrefixRL mode: rebuild prompt for guided prefix rows
        extra = row_dict.get("extra_info", {})
        if self._is_prefixrl_guided_row(extra):
            question = extra.get("question", "")
            prefix_text = extra.get("prefix_text", "")
            if question and prefix_text:
                input_ids, attention_mask, position_ids, raw_prompt_ids = (
                    self._build_prefixrl_prompt(question, prefix_text)
                )
                row_dict["input_ids"] = input_ids
                row_dict["attention_mask"] = attention_mask
                row_dict["position_ids"] = position_ids
                row_dict["raw_prompt_ids"] = raw_prompt_ids
                # Signal to compute_score to prepend prefix_text for reward eval
                if isinstance(extra, dict):
                    extra["prefixrl_prefix_for_reward"] = prefix_text
                    row_dict["extra_info"] = extra

        return row_dict
