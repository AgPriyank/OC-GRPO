"""
Backward-compatible wrapper for apply_chat_template.

Qwen3+ models accept ``enable_thinking`` to toggle internal chain-of-thought.
Older models (Qwen2.5, Llama, OLMo) raise TypeError on the extra kwarg.
This helper catches the TypeError and retries without the flag so a single
codebase works for both families.
"""


def apply_chat_template_compat(tokenizer, messages, enable_thinking=None, **kwargs):
    """Apply chat template, safely forwarding enable_thinking when supported.

    Args:
        tokenizer: HuggingFace tokenizer with apply_chat_template.
        messages: List of chat messages (dicts with role/content).
        enable_thinking: None (don't pass), True, or False.
            False adds <think>\\n\\n</think> on Qwen3 to skip thinking.
        **kwargs: Forwarded to apply_chat_template (e.g. tokenize, add_generation_prompt).

    Returns:
        The result of tokenizer.apply_chat_template().
    """
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(
                messages, enable_thinking=enable_thinking, **kwargs
            )
        except TypeError:
            # Model tokenizer doesn't support enable_thinking — ignore it
            return tokenizer.apply_chat_template(messages, **kwargs)
    return tokenizer.apply_chat_template(messages, **kwargs)
