"""
Frontier model hint client for generating hints via Anthropic/OpenAI APIs.

Used during GRPO training to generate high-quality trajectory-dependent hints
from a frontier model (e.g., Claude Sonnet) instead of the training model.
Supports async batched calls with rate limit handling and retry logic.
"""

import asyncio
import os
import random
import time
from typing import Dict, List, Tuple

API_STATS_KEYS = ("total_retries", "total_failures")


class FrontierHintClient:
    """Generates hints via frontier model API (Anthropic/OpenAI).

    Replaces embedded/external vLLM hint generation with API calls to a
    frontier model. Uses async concurrency with semaphore-based rate limiting
    and exponential backoff with jitter on retries.
    """

    def __init__(
        self,
        api: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        max_concurrency: int = 10,
        max_retries: int = 5,
    ):
        """
        Args:
            api: API provider — "anthropic" or "openai".
            model: Model identifier (e.g., "claude-sonnet-4-20250514", "gpt-4o").
            max_concurrency: Max concurrent API calls (semaphore limit).
            max_retries: Max retries per failed API call.
        """
        self.api = api
        self.model = model
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries

        # Validate API key is available
        if api == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable is required "
                    "for frontier hint generation with api='anthropic'"
                )
        elif api == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required "
                    "for frontier hint generation with api='openai'"
                )
        else:
            raise ValueError(f"Unsupported API provider: {api}. Use 'anthropic' or 'openai'.")

    def generate_hints(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Tuple[List[str], Dict[str, int]]:
        """Generate one hint response per prompt via frontier API.

        Args:
            prompts: Raw prompt texts (one per hint request).
            temperature: Sampling temperature.
            max_tokens: Max output tokens per response.

        Returns:
            Tuple of:
              - responses: List of response texts, same length as prompts.
                Empty string for any that failed after all retries.
              - stats: Dict with 'total_retries' and 'total_failures' counts.
        """
        if not prompts:
            return [], {"total_retries": 0, "total_failures": 0}

        return asyncio.run(
            self._generate_async(prompts, temperature, max_tokens)
        )

    async def _generate_async(
        self,
        prompts: List[str],
        temperature: float,
        max_tokens: int,
    ) -> Tuple[List[str], Dict[str, int]]:
        """Async implementation with semaphore-limited concurrency."""
        semaphore = asyncio.Semaphore(self.max_concurrency)
        stats = {"total_retries": 0, "total_failures": 0}
        stats_lock = asyncio.Lock()

        if self.api == "anthropic":
            import anthropic
            client = anthropic.AsyncAnthropic()

            async def call_api(prompt: str, idx: int) -> Tuple[int, str]:
                async with semaphore:
                    return await self._call_anthropic(
                        client, prompt, idx, temperature, max_tokens, stats, stats_lock
                    )

            tasks = [call_api(p, i) for i, p in enumerate(prompts)]
            results = await asyncio.gather(*tasks)
            await client.close()

        elif self.api == "openai":
            from openai import AsyncOpenAI
            client = AsyncOpenAI()

            async def call_api(prompt: str, idx: int) -> Tuple[int, str]:
                async with semaphore:
                    return await self._call_openai(
                        client, prompt, idx, temperature, max_tokens, stats, stats_lock
                    )

            tasks = [call_api(p, i) for i, p in enumerate(prompts)]
            results = await asyncio.gather(*tasks)
            await client.close()

        # Sort by index to maintain original order
        results.sort(key=lambda x: x[0])
        responses = [r[1] for r in results]

        return responses, stats

    async def _call_anthropic(
        self,
        client,
        prompt: str,
        idx: int,
        temperature: float,
        max_tokens: int,
        stats: dict,
        stats_lock: asyncio.Lock,
    ) -> Tuple[int, str]:
        """Call Anthropic API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = await client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return idx, response.content[0].text

            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate" in error_str.lower()

                # Parse retry-after if available
                retry_after = None
                if hasattr(e, "response") and hasattr(e.response, "headers"):
                    retry_after = e.response.headers.get("retry-after")

                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    if retry_after is not None:
                        wait = max(float(retry_after), 1.0)
                    else:
                        wait = min(2 ** attempt + random.random(), 60.0)

                    async with stats_lock:
                        stats["total_retries"] += 1

                    rl_tag = " [rate-limit]" if is_rate_limit else ""
                    print(f"[FrontierClient] Retry {attempt+1}/{self.max_retries} "
                          f"for prompt {idx}{rl_tag}: {error_str[:100]}... "
                          f"(waiting {wait:.1f}s)")
                    await asyncio.sleep(wait)
                else:
                    async with stats_lock:
                        stats["total_failures"] += 1
                    print(f"[FrontierClient] FAILED prompt {idx} after "
                          f"{self.max_retries} retries: {error_str[:200]}")
                    return idx, ""

        return idx, ""

    async def _call_openai(
        self,
        client,
        prompt: str,
        idx: int,
        temperature: float,
        max_tokens: int,
        stats: dict,
        stats_lock: asyncio.Lock,
    ) -> Tuple[int, str]:
        """Call OpenAI API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                return idx, response.choices[0].message.content

            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate" in error_str.lower()

                retry_after = None
                if hasattr(e, "response") and hasattr(e.response, "headers"):
                    retry_after = e.response.headers.get("retry-after")

                if attempt < self.max_retries - 1:
                    if retry_after is not None:
                        wait = max(float(retry_after), 1.0)
                    else:
                        wait = min(2 ** attempt + random.random(), 60.0)

                    async with stats_lock:
                        stats["total_retries"] += 1

                    rl_tag = " [rate-limit]" if is_rate_limit else ""
                    print(f"[FrontierClient] Retry {attempt+1}/{self.max_retries} "
                          f"for prompt {idx}{rl_tag}: {error_str[:100]}... "
                          f"(waiting {wait:.1f}s)")
                    await asyncio.sleep(wait)
                else:
                    async with stats_lock:
                        stats["total_failures"] += 1
                    print(f"[FrontierClient] FAILED prompt {idx} after "
                          f"{self.max_retries} retries: {error_str[:200]}")
                    return idx, ""

        return idx, ""
