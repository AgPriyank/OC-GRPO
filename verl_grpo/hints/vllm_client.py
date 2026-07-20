"""
External vLLM hint server client for 3-GPU architecture.

Communicates with a standalone vLLM server on a dedicated GPU via the
OpenAI-compatible HTTP API. Supports optional LoRA sync for trained_copy mode.

Modeled on the TRL VLLMClient in train_grpo_separate.py.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from openai import OpenAI
from safetensors.torch import save_file


class VLLMHintClient:
    """Client for an external vLLM hint generation server.

    Three hint model modes:
      - "base": No LoRA sync, uses base model as-is.
      - "trained_copy": Syncs LoRA weights from training to the hint server.
      - "separate": Different model entirely, no LoRA sync.
    """

    LORA_ADAPTER_NAME = "hint_lora"

    def __init__(
        self,
        server_url: str,
        lora_sync_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.lora_sync_path = Path(lora_sync_path) if lora_sync_path else None
        if self.lora_sync_path:
            self.lora_sync_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.client = OpenAI(api_key="EMPTY", base_url=f"{self.server_url}/v1")
        self._lora_loaded = False

    def wait_until_ready(self, timeout: int = 300, poll_interval: int = 5) -> bool:
        """Poll the vLLM server until it responds to /v1/models."""
        self.logger.info(f"Waiting for vLLM hint server at {self.server_url}...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(f"{self.server_url}/v1/models", timeout=5)
                if resp.status_code == 200:
                    models = resp.json()
                    self.logger.info(f"vLLM hint server ready. Models: {models}")
                    return True
            except requests.ConnectionError:
                pass
            time.sleep(poll_interval)
        raise TimeoutError(
            f"vLLM hint server at {self.server_url} not ready after {timeout}s"
        )

    def generate_hints(
        self,
        prompt_texts: List[str],
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.95,
    ) -> List[str]:
        """Generate hint completions via the external vLLM server.

        Args:
            prompt_texts: Chat-formatted prompt strings (from HintGenerator.build_hint_prompts).
            n: Number of completions per prompt.
            temperature: Sampling temperature.
            max_tokens: Max tokens per completion.
            top_p: Nucleus sampling threshold.

        Returns:
            List of response strings, length = len(prompt_texts) * n,
            grouped by prompt (prompt_0_resp_0, ..., prompt_0_resp_n-1, prompt_1_resp_0, ...).
        """
        model_name = self.LORA_ADAPTER_NAME if self._lora_loaded else None
        if model_name is None:
            models = self.client.models.list()
            model_name = models.data[0].id

        self.logger.info(
            f"[HintClient] Generating {len(prompt_texts)} x {n} hints..."
        )
        resp = self.client.completions.create(
            model=model_name,
            prompt=prompt_texts,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
        )

        completions = [choice.text for choice in resp.choices]
        return completions

    def sync_lora(
        self,
        lora_params: Dict,
        peft_config: Dict,
    ) -> None:
        """Save LoRA weights to disk and load them into the external vLLM server.

        Args:
            lora_params: OrderedDict of LoRA parameter tensors (from layered_summon_lora_params).
            peft_config: PEFT adapter config dict (from peft_model.peft_config["default"]).
        """
        if self.lora_sync_path is None:
            raise ValueError(
                "lora_sync_path must be set for LoRA sync (hint_model_mode=trained_copy)"
            )

        adapter_path = str(self.lora_sync_path / "adapter")
        os.makedirs(adapter_path, exist_ok=True)

        # Save LoRA weights
        save_file(lora_params, os.path.join(adapter_path, "adapter_model.safetensors"))

        # Save adapter config
        with open(
            os.path.join(adapter_path, "adapter_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(peft_config, f, ensure_ascii=False, indent=4)

        self.logger.info(f"[HintClient] LoRA adapter saved to {adapter_path}")

        # Unload previous adapter if loaded
        if self._lora_loaded:
            try:
                requests.post(
                    f"{self.server_url}/v1/unload_lora_adapter",
                    json={"lora_name": self.LORA_ADAPTER_NAME},
                    timeout=30,
                )
            except Exception as e:
                self.logger.warning(f"[HintClient] Failed to unload previous LoRA: {e}")

        # Load new adapter
        resp = requests.post(
            f"{self.server_url}/v1/load_lora_adapter",
            json={
                "lora_name": self.LORA_ADAPTER_NAME,
                "lora_path": adapter_path,
            },
            timeout=60,
        )
        if resp.status_code == 200:
            self._lora_loaded = True
            self.logger.info("[HintClient] LoRA adapter loaded into vLLM hint server")
        else:
            raise RuntimeError(
                f"[HintClient] Failed to load LoRA: {resp.status_code} {resp.text}"
            )
