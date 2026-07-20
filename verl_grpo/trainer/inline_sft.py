"""
InlineSFTTrainer: Subclass of veRL's FSDPSFTTrainer that wraps an existing
FSDP actor model for epoch-boundary SFT during GRPO training.

Key differences from FSDPSFTTrainer:
  - Skips model loading — reuses the PPO actor's FSDP model
  - Creates a fresh AdamW optimizer (separate from GRPO's)
  - NoOp LR scheduler (SFT is supplementary)
  - fit_gold() method for mini-batch iteration over gold data
"""

from types import SimpleNamespace

import torch
from torch import optim
from tensordict import TensorDict

from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
from verl.utils.device import get_device_name


class NoOpScheduler:
    """Dummy LR scheduler that does nothing. SFT is supplementary."""

    def __init__(self, optimizer):
        self._lr = optimizer.param_groups[0]["lr"]

    def step(self):
        pass

    def get_last_lr(self):
        return [self._lr]


class SFTConfigAdapter:
    """Maps PPO config + sft_config to the config structure expected by FSDPSFTTrainer.

    FSDPSFTTrainer accesses:
      - config.data.micro_batch_size_per_gpu
      - config.data.balance_dp_token
      - config.optim.clip_grad
      - config.model.strategy
    """

    def __init__(self, sft_micro_batch_size_per_gpu, grad_clip, lr=1e-5,
                 weight_decay=0.01, strategy="fsdp"):
        self.data = SimpleNamespace(
            micro_batch_size_per_gpu=sft_micro_batch_size_per_gpu,
            balance_dp_token=True,
        )
        self.optim = SimpleNamespace(
            clip_grad=grad_clip,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.model = SimpleNamespace(strategy=strategy)


class InlineSFTTrainer(FSDPSFTTrainer):
    """SFT trainer that wraps an existing FSDP actor model.

    Skips model loading — reuses the PPO actor's model only.
    Creates a fresh AdamW optimizer for SFT (discarded after).

    Inherits from FSDPSFTTrainer:
      - _compute_loss_and_backward(batch) — proven loss computation
      - training_step(batch) — micro-batching, grad clip, optimizer step
    """

    def __init__(self, fsdp_model, model_config, sft_config):
        """
        Args:
            fsdp_model: The FSDP-wrapped actor module (same object used by GRPO).
            model_config: HF model config (for vocab_size). Available as
                          self.actor_model_config in fsdp_workers.
            sft_config: SFTConfigAdapter instance.
        """
        # DO NOT call super().__init__() — skip model loading, dataloader, etc.
        self.fsdp_model = fsdp_model
        # FSDPSFTTrainer._compute_loss_and_backward uses self.model.config.vocab_size
        self.model = SimpleNamespace(config=model_config)
        self.config = sft_config
        self.device_name = get_device_name()
        self.use_remove_padding = False
        self.sharding_manager = None

        # Fresh optimizer for SFT — only trainable params (LoRA weights)
        # Avoids allocating AdamW momentum/variance for frozen base model params
        self.optimizer = optim.AdamW(
            [p for p in self.fsdp_model.parameters() if p.requires_grad],
            lr=sft_config.optim.lr,
            weight_decay=sft_config.optim.weight_decay,
        )
        # NoOp LR scheduler — SFT is supplementary, don't step
        self.lr_scheduler = NoOpScheduler(self.optimizer)

    def fit_gold(self, gold_batches, sft_epochs=1):
        """Run SFT on gold data for sft_epochs passes.

        Args:
            gold_batches: List of TensorDict batches (from splitting the full gold buffer).
            sft_epochs: Number of passes over the gold data.

        Returns:
            List of metric dicts from each training_step() call.
        """
        all_metrics = []
        for ep in range(sft_epochs):
            for batch in gold_batches:
                metrics = self.training_step(batch)  # inherited from FSDPSFTTrainer
                all_metrics.append(metrics)
        return all_metrics
