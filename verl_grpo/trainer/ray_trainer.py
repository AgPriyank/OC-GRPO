# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl_grpo.utils.chat_template import apply_chat_template_compat
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, config=None):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = grpo_calculation_mask.size(1)
            # This mask is the one intended for GRPO
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        hint_generator=None,
        prompt_builder=None,
        hint_client=None,
        frontier_client=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
            hint_generator: Optional HintGenerator for adaptive hint injection.
            prompt_builder: Optional HintedPromptBuilder for constructing hinted prompts.
            hint_client: Optional VLLMHintClient for external hint server (3-GPU mode).
            frontier_client: Optional FrontierHintClient for frontier API hints.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        # Adaptive hint injection components
        self.hint_generator = hint_generator
        self.prompt_builder = prompt_builder
        self.hint_client = hint_client  # External vLLM hint server (3-GPU mode)
        self.frontier_client = frontier_client  # Frontier model API client
        self._replacement_problem_log = {}  # Per-problem hint replacement tracking (hierarchical_replacement mode)
        self._prev_hint_correct_lp = None  # Step-to-step hint-correct log-prob delta tracking

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None or config.actor_rollout_ref.rollout.multi_turn.interaction_config_path is not None, "tool_config_path or interaction_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_diagnostic_log(self, step, batch, reward_tensor, n, pad_token_id,
                              diagnostic_hint_data=None):
        """Save detailed per-step diagnostic JSON for trajectory inspection.

        Args:
            step: Global training step
            batch: DataProto batch with responses, prompts, non_tensor_batch
            reward_tensor: Token-level reward tensor [B*n, seq_len]
            n: Number of trajectories per prompt
            pad_token_id: Tokenizer pad token ID
            diagnostic_hint_data: Optional dict {uid: hint_cascade_info} for hint runs
        """
        diag_dir = self.config.trainer.get("diagnostic_log_dir", None)
        if not diag_dir:
            return

        from baseline import extract_answer_math, answers_match_math

        os.makedirs(diag_dir, exist_ok=True)

        uids = batch.non_tensor_batch['uid']
        unique_uids = list(dict.fromkeys(uids))  # Preserve order, deduplicate
        reward_sums = reward_tensor.sum(-1)

        problems = []
        for uid in unique_uids:
            mask = (uids == uid)
            group_indices = np.where(mask)[0]
            group_rewards = reward_sums[mask]
            idx = group_indices[0]

            n_correct = int((group_rewards > 0).sum())
            n_total = len(group_indices)
            if n_correct == 0:
                category = "all_wrong"
            elif n_correct == n_total:
                category = "all_right"
            else:
                category = "mixed"

            trajectories = []
            for i, gi in enumerate(group_indices):
                resp_ids = batch.batch['responses'][gi]
                valid_ids = resp_ids[resp_ids != pad_token_id]
                text = self.tokenizer.decode(valid_ids.tolist(), skip_special_tokens=True)
                predicted = extract_answer_math(text)
                gt = batch.non_tensor_batch['reward_model'][idx]['ground_truth']
                correct = (predicted is not None and answers_match_math(predicted, gt))
                trajectories.append({
                    "index": i,
                    "text": text,
                    "predicted_answer": str(predicted) if predicted else None,
                    "correct": correct,
                })

            problem_entry = {
                "problem_id": batch.non_tensor_batch.get('problem_id', [None]*len(uids))[idx],
                "question": batch.non_tensor_batch['question'][idx],
                "ground_truth": batch.non_tensor_batch['reward_model'][idx]['ground_truth'],
                "solution": batch.non_tensor_batch['solution'][idx],
                "n_total": n_total,
            }

            # Attach hint cascade data if available
            has_cascade = (diagnostic_hint_data and uid in diagnostic_hint_data)
            if has_cascade:
                cascade = diagnostic_hint_data[uid]
                problem_entry["hint_cascade"] = cascade
                # original_trajectories: pre-hint (all failed)
                # final_trajectories: post-replacement (from batch, may contain regen responses)
                # category/n_correct reflect FINAL state (post-replacement)
                problem_entry["original_trajectories"] = cascade.get("original_trajectories", [])
                problem_entry["final_trajectories"] = trajectories
                problem_entry["category_before_hints"] = "all_wrong"
                problem_entry["category_after_hints"] = category
                problem_entry["n_correct_before_hints"] = 0
                problem_entry["n_correct_after_hints"] = n_correct
            else:
                # No hints: trajectories are untouched originals
                problem_entry["original_trajectories"] = trajectories
                problem_entry["final_trajectories"] = trajectories
                problem_entry["category"] = category
                problem_entry["n_correct"] = n_correct

            problems.append(problem_entry)

        output = {
            "step": step,
            "seed": self.config.trainer.get("seed", 0),
            "rollout_seed": self.config.actor_rollout_ref.rollout.get("seed", 0),
            "experiment_name": self.config.trainer.experiment_name,
            "n_trajectories": n,
            "n_problems": len(problems),
            "problems": problems,
        }

        filename = os.path.join(diag_dir, f"step_{step:04d}.json")
        with open(filename, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[Diagnostic] Saved {filename} ({len(problems)} problems)")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, "worker_nsight_options must be set when profile_steps is set"
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(OmegaConf.select(self.config.trainer, "worker_nsight_options"))

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.workers.rollout.async_server import AsyncLLMServerManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

        # Save per-problem replacement metrics (hierarchical_replacement mode)
        if self._replacement_problem_log:
            import json as _json
            metrics_path = os.path.join(local_global_step_folder, "replacement_metrics.json")
            with open(metrics_path, 'w') as f:
                _json.dump(self._replacement_problem_log, f, indent=2)
            print(f"[HierReplace] Saved replacement metrics for {len(self._replacement_problem_log)} "
                  f"problems to {metrics_path}")

        # Sync LoRA to external hint server (checkpoint-level sync for trained_copy mode)
        hints_cfg = self.config.get('hints', {})
        if (self.hint_client is not None
                and hints_cfg.get('lora_sync_frequency', 'never') == 'checkpoint'
                and hints_cfg.get('hint_model_mode', 'base') == 'trained_copy'):
            lora_adapter_path = os.path.join(actor_local_path, "lora_adapter")
            if os.path.exists(os.path.join(lora_adapter_path, "adapter_model.safetensors")):
                from safetensors.torch import load_file
                lora_params = load_file(os.path.join(lora_adapter_path, "adapter_model.safetensors"))
                with open(os.path.join(lora_adapter_path, "adapter_config.json"), "r") as f:
                    peft_config = json.load(f)
                self.hint_client.sync_lora(lora_params, peft_config)
                print(f"[HintSync] Synced LoRA to external hint server at step {self.global_steps}")
            else:
                print(f"[HintSync] Warning: LoRA adapter not found at {lora_adapter_path}, skipping sync")

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _should_skip_mastered(self, problem_id, current_epoch, cfg):
        """Check if problem was >=threshold accuracy in both of the two previous epochs."""
        if not problem_id:
            return False
        start_epoch = cfg.get('start_epoch', 3)
        if current_epoch < start_epoch:
            return False
        threshold = cfg.get('threshold', 0.75)
        consecutive = cfg.get('consecutive_epochs', 2)
        history = self._problem_epoch_accuracy.get(problem_id, {})
        # Check the last `consecutive` epochs before current
        for ep in range(current_epoch - consecutive, current_epoch):
            if ep < 1 or history.get(ep) is None or history[ep] < threshold:
                return False
        return True

    def _flush_epoch_data(self, epoch_num):
        """Flush per-problem accuracy from current epoch into history."""
        for pid, (nc, nt) in self._current_epoch_data.items():
            if pid not in self._problem_epoch_accuracy:
                self._problem_epoch_accuracy[pid] = {}
            self._problem_epoch_accuracy[pid][epoch_num] = nc / nt
        n_tracked = len(self._problem_epoch_accuracy)
        self._current_epoch_data = {}
        print(f"[Curriculum] Flushed epoch {epoch_num} data: {n_tracked} problems tracked")

    def _build_sft_batch(self, gold_buffer):
        """Convert gold data buffer to a DataProto batch for SFT update.

        Each gold item has question + gold_response_text.
        We apply chat template, tokenize, and create loss_mask that masks out
        all prompt tokens (system, chat template, question) — only response tokens
        contribute to the SFT loss.

        Follows the same pattern as veRL's SFTDataset.__getitem__ (sft_dataset.py).
        """
        from verl.utils.model import compute_position_id_with_mask

        tokenizer = self.tokenizer
        max_prompt_length = self.config.data.max_prompt_length
        max_response_length = self.config.data.max_response_length
        max_length = max_prompt_length + max_response_length
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # First pass: tokenize and truncate, collecting unpadded sequences
        unpadded_input_ids = []
        unpadded_prompt_lengths = []
        unpadded_response_lengths = []
        n_skipped = 0

        for item in gold_buffer:
            # Apply chat template to build the full prompt with system tokens
            prompt_chat = [{"role": "user", "content": item['question']}]
            _enable_thinking = self.config.data.get("enable_thinking", None)
            prompt_str = apply_chat_template_compat(
                tokenizer, prompt_chat, enable_thinking=_enable_thinking,
                add_generation_prompt=True, tokenize=False)

            # Tokenize prompt and response separately
            prompt_ids = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            response_str = item['gold_response_text'] + tokenizer.eos_token
            response_ids = tokenizer(response_str, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

            prompt_length = prompt_ids.shape[0]
            response_length = response_ids.shape[0]

            # Truncate if needed
            if prompt_length > max_prompt_length:
                prompt_ids = prompt_ids[-max_prompt_length:]  # Keep rightmost
                prompt_length = max_prompt_length
            if response_length > max_response_length:
                response_ids = response_ids[:max_response_length]
                response_length = max_response_length

            # Concatenate (unpadded)
            input_ids = torch.cat((prompt_ids, response_ids), dim=-1)

            # Truncate to max_length if combined exceeds it
            if input_ids.shape[0] > max_length:
                input_ids = input_ids[:max_length]
                response_length = max_length - prompt_length

            unpadded_input_ids.append(input_ids)
            unpadded_prompt_lengths.append(prompt_length)
            unpadded_response_lengths.append(response_length)

        if not unpadded_input_ids:
            return None

        # Second pass: pad to actual max length in the batch (not config max)
        actual_max_len = max(ids.shape[0] for ids in unpadded_input_ids)
        all_input_ids = []
        all_attention_mask = []
        all_loss_mask = []

        for input_ids, prompt_length, response_length in zip(
                unpadded_input_ids, unpadded_prompt_lengths, unpadded_response_lengths):
            sequence_length = input_ids.shape[0]
            attention_mask = torch.ones(sequence_length, dtype=torch.long)

            # Pad to actual_max_len (right-padded)
            if sequence_length < actual_max_len:
                pad_len = actual_max_len - sequence_length
                input_ids = torch.cat((input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)))
                attention_mask = torch.cat((attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)))

            # Build loss_mask: mask prompt, keep response (following sft_dataset.py convention)
            loss_mask = attention_mask.clone()
            if prompt_length > 1:
                loss_mask[:prompt_length - 1] = 0  # Mask prompt (shifted by 1 for next-token prediction)
            # Mask the last response token (no next token to predict)
            last_pos = min(prompt_length + response_length, loss_mask.size(0)) - 1
            if last_pos >= 0:
                loss_mask[last_pos] = 0

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_loss_mask.append(loss_mask)

        batch_dict = {
            'input_ids': torch.stack(all_input_ids),
            'attention_mask': torch.stack(all_attention_mask),
            'loss_mask': torch.stack(all_loss_mask).float(),
        }
        batch_dict['position_ids'] = compute_position_id_with_mask(batch_dict['attention_mask'])

        from tensordict import TensorDict
        from verl.protocol import DataProtoConfig
        sft_data = DataProto(batch=TensorDict(batch_dict, batch_size=len(all_input_ids)))
        # Enable auto-padding so dispatch can split across GPUs evenly
        sft_data.meta_info[DataProtoConfig.auto_padding_key] = True

        # Set micro batch size for SFT
        sft_cfg = self.config.get('hints', {}).get('sft_config', {})
        sft_micro_bs = sft_cfg.get('sft_micro_batch_size_per_gpu',
                                    self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu)
        sft_data.meta_info['sft_micro_batch_size'] = sft_micro_bs
        sft_data.meta_info['sft_lr'] = sft_cfg.get('sft_lr', 1e-5)
        sft_data.meta_info['sft_epochs'] = sft_cfg.get('sft_epochs', 1)

        print(f"[HintSFT] Built SFT batch: {len(all_input_ids)} samples, "
              f"padded_length={actual_max_len} (config_max={max_length}), skipped={n_skipped}")
        return sft_data

    def _should_skip_beyond_reach(self, problem_id, n_correct_regen, n_total, current_epoch, cfg):
        """Check if hint replacement should be skipped due to high hint success rate.

        When hints consistently solve a problem (high SR) but the model independently
        gets 0/n, the hinted solutions are 'beyond the model's reach'. Training on
        these alien trajectories wastes gradient and can cause forgetting.

        Args:
            problem_id: The problem identifier.
            n_correct_regen: Number of correct trajectories in the hint regeneration.
            n_total: Total trajectories regenerated (typically n=8).
            current_epoch: Current training epoch (1-indexed).
            cfg: The skip_beyond_reach config dict.

        Returns:
            True if this hint replacement should be skipped.
        """
        if not cfg.get('enabled', False):
            return False
        start_epoch = cfg.get('start_epoch', 1)
        if current_epoch < start_epoch:
            return False

        threshold = cfg.get('threshold', 0.75)
        min_obs = cfg.get('min_observations', 1)

        # Update cumulative tracker
        if problem_id not in self._problem_hint_sr:
            self._problem_hint_sr[problem_id] = {'n_correct': 0, 'n_total': 0, 'n_obs': 0}
        tracker = self._problem_hint_sr[problem_id]
        tracker['n_correct'] += n_correct_regen
        tracker['n_total'] += n_total
        tracker['n_obs'] += 1

        if tracker['n_obs'] < min_obs:
            return False

        cumulative_sr = tracker['n_correct'] / tracker['n_total']
        return cumulative_sr >= threshold

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # Per-problem accuracy tracking for curriculum skip_mastered
        self._problem_epoch_accuracy = {}   # {problem_id: {epoch_num: accuracy_float}}
        self._current_epoch_data = {}       # {problem_id: (n_correct, n_total)}
        self._steps_per_epoch = len(self.train_dataloader)  # steps per epoch for epoch boundary detection

        # Per-problem hint success rate tracking for skip_beyond_reach
        self._problem_hint_sr = {}  # {problem_id: {'n_correct': int, 'n_total': int, 'n_obs': int}}

        # Cumulative per-problem tracker for skip_permanently_stuck
        self._problem_cumulative = {}

        # hint_SFT mode: accumulate gold data (hint-derived correct trajectories) per epoch
        self._sft_gold_buffer = []

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.profile_steps is not None else False
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                _step_start = time.time()
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    _generation_start = time.time()
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                    _generation_end = time.time()

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # ===== CUSTOM HOOK 1: Post-reward, pre-log-prob =====
                    # Adaptive hint injection: for zero-reward groups, generate hints,
                    # re-attempt with hinted prompts, and replace improved trajectories.
                    _hook1_start = time.time()
                    n = self.config.actor_rollout_ref.rollout.n
                    response_length = self.config.data.max_response_length
                    reward_sums = reward_tensor.sum(-1)  # [B*n], 0.0 or 1.0 per row
                    uids = batch.non_tensor_batch['uid']
                    unique_uids = np.unique(uids)

                    # --- Log-prob gap diagnostic tracking ---
                    _diag_batch_size = len(reward_sums)
                    hint_replaced = torch.zeros(_diag_batch_size, dtype=torch.bool)
                    hint_level_map = [''] * _diag_batch_size
                    hinted_prompt_texts = [''] * _diag_batch_size  # Store hinted prompt text for IS ratio
                    saved_response_mask = batch.batch.get(
                        'response_mask',
                        batch.batch['attention_mask'][:, -response_length:]
                    ).clone()
                    saved_rollout_lp = None
                    if 'rollout_log_probs' in batch.batch:
                        saved_rollout_lp = batch.batch['rollout_log_probs'].clone()

                    # 1. Identify zero-reward and all-correct groups
                    zero_meta = []
                    _diag_hint_data = {}  # Reset each step (prevents stale data when zero_meta is empty)
                    n_all_right = 0
                    n_mixed = 0
                    all_correct_indices = []
                    skip_indices = []  # Unified skip list for all skip rules
                    n_skipped_mastered = 0
                    curriculum_cfg = self.config.get('curriculum', {})
                    skip_all_correct = curriculum_cfg.get('skip_all_correct', False) if curriculum_cfg else False
                    skip_all_correct_start_epoch = curriculum_cfg.get('skip_all_correct_start_epoch', 1) if curriculum_cfg else 1
                    skip_unhintable_cfg = curriculum_cfg.get('skip_unhintable', False) if curriculum_cfg else False
                    skip_mastered_cfg = curriculum_cfg.get('skip_mastered', {}) if curriculum_cfg else {}
                    skip_beyond_reach_cfg = curriculum_cfg.get('skip_beyond_reach', {}) if curriculum_cfg else {}
                    skip_permanently_stuck_cfg = curriculum_cfg.get('skip_permanently_stuck', {}) if curriculum_cfg else {}

                    # Hint trigger config (generalized threshold)
                    hints_cfg_trigger = self.config.get('hints', {})
                    hint_trigger_threshold = hints_cfg_trigger.get('hint_trigger_threshold', 0) if hints_cfg_trigger else 0
                    hint_trigger_mode = hints_cfg_trigger.get('hint_trigger_mode', 'simple') if hints_cfg_trigger else 'simple'
                    n_explore = self.config.actor_rollout_ref.rollout.n  # e.g. 16
                    n_train = hints_cfg_trigger.get('n_train', None) if hints_cfg_trigger else None
                    if n_train is None:
                        n_train = n_explore  # default: use all trajectories for training

                    # Compute current epoch (1-indexed) for skip_mastered
                    current_epoch = (self.global_steps - 1) // self._steps_per_epoch + 1
                    n_skipped_permanently_stuck = 0
                    n_skipped_sufficient_correct = 0
                    n_partial_correct_hinted = 0

                    for uid in unique_uids:
                        mask = (uids == uid)
                        group_indices = np.where(mask)[0]
                        group_rewards = reward_sums[mask]
                        idx = group_indices[0]
                        n_correct = int((group_rewards > 0).sum())
                        n_total = int(len(group_rewards))

                        # Record per-problem accuracy for curriculum tracking
                        problem_id = batch.non_tensor_batch['problem_id'][idx]
                        if problem_id:
                            self._current_epoch_data[problem_id] = (n_correct, n_total)

                        # Update cumulative tracker
                        if problem_id:
                            cum = self._problem_cumulative.setdefault(problem_id, {
                                'n_correct_total': 0, 'n_total': 0,
                                'n_hint_correct': 0, 'n_hint_total': 0,
                                'n_observations': 0, 'n_hint_observations': 0,
                                'permanently_skipped': False,
                            })
                            cum['n_correct_total'] += n_correct
                            cum['n_total'] += n_total
                            cum['n_observations'] += 1

                        # Check skip_permanently_stuck
                        if (skip_permanently_stuck_cfg and skip_permanently_stuck_cfg.get('enabled', False)
                                and problem_id and self._problem_cumulative.get(problem_id, {}).get('permanently_skipped', False)):
                            skip_indices.extend(group_indices.tolist())
                            n_skipped_permanently_stuck += 1
                            continue

                        # Check skip_mastered: skip if >=threshold in 2 consecutive previous epochs
                        if (skip_mastered_cfg and skip_mastered_cfg.get('enabled', False)
                                and self._should_skip_mastered(problem_id, current_epoch, skip_mastered_cfg)):
                            skip_indices.extend(group_indices.tolist())
                            n_skipped_mastered += 1
                            if n_correct == n_total:
                                n_all_right += 1
                            elif n_correct == 0:
                                pass  # Don't add to zero_meta
                            else:
                                n_mixed += 1
                            continue

                        # Skip sufficient_correct: with intelligent subsample (hints mode),
                        # n_correct >= n_train means ALL subsampled trajectories will be
                        # correct -> GRPO advantage = 0 -> no learning signal. Skip.
                        if (n_train < n_explore and self.hint_generator is not None
                                and n_correct >= n_train):
                            skip_indices.extend(group_indices.tolist())
                            n_skipped_sufficient_correct += 1
                            n_all_right += 1
                            continue

                        # Generalized hint trigger (threshold + two_phase support)
                        should_hint = False
                        if hint_trigger_mode == 'two_phase':
                            half = n_total // 2
                            first_half_correct = int((group_rewards[:half] > 0).sum())
                            if first_half_correct == 0:  # Phase 1: first half all failed
                                if n_correct <= hint_trigger_threshold:  # Phase 2: check full set
                                    should_hint = True
                        else:  # simple mode
                            if n_correct <= hint_trigger_threshold:
                                should_hint = True

                        if should_hint:
                            zero_meta.append({
                                'uid': uid,
                                'question': batch.non_tensor_batch['question'][idx],
                                'ground_truth': batch.non_tensor_batch['reward_model'][idx]['ground_truth'],
                                'solution': batch.non_tensor_batch['solution'][idx],
                                'indices': group_indices,
                                'n_correct_original': n_correct,
                            })
                            if n_correct > 0:
                                n_partial_correct_hinted += 1
                        elif n_correct == n_total:
                            n_all_right += 1
                            if skip_all_correct and current_epoch >= skip_all_correct_start_epoch:
                                all_correct_indices.extend(group_indices.tolist())
                        else:
                            n_mixed += 1

                    metrics["custom/n_questions"] = len(unique_uids)
                    metrics["custom/n_hint_candidates"] = len(zero_meta)
                    metrics["custom/n_all_wrong"] = len(zero_meta)  # backward compat alias
                    metrics["custom/n_all_right"] = n_all_right
                    metrics["custom/n_mixed"] = n_mixed
                    metrics["custom/n_skipped_mastered"] = n_skipped_mastered
                    metrics["custom/n_skipped_permanently_stuck"] = n_skipped_permanently_stuck
                    metrics["custom/n_skipped_sufficient_correct"] = n_skipped_sufficient_correct
                    metrics["custom/n_partial_correct_hinted"] = n_partial_correct_hinted
                    metrics["custom/hint_trigger_threshold"] = hint_trigger_threshold
                    metrics["custom/hint_trigger_mode"] = 0 if hint_trigger_mode == 'simple' else 1
                    metrics["custom/pct_easy_in_batch"] = n_all_right / max(len(unique_uids), 1)
                    metrics["custom/n_permanently_stuck_total"] = sum(
                        1 for c in self._problem_cumulative.values() if c.get('permanently_skipped'))
                    metrics["custom/current_epoch"] = current_epoch

                    # ===== PROMPT MASKED MODE: swap augmented prompts → original =====
                    # For offline augmented data (runs 900/901), guided rows have
                    # augmented prompts (with hints/prefixes) in the parquet. This mode
                    # swaps them back to the original question prompt before training,
                    # so the model learns to solve problems WITHOUT hints while using
                    # hint-generated trajectories as training signal.
                    _pm_hints_cfg = self.config.get('hints', {})
                    _pm_hint_mode = _pm_hints_cfg.get('hint_mode', '') if _pm_hints_cfg else ''
                    prompt_mode = _pm_hints_cfg.get('prompt_mode', 'original') if _pm_hints_cfg else 'original'
                    if (self.prompt_builder is not None
                            and _pm_hint_mode == 'prompt_masked'):
                        from verl_grpo.hints.prompt_builder import compute_position_id_with_mask as _pm_compute_pos

                        _pm_start = time.time()
                        _pm_pad_id = (self.tokenizer.pad_token_id
                                      if self.tokenizer.pad_token_id is not None
                                      else self.tokenizer.eos_token_id)
                        prompt_dim = batch.batch['prompts'].shape[1]
                        guidance_types = batch.non_tensor_batch.get(
                            'guidance_type',
                            np.array(['original'] * len(batch.batch['input_ids']), dtype=object))
                        questions = batch.non_tensor_batch.get('question', None)
                        hint_level_names = batch.non_tensor_batch.get('hint_level_name', None)
                        prefix_fractions = batch.non_tensor_batch.get('prefix_fraction', None)

                        _n_masked = 0
                        _n_masked_hint = 0
                        _n_masked_prefix = 0

                        for _pm_idx in range(len(guidance_types)):
                            _gt = str(guidance_types[_pm_idx])
                            if _gt not in ('hint', 'prefix'):
                                continue

                            # 1. Save current augmented prompt text for IS ratio computation
                            _aug_prompt_ids = batch.batch['prompts'][_pm_idx]
                            _aug_valid = _aug_prompt_ids[_aug_prompt_ids != _pm_pad_id]
                            _aug_text = self.tokenizer.decode(
                                _aug_valid.tolist(), skip_special_tokens=False)
                            hinted_prompt_texts[_pm_idx] = _aug_text

                            # 2. Build original (un-augmented) prompt
                            _question = questions[_pm_idx] if questions is not None else ''
                            _orig_messages = [{"role": "user", "content": _question}]
                            _enable_thinking = self.config.data.get("enable_thinking", None)
                            _orig_text = apply_chat_template_compat(
                                self.tokenizer, _orig_messages, enable_thinking=_enable_thinking,
                                tokenize=False, add_generation_prompt=True)
                            _orig_ids = self.tokenizer.encode(
                                _orig_text, add_special_tokens=False)

                            # Truncate from left if exceeds prompt dimension
                            if len(_orig_ids) > prompt_dim:
                                _orig_ids = _orig_ids[-prompt_dim:]

                            # 3. Left-pad to prompt_dim
                            _orig_len = len(_orig_ids)
                            _pad_len = prompt_dim - _orig_len
                            _new_prompt = torch.full(
                                (prompt_dim,), _pm_pad_id,
                                dtype=batch.batch['prompts'].dtype,
                                device=batch.batch['prompts'].device)
                            _new_prompt[_pad_len:] = torch.tensor(
                                _orig_ids,
                                dtype=batch.batch['prompts'].dtype,
                                device=batch.batch['prompts'].device)

                            _new_attn = torch.zeros(
                                prompt_dim,
                                dtype=batch.batch['attention_mask'].dtype,
                                device=batch.batch['attention_mask'].device)
                            _new_attn[_pad_len:] = 1

                            # 4. Replace prompt in batch tensors
                            batch.batch['prompts'][_pm_idx] = _new_prompt
                            batch.batch['input_ids'][_pm_idx, :prompt_dim] = _new_prompt
                            batch.batch['attention_mask'][_pm_idx, :prompt_dim] = _new_attn
                            batch.batch['position_ids'][_pm_idx] = _pm_compute_pos(
                                batch.batch['attention_mask'][_pm_idx].unsqueeze(0)).squeeze(0)

                            # 5. Mark for IS tracking
                            hint_replaced[_pm_idx] = True
                            if hint_level_names is not None and str(hint_level_names[_pm_idx]):
                                hint_level_map[_pm_idx] = str(hint_level_names[_pm_idx])
                            elif prefix_fractions is not None and str(prefix_fractions[_pm_idx]):
                                hint_level_map[_pm_idx] = f"prefix_{prefix_fractions[_pm_idx]}"
                            else:
                                hint_level_map[_pm_idx] = _gt

                            _n_masked += 1
                            if _gt == 'hint':
                                _n_masked_hint += 1
                            else:
                                _n_masked_prefix += 1

                        _pm_elapsed = time.time() - _pm_start
                        metrics["custom/n_prompt_masked"] = _n_masked
                        metrics["custom/n_prompt_masked_hint"] = _n_masked_hint
                        metrics["custom/n_prompt_masked_prefix"] = _n_masked_prefix
                        metrics["custom/hook1_hint_gen_seconds"] = 0.0
                        print(f"[Hook1] Prompt masked: {_n_masked} guided rows "
                              f"(hint={_n_masked_hint}, prefix={_n_masked_prefix}) "
                              f"swapped to original prompt ({_pm_elapsed:.2f}s)")
                    # ===== END PROMPT MASKED MODE =====

                    # 2-7: Adaptive hint injection (only if enabled and there are zero-reward groups)
                    hints_cfg_for_epoch = self.config.get('hints', {})
                    hint_start_epoch = hints_cfg_for_epoch.get('start_epoch', 1) if hints_cfg_for_epoch else 1
                    if self.hint_generator is not None and len(zero_meta) > 0 and current_epoch < hint_start_epoch and _pm_hint_mode != 'prompt_masked':
                        print(f"[Hook1] Skipping hints: epoch {current_epoch} < start_epoch {hint_start_epoch} "
                              f"({len(zero_meta)} zero-reward groups unassisted)")
                        metrics["custom/hints_epoch_gated"] = 1
                    if self.hint_generator is not None and len(zero_meta) > 0 and current_epoch >= hint_start_epoch and _pm_hint_mode != 'prompt_masked':
                        from baseline import extract_answer_math, answers_match_math

                        _hint_start = time.time()
                        n_gpus = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                        hints_cfg = self.config.get('hints', {})
                        prompt_mode = hints_cfg.get('prompt_mode', 'original') if hints_cfg else 'original'

                        # --- PASS 2: Generate hints ---
                        questions = [m['question'] for m in zero_meta]
                        solutions = [m['solution'] for m in zero_meta]

                        # Extract a random failed trajectory per zero-reward group (for conditional hint mode)
                        failed_trajectories = None
                        hint_mode = hints_cfg.get('hint_mode', 'solution_aware')

                        if hint_mode == 'hierarchical':
                            # ============================================================
                            # HIERARCHICAL HINT ESCALATION: L1 -> L2 -> L3 -> L4 cascade
                            # ============================================================
                            from verl_grpo.hints.hint_generator import HIERARCHICAL_HINT_LEVELS
                            import random

                            hint_levels = hints_cfg.get('hint_levels', HIERARCHICAL_HINT_LEVELS)
                            hint_n_candidates = hints_cfg.get('hint_n_candidates', 2)

                            # --- Extract one random failed trajectory per zero-reward group ---
                            failed_trajectories = []
                            _diag_hint_data = {}  # pidx -> cascade info (for diagnostic logging)
                            _diag_enabled = bool(self.config.trainer.get("diagnostic_log_dir", None))
                            for pidx_ft, meta in enumerate(zero_meta):
                                random_idx = random.choice(meta['indices'].tolist())
                                response_ids = batch.batch['responses'][random_idx]
                                valid_ids = response_ids[response_ids != pad_token_id]
                                trajectory_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                                failed_trajectories.append(trajectory_text)
                                if _diag_enabled:
                                    # Save ALL n original trajectories before cascade overwrites them
                                    _orig_trajs = []
                                    for _oi, _gi in enumerate(meta['indices']):
                                        _resp = batch.batch['responses'][_gi]
                                        _valid = _resp[_resp != pad_token_id]
                                        _text = self.tokenizer.decode(_valid.tolist(), skip_special_tokens=True)
                                        _pred = extract_answer_math(_text)
                                        _orig_trajs.append({
                                            "index": _oi,
                                            "text": _text,
                                            "predicted_answer": str(_pred) if _pred else None,
                                            "correct": False,  # all zero-reward
                                        })
                                    _diag_hint_data[pidx_ft] = {
                                        'original_trajectories': _orig_trajs,
                                        'failed_trajectory_used': trajectory_text,
                                        'failed_trajectory_index': int(random_idx),
                                        'per_level': {},
                                        'solved_at_level': None,
                                    }

                            # --- BATCH GENERATE all hints for all levels at once ---

                            if self.frontier_client is not None:
                                # Frontier API: raw text prompts, one response per prompt
                                all_raw_prompts, prompt_map = self.hint_generator.build_all_hierarchical_prompts_raw(
                                    levels=hint_levels,
                                    questions=questions,
                                    solutions=solutions,
                                    failed_trajectories=failed_trajectories,
                                )
                                print(f"[Hook1-Hierarchy] Generating {len(all_raw_prompts)} hint prompts "
                                      f"({len(hint_levels)} levels × {len(zero_meta)} problems) "
                                      f"via frontier API ({self.frontier_client.model})")

                                hint_responses, api_stats = self.frontier_client.generate_hints(
                                    prompts=all_raw_prompts,
                                    temperature=hints_cfg.get('hint_temperature', 0.7),
                                    max_tokens=hints_cfg.get('hint_max_tokens', 512),
                                )

                                # Track API health metrics
                                metrics["custom/frontier_api_retries"] = api_stats.get('total_retries', 0)
                                metrics["custom/frontier_api_failures"] = api_stats.get('total_failures', 0)

                                # Parse: one response per prompt (no n_candidates grouping)
                                hints_by_level = {}  # {level: {problem_idx: hint_text}}
                                for i, (level, pidx) in enumerate(prompt_map):
                                    hint = self.hint_generator.parse_hierarchical_hint(hint_responses[i])
                                    hints_by_level.setdefault(level, {})[pidx] = hint

                            elif self.hint_client is not None:
                                # External hint server: text-in/text-out
                                all_hint_prompts, prompt_map = self.hint_generator.build_all_hierarchical_prompts(
                                    levels=hint_levels,
                                    questions=questions,
                                    solutions=solutions,
                                    failed_trajectories=failed_trajectories,
                                )
                                print(f"[Hook1-Hierarchy] Generating {len(all_hint_prompts)} hint prompts "
                                      f"({len(hint_levels)} levels × {len(zero_meta)} problems) "
                                      f"via external server (n={hint_n_candidates})")
                                hint_responses = self.hint_client.generate_hints(
                                    prompt_texts=all_hint_prompts,
                                    n=hint_n_candidates,
                                    temperature=hints_cfg.get('hint_temperature', 0.7),
                                    max_tokens=hints_cfg.get('hint_max_tokens', 512),
                                )
                                # Parse: group by (level, problem)
                                hints_by_level = {}  # {level: {problem_idx: hint_text}}
                                for i, (level, pidx) in enumerate(prompt_map):
                                    group = hint_responses[i*hint_n_candidates:(i+1)*hint_n_candidates]
                                    hint = self.hint_generator.parse_hierarchical_hints_from_texts(group)
                                    hints_by_level.setdefault(level, {})[pidx] = hint
                            else:
                                # Embedded vLLM: tokenize, pad, generate, unpad
                                all_hint_prompts, prompt_map = self.hint_generator.build_all_hierarchical_prompts(
                                    levels=hint_levels,
                                    questions=questions,
                                    solutions=solutions,
                                    failed_trajectories=failed_trajectories,
                                )
                                hint_gen_batch = self.prompt_builder.tokenize_prompts(all_hint_prompts)
                                hint_gen_batch.meta_info['do_sample'] = True
                                orig_hint_prompt_count = len(all_hint_prompts)
                                hint_gen_batch, hint_pad_size = pad_dataproto_to_divisor(hint_gen_batch, n_gpus)

                                print(f"[Hook1-Hierarchy] Generating {orig_hint_prompt_count} hint prompts "
                                      f"({len(hint_levels)} levels × {len(zero_meta)} problems) "
                                      f"(padded to {len(hint_gen_batch)}, n={n})")
                                hint_gen_output = self.actor_rollout_wg.generate_sequences(hint_gen_batch)

                                if hint_pad_size > 0:
                                    hint_gen_output = hint_gen_output[:orig_hint_prompt_count * n]

                                # Parse: use only hint_n_candidates from each group of n responses
                                hints_by_level = {}
                                for i, (level, pidx) in enumerate(prompt_map):
                                    group_responses = hint_gen_output.batch['responses'][i*n:i*n+hint_n_candidates]
                                    hint = self.hint_generator.parse_hierarchical_hints_from_group(
                                        group_responses, hint_n_candidates, self.tokenizer)
                                    hints_by_level.setdefault(level, {})[pidx] = hint

                            _hint_gen_elapsed = time.time() - _hint_start
                            metrics["custom/hook1_hint_gen_seconds"] = _hint_gen_elapsed

                            # --- CASCADE application: L1 -> L2 -> L3 -> L4 ---
                            remaining_indices = set(range(len(zero_meta)))
                            n_improved = 0
                            beyond_reach_indices = []
                            n_skipped_beyond_reach = 0

                            for level in hint_levels:
                                if not remaining_indices:
                                    break

                                # Collect hints for remaining problems at this level
                                valid_for_regen = []
                                for pidx in sorted(remaining_indices):
                                    hint = hints_by_level.get(level, {}).get(pidx)
                                    if hint:
                                        valid_for_regen.append((pidx, zero_meta[pidx], hint))
                                    elif _diag_enabled and pidx in _diag_hint_data:
                                        # Record that this level had no valid hint for this problem
                                        _diag_hint_data[pidx]['per_level'][level] = {
                                            'hint_prompt': None,
                                            'hint_parsed': None,
                                            'hinted_prompt': None,
                                            'regen_trajectories': [],
                                            'n_correct_regen': 0,
                                            'solved': False,
                                            'skipped_reason': 'hint_parse_failed',
                                        }

                                if not valid_for_regen:
                                    metrics[f"custom/n_solved_at_{level}"] = 0
                                    continue

                                # Build hinted prompts + re-generate
                                hinted_texts = [
                                    self.prompt_builder.build_hinted_prompt_text(meta['question'], hint)
                                    for _, meta, hint in valid_for_regen
                                ]
                                regen_batch = self.prompt_builder.tokenize_prompts(hinted_texts)
                                regen_batch.meta_info['do_sample'] = True
                                orig_count = len(valid_for_regen)
                                regen_batch, regen_pad = pad_dataproto_to_divisor(regen_batch, n_gpus)

                                print(f"[Hook1-Hierarchy] {level}: re-generating with {orig_count} hinted prompts "
                                      f"(padded to {len(regen_batch)}, n={n})")
                                regen_output = self.actor_rollout_wg.generate_sequences(regen_batch)

                                if regen_pad > 0:
                                    regen_output = regen_output[:orig_count * n]

                                # Check rewards + replace
                                solved_at_level = []
                                for k, (pidx, meta, hint) in enumerate(valid_for_regen):
                                    regen_responses = regen_output.batch['responses'][k*n:(k+1)*n]
                                    regen_attn = regen_output.batch['attention_mask'][k*n:(k+1)*n, -response_length:]

                                    any_correct = False
                                    _diag_regen_trajs = []  # For diagnostic logging
                                    _diag_n_correct_regen = 0
                                    for i in range(n):
                                        text = self.tokenizer.decode(
                                            regen_responses[i].tolist(), skip_special_tokens=True)
                                        predicted = extract_answer_math(text)
                                        is_correct = (predicted is not None
                                                      and answers_match_math(predicted, meta['ground_truth']))
                                        if is_correct:
                                            any_correct = True
                                            _diag_n_correct_regen += 1
                                        if _diag_enabled:
                                            _diag_regen_trajs.append({
                                                "index": i,
                                                "text": text,
                                                "predicted_answer": str(predicted) if predicted else None,
                                                "correct": is_correct,
                                            })

                                    # Record diagnostic data for this level
                                    if _diag_enabled and pidx in _diag_hint_data:
                                        # Get the raw hint prompt text
                                        hint_prompt_raw = self.hint_generator._format_hierarchical_prompt(
                                            level, meta['question'], meta['solution'],
                                            failed_trajectories[pidx])
                                        _diag_hint_data[pidx]['per_level'][level] = {
                                            'hint_prompt': hint_prompt_raw,
                                            'hint_parsed': hint,
                                            'hinted_prompt': hinted_texts[k],
                                            'regen_trajectories': _diag_regen_trajs,
                                            'n_correct_regen': _diag_n_correct_regen,
                                            'solved': any_correct,
                                        }

                                    if any_correct:
                                        # Check skip_beyond_reach
                                        _pid = batch.non_tensor_batch['problem_id'][meta['indices'][0]]
                                        if (skip_beyond_reach_cfg and
                                                self._should_skip_beyond_reach(
                                                    _pid, _diag_n_correct_regen, n,
                                                    current_epoch, skip_beyond_reach_cfg)):
                                            beyond_reach_indices.extend(meta['indices'].tolist())
                                            solved_at_level.append(pidx)
                                            n_skipped_beyond_reach += 1
                                            if _diag_enabled and pidx in _diag_hint_data:
                                                _diag_hint_data[pidx]['solved_at_level'] = f"{level}_SKIPPED_BEYOND_REACH"
                                            print(f"[Hook1-Hierarchy] Skipping beyond-reach problem "
                                                  f"{_pid} at {level} (SR={_diag_n_correct_regen}/{n})")
                                            continue

                                        main_indices = meta['indices']
                                        batch.batch['responses'][main_indices] = regen_responses
                                        batch.batch['input_ids'][main_indices, -response_length:] = regen_responses
                                        batch.batch['attention_mask'][main_indices, -response_length:] = regen_attn
                                        batch.batch['position_ids'][main_indices] = compute_position_id_with_mask(
                                            batch.batch['attention_mask'][main_indices])
                                        # Log-prob gap: mark replaced trajectories
                                        hint_replaced[main_indices] = True
                                        for _ri in main_indices:
                                            hint_level_map[_ri] = level
                                            hinted_prompt_texts[_ri] = hinted_texts[k]

                                        # Update reward_tensor for this group
                                        for i, idx in enumerate(main_indices):
                                            text = self.tokenizer.decode(
                                                regen_responses[i].tolist(), skip_special_tokens=True)
                                            predicted = extract_answer_math(text)
                                            correct = (predicted is not None
                                                       and answers_match_math(predicted, meta['ground_truth']))
                                            reward_tensor[idx] = 0.0
                                            if correct:
                                                valid_pos = torch.where(regen_attn[i] > 0)[0]
                                                if len(valid_pos) > 0:
                                                    reward_tensor[idx, valid_pos[-1].item()] = 1.0

                                        solved_at_level.append(pidx)
                                        n_improved += 1
                                        if _diag_enabled and pidx in _diag_hint_data:
                                            _diag_hint_data[pidx]['solved_at_level'] = level

                                remaining_indices -= set(solved_at_level)
                                metrics[f"custom/n_solved_at_{level}"] = len(solved_at_level)
                                print(f"[Hook1-Hierarchy] {level}: {len(solved_at_level)} solved, "
                                      f"{len(remaining_indices)} remaining")

                            # Recompute response_mask for the full batch
                            batch.batch['response_mask'] = compute_response_mask(batch)

                            metrics["custom/n_still_zero_after_hierarchy"] = len(remaining_indices)
                            metrics["custom/n_hints_improved"] = n_improved
                            metrics["custom/n_hints_still_zero"] = len(zero_meta) - n_improved

                            # Per-problem metrics: count correct after hint for converted groups
                            converted_correct_counts = []
                            reward_sums_updated = reward_tensor.sum(-1)
                            for pidx in range(len(zero_meta)):
                                if pidx not in remaining_indices:
                                    meta = zero_meta[pidx]
                                    nc = int((reward_sums_updated[meta['indices']] > 0).sum())
                                    converted_correct_counts.append(nc)
                            if converted_correct_counts:
                                metrics["custom/mean_correct_after_hint"] = np.mean(converted_correct_counts)
                                # Distribution buckets
                                for bucket in range(1, n + 1):
                                    metrics[f"custom/n_converted_{bucket}of{n}"] = sum(
                                        1 for c in converted_correct_counts if c == bucket)

                            # Skip unhintable: problems that remained zero after all hint levels
                            n_skipped_unhintable = 0
                            if skip_unhintable_cfg and remaining_indices:
                                for pidx in remaining_indices:
                                    skip_indices.extend(zero_meta[pidx]['indices'].tolist())
                                    n_skipped_unhintable += 1
                            metrics["custom/n_skipped_unhintable"] = n_skipped_unhintable

                            # Skip beyond_reach: add to skip_indices (same as unhintable)
                            if beyond_reach_indices:
                                skip_indices.extend(beyond_reach_indices)
                            metrics["custom/n_skipped_beyond_reach"] = n_skipped_beyond_reach

                            _hint_total_elapsed = time.time() - _hint_start
                            metrics["custom/hook1_hint_total_seconds"] = _hint_total_elapsed
                            print(f"[Hook1-Hierarchy] Done: {len(zero_meta)} zero-reward, "
                                  f"{n_improved} improved, {n_skipped_beyond_reach} beyond-reach "
                                  f"across {len(hint_levels)} levels "
                                  f"({_hint_total_elapsed:.1f}s)")

                        elif hint_mode == 'hierarchical2':
                            # ============================================================
                            # HIERARCHICAL2 HINT ESCALATION: Configurable L1-L5 cascade
                            # with anti-repetition system prompt and hint probability gating.
                            # L5 uses reference solution directly (no LLM generation).
                            # ============================================================
                            from verl_grpo.hints.hint_generator import HIERARCHICAL2_HINT_LEVELS
                            import random

                            hint_levels = hints_cfg.get('hint_levels', HIERARCHICAL2_HINT_LEVELS)
                            hint_n_candidates = hints_cfg.get('hint_n_candidates', 2)
                            hint_probability = hints_cfg.get('hint_probability', 1.0)
                            anti_rep_prompt = hints_cfg.get('anti_repetition_system_prompt', None)
                            has_l5 = "L5_solution" in hint_levels
                            gen_levels = [l for l in hint_levels if l != "L5_solution"]
                            parallel_cascade = hints_cfg.get('parallel_cascade', False)

                            # --- Probability gate: skip hints for this batch? ---
                            if random.random() >= hint_probability:
                                print(f"[Hook1-Hierarchical2] Skipping hints "
                                      f"(probability={hint_probability:.2f})")
                                metrics["custom/n_hints_improved"] = 0
                                metrics["custom/n_hints_still_zero"] = len(zero_meta)
                                metrics["custom/hint_skipped_by_probability"] = 1
                            else:
                                metrics["custom/hint_skipped_by_probability"] = 0

                                # --- Extract one random failed trajectory per zero-reward group ---
                                failed_trajectories = []
                                _diag_hint_data = {}
                                _diag_enabled = bool(self.config.trainer.get("diagnostic_log_dir", None))
                                for pidx_ft, meta in enumerate(zero_meta):
                                    random_idx = random.choice(meta['indices'].tolist())
                                    response_ids = batch.batch['responses'][random_idx]
                                    valid_ids = response_ids[response_ids != pad_token_id]
                                    trajectory_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                                    failed_trajectories.append(trajectory_text)
                                    if _diag_enabled:
                                        _orig_trajs = []
                                        for _oi, _gi in enumerate(meta['indices']):
                                            _resp = batch.batch['responses'][_gi]
                                            _valid = _resp[_resp != pad_token_id]
                                            _text = self.tokenizer.decode(_valid.tolist(), skip_special_tokens=True)
                                            _pred = extract_answer_math(_text)
                                            _orig_trajs.append({
                                                "index": _oi,
                                                "text": _text,
                                                "predicted_answer": str(_pred) if _pred else None,
                                                "correct": False,
                                            })
                                        _diag_hint_data[pidx_ft] = {
                                            'original_trajectories': _orig_trajs,
                                            'failed_trajectory_used': trajectory_text,
                                            'failed_trajectory_index': int(random_idx),
                                            'per_level': {},
                                            'solved_at_level': None,
                                        }

                                # --- BATCH GENERATE hints for gen_levels (L1-L4, excluding L5) ---
                                if gen_levels:
                                    if self.frontier_client is not None:
                                        all_raw_prompts, prompt_map = self.hint_generator.build_all_hierarchical_prompts_raw(
                                            levels=gen_levels,
                                            questions=questions,
                                            solutions=solutions,
                                            failed_trajectories=failed_trajectories,
                                        )
                                        print(f"[Hook1-Hierarchical2] Generating {len(all_raw_prompts)} hint prompts "
                                              f"({len(gen_levels)} levels × {len(zero_meta)} problems) "
                                              f"via frontier API ({self.frontier_client.model})")

                                        hint_responses, api_stats = self.frontier_client.generate_hints(
                                            prompts=all_raw_prompts,
                                            temperature=hints_cfg.get('hint_temperature', 0.7),
                                            max_tokens=hints_cfg.get('hint_max_tokens', 512),
                                        )
                                        metrics["custom/frontier_api_retries"] = api_stats.get('total_retries', 0)
                                        metrics["custom/frontier_api_failures"] = api_stats.get('total_failures', 0)

                                        hints_by_level = {}
                                        for i, (level, pidx) in enumerate(prompt_map):
                                            hint = self.hint_generator.parse_hierarchical_hint(hint_responses[i])
                                            hints_by_level.setdefault(level, {})[pidx] = hint

                                    elif self.hint_client is not None:
                                        all_hint_prompts, prompt_map = self.hint_generator.build_all_hierarchical_prompts(
                                            levels=gen_levels,
                                            questions=questions,
                                            solutions=solutions,
                                            failed_trajectories=failed_trajectories,
                                        )
                                        print(f"[Hook1-Hierarchical2] Generating {len(all_hint_prompts)} hint prompts "
                                              f"({len(gen_levels)} levels × {len(zero_meta)} problems) "
                                              f"via external server (n={hint_n_candidates})")
                                        hint_responses = self.hint_client.generate_hints(
                                            prompt_texts=all_hint_prompts,
                                            n=hint_n_candidates,
                                            temperature=hints_cfg.get('hint_temperature', 0.7),
                                            max_tokens=hints_cfg.get('hint_max_tokens', 512),
                                        )
                                        hints_by_level = {}
                                        for i, (level, pidx) in enumerate(prompt_map):
                                            group = hint_responses[i*hint_n_candidates:(i+1)*hint_n_candidates]
                                            hint = self.hint_generator.parse_hierarchical_hints_from_texts(group)
                                            hints_by_level.setdefault(level, {})[pidx] = hint
                                    else:
                                        all_hint_prompts, prompt_map = self.hint_generator.build_all_hierarchical_prompts(
                                            levels=gen_levels,
                                            questions=questions,
                                            solutions=solutions,
                                            failed_trajectories=failed_trajectories,
                                        )
                                        hint_gen_batch = self.prompt_builder.tokenize_prompts(all_hint_prompts)
                                        hint_gen_batch.meta_info['do_sample'] = True
                                        orig_hint_prompt_count = len(all_hint_prompts)
                                        hint_gen_batch, hint_pad_size = pad_dataproto_to_divisor(hint_gen_batch, n_gpus)

                                        print(f"[Hook1-Hierarchical2] Generating {orig_hint_prompt_count} hint prompts "
                                              f"({len(gen_levels)} levels × {len(zero_meta)} problems) "
                                              f"(padded to {len(hint_gen_batch)}, n={n})")
                                        hint_gen_output = self.actor_rollout_wg.generate_sequences(hint_gen_batch)

                                        if hint_pad_size > 0:
                                            hint_gen_output = hint_gen_output[:orig_hint_prompt_count * n]

                                        hints_by_level = {}
                                        for i, (level, pidx) in enumerate(prompt_map):
                                            group_responses = hint_gen_output.batch['responses'][i*n:i*n+hint_n_candidates]
                                            hint = self.hint_generator.parse_hierarchical_hints_from_group(
                                                group_responses, hint_n_candidates, self.tokenizer)
                                            hints_by_level.setdefault(level, {})[pidx] = hint
                                else:
                                    hints_by_level = {}

                                _hint_gen_elapsed = time.time() - _hint_start
                                metrics["custom/hook1_hint_gen_seconds"] = _hint_gen_elapsed

                                # --- CASCADE over gen_levels: PARALLEL or SEQUENTIAL ---
                                remaining_indices = set(range(len(zero_meta)))
                                n_improved = 0
                                _solved_level_map = {}  # pidx -> level that solved it (independent of diagnostics)
                                beyond_reach_indices = []  # indices to skip from actor update
                                n_skipped_beyond_reach = 0
                                regen_n = n_train if n_train < n_explore else n  # trajectories per regen prompt

                                if parallel_cascade:
                                    # ============================================================
                                    # PARALLEL CASCADE: batch ALL regen levels in ONE vLLM call
                                    # When n_train < n_explore, regen uses n_train trajectories
                                    # and success = "n_correct_regen > m" (improvement over original)
                                    # ============================================================
                                    all_levels_ordered = gen_levels + (["L5_solution"] if has_l5 else [])

                                    # --- Step 1: Build ALL regen items (all problems × all levels) ---
                                    regen_items = []  # list of (pidx, level, meta, hint_text, is_l5)
                                    for level in gen_levels:
                                        for pidx in range(len(zero_meta)):
                                            hint = hints_by_level.get(level, {}).get(pidx)
                                            if hint:
                                                regen_items.append((pidx, level, zero_meta[pidx], hint, False))
                                            elif _diag_enabled and pidx in _diag_hint_data:
                                                _diag_hint_data[pidx]['per_level'][level] = {
                                                    'hint_prompt': None,
                                                    'hint_parsed': None,
                                                    'hinted_prompt': None,
                                                    'regen_trajectories': [],
                                                    'n_correct_regen': 0,
                                                    'solved': False,
                                                    'skipped_reason': 'hint_parse_failed',
                                                }
                                    if has_l5:
                                        for pidx in range(len(zero_meta)):
                                            meta = zero_meta[pidx]
                                            solution = meta['solution']
                                            if solution and solution.strip():
                                                regen_items.append((pidx, "L5_solution", meta, solution, True))
                                            elif _diag_enabled and pidx in _diag_hint_data:
                                                _diag_hint_data[pidx]['per_level']["L5_solution"] = {
                                                    'hint_prompt': None,
                                                    'hint_parsed': None,
                                                    'hinted_prompt': None,
                                                    'regen_trajectories': [],
                                                    'n_correct_regen': 0,
                                                    'solved': False,
                                                    'skipped_reason': 'no_reference_solution',
                                                }

                                    if regen_items:
                                        # --- Step 2: Build ALL hinted prompts + single vLLM call ---
                                        hinted_texts_all = []
                                        for pidx, level, meta, hint_text, is_l5 in regen_items:
                                            if is_l5:
                                                hinted_texts_all.append(
                                                    self.prompt_builder.build_hinted_prompt_text_v2(
                                                        meta['question'], hint_text,
                                                        system_prompt=None, is_l5=True))
                                            else:
                                                hinted_texts_all.append(
                                                    self.prompt_builder.build_hinted_prompt_text_v2(
                                                        meta['question'], hint_text,
                                                        system_prompt=anti_rep_prompt, is_l5=False))

                                        regen_batch_all = self.prompt_builder.tokenize_prompts(hinted_texts_all)
                                        regen_batch_all.meta_info['do_sample'] = True
                                        if regen_n != n:
                                            regen_batch_all.meta_info['n_override'] = regen_n
                                        orig_count_all = len(regen_items)
                                        regen_batch_all, regen_pad_all = pad_dataproto_to_divisor(regen_batch_all, n_gpus)

                                        print(f"[Hook1-Hierarchical2-Parallel] Re-generating {orig_count_all} "
                                              f"hinted prompts ({len(all_levels_ordered)} levels × "
                                              f"{len(zero_meta)} problems, padded to {len(regen_batch_all)}, n={regen_n})")
                                        regen_output_all = self.actor_rollout_wg.generate_sequences(regen_batch_all)

                                        # Validate n_override took effect + remove padding
                                        expected_regen_size = orig_count_all * regen_n
                                        padded_expected = (orig_count_all + regen_pad_all) * regen_n
                                        actual_size = len(regen_output_all)
                                        if actual_size != padded_expected:
                                            raise RuntimeError(
                                                f"[Hierarchical2-Parallel] Regen output size mismatch: "
                                                f"expected {padded_expected} ({orig_count_all}+{regen_pad_all} prompts × {regen_n}), "
                                                f"got {actual_size}. Is n_override support missing from verl? "
                                                f"Ensure verl is on branch custom-patches-v0.4.1.")
                                        regen_output_all = regen_output_all[:expected_regen_size]

                                        # Dimension alignment for hinted prompt_mode
                                        if prompt_mode == 'hinted':
                                            main_prompt_len = batch.batch['prompts'].shape[1]
                                            regen_prompt_len = regen_output_all.batch['prompts'].shape[1]
                                            target_len = max(main_prompt_len, regen_prompt_len)
                                            if main_prompt_len < target_len:
                                                _expand_batch_prompt_dim(batch, target_len, pad_token_id)
                                            if regen_prompt_len < target_len:
                                                _expand_batch_prompt_dim(regen_output_all, target_len, pad_token_id)

                                        # --- Step 3: Check correctness for ALL (problem, level) pairs ---
                                        regen_results = {}  # (pidx, level) -> dict
                                        for k, (pidx, level, meta, hint_text, is_l5) in enumerate(regen_items):
                                            regen_responses = regen_output_all.batch['responses'][k*regen_n:(k+1)*regen_n]
                                            regen_attn = regen_output_all.batch['attention_mask'][k*regen_n:(k+1)*regen_n, -response_length:]

                                            any_correct = False
                                            _diag_regen_trajs = []
                                            _diag_n_correct_regen = 0
                                            for i in range(regen_n):
                                                text = self.tokenizer.decode(
                                                    regen_responses[i].tolist(), skip_special_tokens=True)
                                                predicted = extract_answer_math(text)
                                                is_correct = (predicted is not None
                                                              and answers_match_math(predicted, meta['ground_truth']))
                                                if is_correct:
                                                    any_correct = True
                                                    _diag_n_correct_regen += 1
                                                if _diag_enabled:
                                                    _diag_regen_trajs.append({
                                                        "index": i,
                                                        "text": text,
                                                        "predicted_answer": str(predicted) if predicted else None,
                                                        "correct": is_correct,
                                                    })

                                            regen_results[(pidx, level)] = {
                                                'any_correct': any_correct,
                                                'n_correct': _diag_n_correct_regen,
                                                'regen_responses': regen_responses,
                                                'regen_attn': regen_attn,
                                                '_diag_regen_trajs': _diag_regen_trajs,
                                                'hint_text': hint_text,
                                                'hinted_prompt_text': hinted_texts_all[k],
                                                'is_l5': is_l5,
                                            }
                                            if prompt_mode == 'hinted':
                                                regen_slice = slice(k*regen_n, (k+1)*regen_n)
                                                regen_results[(pidx, level)]['regen_prompts'] = regen_output_all.batch['prompts'][regen_slice]
                                                regen_results[(pidx, level)]['regen_input_ids'] = regen_output_all.batch['input_ids'][regen_slice]
                                                regen_results[(pidx, level)]['regen_attention_mask'] = regen_output_all.batch['attention_mask'][regen_slice]
                                                regen_results[(pidx, level)]['regen_position_ids'] = regen_output_all.batch['position_ids'][regen_slice]

                                        # --- Step 4: For each problem, pick lowest solving level ---
                                        solved_at_counts = {lvl: 0 for lvl in all_levels_ordered}
                                        for pidx in range(len(zero_meta)):
                                            meta = zero_meta[pidx]
                                            problem_solved = False

                                            # Record diagnostics for ALL levels (richer than sequential)
                                            for level in all_levels_ordered:
                                                result = regen_results.get((pidx, level))
                                                if result is None:
                                                    continue  # hint_parse_failed or no_reference_solution (already recorded)

                                                if _diag_enabled and pidx in _diag_hint_data:
                                                    if result['is_l5']:
                                                        hint_prompt_raw = None
                                                        hint_parsed = f"[reference_solution, len={len(result['hint_text'])}]"
                                                    else:
                                                        hint_prompt_raw = self.hint_generator._format_hierarchical_prompt(
                                                            level, meta['question'], meta['solution'],
                                                            failed_trajectories[pidx])
                                                        hint_parsed = result['hint_text']
                                                    _diag_hint_data[pidx]['per_level'][level] = {
                                                        'hint_prompt': hint_prompt_raw,
                                                        'hint_parsed': hint_parsed,
                                                        'hinted_prompt': result['hinted_prompt_text'],
                                                        'regen_trajectories': result['_diag_regen_trajs'],
                                                        'n_correct_regen': result['n_correct'],
                                                        'solved': result['any_correct'],
                                                    }

                                                # Pick lowest level that IMPROVES over original
                                                # Success = n_correct_regen > m (not just any_correct)
                                                m_original = meta['n_correct_original']
                                                if not problem_solved and result['n_correct'] > m_original:
                                                    # Check skip_beyond_reach
                                                    _pid = batch.non_tensor_batch['problem_id'][meta['indices'][0]]
                                                    if (skip_beyond_reach_cfg and
                                                            self._should_skip_beyond_reach(
                                                                _pid, result['n_correct'], regen_n,
                                                                current_epoch, skip_beyond_reach_cfg)):
                                                        beyond_reach_indices.extend(meta['indices'].tolist())
                                                        n_skipped_beyond_reach += 1
                                                        solved_at_counts[level] += 1
                                                        remaining_indices.discard(pidx)
                                                        problem_solved = True
                                                        if _diag_enabled and pidx in _diag_hint_data:
                                                            _diag_hint_data[pidx]['solved_at_level'] = f"{level}_SKIPPED_BEYOND_REACH"
                                                        print(f"[Hook1-Hierarchical2-Parallel] Skipping beyond-reach problem "
                                                              f"{_pid} at {level} (SR={result['n_correct']}/{regen_n})")
                                                        continue

                                                    # Replace first regen_n slots with this level's regen output
                                                    main_indices = meta['indices']
                                                    replace_indices = main_indices[:regen_n]
                                                    batch.batch['responses'][replace_indices] = result['regen_responses']

                                                    if prompt_mode == 'hinted':
                                                        # Replace full prompt+response (dims already aligned)
                                                        batch.batch['prompts'][replace_indices] = result['regen_prompts']
                                                        batch.batch['input_ids'][replace_indices] = result['regen_input_ids']
                                                        batch.batch['attention_mask'][replace_indices] = result['regen_attention_mask']
                                                        batch.batch['position_ids'][replace_indices] = result['regen_position_ids']
                                                    else:
                                                        # Original mode: keep original prompts, replace response part only
                                                        batch.batch['input_ids'][replace_indices, -response_length:] = result['regen_responses']
                                                        batch.batch['attention_mask'][replace_indices, -response_length:] = result['regen_attn']
                                                        batch.batch['position_ids'][replace_indices] = compute_position_id_with_mask(
                                                            batch.batch['attention_mask'][replace_indices])

                                                    # Mark replaced trajectories for log-prob gap analysis
                                                    hint_replaced[replace_indices] = True
                                                    for _ri in replace_indices:
                                                        hint_level_map[_ri] = level
                                                        hinted_prompt_texts[_ri] = result['hinted_prompt_text']

                                                    # Update rewards for replaced slots
                                                    for i, idx in enumerate(replace_indices):
                                                        text = self.tokenizer.decode(
                                                            result['regen_responses'][i].tolist(), skip_special_tokens=True)
                                                        predicted = extract_answer_math(text)
                                                        correct = (predicted is not None
                                                                   and answers_match_math(predicted, meta['ground_truth']))
                                                        reward_tensor[idx] = 0.0
                                                        if correct:
                                                            valid_pos = torch.where(result['regen_attn'][i] > 0)[0]
                                                            if len(valid_pos) > 0:
                                                                reward_tensor[idx, valid_pos[-1].item()] = 1.0

                                                    solved_at_counts[level] += 1
                                                    n_improved += 1
                                                    _solved_level_map[pidx] = level
                                                    remaining_indices.discard(pidx)
                                                    problem_solved = True
                                                    if _diag_enabled and pidx in _diag_hint_data:
                                                        _diag_hint_data[pidx]['solved_at_level'] = level

                                        # --- Fallback for hint-triggered problems where cascade didn't improve ---
                                        # Rearrange best regen_n trajectories into first regen_n slots
                                        # so that n_train subsample (uid_indices[:n_train]) picks them up
                                        n_cascade_fallback = 0
                                        if regen_n < n_explore:
                                            for pidx in list(remaining_indices):
                                                meta = zero_meta[pidx]
                                                main_indices = meta['indices']
                                                m = meta['n_correct_original']
                                                rewards_group = reward_tensor[main_indices].sum(-1)
                                                correct_mask = (rewards_group > 0)
                                                correct_idx = main_indices[correct_mask].tolist()
                                                incorrect_idx = main_indices[~correct_mask].tolist()
                                                # Keep m correct + (regen_n - m) incorrect = regen_n total
                                                keep = correct_idx[:regen_n] + incorrect_idx[:max(0, regen_n - len(correct_idx[:regen_n]))]
                                                keep = sorted(keep[:regen_n])
                                                # Rearrange: swap best into first regen_n positions
                                                first_slots = main_indices[:regen_n].tolist()
                                                if set(keep) != set(first_slots):
                                                    # Need to swap: copy keep into first_slots
                                                    swap_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
                                                    if prompt_mode == 'hinted':
                                                        swap_keys.append('prompts')
                                                    for batch_key in swap_keys:
                                                        temp = batch.batch[batch_key][keep].clone()
                                                        batch.batch[batch_key][main_indices[:regen_n]] = temp
                                                    # Also swap rewards
                                                    temp_rew = reward_tensor[keep].clone()
                                                    reward_tensor[main_indices[:regen_n]] = temp_rew
                                                n_cascade_fallback += 1

                                        # Emit per-level metrics
                                        for level in all_levels_ordered:
                                            metrics[f"custom/n_solved_at_{level}"] = solved_at_counts[level]
                                        metrics["custom/n_cascade_fallback"] = n_cascade_fallback
                                        print(f"[Hook1-Hierarchical2-Parallel] Done cascade: "
                                              f"{n_improved} improved, {n_cascade_fallback} fallback, "
                                              f"{n_skipped_beyond_reach} beyond-reach "
                                              f"(regen_n={regen_n}, "
                                              f"{', '.join(f'{l}={solved_at_counts[l]}' for l in all_levels_ordered)})")

                                    else:
                                        # No valid regen items (all hints failed to parse + no L5)
                                        for level in all_levels_ordered:
                                            metrics[f"custom/n_solved_at_{level}"] = 0

                                else:
                                    # --- SEQUENTIAL CASCADE over gen_levels (L1-L4): with system prompt ---
                                    for level in gen_levels:
                                        if not remaining_indices:
                                            break

                                        valid_for_regen = []
                                        for pidx in sorted(remaining_indices):
                                            hint = hints_by_level.get(level, {}).get(pidx)
                                            if hint:
                                                valid_for_regen.append((pidx, zero_meta[pidx], hint))
                                            elif _diag_enabled and pidx in _diag_hint_data:
                                                _diag_hint_data[pidx]['per_level'][level] = {
                                                    'hint_prompt': None,
                                                    'hint_parsed': None,
                                                    'hinted_prompt': None,
                                                    'regen_trajectories': [],
                                                    'n_correct_regen': 0,
                                                    'solved': False,
                                                    'skipped_reason': 'hint_parse_failed',
                                                }

                                        if not valid_for_regen:
                                            metrics[f"custom/n_solved_at_{level}"] = 0
                                            continue

                                        # Build hinted prompts with v2 (system prompt for L1-L4)
                                        hinted_texts = [
                                            self.prompt_builder.build_hinted_prompt_text_v2(
                                                meta['question'], hint,
                                                system_prompt=anti_rep_prompt, is_l5=False)
                                            for _, meta, hint in valid_for_regen
                                        ]
                                        regen_batch = self.prompt_builder.tokenize_prompts(hinted_texts)
                                        regen_batch.meta_info['do_sample'] = True
                                        orig_count = len(valid_for_regen)
                                        regen_batch, regen_pad = pad_dataproto_to_divisor(regen_batch, n_gpus)

                                        print(f"[Hook1-Hierarchical2] {level}: re-generating with {orig_count} hinted prompts "
                                              f"(padded to {len(regen_batch)}, n={n})")
                                        regen_output = self.actor_rollout_wg.generate_sequences(regen_batch)

                                        if regen_pad > 0:
                                            regen_output = regen_output[:orig_count * n]

                                        # Dimension alignment for hinted prompt_mode
                                        if prompt_mode == 'hinted':
                                            main_prompt_len = batch.batch['prompts'].shape[1]
                                            regen_prompt_len = regen_output.batch['prompts'].shape[1]
                                            target_len = max(main_prompt_len, regen_prompt_len)
                                            if main_prompt_len < target_len:
                                                _expand_batch_prompt_dim(batch, target_len, pad_token_id)
                                            if regen_prompt_len < target_len:
                                                _expand_batch_prompt_dim(regen_output, target_len, pad_token_id)

                                        # Check rewards + replace
                                        solved_at_level = []
                                        for k, (pidx, meta, hint) in enumerate(valid_for_regen):
                                            regen_responses = regen_output.batch['responses'][k*n:(k+1)*n]
                                            regen_attn = regen_output.batch['attention_mask'][k*n:(k+1)*n, -response_length:]

                                            any_correct = False
                                            _diag_regen_trajs = []
                                            _diag_n_correct_regen = 0
                                            for i in range(n):
                                                text = self.tokenizer.decode(
                                                    regen_responses[i].tolist(), skip_special_tokens=True)
                                                predicted = extract_answer_math(text)
                                                is_correct = (predicted is not None
                                                              and answers_match_math(predicted, meta['ground_truth']))
                                                if is_correct:
                                                    any_correct = True
                                                    _diag_n_correct_regen += 1
                                                if _diag_enabled:
                                                    _diag_regen_trajs.append({
                                                        "index": i,
                                                        "text": text,
                                                        "predicted_answer": str(predicted) if predicted else None,
                                                        "correct": is_correct,
                                                    })

                                            if _diag_enabled and pidx in _diag_hint_data:
                                                hint_prompt_raw = self.hint_generator._format_hierarchical_prompt(
                                                    level, meta['question'], meta['solution'],
                                                    failed_trajectories[pidx])
                                                _diag_hint_data[pidx]['per_level'][level] = {
                                                    'hint_prompt': hint_prompt_raw,
                                                    'hint_parsed': hint,
                                                    'hinted_prompt': hinted_texts[k],
                                                    'regen_trajectories': _diag_regen_trajs,
                                                    'n_correct_regen': _diag_n_correct_regen,
                                                    'solved': any_correct,
                                                }

                                            if any_correct:
                                                # Check skip_beyond_reach: skip if hint SR is too high
                                                _pid = batch.non_tensor_batch['problem_id'][meta['indices'][0]]
                                                if (skip_beyond_reach_cfg and
                                                        self._should_skip_beyond_reach(
                                                            _pid, _diag_n_correct_regen, n,
                                                            current_epoch, skip_beyond_reach_cfg)):
                                                    # Beyond reach: do NOT replace, add to skip list
                                                    beyond_reach_indices.extend(meta['indices'].tolist())
                                                    solved_at_level.append(pidx)  # remove from cascade
                                                    n_skipped_beyond_reach += 1
                                                    if _diag_enabled and pidx in _diag_hint_data:
                                                        _diag_hint_data[pidx]['solved_at_level'] = f"{level}_SKIPPED_BEYOND_REACH"
                                                    print(f"[Hook1-Hierarchical2] Skipping beyond-reach problem "
                                                          f"{_pid} at {level} (SR={_diag_n_correct_regen}/{n})")
                                                    continue

                                                main_indices = meta['indices']
                                                batch.batch['responses'][main_indices] = regen_responses

                                                if prompt_mode == 'hinted':
                                                    # Replace full prompt+response (dims already aligned)
                                                    regen_slice = slice(k*n, (k+1)*n)
                                                    batch.batch['prompts'][main_indices] = regen_output.batch['prompts'][regen_slice]
                                                    batch.batch['input_ids'][main_indices] = regen_output.batch['input_ids'][regen_slice]
                                                    batch.batch['attention_mask'][main_indices] = regen_output.batch['attention_mask'][regen_slice]
                                                    batch.batch['position_ids'][main_indices] = regen_output.batch['position_ids'][regen_slice]
                                                else:
                                                    # Original mode: keep original prompts, replace response part only
                                                    batch.batch['input_ids'][main_indices, -response_length:] = regen_responses
                                                    batch.batch['attention_mask'][main_indices, -response_length:] = regen_attn
                                                    batch.batch['position_ids'][main_indices] = compute_position_id_with_mask(
                                                        batch.batch['attention_mask'][main_indices])
                                                # Log-prob gap: mark replaced trajectories
                                                hint_replaced[main_indices] = True
                                                for _ri in main_indices:
                                                    hint_level_map[_ri] = level
                                                    hinted_prompt_texts[_ri] = hinted_texts[k]

                                                for i, idx in enumerate(main_indices):
                                                    text = self.tokenizer.decode(
                                                        regen_responses[i].tolist(), skip_special_tokens=True)
                                                    predicted = extract_answer_math(text)
                                                    correct = (predicted is not None
                                                               and answers_match_math(predicted, meta['ground_truth']))
                                                    reward_tensor[idx] = 0.0
                                                    if correct:
                                                        valid_pos = torch.where(regen_attn[i] > 0)[0]
                                                        if len(valid_pos) > 0:
                                                            reward_tensor[idx, valid_pos[-1].item()] = 1.0

                                                solved_at_level.append(pidx)
                                                n_improved += 1
                                                _solved_level_map[pidx] = level
                                                if _diag_enabled and pidx in _diag_hint_data:
                                                    _diag_hint_data[pidx]['solved_at_level'] = level

                                        remaining_indices -= set(solved_at_level)
                                        metrics[f"custom/n_solved_at_{level}"] = len(solved_at_level)
                                        print(f"[Hook1-Hierarchical2] {level}: {len(solved_at_level)} solved, "
                                              f"{len(remaining_indices)} remaining")

                                    # --- L5 CASCADE STEP (only if L5_solution in hint_levels) ---
                                    if has_l5 and remaining_indices:
                                        level = "L5_solution"
                                        valid_for_regen_l5 = []
                                        for pidx in sorted(remaining_indices):
                                            meta = zero_meta[pidx]
                                            solution = meta['solution']
                                            if solution and solution.strip():
                                                valid_for_regen_l5.append((pidx, meta, solution))
                                            elif _diag_enabled and pidx in _diag_hint_data:
                                                _diag_hint_data[pidx]['per_level'][level] = {
                                                    'hint_prompt': None,
                                                    'hint_parsed': None,
                                                    'hinted_prompt': None,
                                                    'regen_trajectories': [],
                                                    'n_correct_regen': 0,
                                                    'solved': False,
                                                    'skipped_reason': 'no_reference_solution',
                                                }

                                        if valid_for_regen_l5:
                                            # Build L5 prompts (is_l5=True, no system prompt)
                                            hinted_texts_l5 = [
                                                self.prompt_builder.build_hinted_prompt_text_v2(
                                                    meta['question'], solution,
                                                    system_prompt=None, is_l5=True)
                                                for _, meta, solution in valid_for_regen_l5
                                            ]
                                            regen_batch_l5 = self.prompt_builder.tokenize_prompts(hinted_texts_l5)
                                            regen_batch_l5.meta_info['do_sample'] = True
                                            orig_count_l5 = len(valid_for_regen_l5)
                                            regen_batch_l5, regen_pad_l5 = pad_dataproto_to_divisor(regen_batch_l5, n_gpus)

                                            print(f"[Hook1-Hierarchical2] L5_solution: re-generating with "
                                                  f"{orig_count_l5} reference solution prompts "
                                                  f"(padded to {len(regen_batch_l5)}, n={n})")
                                            regen_output_l5 = self.actor_rollout_wg.generate_sequences(regen_batch_l5)

                                            if regen_pad_l5 > 0:
                                                regen_output_l5 = regen_output_l5[:orig_count_l5 * n]

                                            # Dimension alignment for hinted prompt_mode
                                            if prompt_mode == 'hinted':
                                                main_prompt_len = batch.batch['prompts'].shape[1]
                                                regen_prompt_len = regen_output_l5.batch['prompts'].shape[1]
                                                target_len = max(main_prompt_len, regen_prompt_len)
                                                if main_prompt_len < target_len:
                                                    _expand_batch_prompt_dim(batch, target_len, pad_token_id)
                                                if regen_prompt_len < target_len:
                                                    _expand_batch_prompt_dim(regen_output_l5, target_len, pad_token_id)

                                            solved_at_l5 = []
                                            for k, (pidx, meta, solution) in enumerate(valid_for_regen_l5):
                                                regen_responses = regen_output_l5.batch['responses'][k*n:(k+1)*n]
                                                regen_attn = regen_output_l5.batch['attention_mask'][k*n:(k+1)*n, -response_length:]

                                                any_correct = False
                                                _diag_regen_trajs = []
                                                _diag_n_correct_regen = 0
                                                for i in range(n):
                                                    text = self.tokenizer.decode(
                                                        regen_responses[i].tolist(), skip_special_tokens=True)
                                                    predicted = extract_answer_math(text)
                                                    is_correct = (predicted is not None
                                                                  and answers_match_math(predicted, meta['ground_truth']))
                                                    if is_correct:
                                                        any_correct = True
                                                        _diag_n_correct_regen += 1
                                                    if _diag_enabled:
                                                        _diag_regen_trajs.append({
                                                            "index": i,
                                                            "text": text,
                                                            "predicted_answer": str(predicted) if predicted else None,
                                                            "correct": is_correct,
                                                        })

                                                if _diag_enabled and pidx in _diag_hint_data:
                                                    _diag_hint_data[pidx]['per_level'][level] = {
                                                        'hint_prompt': None,  # L5 has no generation prompt
                                                        'hint_parsed': f"[reference_solution, len={len(solution)}]",
                                                        'hinted_prompt': hinted_texts_l5[k],
                                                        'regen_trajectories': _diag_regen_trajs,
                                                        'n_correct_regen': _diag_n_correct_regen,
                                                        'solved': any_correct,
                                                    }

                                                if any_correct:
                                                    # Check skip_beyond_reach for L5
                                                    _pid = batch.non_tensor_batch['problem_id'][meta['indices'][0]]
                                                    if (skip_beyond_reach_cfg and
                                                            self._should_skip_beyond_reach(
                                                                _pid, _diag_n_correct_regen, n,
                                                                current_epoch, skip_beyond_reach_cfg)):
                                                        beyond_reach_indices.extend(meta['indices'].tolist())
                                                        solved_at_l5.append(pidx)
                                                        n_skipped_beyond_reach += 1
                                                        if _diag_enabled and pidx in _diag_hint_data:
                                                            _diag_hint_data[pidx]['solved_at_level'] = "L5_solution_SKIPPED_BEYOND_REACH"
                                                        print(f"[Hook1-Hierarchical2] Skipping beyond-reach problem "
                                                              f"{_pid} at L5 (SR={_diag_n_correct_regen}/{n})")
                                                        continue

                                                    main_indices = meta['indices']
                                                    batch.batch['responses'][main_indices] = regen_responses

                                                    if prompt_mode == 'hinted':
                                                        # Replace full prompt+response (dims already aligned)
                                                        regen_slice = slice(k*n, (k+1)*n)
                                                        batch.batch['prompts'][main_indices] = regen_output_l5.batch['prompts'][regen_slice]
                                                        batch.batch['input_ids'][main_indices] = regen_output_l5.batch['input_ids'][regen_slice]
                                                        batch.batch['attention_mask'][main_indices] = regen_output_l5.batch['attention_mask'][regen_slice]
                                                        batch.batch['position_ids'][main_indices] = regen_output_l5.batch['position_ids'][regen_slice]
                                                    else:
                                                        # Original mode: keep original prompts, replace response part only
                                                        batch.batch['input_ids'][main_indices, -response_length:] = regen_responses
                                                        batch.batch['attention_mask'][main_indices, -response_length:] = regen_attn
                                                        batch.batch['position_ids'][main_indices] = compute_position_id_with_mask(
                                                            batch.batch['attention_mask'][main_indices])

                                                    # Log-prob gap: mark replaced trajectories
                                                    hint_replaced[main_indices] = True
                                                    for _ri in main_indices:
                                                        hint_level_map[_ri] = level  # = "L5_solution"
                                                        hinted_prompt_texts[_ri] = hinted_texts_l5[k]

                                                    for i, idx in enumerate(main_indices):
                                                        text = self.tokenizer.decode(
                                                            regen_responses[i].tolist(), skip_special_tokens=True)
                                                        predicted = extract_answer_math(text)
                                                        correct = (predicted is not None
                                                                   and answers_match_math(predicted, meta['ground_truth']))
                                                        reward_tensor[idx] = 0.0
                                                        if correct:
                                                            valid_pos = torch.where(regen_attn[i] > 0)[0]
                                                            if len(valid_pos) > 0:
                                                                reward_tensor[idx, valid_pos[-1].item()] = 1.0

                                                    solved_at_l5.append(pidx)
                                                    n_improved += 1
                                                    _solved_level_map[pidx] = level
                                                    if _diag_enabled and pidx in _diag_hint_data:
                                                        _diag_hint_data[pidx]['solved_at_level'] = level

                                            remaining_indices -= set(solved_at_l5)
                                            metrics[f"custom/n_solved_at_L5_solution"] = len(solved_at_l5)
                                            print(f"[Hook1-Hierarchical2] L5_solution: {len(solved_at_l5)} solved, "
                                                  f"{len(remaining_indices)} remaining")
                                        else:
                                            metrics[f"custom/n_solved_at_L5_solution"] = 0

                                # Recompute response_mask for the full batch
                                batch.batch['response_mask'] = compute_response_mask(batch)

                                metrics["custom/n_still_zero_after_hierarchy"] = len(remaining_indices)
                                metrics["custom/n_hints_improved"] = n_improved
                                metrics["custom/n_hints_still_zero"] = len(zero_meta) - n_improved

                                # Per-problem metrics: count correct after hint for converted groups
                                # Only count in the first regen_n slots (the ones kept for training)
                                converted_correct_counts = []
                                reward_sums_updated = reward_tensor.sum(-1)
                                for pidx in range(len(zero_meta)):
                                    if pidx not in remaining_indices:
                                        meta = zero_meta[pidx]
                                        kept_indices = meta['indices'][:regen_n]
                                        nc = int((reward_sums_updated[kept_indices] > 0).sum())
                                        converted_correct_counts.append(nc)
                                if converted_correct_counts:
                                    metrics["custom/mean_correct_after_hint"] = np.mean(converted_correct_counts)
                                    for bucket in range(1, regen_n + 1):
                                        metrics[f"custom/n_converted_{bucket}of{regen_n}"] = sum(
                                            1 for c in converted_correct_counts if c == bucket)

                                # Per-problem tracking: which level succeeded for each problem
                                for pidx, meta in enumerate(zero_meta):
                                    problem_id = batch.non_tensor_batch['problem_id'][meta['indices'][0]]
                                    if problem_id:
                                        self._replacement_problem_log[problem_id] = {
                                            'step': self.global_steps,
                                            'epoch': current_epoch,
                                            'hint_level_succeeded': _solved_level_map.get(pidx),
                                            'replaced': pidx not in remaining_indices,
                                        }

                                # Skip unhintable
                                n_skipped_unhintable = 0
                                if skip_unhintable_cfg and remaining_indices:
                                    for pidx in remaining_indices:
                                        skip_indices.extend(zero_meta[pidx]['indices'].tolist())
                                        n_skipped_unhintable += 1
                                metrics["custom/n_skipped_unhintable"] = n_skipped_unhintable

                                # Skip beyond_reach: add to skip_indices (same as unhintable)
                                if beyond_reach_indices:
                                    skip_indices.extend(beyond_reach_indices)
                                metrics["custom/n_skipped_beyond_reach"] = n_skipped_beyond_reach

                                # Update cumulative hint stats for skip_permanently_stuck
                                for pidx, meta in enumerate(zero_meta):
                                    pid = meta.get('question', '')  # use question as fallback id
                                    # Try to get problem_id from batch
                                    _pid_idx = meta['indices'][0]
                                    pid = batch.non_tensor_batch['problem_id'][_pid_idx] if 'problem_id' in batch.non_tensor_batch else pid
                                    if pid and pid in self._problem_cumulative:
                                        cum = self._problem_cumulative[pid]
                                        cum['n_hint_observations'] += 1
                                        if pidx not in remaining_indices:
                                            # Hint cascade solved this problem
                                            cum['n_hint_correct'] += 1
                                            cum['n_hint_total'] += 1
                                        else:
                                            cum['n_hint_total'] += 1

                                # Mark permanently_stuck problems
                                if skip_permanently_stuck_cfg and skip_permanently_stuck_cfg.get('enabled', False):
                                    min_obs = skip_permanently_stuck_cfg.get('min_observations', 3)
                                    min_hint_obs = skip_permanently_stuck_cfg.get('min_hint_observations', 2)
                                    start_step = skip_permanently_stuck_cfg.get('start_step', 10)
                                    if self.global_steps >= start_step:
                                        for pidx in remaining_indices:
                                            _pid_idx2 = zero_meta[pidx]['indices'][0]
                                            pid2 = batch.non_tensor_batch['problem_id'][_pid_idx2] if 'problem_id' in batch.non_tensor_batch else None
                                            if pid2 and pid2 in self._problem_cumulative:
                                                cum2 = self._problem_cumulative[pid2]
                                                if (cum2['n_observations'] >= min_obs
                                                        and cum2['n_hint_observations'] >= min_hint_obs
                                                        and cum2['n_correct_total'] == 0
                                                        and cum2['n_hint_correct'] == 0):
                                                    cum2['permanently_skipped'] = True

                                _hint_total_elapsed = time.time() - _hint_start
                                metrics["custom/hook1_hint_total_seconds"] = _hint_total_elapsed
                                print(f"[Hook1-Hierarchical2] Done: {len(zero_meta)} zero-reward, "
                                      f"{n_improved} improved, {n_skipped_beyond_reach} beyond-reach "
                                      f"across {len(hint_levels)} levels "
                                      f"(L5={has_l5}, sys_prompt={'yes' if anti_rep_prompt else 'no'}, "
                                      f"p={hint_probability:.2f}, {_hint_total_elapsed:.1f}s)")

                        elif hint_mode == 'hierarchical_replacement':
                            # ============================================================
                            # HIERARCHICAL REPLACEMENT: L1 -> L2 -> L3 -> L4 cascade
                            # Scaf-GRPO style: replace only 1 trajectory per group
                            # ============================================================
                            from verl_grpo.hints.hint_generator import HIERARCHICAL_HINT_LEVELS
                            import random

                            hint_levels = hints_cfg.get('hint_levels', HIERARCHICAL_HINT_LEVELS)
                            hint_n_candidates = hints_cfg.get('hint_n_candidates', 2)

                            # --- Phase A: Extract one random failed trajectory per zero-reward group ---
                            failed_trajectories = []
                            for meta in zero_meta:
                                random_idx = random.choice(meta['indices'].tolist())
                                response_ids = batch.batch['responses'][random_idx]
                                valid_ids = response_ids[response_ids != pad_token_id]
                                trajectory_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                                failed_trajectories.append(trajectory_text)

                            # --- Phase B: Batch generate ALL hints for ALL levels at once ---
                            # (Duplicated from hierarchical mode for safety — avoids touching working code)

                            if self.frontier_client is not None:
                                # Frontier API: raw text prompts, one response per prompt
                                all_raw_prompts, prompt_map = self.hint_generator.build_all_hierarchical_prompts_raw(
                                    levels=hint_levels,
                                    questions=questions,
                                    solutions=solutions,
                                    failed_trajectories=failed_trajectories,
                                )
                                print(f"[Hook1-HierReplace] Generating {len(all_raw_prompts)} hint prompts "
                                      f"({len(hint_levels)} levels × {len(zero_meta)} problems) "
                                      f"via frontier API ({self.frontier_client.model})")

                                hint_responses, api_stats = self.frontier_client.generate_hints(
                                    prompts=all_raw_prompts,
                                    temperature=hints_cfg.get('hint_temperature', 0.7),
                                    max_tokens=hints_cfg.get('hint_max_tokens', 512),
                                )

                                metrics["custom/frontier_api_retries"] = api_stats.get('total_retries', 0)
                                metrics["custom/frontier_api_failures"] = api_stats.get('total_failures', 0)

                                hints_by_level = {}
                                for i, (level, pidx) in enumerate(prompt_map):
                                    hint = self.hint_generator.parse_hierarchical_hint(hint_responses[i])
                                    hints_by_level.setdefault(level, {})[pidx] = hint

                            elif self.hint_client is not None:
                                # External hint server: text-in/text-out
                                all_hint_prompts, prompt_map = self.hint_generator.build_all_hierarchical_prompts(
                                    levels=hint_levels,
                                    questions=questions,
                                    solutions=solutions,
                                    failed_trajectories=failed_trajectories,
                                )
                                print(f"[Hook1-HierReplace] Generating {len(all_hint_prompts)} hint prompts "
                                      f"({len(hint_levels)} levels × {len(zero_meta)} problems) "
                                      f"via external server (n={hint_n_candidates})")
                                hint_responses = self.hint_client.generate_hints(
                                    prompt_texts=all_hint_prompts,
                                    n=hint_n_candidates,
                                    temperature=hints_cfg.get('hint_temperature', 0.7),
                                    max_tokens=hints_cfg.get('hint_max_tokens', 512),
                                )
                                hints_by_level = {}
                                for i, (level, pidx) in enumerate(prompt_map):
                                    group = hint_responses[i*hint_n_candidates:(i+1)*hint_n_candidates]
                                    hint = self.hint_generator.parse_hierarchical_hints_from_texts(group)
                                    hints_by_level.setdefault(level, {})[pidx] = hint
                            else:
                                # Embedded vLLM: tokenize, pad, generate, unpad
                                all_hint_prompts, prompt_map = self.hint_generator.build_all_hierarchical_prompts(
                                    levels=hint_levels,
                                    questions=questions,
                                    solutions=solutions,
                                    failed_trajectories=failed_trajectories,
                                )
                                hint_gen_batch = self.prompt_builder.tokenize_prompts(all_hint_prompts)
                                hint_gen_batch.meta_info['do_sample'] = True
                                orig_hint_prompt_count = len(all_hint_prompts)
                                hint_gen_batch, hint_pad_size = pad_dataproto_to_divisor(hint_gen_batch, n_gpus)

                                print(f"[Hook1-HierReplace] Generating {orig_hint_prompt_count} hint prompts "
                                      f"({len(hint_levels)} levels × {len(zero_meta)} problems) "
                                      f"(padded to {len(hint_gen_batch)}, n={n})")
                                hint_gen_output = self.actor_rollout_wg.generate_sequences(hint_gen_batch)

                                if hint_pad_size > 0:
                                    hint_gen_output = hint_gen_output[:orig_hint_prompt_count * n]

                                hints_by_level = {}
                                for i, (level, pidx) in enumerate(prompt_map):
                                    group_responses = hint_gen_output.batch['responses'][i*n:i*n+hint_n_candidates]
                                    hint = self.hint_generator.parse_hierarchical_hints_from_group(
                                        group_responses, hint_n_candidates, self.tokenizer)
                                    hints_by_level.setdefault(level, {})[pidx] = hint

                            _hint_gen_elapsed = time.time() - _hint_start
                            metrics["custom/hook1_hint_gen_seconds"] = _hint_gen_elapsed

                            # --- Phase C: CASCADE with single-trajectory replacement ---
                            remaining_indices = set(range(len(zero_meta)))
                            n_improved = 0
                            beyond_reach_indices = []
                            n_skipped_beyond_reach = 0
                            level_solved_map = {}       # {problem_idx: level_that_solved_it}
                            level_n_correct_map = {}    # {problem_idx: n_correct_at_success_level}
                            level_attempted_map = {}    # {problem_idx: [levels_attempted]}

                            for level in hint_levels:
                                if not remaining_indices:
                                    break

                                # Track which problems attempted this level
                                for pidx in list(remaining_indices):
                                    level_attempted_map.setdefault(pidx, []).append(level)

                                # Collect hints for remaining problems at this level
                                valid_for_regen = []
                                for pidx in sorted(remaining_indices):
                                    hint = hints_by_level.get(level, {}).get(pidx)
                                    if hint:
                                        valid_for_regen.append((pidx, zero_meta[pidx], hint))

                                if not valid_for_regen:
                                    metrics[f"custom/n_solved_at_{level}"] = 0
                                    continue

                                # Build hinted prompts + re-generate
                                hinted_texts = [
                                    self.prompt_builder.build_hinted_prompt_text(meta['question'], hint)
                                    for _, meta, hint in valid_for_regen
                                ]
                                regen_batch = self.prompt_builder.tokenize_prompts(hinted_texts)
                                regen_batch.meta_info['do_sample'] = True
                                orig_count = len(valid_for_regen)
                                regen_batch, regen_pad = pad_dataproto_to_divisor(regen_batch, n_gpus)

                                print(f"[Hook1-HierReplace] {level}: re-generating with {orig_count} "
                                      f"hinted prompts (padded to {len(regen_batch)}, n={n})")
                                regen_output = self.actor_rollout_wg.generate_sequences(regen_batch)

                                if regen_pad > 0:
                                    regen_output = regen_output[:orig_count * n]

                                # Check ALL n re-gen trajectories for correctness + selective replacement
                                solved_at_level = []
                                for k, (pidx, meta, hint) in enumerate(valid_for_regen):
                                    regen_responses = regen_output.batch['responses'][k*n:(k+1)*n]
                                    regen_attn = regen_output.batch['attention_mask'][k*n:(k+1)*n, -response_length:]

                                    # Find ALL correct trajectories among the n re-generated ones
                                    correct_regen_indices = []
                                    for i in range(n):
                                        text = self.tokenizer.decode(
                                            regen_responses[i].tolist(), skip_special_tokens=True)
                                        predicted = extract_answer_math(text)
                                        if predicted is not None and answers_match_math(
                                                predicted, meta['ground_truth']):
                                            correct_regen_indices.append(i)

                                    if not correct_regen_indices:
                                        continue

                                    # Check skip_beyond_reach: SR uses all n regen trajectories
                                    _pid = batch.non_tensor_batch['problem_id'][meta['indices'][0]]
                                    _n_correct_regen = len(correct_regen_indices)
                                    if (skip_beyond_reach_cfg and
                                            self._should_skip_beyond_reach(
                                                _pid, _n_correct_regen, n,
                                                current_epoch, skip_beyond_reach_cfg)):
                                        beyond_reach_indices.extend(meta['indices'].tolist())
                                        solved_at_level.append(pidx)
                                        n_skipped_beyond_reach += 1
                                        print(f"[Hook1-HierReplace] Skipping beyond-reach problem "
                                              f"{_pid} at {level} (SR={_n_correct_regen}/{n})")
                                        continue

                                    # === KEY DIFFERENCE: Replace only 1 trajectory ===
                                    # Randomly pick ONE correct trajectory from re-gen
                                    chosen_regen_idx = random.choice(correct_regen_indices)
                                    # Randomly pick ONE index from original group to replace
                                    main_indices = meta['indices']
                                    replace_pos = random.choice(range(len(main_indices)))
                                    orig_batch_idx = main_indices[replace_pos]

                                    # Replace response tokens
                                    batch.batch['responses'][orig_batch_idx] = regen_responses[chosen_regen_idx]
                                    # Replace response portion of input_ids (keep original prompt)
                                    batch.batch['input_ids'][orig_batch_idx, -response_length:] = \
                                        regen_responses[chosen_regen_idx]
                                    # Replace attention mask for response portion
                                    batch.batch['attention_mask'][orig_batch_idx, -response_length:] = \
                                        regen_attn[chosen_regen_idx]
                                    # Recompute position IDs for the replaced row
                                    batch.batch['position_ids'][orig_batch_idx] = compute_position_id_with_mask(
                                        batch.batch['attention_mask'][orig_batch_idx:orig_batch_idx+1])[0]
                                    # Log-prob gap: mark replaced trajectory
                                    hint_replaced[orig_batch_idx] = True
                                    hint_level_map[orig_batch_idx] = level
                                    hinted_prompt_texts[orig_batch_idx] = hinted_texts[k]

                                    # Update reward_tensor ONLY for the replaced index
                                    reward_tensor[orig_batch_idx] = 0.0
                                    valid_pos = torch.where(regen_attn[chosen_regen_idx] > 0)[0]
                                    if len(valid_pos) > 0:
                                        reward_tensor[orig_batch_idx, valid_pos[-1].item()] = 1.0

                                    solved_at_level.append(pidx)
                                    level_solved_map[pidx] = level
                                    level_n_correct_map[pidx] = len(correct_regen_indices)
                                    n_improved += 1

                                remaining_indices -= set(solved_at_level)
                                metrics[f"custom/n_solved_at_{level}"] = len(solved_at_level)
                                print(f"[Hook1-HierReplace] {level}: {len(solved_at_level)} solved, "
                                      f"{len(remaining_indices)} remaining")

                            # Recompute response_mask for the full batch
                            batch.batch['response_mask'] = compute_response_mask(batch)

                            # --- Aggregate metrics ---
                            metrics["custom/n_still_zero_after_hierarchy"] = len(remaining_indices)
                            metrics["custom/n_hints_improved"] = n_improved
                            metrics["custom/n_hints_still_zero"] = len(zero_meta) - n_improved
                            metrics["custom/n_replaced_total"] = n_improved  # 1 replacement per improved group

                            # Per-problem correct count after replacement (should be exactly 1 for each)
                            converted_correct_counts = []
                            reward_sums_updated = reward_tensor.sum(-1)
                            for pidx in range(len(zero_meta)):
                                if pidx not in remaining_indices:
                                    meta = zero_meta[pidx]
                                    nc = int((reward_sums_updated[meta['indices']] > 0).sum())
                                    converted_correct_counts.append(nc)
                            if converted_correct_counts:
                                metrics["custom/mean_correct_after_hint"] = np.mean(converted_correct_counts)
                                for bucket in range(1, n + 1):
                                    metrics[f"custom/n_converted_{bucket}of{n}"] = sum(
                                        1 for c in converted_correct_counts if c == bucket)

                            # Level distribution for solved problems
                            for level in hint_levels:
                                count = sum(1 for v in level_solved_map.values() if v == level)
                                metrics[f"custom/replacement_level_{level}"] = count

                            # Per-problem metric logging
                            for pidx, meta in enumerate(zero_meta):
                                problem_id = batch.non_tensor_batch['problem_id'][meta['indices'][0]]
                                if problem_id:
                                    self._replacement_problem_log[problem_id] = {
                                        'step': self.global_steps,
                                        'epoch': current_epoch,
                                        'hint_level_succeeded': level_solved_map.get(pidx),
                                        'levels_attempted': level_attempted_map.get(pidx, []),
                                        'n_correct_at_success_level': level_n_correct_map.get(pidx, 0),
                                        'replaced': pidx not in remaining_indices,
                                    }

                            # Skip unhintable: problems that remained zero after all hint levels
                            n_skipped_unhintable = 0
                            if skip_unhintable_cfg and remaining_indices:
                                for pidx in remaining_indices:
                                    skip_indices.extend(zero_meta[pidx]['indices'].tolist())
                                    n_skipped_unhintable += 1
                            metrics["custom/n_skipped_unhintable"] = n_skipped_unhintable

                            # Skip beyond_reach: add to skip_indices (same as unhintable)
                            if beyond_reach_indices:
                                skip_indices.extend(beyond_reach_indices)
                            metrics["custom/n_skipped_beyond_reach"] = n_skipped_beyond_reach

                            _hint_total_elapsed = time.time() - _hint_start
                            metrics["custom/hook1_hint_total_seconds"] = _hint_total_elapsed
                            print(f"[Hook1-HierReplace] Done: {len(zero_meta)} zero-reward, "
                                  f"{n_improved} improved (1 replaced each), "
                                  f"{n_skipped_beyond_reach} beyond-reach across "
                                  f"{len(hint_levels)} levels ({_hint_total_elapsed:.1f}s)")

                        elif hint_mode == 'replacement2':
                            # ============================================================
                            # REPLACEMENT2: hierarchical2-style v2 prompts + Scaf-GRPO
                            # minimal replacement. All hint levels (L1-L5) evaluated in
                            # parallel, lowest successful level selected, replace_count
                            # (1-2) trajectories replaced. IS correction compatible.
                            # ============================================================
                            from verl_grpo.hints.hint_generator import HIERARCHICAL2_HINT_LEVELS
                            import random

                            # --- Phase A: Configuration + Probability Gating ---
                            hint_levels = hints_cfg.get('hint_levels', HIERARCHICAL2_HINT_LEVELS)
                            hint_n_candidates = hints_cfg.get('hint_n_candidates', 2)
                            hint_probability = hints_cfg.get('hint_probability', 1.0)
                            anti_rep_prompt = hints_cfg.get('anti_repetition_system_prompt', None)
                            replace_count = hints_cfg.get('replace_count', None)  # None = replace all correct candidates
                            has_l5 = "L5_solution" in hint_levels
                            gen_levels = [l for l in hint_levels if l != "L5_solution"]
                            all_levels_ordered = gen_levels + (["L5_solution"] if has_l5 else [])

                            if random.random() >= hint_probability:
                                print(f"[Hook1-Replacement2] Skipping hints "
                                      f"(probability={hint_probability:.2f})")
                                metrics["custom/n_hints_improved"] = 0
                                metrics["custom/n_hints_still_zero"] = len(zero_meta)
                                metrics["custom/hint_skipped_by_probability"] = 1
                            else:
                                metrics["custom/hint_skipped_by_probability"] = 0

                                # --- Phase B: Extract failed trajectories ---
                                failed_trajectories = []
                                _diag_hint_data = {}
                                _diag_enabled = bool(self.config.trainer.get("diagnostic_log_dir", None))
                                for pidx_ft, meta in enumerate(zero_meta):
                                    random_idx = random.choice(meta['indices'].tolist())
                                    response_ids = batch.batch['responses'][random_idx]
                                    valid_ids = response_ids[response_ids != pad_token_id]
                                    trajectory_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                                    failed_trajectories.append(trajectory_text)
                                    if _diag_enabled:
                                        _orig_trajs = []
                                        for _oi, _gi in enumerate(meta['indices']):
                                            _resp = batch.batch['responses'][_gi]
                                            _valid = _resp[_resp != pad_token_id]
                                            _text = self.tokenizer.decode(_valid.tolist(), skip_special_tokens=True)
                                            _pred = extract_answer_math(_text)
                                            _orig_trajs.append({
                                                "index": _oi,
                                                "text": _text,
                                                "predicted_answer": str(_pred) if _pred else None,
                                                "correct": False,
                                            })
                                        _diag_hint_data[pidx_ft] = {
                                            'original_trajectories': _orig_trajs,
                                            'failed_trajectory_used': trajectory_text,
                                            'failed_trajectory_index': int(random_idx),
                                            'per_level': {},
                                            'solved_at_level': None,
                                        }

                                # --- Phase C: Batch generate hints for L1-L4 ---
                                if gen_levels:
                                    if self.frontier_client is not None:
                                        all_raw_prompts, prompt_map = self.hint_generator.build_all_hierarchical_prompts_raw(
                                            levels=gen_levels,
                                            questions=questions,
                                            solutions=solutions,
                                            failed_trajectories=failed_trajectories,
                                        )
                                        print(f"[Hook1-Replacement2] Generating {len(all_raw_prompts)} hint prompts "
                                              f"({len(gen_levels)} levels x {len(zero_meta)} problems) "
                                              f"via frontier API ({self.frontier_client.model})")
                                        hint_responses, api_stats = self.frontier_client.generate_hints(
                                            prompts=all_raw_prompts,
                                            temperature=hints_cfg.get('hint_temperature', 0.7),
                                            max_tokens=hints_cfg.get('hint_max_tokens', 512),
                                        )
                                        metrics["custom/frontier_api_retries"] = api_stats.get('total_retries', 0)
                                        metrics["custom/frontier_api_failures"] = api_stats.get('total_failures', 0)
                                        hints_by_level = {}
                                        for i, (level, pidx) in enumerate(prompt_map):
                                            hint = self.hint_generator.parse_hierarchical_hint(hint_responses[i])
                                            hints_by_level.setdefault(level, {})[pidx] = hint

                                    elif self.hint_client is not None:
                                        all_hint_prompts, prompt_map = self.hint_generator.build_all_hierarchical_prompts(
                                            levels=gen_levels,
                                            questions=questions,
                                            solutions=solutions,
                                            failed_trajectories=failed_trajectories,
                                        )
                                        print(f"[Hook1-Replacement2] Generating {len(all_hint_prompts)} hint prompts "
                                              f"({len(gen_levels)} levels x {len(zero_meta)} problems) "
                                              f"via external server (n={hint_n_candidates})")
                                        hint_responses = self.hint_client.generate_hints(
                                            prompt_texts=all_hint_prompts,
                                            n=hint_n_candidates,
                                            temperature=hints_cfg.get('hint_temperature', 0.7),
                                            max_tokens=hints_cfg.get('hint_max_tokens', 512),
                                        )
                                        hints_by_level = {}
                                        for i, (level, pidx) in enumerate(prompt_map):
                                            group = hint_responses[i*hint_n_candidates:(i+1)*hint_n_candidates]
                                            hint = self.hint_generator.parse_hierarchical_hints_from_texts(group)
                                            hints_by_level.setdefault(level, {})[pidx] = hint
                                    else:
                                        all_hint_prompts, prompt_map = self.hint_generator.build_all_hierarchical_prompts(
                                            levels=gen_levels,
                                            questions=questions,
                                            solutions=solutions,
                                            failed_trajectories=failed_trajectories,
                                        )
                                        hint_gen_batch = self.prompt_builder.tokenize_prompts(all_hint_prompts)
                                        hint_gen_batch.meta_info['do_sample'] = True
                                        orig_hint_prompt_count = len(all_hint_prompts)
                                        hint_gen_batch, hint_pad_size = pad_dataproto_to_divisor(hint_gen_batch, n_gpus)

                                        print(f"[Hook1-Replacement2] Generating {orig_hint_prompt_count} hint prompts "
                                              f"({len(gen_levels)} levels x {len(zero_meta)} problems) "
                                              f"(padded to {len(hint_gen_batch)}, n={n})")
                                        hint_gen_output = self.actor_rollout_wg.generate_sequences(hint_gen_batch)

                                        if hint_pad_size > 0:
                                            hint_gen_output = hint_gen_output[:orig_hint_prompt_count * n]

                                        hints_by_level = {}
                                        for i, (level, pidx) in enumerate(prompt_map):
                                            group_responses = hint_gen_output.batch['responses'][i*n:i*n+hint_n_candidates]
                                            hint = self.hint_generator.parse_hierarchical_hints_from_group(
                                                group_responses, hint_n_candidates, self.tokenizer)
                                            hints_by_level.setdefault(level, {})[pidx] = hint
                                else:
                                    hints_by_level = {}

                                _hint_gen_elapsed = time.time() - _hint_start
                                metrics["custom/hook1_hint_gen_seconds"] = _hint_gen_elapsed

                                # --- Phase D: Build ALL regen items + ONE vLLM call ---
                                _regen_start = time.time()
                                regen_items = []  # (pidx, level, meta, hint_text, is_l5)
                                hinted_texts_all = []

                                for level in gen_levels:
                                    for pidx in range(len(zero_meta)):
                                        hint = hints_by_level.get(level, {}).get(pidx)
                                        if hint:
                                            regen_items.append((pidx, level, zero_meta[pidx], hint, False))
                                            hinted_texts_all.append(
                                                self.prompt_builder.build_hinted_prompt_text_v2(
                                                    zero_meta[pidx]['question'], hint,
                                                    system_prompt=anti_rep_prompt, is_l5=False))
                                        elif _diag_enabled and pidx in _diag_hint_data:
                                            _diag_hint_data[pidx]['per_level'][level] = {
                                                'hint_prompt': None,
                                                'hint_parsed': None,
                                                'hinted_prompt': None,
                                                'regen_trajectories': [],
                                                'n_correct_regen': 0,
                                                'solved': False,
                                                'skipped_reason': 'hint_parse_failed',
                                            }

                                if has_l5:
                                    for pidx in range(len(zero_meta)):
                                        meta = zero_meta[pidx]
                                        solution = meta['solution']
                                        if solution and solution.strip():
                                            regen_items.append((pidx, "L5_solution", meta, solution, True))
                                            hinted_texts_all.append(
                                                self.prompt_builder.build_hinted_prompt_text_v2(
                                                    meta['question'], solution,
                                                    system_prompt=None, is_l5=True))
                                        elif _diag_enabled and pidx in _diag_hint_data:
                                            _diag_hint_data[pidx]['per_level']["L5_solution"] = {
                                                'hint_prompt': None,
                                                'hint_parsed': None,
                                                'hinted_prompt': None,
                                                'regen_trajectories': [],
                                                'n_correct_regen': 0,
                                                'solved': False,
                                                'skipped_reason': 'no_reference_solution',
                                            }

                                if regen_items:
                                    regen_batch_all = self.prompt_builder.tokenize_prompts(hinted_texts_all)
                                    regen_batch_all.meta_info['do_sample'] = True
                                    if hint_n_candidates != n:
                                        regen_batch_all.meta_info['n_override'] = hint_n_candidates
                                    orig_count_all = len(regen_items)
                                    regen_batch_all, regen_pad_all = pad_dataproto_to_divisor(regen_batch_all, n_gpus)

                                    print(f"[Hook1-Replacement2] Re-generating {orig_count_all} "
                                          f"hinted prompts ({len(all_levels_ordered)} levels x "
                                          f"{len(zero_meta)} problems, n_cand={hint_n_candidates}, "
                                          f"padded to {len(regen_batch_all)})")
                                    regen_output_all = self.actor_rollout_wg.generate_sequences(regen_batch_all)

                                    # Validate n_override took effect + remove padding
                                    expected_regen_size = orig_count_all * hint_n_candidates
                                    padded_expected = (orig_count_all + regen_pad_all) * hint_n_candidates
                                    actual_size = len(regen_output_all)
                                    if actual_size != padded_expected:
                                        raise RuntimeError(
                                            f"[Replacement2] Regen output size mismatch: "
                                            f"expected {padded_expected} ({orig_count_all}+{regen_pad_all} prompts × {hint_n_candidates}), "
                                            f"got {actual_size}. Is n_override support missing from verl? "
                                            f"Ensure verl is on branch custom-patches-v0.4.1.")
                                    regen_output_all = regen_output_all[:expected_regen_size]

                                    # --- Phase E: Evaluate correctness at ALL levels ---
                                    regen_results = {}  # (pidx, level) -> dict
                                    for k, (pidx, level, meta, hint_text, is_l5) in enumerate(regen_items):
                                        regen_responses = regen_output_all.batch['responses'][
                                            k*hint_n_candidates:(k+1)*hint_n_candidates]
                                        regen_attn = regen_output_all.batch['attention_mask'][
                                            k*hint_n_candidates:(k+1)*hint_n_candidates, -response_length:]

                                        correct_indices = []
                                        _diag_regen_trajs = []
                                        for i in range(hint_n_candidates):
                                            text = self.tokenizer.decode(
                                                regen_responses[i].tolist(), skip_special_tokens=True)
                                            predicted = extract_answer_math(text)
                                            is_correct = (predicted is not None
                                                          and answers_match_math(predicted, meta['ground_truth']))
                                            if is_correct:
                                                correct_indices.append(i)
                                            if _diag_enabled:
                                                _diag_regen_trajs.append({
                                                    "index": i,
                                                    "text": text,
                                                    "predicted_answer": str(predicted) if predicted else None,
                                                    "correct": is_correct,
                                                })

                                        regen_results[(pidx, level)] = {
                                            'any_correct': len(correct_indices) > 0,
                                            'n_correct': len(correct_indices),
                                            'correct_indices': correct_indices,
                                            'regen_responses': regen_responses,
                                            'regen_attn': regen_attn,
                                            'hint_text': hint_text,
                                            'hinted_prompt_text': hinted_texts_all[k],
                                            'is_l5': is_l5,
                                            '_diag_regen_trajs': _diag_regen_trajs,
                                        }

                                    # --- Phase F: Find lowest successful level + replace ---
                                    remaining_indices = set(range(len(zero_meta)))
                                    n_improved = 0
                                    beyond_reach_indices = []
                                    n_skipped_beyond_reach = 0
                                    solved_at_counts = {lvl: 0 for lvl in all_levels_ordered}
                                    _solved_level_map = {}
                                    level_n_correct_map = {}
                                    candidate_levels_per_problem = []
                                    all_levels_with_correct_map = {}  # pidx -> [levels with correct]
                                    n_replaced_total = 0

                                    for pidx in range(len(zero_meta)):
                                        meta = zero_meta[pidx]

                                        # Count how many levels have correct trajectories (diagnostic)
                                        levels_with_correct = []
                                        for level in all_levels_ordered:
                                            result = regen_results.get((pidx, level))
                                            if result and result['any_correct']:
                                                levels_with_correct.append(level)
                                        candidate_levels_per_problem.append(len(levels_with_correct))
                                        all_levels_with_correct_map[pidx] = levels_with_correct

                                        # Record diagnostics for ALL levels
                                        if _diag_enabled and pidx in _diag_hint_data:
                                            for level in all_levels_ordered:
                                                result = regen_results.get((pidx, level))
                                                if result is None:
                                                    continue
                                                if result['is_l5']:
                                                    hint_prompt_raw = None
                                                    hint_parsed = f"[reference_solution, len={len(result['hint_text'])}]"
                                                else:
                                                    hint_prompt_raw = self.hint_generator._format_hierarchical_prompt(
                                                        level, meta['question'], meta['solution'],
                                                        failed_trajectories[pidx])
                                                    hint_parsed = result['hint_text']
                                                _diag_hint_data[pidx]['per_level'][level] = {
                                                    'hint_prompt': hint_prompt_raw,
                                                    'hint_parsed': hint_parsed,
                                                    'hinted_prompt': result['hinted_prompt_text'],
                                                    'regen_trajectories': result['_diag_regen_trajs'],
                                                    'n_correct_regen': result['n_correct'],
                                                    'solved': result['any_correct'],
                                                }

                                        # Find LOWEST level with any correct trajectory
                                        selected_level = None
                                        selected_result = None
                                        for level in all_levels_ordered:
                                            result = regen_results.get((pidx, level))
                                            if result and result['any_correct']:
                                                selected_level = level
                                                selected_result = result
                                                break

                                        if selected_level is None:
                                            continue  # no level solved this problem

                                        # Check skip_beyond_reach
                                        _pid = batch.non_tensor_batch['problem_id'][meta['indices'][0]]
                                        if (skip_beyond_reach_cfg and
                                                self._should_skip_beyond_reach(
                                                    _pid, selected_result['n_correct'], hint_n_candidates,
                                                    current_epoch, skip_beyond_reach_cfg)):
                                            beyond_reach_indices.extend(meta['indices'].tolist())
                                            n_skipped_beyond_reach += 1
                                            solved_at_counts[selected_level] += 1
                                            remaining_indices.discard(pidx)
                                            if _diag_enabled and pidx in _diag_hint_data:
                                                _diag_hint_data[pidx]['solved_at_level'] = f"{selected_level}_SKIPPED_BEYOND_REACH"
                                            print(f"[Hook1-Replacement2] Skipping beyond-reach problem "
                                                  f"{_pid} at {selected_level} "
                                                  f"(SR={selected_result['n_correct']}/{hint_n_candidates})")
                                            continue

                                        # === REPLACEMENT: replace ALL correct candidates ===
                                        # Default: replace as many as are correct from hint_n_candidates.
                                        # replace_count (if set) acts as optional upper bound.
                                        correct_regen_idx_list = selected_result['correct_indices']
                                        n_to_replace = len(correct_regen_idx_list)
                                        if replace_count is not None:
                                            n_to_replace = min(n_to_replace, replace_count)
                                        n_to_replace = min(n_to_replace, len(meta['indices']))
                                        chosen_regen_indices = random.sample(correct_regen_idx_list, n_to_replace)
                                        replace_positions = random.sample(range(len(meta['indices'])), n_to_replace)

                                        for r_pos, chosen_regen_idx in zip(replace_positions, chosen_regen_indices):
                                            orig_batch_idx = meta['indices'][r_pos]

                                            # Replace response tokens
                                            batch.batch['responses'][orig_batch_idx] = \
                                                selected_result['regen_responses'][chosen_regen_idx]
                                            batch.batch['input_ids'][orig_batch_idx, -response_length:] = \
                                                selected_result['regen_responses'][chosen_regen_idx]
                                            batch.batch['attention_mask'][orig_batch_idx, -response_length:] = \
                                                selected_result['regen_attn'][chosen_regen_idx]
                                            batch.batch['position_ids'][orig_batch_idx] = compute_position_id_with_mask(
                                                batch.batch['attention_mask'][orig_batch_idx:orig_batch_idx+1])[0]

                                            # IS correction: store hinted prompt text
                                            hint_replaced[orig_batch_idx] = True
                                            hint_level_map[orig_batch_idx] = selected_level
                                            hinted_prompt_texts[orig_batch_idx] = selected_result['hinted_prompt_text']

                                            # Update reward for the replaced index
                                            reward_tensor[orig_batch_idx] = 0.0
                                            valid_pos = torch.where(
                                                selected_result['regen_attn'][chosen_regen_idx] > 0)[0]
                                            if len(valid_pos) > 0:
                                                reward_tensor[orig_batch_idx, valid_pos[-1].item()] = 1.0

                                            n_replaced_total += 1

                                        solved_at_counts[selected_level] += 1
                                        n_improved += 1
                                        _solved_level_map[pidx] = selected_level
                                        level_n_correct_map[pidx] = selected_result['n_correct']
                                        remaining_indices.discard(pidx)
                                        if _diag_enabled and pidx in _diag_hint_data:
                                            _diag_hint_data[pidx]['solved_at_level'] = selected_level

                                    # --- Phase G: Post-processing ---
                                    batch.batch['response_mask'] = compute_response_mask(batch)

                                    # Skip unhintable
                                    n_skipped_unhintable = 0
                                    if skip_unhintable_cfg and remaining_indices:
                                        for pidx in remaining_indices:
                                            skip_indices.extend(zero_meta[pidx]['indices'].tolist())
                                            n_skipped_unhintable += 1
                                    metrics["custom/n_skipped_unhintable"] = n_skipped_unhintable

                                    if beyond_reach_indices:
                                        skip_indices.extend(beyond_reach_indices)
                                    metrics["custom/n_skipped_beyond_reach"] = n_skipped_beyond_reach

                                    # Update cumulative hint stats for skip_permanently_stuck
                                    for pidx, meta in enumerate(zero_meta):
                                        pid = meta.get('question', '')
                                        _pid_idx = meta['indices'][0]
                                        pid = batch.non_tensor_batch['problem_id'][_pid_idx] if 'problem_id' in batch.non_tensor_batch else pid
                                        if pid and pid in self._problem_cumulative:
                                            cum = self._problem_cumulative[pid]
                                            cum['n_hint_observations'] += 1
                                            if pidx not in remaining_indices:
                                                cum['n_hint_correct'] += 1
                                                cum['n_hint_total'] += 1
                                            else:
                                                cum['n_hint_total'] += 1

                                    # Mark permanently_stuck problems
                                    if skip_permanently_stuck_cfg and skip_permanently_stuck_cfg.get('enabled', False):
                                        min_obs = skip_permanently_stuck_cfg.get('min_observations', 3)
                                        min_hint_obs = skip_permanently_stuck_cfg.get('min_hint_observations', 2)
                                        start_step = skip_permanently_stuck_cfg.get('start_step', 10)
                                        if self.global_steps >= start_step:
                                            for pidx in remaining_indices:
                                                _pid_idx2 = zero_meta[pidx]['indices'][0]
                                                pid2 = batch.non_tensor_batch['problem_id'][_pid_idx2] if 'problem_id' in batch.non_tensor_batch else None
                                                if pid2 and pid2 in self._problem_cumulative:
                                                    cum2 = self._problem_cumulative[pid2]
                                                    if (cum2['n_observations'] >= min_obs
                                                            and cum2['n_hint_observations'] >= min_hint_obs
                                                            and cum2['n_correct_total'] == 0
                                                            and cum2['n_hint_correct'] == 0):
                                                        cum2['permanently_skipped'] = True

                                    # --- Phase H: Comprehensive metrics ---
                                    metrics["custom/n_still_zero_after_hierarchy"] = len(remaining_indices)
                                    metrics["custom/n_hints_improved"] = n_improved
                                    metrics["custom/n_hints_still_zero"] = len(zero_meta) - n_improved
                                    metrics["custom/n_replaced_total"] = n_replaced_total

                                    # Per-level: problems where this was the LOWEST successful level
                                    for level in all_levels_ordered:
                                        metrics[f"custom/n_solved_at_{level}"] = solved_at_counts.get(level, 0)

                                    # Per-level: total correct trajectories at this level (even if not selected)
                                    for level in all_levels_ordered:
                                        total_correct_at_level = sum(
                                            regen_results.get((pidx, level), {}).get('n_correct', 0)
                                            for pidx in range(len(zero_meta)))
                                        metrics[f"custom/n_correct_at_{level}"] = total_correct_at_level

                                    # Level distribution for selected (lowest successful) levels
                                    for level in all_levels_ordered:
                                        count = sum(1 for v in _solved_level_map.values() if v == level)
                                        metrics[f"custom/replacement_level_{level}"] = count

                                    # Mean candidate levels per problem (how many levels COULD have worked)
                                    if candidate_levels_per_problem:
                                        metrics["custom/n_candidate_levels_per_problem"] = np.mean(candidate_levels_per_problem)
                                        metrics["custom/max_candidate_levels"] = max(candidate_levels_per_problem)

                                    # Per-problem correct count after replacement
                                    converted_correct_counts = []
                                    reward_sums_updated = reward_tensor.sum(-1)
                                    for pidx in range(len(zero_meta)):
                                        if pidx not in remaining_indices:
                                            meta = zero_meta[pidx]
                                            nc = int((reward_sums_updated[meta['indices']] > 0).sum())
                                            converted_correct_counts.append(nc)
                                    if converted_correct_counts:
                                        metrics["custom/mean_correct_after_hint"] = np.mean(converted_correct_counts)
                                        for bucket in range(1, n + 1):
                                            metrics[f"custom/n_converted_{bucket}of{n}"] = sum(
                                                1 for c in converted_correct_counts if c == bucket)

                                    # Per-problem tracking
                                    for pidx, meta in enumerate(zero_meta):
                                        problem_id = batch.non_tensor_batch['problem_id'][meta['indices'][0]]
                                        if problem_id:
                                            self._replacement_problem_log[problem_id] = {
                                                'step': self.global_steps,
                                                'epoch': current_epoch,
                                                'hint_level_succeeded': _solved_level_map.get(pidx),
                                                'all_levels_with_correct': all_levels_with_correct_map.get(pidx, []),
                                                'n_correct_at_success_level': level_n_correct_map.get(pidx, 0),
                                                'replace_count_actual': min(level_n_correct_map.get(pidx, 0), replace_count) if (pidx not in remaining_indices and replace_count is not None) else level_n_correct_map.get(pidx, 0) if pidx not in remaining_indices else 0,
                                                'replaced': pidx not in remaining_indices,
                                            }

                                    _regen_elapsed = time.time() - _regen_start
                                    metrics["custom/hook1_regen_seconds"] = _regen_elapsed

                                else:
                                    # No valid regen items (all hints failed to parse + no L5)
                                    remaining_indices = set(range(len(zero_meta)))
                                    n_improved = 0
                                    n_replaced_total = 0
                                    n_skipped_beyond_reach = 0
                                    solved_at_counts = {lvl: 0 for lvl in all_levels_ordered}
                                    for level in all_levels_ordered:
                                        metrics[f"custom/n_solved_at_{level}"] = 0
                                        metrics[f"custom/n_correct_at_{level}"] = 0
                                        metrics[f"custom/replacement_level_{level}"] = 0
                                    metrics["custom/n_still_zero_after_hierarchy"] = len(zero_meta)
                                    metrics["custom/n_hints_improved"] = 0
                                    metrics["custom/n_hints_still_zero"] = len(zero_meta)
                                    metrics["custom/n_replaced_total"] = 0
                                    metrics["custom/n_skipped_unhintable"] = 0
                                    metrics["custom/n_skipped_beyond_reach"] = 0

                                _hint_total_elapsed = time.time() - _hint_start
                                metrics["custom/hook1_hint_total_seconds"] = _hint_total_elapsed
                                print(f"[Hook1-Replacement2] Done: {len(zero_meta)} zero-reward, "
                                      f"{n_improved} improved ({n_replaced_total} replacements, "
                                      f"replace_count={replace_count}), "
                                      f"{n_skipped_beyond_reach} beyond-reach "
                                      f"across {len(hint_levels)} levels "
                                      f"({', '.join(f'{l}={solved_at_counts.get(l,0)}' for l in all_levels_ordered)}), "
                                      f"{_hint_total_elapsed:.1f}s)")

                        elif hint_mode == 'hint_SFT':
                            # ============================================================
                            # HINT_SFT: Generate hint-based trajectories like replacement2,
                            # but instead of replacing in the GRPO batch, collect gold data
                            # (correct trajectory + original prompt) for SFT at epoch boundary.
                            # GRPO trains only on natural trajectories.
                            # ============================================================
                            from verl_grpo.hints.hint_generator import HIERARCHICAL2_HINT_LEVELS
                            import random

                            # --- Phase A: Configuration + Probability Gating ---
                            hint_levels = hints_cfg.get('hint_levels', HIERARCHICAL2_HINT_LEVELS)
                            hint_n_candidates = hints_cfg.get('hint_n_candidates', 2)
                            hint_probability = hints_cfg.get('hint_probability', 1.0)
                            anti_rep_prompt = hints_cfg.get('anti_repetition_system_prompt', None)
                            has_l5 = "L5_solution" in hint_levels
                            gen_levels = [l for l in hint_levels if l != "L5_solution"]
                            all_levels_ordered = gen_levels + (["L5_solution"] if has_l5 else [])

                            if random.random() >= hint_probability:
                                print(f"[Hook1-HintSFT] Skipping hints "
                                      f"(probability={hint_probability:.2f})")
                                metrics["custom/hint_sft_n_gold_collected"] = 0
                                metrics["custom/hint_sft_n_unhintable"] = len(zero_meta)
                                metrics["custom/hint_skipped_by_probability"] = 1
                            else:
                                metrics["custom/hint_skipped_by_probability"] = 0

                                # --- Phase B: Extract failed trajectories ---
                                failed_trajectories = []
                                _diag_hint_data = {}
                                _diag_enabled = bool(self.config.trainer.get("diagnostic_log_dir", None))
                                for pidx_ft, meta in enumerate(zero_meta):
                                    random_idx = random.choice(meta['indices'].tolist())
                                    response_ids = batch.batch['responses'][random_idx]
                                    valid_ids = response_ids[response_ids != pad_token_id]
                                    trajectory_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                                    failed_trajectories.append(trajectory_text)
                                    if _diag_enabled:
                                        _orig_trajs = []
                                        for _oi, _gi in enumerate(meta['indices']):
                                            _resp = batch.batch['responses'][_gi]
                                            _valid = _resp[_resp != pad_token_id]
                                            _text = self.tokenizer.decode(_valid.tolist(), skip_special_tokens=True)
                                            _pred = extract_answer_math(_text)
                                            _orig_trajs.append({
                                                "index": _oi,
                                                "text": _text,
                                                "predicted_answer": str(_pred) if _pred else None,
                                                "correct": False,
                                            })
                                        _diag_hint_data[pidx_ft] = {
                                            'original_trajectories': _orig_trajs,
                                            'failed_trajectory_used': trajectory_text,
                                            'failed_trajectory_index': int(random_idx),
                                            'per_level': {},
                                            'solved_at_level': None,
                                        }

                                # --- Phase C: Batch generate hints for L1-L4 ---
                                if gen_levels:
                                    if self.frontier_client is not None:
                                        all_raw_prompts, prompt_map = self.hint_generator.build_all_hierarchical_prompts_raw(
                                            levels=gen_levels,
                                            questions=questions,
                                            solutions=solutions,
                                            failed_trajectories=failed_trajectories,
                                        )
                                        print(f"[Hook1-HintSFT] Generating {len(all_raw_prompts)} hint prompts "
                                              f"({len(gen_levels)} levels x {len(zero_meta)} problems) "
                                              f"via frontier API ({self.frontier_client.model})")
                                        hint_responses, api_stats = self.frontier_client.generate_hints(
                                            prompts=all_raw_prompts,
                                            temperature=hints_cfg.get('hint_temperature', 0.7),
                                            max_tokens=hints_cfg.get('hint_max_tokens', 512),
                                        )
                                        metrics["custom/frontier_api_retries"] = api_stats.get('total_retries', 0)
                                        metrics["custom/frontier_api_failures"] = api_stats.get('total_failures', 0)
                                        hints_by_level = {}
                                        for i, (level, pidx) in enumerate(prompt_map):
                                            hint = self.hint_generator.parse_hierarchical_hint(hint_responses[i])
                                            hints_by_level.setdefault(level, {})[pidx] = hint

                                    elif self.hint_client is not None:
                                        all_hint_prompts, prompt_map = self.hint_generator.build_all_hierarchical_prompts(
                                            levels=gen_levels,
                                            questions=questions,
                                            solutions=solutions,
                                            failed_trajectories=failed_trajectories,
                                        )
                                        print(f"[Hook1-HintSFT] Generating {len(all_hint_prompts)} hint prompts "
                                              f"({len(gen_levels)} levels x {len(zero_meta)} problems) "
                                              f"via external server (n={hint_n_candidates})")
                                        hint_responses = self.hint_client.generate_hints(
                                            prompt_texts=all_hint_prompts,
                                            n=hint_n_candidates,
                                            temperature=hints_cfg.get('hint_temperature', 0.7),
                                            max_tokens=hints_cfg.get('hint_max_tokens', 512),
                                        )
                                        hints_by_level = {}
                                        for i, (level, pidx) in enumerate(prompt_map):
                                            group = hint_responses[i*hint_n_candidates:(i+1)*hint_n_candidates]
                                            hint = self.hint_generator.parse_hierarchical_hints_from_texts(group)
                                            hints_by_level.setdefault(level, {})[pidx] = hint
                                    else:
                                        all_hint_prompts, prompt_map = self.hint_generator.build_all_hierarchical_prompts(
                                            levels=gen_levels,
                                            questions=questions,
                                            solutions=solutions,
                                            failed_trajectories=failed_trajectories,
                                        )
                                        hint_gen_batch = self.prompt_builder.tokenize_prompts(all_hint_prompts)
                                        hint_gen_batch.meta_info['do_sample'] = True
                                        orig_hint_prompt_count = len(all_hint_prompts)
                                        hint_gen_batch, hint_pad_size = pad_dataproto_to_divisor(hint_gen_batch, n_gpus)

                                        print(f"[Hook1-HintSFT] Generating {orig_hint_prompt_count} hint prompts "
                                              f"({len(gen_levels)} levels x {len(zero_meta)} problems) "
                                              f"(padded to {len(hint_gen_batch)}, n={n})")
                                        hint_gen_output = self.actor_rollout_wg.generate_sequences(hint_gen_batch)

                                        if hint_pad_size > 0:
                                            hint_gen_output = hint_gen_output[:orig_hint_prompt_count * n]

                                        hints_by_level = {}
                                        for i, (level, pidx) in enumerate(prompt_map):
                                            group_responses = hint_gen_output.batch['responses'][i*n:i*n+hint_n_candidates]
                                            hint = self.hint_generator.parse_hierarchical_hints_from_group(
                                                group_responses, hint_n_candidates, self.tokenizer)
                                            hints_by_level.setdefault(level, {})[pidx] = hint
                                else:
                                    hints_by_level = {}

                                _hint_gen_elapsed = time.time() - _hint_start
                                metrics["custom/hook1_hint_gen_seconds"] = _hint_gen_elapsed

                                # --- Phase D: Build ALL regen items + ONE vLLM call ---
                                _regen_start = time.time()
                                regen_items = []  # (pidx, level, meta, hint_text, is_l5)
                                hinted_texts_all = []

                                for level in gen_levels:
                                    for pidx in range(len(zero_meta)):
                                        hint = hints_by_level.get(level, {}).get(pidx)
                                        if hint:
                                            regen_items.append((pidx, level, zero_meta[pidx], hint, False))
                                            hinted_texts_all.append(
                                                self.prompt_builder.build_hinted_prompt_text_v2(
                                                    zero_meta[pidx]['question'], hint,
                                                    system_prompt=anti_rep_prompt, is_l5=False))
                                        elif _diag_enabled and pidx in _diag_hint_data:
                                            _diag_hint_data[pidx]['per_level'][level] = {
                                                'hint_prompt': None,
                                                'hint_parsed': None,
                                                'hinted_prompt': None,
                                                'regen_trajectories': [],
                                                'n_correct_regen': 0,
                                                'solved': False,
                                                'skipped_reason': 'hint_parse_failed',
                                            }

                                if has_l5:
                                    for pidx in range(len(zero_meta)):
                                        meta = zero_meta[pidx]
                                        solution = meta['solution']
                                        if solution and solution.strip():
                                            regen_items.append((pidx, "L5_solution", meta, solution, True))
                                            hinted_texts_all.append(
                                                self.prompt_builder.build_hinted_prompt_text_v2(
                                                    meta['question'], solution,
                                                    system_prompt=None, is_l5=True))
                                        elif _diag_enabled and pidx in _diag_hint_data:
                                            _diag_hint_data[pidx]['per_level']["L5_solution"] = {
                                                'hint_prompt': None,
                                                'hint_parsed': None,
                                                'hinted_prompt': None,
                                                'regen_trajectories': [],
                                                'n_correct_regen': 0,
                                                'solved': False,
                                                'skipped_reason': 'no_reference_solution',
                                            }

                                if regen_items:
                                    regen_batch_all = self.prompt_builder.tokenize_prompts(hinted_texts_all)
                                    regen_batch_all.meta_info['do_sample'] = True
                                    if hint_n_candidates != n:
                                        regen_batch_all.meta_info['n_override'] = hint_n_candidates
                                    orig_count_all = len(regen_items)
                                    regen_batch_all, regen_pad_all = pad_dataproto_to_divisor(regen_batch_all, n_gpus)

                                    print(f"[Hook1-HintSFT] Re-generating {orig_count_all} "
                                          f"hinted prompts ({len(all_levels_ordered)} levels x "
                                          f"{len(zero_meta)} problems, n_cand={hint_n_candidates}, "
                                          f"padded to {len(regen_batch_all)})")
                                    regen_output_all = self.actor_rollout_wg.generate_sequences(regen_batch_all)

                                    # Validate n_override took effect + remove padding
                                    expected_regen_size = orig_count_all * hint_n_candidates
                                    padded_expected = (orig_count_all + regen_pad_all) * hint_n_candidates
                                    actual_size = len(regen_output_all)
                                    if actual_size != padded_expected:
                                        raise RuntimeError(
                                            f"[HintSFT] Regen output size mismatch: "
                                            f"expected {padded_expected} ({orig_count_all}+{regen_pad_all} prompts × {hint_n_candidates}), "
                                            f"got {actual_size}. Is n_override support missing from verl? "
                                            f"Ensure verl is on branch custom-patches-v0.4.1.")
                                    regen_output_all = regen_output_all[:expected_regen_size]

                                    # --- Phase E: Evaluate correctness at ALL levels ---
                                    regen_results = {}  # (pidx, level) -> dict
                                    for k, (pidx, level, meta, hint_text, is_l5) in enumerate(regen_items):
                                        regen_responses = regen_output_all.batch['responses'][
                                            k*hint_n_candidates:(k+1)*hint_n_candidates]

                                        correct_indices = []
                                        _diag_regen_trajs = []
                                        for i in range(hint_n_candidates):
                                            text = self.tokenizer.decode(
                                                regen_responses[i].tolist(), skip_special_tokens=True)
                                            predicted = extract_answer_math(text)
                                            is_correct = (predicted is not None
                                                          and answers_match_math(predicted, meta['ground_truth']))
                                            if is_correct:
                                                correct_indices.append(i)
                                            if _diag_enabled:
                                                _diag_regen_trajs.append({
                                                    "index": i,
                                                    "text": text,
                                                    "predicted_answer": str(predicted) if predicted else None,
                                                    "correct": is_correct,
                                                })

                                        regen_results[(pidx, level)] = {
                                            'any_correct': len(correct_indices) > 0,
                                            'n_correct': len(correct_indices),
                                            'correct_indices': correct_indices,
                                            'regen_responses': regen_responses,
                                            'hint_text': hint_text,
                                            'hinted_prompt_text': hinted_texts_all[k],
                                            'is_l5': is_l5,
                                            '_diag_regen_trajs': _diag_regen_trajs,
                                        }

                                    # --- Phase F: Find lowest successful level + collect gold ---
                                    # Unlike replacement2, we do NOT replace in the GRPO batch.
                                    # Instead, we collect gold data for SFT at epoch boundary.
                                    n_gold_collected = 0
                                    n_unhintable = 0
                                    solved_at_counts = {lvl: 0 for lvl in all_levels_ordered}
                                    candidate_levels_per_problem = []

                                    for pidx in range(len(zero_meta)):
                                        meta = zero_meta[pidx]

                                        # Count how many levels have correct trajectories (diagnostic)
                                        levels_with_correct = []
                                        for level in all_levels_ordered:
                                            result = regen_results.get((pidx, level))
                                            if result and result['any_correct']:
                                                levels_with_correct.append(level)
                                        candidate_levels_per_problem.append(len(levels_with_correct))

                                        # Record diagnostics for ALL levels
                                        if _diag_enabled and pidx in _diag_hint_data:
                                            for level in all_levels_ordered:
                                                result = regen_results.get((pidx, level))
                                                if result is None:
                                                    continue
                                                if result['is_l5']:
                                                    hint_prompt_raw = None
                                                    hint_parsed = f"[reference_solution, len={len(result['hint_text'])}]"
                                                else:
                                                    hint_prompt_raw = self.hint_generator._format_hierarchical_prompt(
                                                        level, meta['question'], meta['solution'],
                                                        failed_trajectories[pidx])
                                                    hint_parsed = result['hint_text']
                                                _diag_hint_data[pidx]['per_level'][level] = {
                                                    'hint_prompt': hint_prompt_raw,
                                                    'hint_parsed': hint_parsed,
                                                    'hinted_prompt': result['hinted_prompt_text'],
                                                    'regen_trajectories': result['_diag_regen_trajs'],
                                                    'n_correct_regen': result['n_correct'],
                                                    'solved': result['any_correct'],
                                                }

                                        # Find LOWEST level with any correct trajectory
                                        selected_level = None
                                        selected_result = None
                                        for level in all_levels_ordered:
                                            result = regen_results.get((pidx, level))
                                            if result and result['any_correct']:
                                                selected_level = level
                                                selected_result = result
                                                break

                                        if selected_level is None:
                                            n_unhintable += 1
                                            if _diag_enabled and pidx in _diag_hint_data:
                                                _diag_hint_data[pidx]['solved_at_level'] = None
                                            continue

                                        # Randomly pick ONE correct trajectory from the lowest successful level
                                        chosen_idx = random.choice(selected_result['correct_indices'])
                                        gold_response_text = self.tokenizer.decode(
                                            selected_result['regen_responses'][chosen_idx].tolist(),
                                            skip_special_tokens=True)

                                        # Get original prompt text (question, NOT hinted)
                                        original_question = meta['question']

                                        # Get problem_id
                                        _pid_idx = meta['indices'][0]
                                        problem_id = batch.non_tensor_batch['problem_id'][_pid_idx] if 'problem_id' in batch.non_tensor_batch else None

                                        # Append to gold buffer
                                        self._sft_gold_buffer.append({
                                            'problem_id': problem_id,
                                            'question': original_question,
                                            'gold_response_text': gold_response_text,
                                            'hint_level': selected_level,
                                            'step': self.global_steps,
                                        })

                                        solved_at_counts[selected_level] += 1
                                        n_gold_collected += 1
                                        if _diag_enabled and pidx in _diag_hint_data:
                                            _diag_hint_data[pidx]['solved_at_level'] = selected_level

                                    # --- Phase G: Metrics (no replacement, no skip logic needed) ---
                                    metrics["custom/hint_sft_n_gold_collected"] = n_gold_collected
                                    metrics["custom/hint_sft_n_unhintable"] = n_unhintable
                                    metrics["custom/hint_sft_gold_buffer_size"] = len(self._sft_gold_buffer)

                                    # Per-level: problems where this was the LOWEST successful level
                                    for level in all_levels_ordered:
                                        metrics[f"custom/n_solved_at_{level}"] = solved_at_counts.get(level, 0)

                                    # Per-level: total correct trajectories at this level
                                    for level in all_levels_ordered:
                                        total_correct_at_level = sum(
                                            regen_results.get((pidx, level), {}).get('n_correct', 0)
                                            for pidx in range(len(zero_meta)))
                                        metrics[f"custom/n_correct_at_{level}"] = total_correct_at_level

                                    # Mean candidate levels per problem
                                    if candidate_levels_per_problem:
                                        metrics["custom/n_candidate_levels_per_problem"] = np.mean(candidate_levels_per_problem)
                                        metrics["custom/max_candidate_levels"] = max(candidate_levels_per_problem)

                                    # Natural success tracking
                                    n_naturally_correct = len(unique_uids) - len(zero_meta)
                                    metrics["custom/n_naturally_correct_groups"] = n_naturally_correct
                                    metrics["custom/hint_sft_pct_naturally_correct"] = n_naturally_correct / max(len(unique_uids), 1)

                                    _regen_elapsed = time.time() - _regen_start
                                    metrics["custom/hook1_regen_seconds"] = _regen_elapsed

                                else:
                                    # No valid regen items (all hints failed to parse + no L5)
                                    n_gold_collected = 0
                                    n_unhintable = len(zero_meta)
                                    for level in all_levels_ordered:
                                        metrics[f"custom/n_solved_at_{level}"] = 0
                                        metrics[f"custom/n_correct_at_{level}"] = 0
                                    metrics["custom/hint_sft_n_gold_collected"] = 0
                                    metrics["custom/hint_sft_n_unhintable"] = n_unhintable
                                    metrics["custom/hint_sft_gold_buffer_size"] = len(self._sft_gold_buffer)
                                    metrics["custom/n_naturally_correct_groups"] = len(unique_uids) - len(zero_meta)

                                _hint_total_elapsed = time.time() - _hint_start
                                metrics["custom/hook1_hint_total_seconds"] = _hint_total_elapsed
                                print(f"[Hook1-HintSFT] Done: {len(zero_meta)} zero-reward, "
                                      f"{n_gold_collected} gold collected, "
                                      f"{n_unhintable} unhintable, "
                                      f"buffer={len(self._sft_gold_buffer)}, "
                                      f"across {len(hint_levels)} levels "
                                      f"({', '.join(f'{l}={solved_at_counts.get(l,0)}' for l in all_levels_ordered)}), "
                                      f"{_hint_total_elapsed:.1f}s)")

                        elif hint_mode == 'prefix':
                            # ============================================================
                            # SOLUTION PREFIX CASCADE: prefix_20 -> prefix_40 -> ... -> prefix_100
                            # No LLM hint generation — fully algorithmic from reference solution.
                            # ============================================================
                            from verl_grpo.hints.hint_generator import HintGenerator as _HG

                            prefix_fractions = hints_cfg.get('prefix_fractions', [0.2, 0.4, 0.6, 0.8, 1.0])
                            prefix_labels = [f"prefix_{int(f*100)}" for f in prefix_fractions]

                            _diag_hint_data = {}
                            _diag_enabled = bool(self.config.trainer.get("diagnostic_log_dir", None))

                            # Pre-compute all prefixes for all zero-reward problems at all levels.
                            # Pure string manipulation — no GPU work, no vLLM calls.
                            prefixes_by_level = {}
                            for frac, label in zip(prefix_fractions, prefix_labels):
                                prefixes_by_level[label] = {}
                                for pidx, meta in enumerate(zero_meta):
                                    solution = meta['solution']
                                    if not solution or not solution.strip():
                                        prefixes_by_level[label][pidx] = None
                                        continue
                                    prefix = _HG.get_solution_prefix(solution, frac)
                                    prefixes_by_level[label][pidx] = prefix if prefix else None

                            # Diagnostic: capture original trajectories before cascade
                            if _diag_enabled:
                                for pidx, meta in enumerate(zero_meta):
                                    _orig_trajs = []
                                    for _oi, _gi in enumerate(meta['indices']):
                                        _resp = batch.batch['responses'][_gi]
                                        _valid = _resp[_resp != pad_token_id]
                                        _text = self.tokenizer.decode(_valid.tolist(), skip_special_tokens=True)
                                        _pred = extract_answer_math(_text)
                                        _orig_trajs.append({
                                            "index": _oi,
                                            "text": _text,
                                            "predicted_answer": str(_pred) if _pred else None,
                                            "correct": False,
                                        })
                                    _diag_hint_data[pidx] = {
                                        'original_trajectories': _orig_trajs,
                                        'per_level': {},
                                        'solved_at_level': None,
                                    }

                            metrics["custom/hook1_hint_gen_seconds"] = 0.0  # No LLM generation

                            # --- CASCADE: try each prefix level, stop at first success ---
                            remaining_indices = set(range(len(zero_meta)))
                            n_improved = 0

                            for frac, label in zip(prefix_fractions, prefix_labels):
                                if not remaining_indices:
                                    break

                                # Collect valid prefixes for remaining problems
                                valid_for_regen = []
                                for pidx in sorted(remaining_indices):
                                    prefix = prefixes_by_level[label].get(pidx)
                                    if prefix:
                                        valid_for_regen.append((pidx, zero_meta[pidx], prefix))
                                    elif _diag_enabled and pidx in _diag_hint_data:
                                        _diag_hint_data[pidx]['per_level'][label] = {
                                            'prefix_fraction': frac,
                                            'prefix_text': None,
                                            'hinted_prompt': None,
                                            'regen_trajectories': [],
                                            'n_correct_regen': 0,
                                            'solved': False,
                                            'skipped_reason': 'empty_solution',
                                        }

                                if not valid_for_regen:
                                    metrics[f"custom/n_solved_at_{label}"] = 0
                                    continue

                                # Build prefix-hinted prompts (with optional anti-repetition)
                                _prefix_anti_rep = hints_cfg.get('anti_repetition_system_prompt', None)
                                hinted_texts = [
                                    self.prompt_builder.build_prefix_prompt_text(
                                        meta['question'], prefix, frac,
                                        system_prompt=_prefix_anti_rep)
                                    for _, meta, prefix in valid_for_regen
                                ]

                                # Re-generate with prefix-hinted prompts
                                regen_batch = self.prompt_builder.tokenize_prompts(hinted_texts)
                                regen_batch.meta_info['do_sample'] = True
                                orig_count = len(valid_for_regen)
                                regen_batch, regen_pad = pad_dataproto_to_divisor(regen_batch, n_gpus)

                                print(f"[Hook1-Prefix] {label}: re-generating with {orig_count} "
                                      f"prefix-hinted prompts (padded to {len(regen_batch)}, n={n})")
                                regen_output = self.actor_rollout_wg.generate_sequences(regen_batch)

                                if regen_pad > 0:
                                    regen_output = regen_output[:orig_count * n]

                                # Dimension alignment for hinted prompt_mode
                                if prompt_mode == 'hinted':
                                    main_prompt_len = batch.batch['prompts'].shape[1]
                                    regen_prompt_len = regen_output.batch['prompts'].shape[1]
                                    target_len = max(main_prompt_len, regen_prompt_len)
                                    if main_prompt_len < target_len:
                                        _expand_batch_prompt_dim(batch, target_len, pad_token_id)
                                    if regen_prompt_len < target_len:
                                        _expand_batch_prompt_dim(regen_output, target_len, pad_token_id)

                                # Check rewards + replace (same logic as hierarchical)
                                solved_at_level = []
                                for k, (pidx, meta, prefix) in enumerate(valid_for_regen):
                                    regen_responses = regen_output.batch['responses'][k*n:(k+1)*n]
                                    regen_attn = regen_output.batch['attention_mask'][k*n:(k+1)*n, -response_length:]

                                    any_correct = False
                                    _diag_regen_trajs = []
                                    _diag_n_correct_regen = 0
                                    for i in range(n):
                                        text = self.tokenizer.decode(
                                            regen_responses[i].tolist(), skip_special_tokens=True)
                                        predicted = extract_answer_math(text)
                                        is_correct = (predicted is not None
                                                      and answers_match_math(predicted, meta['ground_truth']))
                                        if is_correct:
                                            any_correct = True
                                            _diag_n_correct_regen += 1
                                        if _diag_enabled:
                                            _diag_regen_trajs.append({
                                                "index": i,
                                                "text": text,
                                                "predicted_answer": str(predicted) if predicted else None,
                                                "correct": is_correct,
                                            })

                                    # Record diagnostic data
                                    if _diag_enabled and pidx in _diag_hint_data:
                                        _diag_hint_data[pidx]['per_level'][label] = {
                                            'prefix_fraction': frac,
                                            'prefix_text': prefix[:200] + '...' if len(prefix) > 200 else prefix,
                                            'hinted_prompt': hinted_texts[k][:300] + '...' if len(hinted_texts[k]) > 300 else hinted_texts[k],
                                            'regen_trajectories': _diag_regen_trajs,
                                            'n_correct_regen': _diag_n_correct_regen,
                                            'solved': any_correct,
                                        }

                                    if any_correct:
                                        main_indices = meta['indices']
                                        batch.batch['responses'][main_indices] = regen_responses

                                        if prompt_mode == 'hinted':
                                            # Replace full prompt+response (dims already aligned)
                                            regen_slice = slice(k*n, (k+1)*n)
                                            batch.batch['prompts'][main_indices] = regen_output.batch['prompts'][regen_slice]
                                            batch.batch['input_ids'][main_indices] = regen_output.batch['input_ids'][regen_slice]
                                            batch.batch['attention_mask'][main_indices] = regen_output.batch['attention_mask'][regen_slice]
                                            batch.batch['position_ids'][main_indices] = regen_output.batch['position_ids'][regen_slice]
                                        else:
                                            # Original mode: keep original prompts, replace response part only
                                            batch.batch['input_ids'][main_indices, -response_length:] = regen_responses
                                            batch.batch['attention_mask'][main_indices, -response_length:] = regen_attn
                                            batch.batch['position_ids'][main_indices] = compute_position_id_with_mask(
                                                batch.batch['attention_mask'][main_indices])

                                        # Log-prob gap: mark replaced trajectories
                                        hint_replaced[main_indices] = True
                                        for _ri in main_indices:
                                            hint_level_map[_ri] = label
                                            hinted_prompt_texts[_ri] = hinted_texts[k]

                                        # Update reward_tensor
                                        for i, idx in enumerate(main_indices):
                                            text = self.tokenizer.decode(
                                                regen_responses[i].tolist(), skip_special_tokens=True)
                                            predicted = extract_answer_math(text)
                                            correct = (predicted is not None
                                                       and answers_match_math(predicted, meta['ground_truth']))
                                            reward_tensor[idx] = 0.0
                                            if correct:
                                                valid_pos = torch.where(regen_attn[i] > 0)[0]
                                                if len(valid_pos) > 0:
                                                    reward_tensor[idx, valid_pos[-1].item()] = 1.0

                                        solved_at_level.append(pidx)
                                        n_improved += 1
                                        if _diag_enabled and pidx in _diag_hint_data:
                                            _diag_hint_data[pidx]['solved_at_level'] = label

                                remaining_indices -= set(solved_at_level)
                                metrics[f"custom/n_solved_at_{label}"] = len(solved_at_level)
                                print(f"[Hook1-Prefix] {label}: {len(solved_at_level)} solved, "
                                      f"{len(remaining_indices)} remaining")

                            # Recompute response_mask for the full batch
                            batch.batch['response_mask'] = compute_response_mask(batch)

                            metrics["custom/n_still_zero_after_hierarchy"] = len(remaining_indices)
                            metrics["custom/n_hints_improved"] = n_improved
                            metrics["custom/n_hints_still_zero"] = len(zero_meta) - n_improved

                            # Per-problem metrics
                            converted_correct_counts = []
                            reward_sums_updated = reward_tensor.sum(-1)
                            for pidx in range(len(zero_meta)):
                                if pidx not in remaining_indices:
                                    meta = zero_meta[pidx]
                                    nc = int((reward_sums_updated[meta['indices']] > 0).sum())
                                    converted_correct_counts.append(nc)
                            if converted_correct_counts:
                                metrics["custom/mean_correct_after_hint"] = np.mean(converted_correct_counts)
                                for bucket in range(1, n + 1):
                                    metrics[f"custom/n_converted_{bucket}of{n}"] = sum(
                                        1 for c in converted_correct_counts if c == bucket)

                            # Skip unhintable
                            n_skipped_unhintable = 0
                            if skip_unhintable_cfg and remaining_indices:
                                for pidx in remaining_indices:
                                    skip_indices.extend(zero_meta[pidx]['indices'].tolist())
                                    n_skipped_unhintable += 1
                            metrics["custom/n_skipped_unhintable"] = n_skipped_unhintable

                            _hint_total_elapsed = time.time() - _hint_start
                            metrics["custom/hook1_hint_total_seconds"] = _hint_total_elapsed
                            print(f"[Hook1-Prefix] Done: {len(zero_meta)} zero-reward, "
                                  f"{n_improved} improved across {len(prefix_labels)} prefix levels "
                                  f"({_hint_total_elapsed:.1f}s)")

                        elif hint_mode == 'prefix_continuation_cascade':
                            # ============================================================
                            # PREFIXRL-STYLE PREFIX CASCADE:
                            # Prefix placed after <|im_start|>assistant (not in user msg).
                            # Model continues from prefix. Gradients only on continuation.
                            # Sequential cascade: 20% -> 40% -> 60% -> 80%.
                            # ============================================================
                            from verl_grpo.hints.hint_generator import HintGenerator as _HG

                            prefix_fractions = hints_cfg.get('prefix_fractions', [0.2, 0.4, 0.6, 0.8])
                            prefix_labels = [f"prefixrl_{int(f*100)}" for f in prefix_fractions]

                            # Pre-compute all prefixes (pure string manipulation)
                            prefixes_by_level = {}
                            for frac, label in zip(prefix_fractions, prefix_labels):
                                prefixes_by_level[label] = {}
                                for pidx, meta in enumerate(zero_meta):
                                    solution = meta['solution']
                                    if not solution or not solution.strip():
                                        prefixes_by_level[label][pidx] = None
                                        continue
                                    prefix = _HG.get_solution_prefix(solution, frac)
                                    prefixes_by_level[label][pidx] = prefix if prefix else None

                            metrics["custom/hook1_hint_gen_seconds"] = 0.0

                            # --- CASCADE: try each prefix level, stop at first success ---
                            remaining_indices = set(range(len(zero_meta)))
                            n_improved = 0

                            for frac, label in zip(prefix_fractions, prefix_labels):
                                if not remaining_indices:
                                    break

                                valid_for_regen = []
                                for pidx in sorted(remaining_indices):
                                    prefix = prefixes_by_level[label].get(pidx)
                                    if prefix:
                                        valid_for_regen.append((pidx, zero_meta[pidx], prefix))

                                if not valid_for_regen:
                                    metrics[f"custom/n_solved_at_{label}"] = 0
                                    continue

                                # Build PrefixRL-style prompts: chat_template(question) + prefix tokens
                                regen_questions = [meta['question'] for _, meta, _ in valid_for_regen]
                                regen_prefixes = [prefix for _, _, prefix in valid_for_regen]
                                regen_batch = self.prompt_builder.build_prefixrl_prompts(
                                    regen_questions, regen_prefixes)
                                regen_batch.meta_info['do_sample'] = True
                                orig_count = len(valid_for_regen)
                                regen_batch, regen_pad = pad_dataproto_to_divisor(regen_batch, n_gpus)

                                print(f"[Hook1-PrefixRL] {label}: re-generating with {orig_count} "
                                      f"PrefixRL prompts (padded to {len(regen_batch)}, n={n})")
                                regen_output = self.actor_rollout_wg.generate_sequences(regen_batch)

                                if regen_pad > 0:
                                    regen_output = regen_output[:orig_count * n]

                                # Dimension alignment: PrefixRL prompts are longer than original
                                main_prompt_len = batch.batch['prompts'].shape[1]
                                regen_prompt_len = regen_output.batch['prompts'].shape[1]
                                target_len = max(main_prompt_len, regen_prompt_len)
                                if main_prompt_len < target_len:
                                    _expand_batch_prompt_dim(batch, target_len, pad_token_id)
                                if regen_prompt_len < target_len:
                                    _expand_batch_prompt_dim(regen_output, target_len, pad_token_id)

                                # Check rewards + replace
                                solved_at_level = []
                                for k, (pidx, meta, prefix) in enumerate(valid_for_regen):
                                    regen_responses = regen_output.batch['responses'][k*n:(k+1)*n]
                                    regen_attn = regen_output.batch['attention_mask'][k*n:(k+1)*n, -response_length:]

                                    any_correct = False
                                    for i in range(n):
                                        text = self.tokenizer.decode(
                                            regen_responses[i].tolist(), skip_special_tokens=True)
                                        # PrefixRL: prepend prefix_text for answer extraction
                                        full_text = prefix + text
                                        predicted = extract_answer_math(full_text)
                                        is_correct = (predicted is not None
                                                      and answers_match_math(predicted, meta['ground_truth']))
                                        if is_correct:
                                            any_correct = True
                                            break

                                    if any_correct:
                                        main_indices = meta['indices']
                                        # Full replacement (prompt includes prefix)
                                        regen_slice = slice(k*n, (k+1)*n)
                                        batch.batch['responses'][main_indices] = regen_responses
                                        batch.batch['prompts'][main_indices] = regen_output.batch['prompts'][regen_slice]
                                        batch.batch['input_ids'][main_indices] = regen_output.batch['input_ids'][regen_slice]
                                        batch.batch['attention_mask'][main_indices] = regen_output.batch['attention_mask'][regen_slice]
                                        batch.batch['position_ids'][main_indices] = regen_output.batch['position_ids'][regen_slice]

                                        hint_replaced[main_indices] = True
                                        for _ri in main_indices:
                                            hint_level_map[_ri] = label

                                        # Update reward_tensor (prefix_text + continuation for answer check)
                                        for i, idx in enumerate(main_indices):
                                            text = self.tokenizer.decode(
                                                regen_responses[i].tolist(), skip_special_tokens=True)
                                            full_text = prefix + text
                                            predicted = extract_answer_math(full_text)
                                            correct = (predicted is not None
                                                       and answers_match_math(predicted, meta['ground_truth']))
                                            reward_tensor[idx] = 0.0
                                            if correct:
                                                valid_pos = torch.where(regen_attn[i] > 0)[0]
                                                if len(valid_pos) > 0:
                                                    reward_tensor[idx, valid_pos[-1].item()] = 1.0

                                        solved_at_level.append(pidx)
                                        n_improved += 1

                                remaining_indices -= set(solved_at_level)
                                metrics[f"custom/n_solved_at_{label}"] = len(solved_at_level)
                                print(f"[Hook1-PrefixRL] {label}: {len(solved_at_level)} solved, "
                                      f"{len(remaining_indices)} remaining")

                            # Recompute response_mask for the full batch
                            batch.batch['response_mask'] = compute_response_mask(batch)

                            metrics["custom/n_still_zero_after_hierarchy"] = len(remaining_indices)
                            metrics["custom/n_hints_improved"] = n_improved
                            metrics["custom/n_hints_still_zero"] = len(zero_meta) - n_improved

                            converted_correct_counts = []
                            reward_sums_updated = reward_tensor.sum(-1)
                            for pidx in range(len(zero_meta)):
                                if pidx not in remaining_indices:
                                    meta = zero_meta[pidx]
                                    nc = int((reward_sums_updated[meta['indices']] > 0).sum())
                                    converted_correct_counts.append(nc)
                            if converted_correct_counts:
                                metrics["custom/mean_correct_after_hint"] = np.mean(converted_correct_counts)

                            n_skipped_unhintable = 0
                            if skip_unhintable_cfg and remaining_indices:
                                for pidx in remaining_indices:
                                    skip_indices.extend(zero_meta[pidx]['indices'].tolist())
                                    n_skipped_unhintable += 1
                            metrics["custom/n_skipped_unhintable"] = n_skipped_unhintable

                            _hint_total_elapsed = time.time() - _hint_start
                            metrics["custom/hook1_hint_total_seconds"] = _hint_total_elapsed
                            print(f"[Hook1-PrefixRL] Done: {len(zero_meta)} zero-reward, "
                                  f"{n_improved} improved across {len(prefix_labels)} prefix levels "
                                  f"({_hint_total_elapsed:.1f}s)")

                        else:
                            # ============================================================
                            # SINGLE-HINT path (existing: solution_aware, solution_blind,
                            # solution_aware_failed_conditional)
                            # ============================================================
                            if hint_mode == 'solution_aware_failed_conditional':
                                import random
                                failed_trajectories = []
                                for meta in zero_meta:
                                    random_idx = random.choice(meta['indices'].tolist())
                                    response_ids = batch.batch['responses'][random_idx]
                                    valid_ids = response_ids[response_ids != pad_token_id]
                                    trajectory_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                                    failed_trajectories.append(trajectory_text)

                            hint_prompt_texts = self.hint_generator.build_hint_prompts(
                                questions, solutions, failed_trajectories=failed_trajectories
                            )
                            orig_hint_count = len(zero_meta)

                            valid_hints = []  # list of (zero_meta_idx, meta_dict, hint_text)

                            if self.hint_client is not None:
                                # --- PASS 2a (external): Generate hints via external vLLM server ---
                                print(f"[Hook1] Generating hints for {orig_hint_count} zero-reward problems "
                                      f"via external server (n={n})")
                                hint_responses = self.hint_client.generate_hints(
                                    prompt_texts=hint_prompt_texts,
                                    n=n,
                                    temperature=hints_cfg.get('hint_temperature', 0.7),
                                    max_tokens=hints_cfg.get('hint_max_tokens', 512),
                                )

                                _hint_gen_elapsed = time.time() - _hint_start
                                metrics["custom/hook1_hint_gen_seconds"] = _hint_gen_elapsed

                                # Parse hints from text responses
                                for j, meta in enumerate(zero_meta):
                                    group_texts = hint_responses[j*n:(j+1)*n]
                                    best_hint = self.hint_generator.parse_hints_from_texts(group_texts)
                                    if best_hint is not None:
                                        valid_hints.append((j, meta, best_hint))

                            else:
                                # --- PASS 2b (embedded): Generate hints via embedded vLLM ---
                                hint_gen_batch = self.prompt_builder.tokenize_prompts(hint_prompt_texts)
                                hint_gen_batch.meta_info['do_sample'] = True

                                hint_gen_batch, hint_pad_size = pad_dataproto_to_divisor(hint_gen_batch, n_gpus)

                                print(f"[Hook1] Generating hints for {orig_hint_count} zero-reward problems "
                                      f"(padded to {len(hint_gen_batch)}, n={n})")
                                hint_gen_output = self.actor_rollout_wg.generate_sequences(hint_gen_batch)

                                # Unpad: keep only orig_hint_count * n rows
                                if hint_pad_size > 0:
                                    hint_gen_output = hint_gen_output[:orig_hint_count * n]

                                _hint_gen_elapsed = time.time() - _hint_start
                                metrics["custom/hook1_hint_gen_seconds"] = _hint_gen_elapsed

                                # Parse hints from token responses
                                for j, meta in enumerate(zero_meta):
                                    group_responses = hint_gen_output.batch['responses'][j*n:(j+1)*n]
                                    best_hint = self.hint_generator.parse_hints_from_group(
                                        group_responses, n, self.tokenizer)
                                    if best_hint is not None:
                                        valid_hints.append((j, meta, best_hint))

                            metrics["custom/n_hints_generated"] = len(valid_hints)
                            metrics["custom/n_hints_parse_failed"] = len(zero_meta) - len(valid_hints)

                            n_improved = 0
                            if len(valid_hints) > 0:
                                # --- PASS 3: Re-generate with hinted prompts ---
                                _regen_start = time.time()
                                hinted_texts = [
                                    self.prompt_builder.build_hinted_prompt_text(meta['question'], hint)
                                    for _, meta, hint in valid_hints
                                ]

                                regen_batch = self.prompt_builder.tokenize_prompts(hinted_texts)
                                regen_batch.meta_info['do_sample'] = True

                                orig_regen_count = len(valid_hints)
                                regen_batch, regen_pad_size = pad_dataproto_to_divisor(regen_batch, n_gpus)

                                print(f"[Hook1] Re-generating with {orig_regen_count} hinted prompts "
                                      f"(padded to {len(regen_batch)}, n={n})")
                                regen_output = self.actor_rollout_wg.generate_sequences(regen_batch)

                                if regen_pad_size > 0:
                                    regen_output = regen_output[:orig_regen_count * n]

                                _regen_elapsed = time.time() - _regen_start
                                metrics["custom/hook1_regen_seconds"] = _regen_elapsed

                                # Align prompt dimensions if needed (for hinted prompt_mode)
                                if prompt_mode == 'hinted':
                                    main_prompt_len = batch.batch['prompts'].shape[1]
                                    regen_prompt_len = regen_output.batch['prompts'].shape[1]
                                    target_len = max(main_prompt_len, regen_prompt_len)
                                    if main_prompt_len < target_len:
                                        _expand_batch_prompt_dim(batch, target_len, pad_token_id)
                                    if regen_prompt_len < target_len:
                                        _expand_batch_prompt_dim(regen_output, target_len, pad_token_id)

                                # --- STEP 6-7: Local reward check + selective replacement ---
                                for k, (j, meta, hint) in enumerate(valid_hints):
                                    regen_responses = regen_output.batch['responses'][k*n:(k+1)*n]
                                    regen_response_attn = regen_output.batch['attention_mask'][k*n:(k+1)*n, -response_length:]

                                    # Check if ANY re-generated response is correct
                                    any_correct = False
                                    for i in range(n):
                                        text = self.tokenizer.decode(
                                            regen_responses[i].tolist(), skip_special_tokens=True)
                                        predicted = extract_answer_math(text)
                                        if predicted is not None and answers_match_math(predicted, meta['ground_truth']):
                                            any_correct = True
                                            break

                                    if not any_correct:
                                        continue

                                    n_improved += 1
                                    main_indices = meta['indices']

                                    # Replace responses in main batch
                                    batch.batch['responses'][main_indices] = regen_responses

                                    if prompt_mode == 'hinted':
                                        # Replace all tensors (prompts already dimension-aligned)
                                        regen_slice = slice(k*n, (k+1)*n)
                                        batch.batch['prompts'][main_indices] = regen_output.batch['prompts'][regen_slice]
                                        batch.batch['input_ids'][main_indices] = regen_output.batch['input_ids'][regen_slice]
                                        batch.batch['attention_mask'][main_indices] = regen_output.batch['attention_mask'][regen_slice]
                                        batch.batch['position_ids'][main_indices] = regen_output.batch['position_ids'][regen_slice]
                                    else:
                                        # prompt_mode == 'original': keep original prompts, replace response part
                                        batch.batch['input_ids'][main_indices, -response_length:] = regen_responses
                                        batch.batch['attention_mask'][main_indices, -response_length:] = regen_response_attn
                                        batch.batch['position_ids'][main_indices] = compute_position_id_with_mask(
                                            batch.batch['attention_mask'][main_indices])
                                    # Log-prob gap: mark replaced trajectories
                                    hint_replaced[main_indices] = True
                                    for _ri in main_indices:
                                        hint_level_map[_ri] = 'single'
                                        hinted_prompt_texts[_ri] = hinted_texts[k]

                                    # Update reward_tensor for this group
                                    for i, idx in enumerate(main_indices):
                                        text = self.tokenizer.decode(
                                            regen_responses[i].tolist(), skip_special_tokens=True)
                                        predicted = extract_answer_math(text)
                                        correct = (predicted is not None
                                                   and answers_match_math(predicted, meta['ground_truth']))
                                        reward_tensor[idx] = 0.0
                                        if correct:
                                            valid_pos = torch.where(regen_response_attn[i] > 0)[0]
                                            if len(valid_pos) > 0:
                                                reward_tensor[idx, valid_pos[-1].item()] = 1.0

                                # Recompute response_mask for the full batch
                                batch.batch['response_mask'] = compute_response_mask(batch)

                            metrics["custom/n_hints_improved"] = n_improved
                            metrics["custom/n_hints_still_zero"] = len(valid_hints) - n_improved

                            # Per-problem metrics: count correct after hint for converted groups
                            converted_correct_counts = []
                            reward_sums_updated = reward_tensor.sum(-1)
                            improved_set = set()
                            for k, (j, meta, hint) in enumerate(valid_hints):
                                nc = int((reward_sums_updated[meta['indices']] > 0).sum())
                                if nc > 0:
                                    converted_correct_counts.append(nc)
                                    improved_set.add(j)
                            if converted_correct_counts:
                                metrics["custom/mean_correct_after_hint"] = np.mean(converted_correct_counts)
                                for bucket in range(1, n + 1):
                                    metrics[f"custom/n_converted_{bucket}of{n}"] = sum(
                                        1 for c in converted_correct_counts if c == bucket)

                            # Skip unhintable: problems still zero after hint attempt
                            n_skipped_unhintable = 0
                            if skip_unhintable_cfg:
                                for j, meta in enumerate(zero_meta):
                                    if j not in improved_set:
                                        skip_indices.extend(meta['indices'].tolist())
                                        n_skipped_unhintable += 1
                            metrics["custom/n_skipped_unhintable"] = n_skipped_unhintable

                            _hint_total_elapsed = time.time() - _hint_start
                            metrics["custom/hook1_hint_total_seconds"] = _hint_total_elapsed
                            print(f"[Hook1] Done: {len(zero_meta)} zero-reward, "
                                  f"{len(valid_hints)} hints parsed, {n_improved} improved "
                                  f"({_hint_total_elapsed:.1f}s)")

                    # For vanilla GRPO (no hints): all zero-reward groups are unhintable
                    if self.hint_generator is None and skip_unhintable_cfg and len(zero_meta) > 0:
                        n_skipped_unhintable = len(zero_meta)
                        for meta in zero_meta:
                            skip_indices.extend(meta['indices'].tolist())
                        metrics["custom/n_skipped_unhintable"] = n_skipped_unhintable

                    # Epoch boundary flush for per-problem tracking
                    current_epoch = (self.global_steps - 1) // self._steps_per_epoch + 1
                    if self.global_steps % self._steps_per_epoch == 0:
                        self._flush_epoch_data(current_epoch)

                    # --- Diagnostic trajectory logging (if enabled) ---
                    if self.config.trainer.get("diagnostic_log_dir", None):
                        _diag_pad_id = (self.tokenizer.pad_token_id
                                        if self.tokenizer.pad_token_id is not None
                                        else self.tokenizer.eos_token_id)
                        _diag_data_by_uid = {}
                        # If hints were generated, attach per-problem hint cascade data
                        try:
                            if _diag_hint_data:
                                for pidx_d, data_d in _diag_hint_data.items():
                                    uid_d = zero_meta[pidx_d]['uid']
                                    _diag_data_by_uid[uid_d] = data_d
                        except (NameError, UnboundLocalError, IndexError):
                            pass  # _diag_hint_data not defined or stale index
                        self._dump_diagnostic_log(
                            self.global_steps, batch, reward_tensor, n,
                            _diag_pad_id, diagnostic_hint_data=_diag_data_by_uid)

                    _hook1_elapsed = time.time() - _hook1_start
                    metrics["custom/hook1_post_reward_seconds"] = _hook1_elapsed
                    # ===== END HOOK 1 =====

                    # ===== N_TRAIN SUBSAMPLE: reduce from n_explore to n_train per group =====
                    # When n_train < n_explore (e.g. 8 < 16), subsample each group down
                    # to n_train trajectories BEFORE computing log-probs (saves compute).
                    if n_train < n_explore:
                        selected_indices = []
                        _hint_replaced_uids = set()
                        if hint_replaced is not None and hint_replaced.any():
                            # Track which UIDs had hint-replaced trajectories
                            for uid in unique_uids:
                                uid_mask = (uids == uid)
                                uid_indices = np.where(uid_mask)[0]
                                if hint_replaced[uid_indices].any():
                                    _hint_replaced_uids.add(uid)

                        # Intelligent subsample: prioritize correct trajectories (hint mode only)
                        _use_intelligent_subsample = (self.hint_generator is not None)
                        _n_reordered = 0

                        for uid in unique_uids:
                            uid_mask = (uids == uid)
                            uid_indices = np.where(uid_mask)[0]

                            if _use_intelligent_subsample:
                                # Intelligent selection: keep all correct, fill with incorrect
                                group_rewards = reward_tensor[uid_indices].sum(-1)
                                correct_mask = (group_rewards > 0)
                                correct_idx = uid_indices[correct_mask].tolist()
                                incorrect_idx = uid_indices[~correct_mask].tolist()

                                if len(correct_idx) >= n_train:
                                    # More correct than n_train — keep first n_train correct
                                    keep = correct_idx[:n_train]
                                else:
                                    # Keep all correct, fill remainder with incorrect
                                    keep = correct_idx + incorrect_idx[:n_train - len(correct_idx)]

                                # Track if this differed from blind [:n_train]
                                blind = set(uid_indices[:n_train].tolist())
                                if set(keep) != blind:
                                    _n_reordered += 1

                                selected_indices.extend(sorted(keep))
                            else:
                                # Vanilla GRPO: blind first n_train
                                selected_indices.extend(uid_indices[:n_train].tolist())

                        selected_indices = sorted(selected_indices)
                        n_before = len(batch)
                        keep_mask_sub = torch.zeros(n_before, dtype=torch.bool)
                        for si in selected_indices:
                            keep_mask_sub[si] = True
                        batch = batch[keep_mask_sub]
                        reward_tensor = reward_tensor[keep_mask_sub]
                        # Update hint_replaced mask
                        if hint_replaced is not None:
                            hint_replaced = hint_replaced[keep_mask_sub]
                        # Subsample hint_level_map and hinted_prompt_texts (Python lists)
                        hint_level_map = [hint_level_map[i] for i in selected_indices]
                        hinted_prompt_texts = [hinted_prompt_texts[i] for i in selected_indices]
                        # Update saved rollout log-probs
                        if 'saved_rollout_lp' in dir() and saved_rollout_lp is not None:
                            saved_rollout_lp = saved_rollout_lp[keep_mask_sub]
                        if 'saved_response_mask' in dir() and saved_response_mask is not None:
                            saved_response_mask = saved_response_mask[keep_mask_sub]
                        # Remap skip/curriculum indices from old batch positions to new positions
                        old_to_new = {old: new for new, old in enumerate(selected_indices)}
                        all_correct_indices = [old_to_new[i] for i in all_correct_indices if i in old_to_new]
                        skip_indices = [old_to_new[i] for i in skip_indices if i in old_to_new]
                        # Recompute uid-related arrays for downstream
                        uids = batch.non_tensor_batch['uid']
                        unique_uids = list(dict.fromkeys(uids))
                        _diag_batch_size = len(batch)
                        n = n_train  # GRPO group size is now n_train
                        metrics["custom/n_train_subsample"] = n_train
                        metrics["custom/batch_size_after_subsample"] = len(batch)
                        if _use_intelligent_subsample:
                            metrics["custom/n_subsample_reordered"] = _n_reordered
                        print(f"[Subsample] {n_before} -> {len(batch)} trajectories "
                              f"(n_explore={n_explore}, n_train={n_train}"
                              f"{f', {_n_reordered} reordered' if _use_intelligent_subsample else ''})")
                    # ===== END N_TRAIN SUBSAMPLE =====

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        # --- Log-prob gap diagnostic (zero extra forward passes) ---
                        if hint_replaced.any() and self.hint_generator is not None:
                            _old_lp = batch.batch['old_log_probs']           # [B, resp_len] — fresh
                            _resp_mask = batch.batch['response_mask']         # [B, resp_len] — current
                            _reward_sums_diag = reward_tensor.sum(-1)         # [B]

                            def _masked_mean_lp(lp, mask):
                                """Mean per-token log-prob, masked."""
                                return (lp * mask).sum(-1) / mask.sum(-1).clamp(min=1)

                            # Metric 1: Within-problem (same problem: hint vs own-wrong)
                            if saved_rollout_lp is not None:
                                _own_wrong_lp = _masked_mean_lp(
                                    saved_rollout_lp[hint_replaced],
                                    saved_response_mask[hint_replaced])
                                _hint_new_lp = _masked_mean_lp(
                                    _old_lp[hint_replaced],
                                    _resp_mask[hint_replaced])
                                _gap_within = (_hint_new_lp - _own_wrong_lp).mean().item()
                                metrics['diagnostics/gap_within_problem'] = _gap_within
                                metrics['diagnostics/own_wrong_mean_lp'] = _own_wrong_lp.mean().item()
                                metrics['diagnostics/hint_new_mean_lp'] = _hint_new_lp.mean().item()

                            # Metric 2: Cross-problem (hint_correct vs own_correct)
                            _own_mask = ~hint_replaced
                            _own_correct_mask = _own_mask & (_reward_sums_diag > 0)
                            _hint_correct_mask = hint_replaced & (_reward_sums_diag > 0)
                            _hint_wrong_mask = hint_replaced & (_reward_sums_diag == 0)

                            if _own_correct_mask.any():
                                metrics['diagnostics/own_correct_mean_lp'] = _masked_mean_lp(
                                    _old_lp[_own_correct_mask],
                                    _resp_mask[_own_correct_mask]).mean().item()
                            if _hint_correct_mask.any():
                                metrics['diagnostics/hint_correct_mean_lp'] = _masked_mean_lp(
                                    _old_lp[_hint_correct_mask],
                                    _resp_mask[_hint_correct_mask]).mean().item()
                            if _hint_wrong_mask.any():
                                metrics['diagnostics/hint_wrong_mean_lp'] = _masked_mean_lp(
                                    _old_lp[_hint_wrong_mask],
                                    _resp_mask[_hint_wrong_mask]).mean().item()
                            if _own_correct_mask.any() and _hint_correct_mask.any():
                                metrics['diagnostics/gap_cross_problem'] = (
                                    metrics['diagnostics/hint_correct_mean_lp'] -
                                    metrics['diagnostics/own_correct_mean_lp'])

                            # Metric 3: Per-level breakdown
                            for _lvl_name in set(hint_level_map):
                                if not _lvl_name:
                                    continue
                                _lvl_mask = torch.tensor(
                                    [h == _lvl_name for h in hint_level_map],
                                    dtype=torch.bool)
                                _lvl_correct = _lvl_mask & (_reward_sums_diag > 0)
                                if _lvl_correct.any():
                                    metrics[f'diagnostics/hint_correct_lp_{_lvl_name}'] = \
                                        _masked_mean_lp(
                                            _old_lp[_lvl_correct],
                                            _resp_mask[_lvl_correct]).mean().item()
                                if saved_rollout_lp is not None and _lvl_mask.any():
                                    _own_lp_lvl = _masked_mean_lp(
                                        saved_rollout_lp[_lvl_mask],
                                        saved_response_mask[_lvl_mask])
                                    _hint_lp_lvl = _masked_mean_lp(
                                        _old_lp[_lvl_mask],
                                        _resp_mask[_lvl_mask])
                                    metrics[f'diagnostics/gap_within_{_lvl_name}'] = (
                                        _hint_lp_lvl - _own_lp_lvl).mean().item()
                                metrics[f'diagnostics/n_replaced_{_lvl_name}'] = \
                                    int(_lvl_mask.sum())

                            # Sanity check: rollout vs actor for non-replaced
                            if saved_rollout_lp is not None and _own_mask.any():
                                _rl_own = _masked_mean_lp(
                                    saved_rollout_lp[_own_mask],
                                    saved_response_mask[_own_mask])
                                _al_own = _masked_mean_lp(
                                    _old_lp[_own_mask],
                                    _resp_mask[_own_mask])
                                metrics['diagnostics/sanity_rollout_actor_diff'] = \
                                    (_al_own - _rl_own).mean().item()

                            # Summary counts
                            metrics['diagnostics/n_hint_replaced'] = int(hint_replaced.sum())
                            metrics['diagnostics/n_hint_correct'] = int(_hint_correct_mask.sum())
                            metrics['diagnostics/n_own_correct'] = int(_own_correct_mask.sum())
                            metrics['diagnostics/hint_correct_rate'] = (
                                int(_hint_correct_mask.sum()) /
                                max(int(hint_replaced.sum()), 1))

                            # --- Group B: Aggregated log-prob by hint status (ALL, not just correct) ---
                            _all_hinted_lp = _masked_mean_lp(
                                _old_lp[hint_replaced], _resp_mask[hint_replaced])
                            metrics['diagnostics/all_hinted_mean_lp'] = _all_hinted_lp.mean().item()
                            if (~hint_replaced).any():
                                _all_own_lp = _masked_mean_lp(
                                    _old_lp[~hint_replaced], _resp_mask[~hint_replaced])
                                metrics['diagnostics/all_own_mean_lp'] = _all_own_lp.mean().item()

                            # Learning velocity: step-to-step delta for hint-correct log-prob
                            if _hint_correct_mask.any():
                                _current_hc_lp = _masked_mean_lp(
                                    _old_lp[_hint_correct_mask],
                                    _resp_mask[_hint_correct_mask]).mean().item()
                                if self._prev_hint_correct_lp is not None:
                                    metrics['diagnostics/hinted_correct_lp_delta'] = (
                                        _current_hc_lp - self._prev_hint_correct_lp)
                                self._prev_hint_correct_lp = _current_hc_lp

                            # --- Group C: Variance metrics ---
                            metrics['diagnostics/hinted_lp_std'] = _all_hinted_lp.std().item()

                            # --- Group E: IS preparation ---
                            # Sum (not mean) of log-probs for hinted trajectories
                            _hinted_lp_sum = (
                                _old_lp[hint_replaced] * _resp_mask[hint_replaced]
                            ).sum(-1)
                            metrics['diagnostics/hinted_old_lp_sum_mean'] = _hinted_lp_sum.mean().item()

                            # Per-trajectory gap tensor for IS insertion point (local var, not logged)
                            per_traj_gap = torch.zeros(len(batch), device=_old_lp.device)
                            if saved_rollout_lp is not None:
                                _gap_per_traj = _masked_mean_lp(
                                    _old_lp[hint_replaced],
                                    _resp_mask[hint_replaced]) - _masked_mean_lp(
                                    saved_rollout_lp[hint_replaced],
                                    saved_response_mask[hint_replaced])
                                per_traj_gap[hint_replaced] = _gap_per_traj

                            # Write per-problem gap JSON (supplementary to diagnostic log)
                            _gap_log_dir = self.config.trainer.get(
                                "diagnostic_log_dir", None)
                            if _gap_log_dir and saved_rollout_lp is not None:
                                _gap_log = {
                                    'step': self.global_steps,
                                    'summary': {
                                        k: v for k, v in metrics.items()
                                        if k.startswith('diagnostics/')
                                    },
                                    'per_trajectory': [],
                                }
                                _replaced_idxs = torch.where(hint_replaced)[0]
                                for _ridx in _replaced_idxs:
                                    _ri = _ridx.item()
                                    _own_lp_i = _masked_mean_lp(
                                        saved_rollout_lp[_ri:_ri+1],
                                        saved_response_mask[_ri:_ri+1]).item()
                                    _hint_lp_i = _masked_mean_lp(
                                        _old_lp[_ri:_ri+1],
                                        _resp_mask[_ri:_ri+1]).item()
                                    _pid = batch.non_tensor_batch.get(
                                        'problem_id', [None] * _diag_batch_size)[_ri]
                                    _gap_log['per_trajectory'].append({
                                        'batch_idx': _ri,
                                        'problem_id': str(_pid) if _pid else None,
                                        'uid': str(uids[_ri]),
                                        'hint_level': hint_level_map[_ri],
                                        'correct': bool(_reward_sums_diag[_ri] > 0),
                                        'own_wrong_mean_lp': round(_own_lp_i, 4),
                                        'hint_new_mean_lp': round(_hint_lp_i, 4),
                                        'gap': round(_hint_lp_i - _own_lp_i, 4),
                                    })
                                _gap_path = os.path.join(
                                    _gap_log_dir,
                                    f"step_{self.global_steps:04d}_logprob_gap.json")
                                os.makedirs(_gap_log_dir, exist_ok=True)
                                with open(_gap_path, 'w') as _gf:
                                    json.dump(_gap_log, _gf, indent=2)
                        # --- End log-prob gap diagnostic ---

                        # --- IS Ratio Diagnostic: compute log π_θ_old(y|x̃) ---
                        # For hinted trajectories, the corrected PPO ratio needs
                        # π_θ(y|x) / π_θ_old(y|x̃) instead of π_θ(y|x) / π_θ_old(y|x).
                        # This block computes log π_θ_old(y|x̃) via an additional actor
                        # forward pass and logs IS ratios as diagnostic metrics.
                        _hints_cfg = self.config.get('hints', {})
                        _compute_is = _hints_cfg.get('compute_is_ratio', False) if _hints_cfg else False
                        n_gpus = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                        if (_compute_is and hint_replaced.any()
                                and (self.hint_generator is not None
                                     or _pm_hint_mode == 'prompt_masked')
                                and prompt_mode != 'hinted'):

                            _is_start = time.time()
                            print(f"[IS Ratio] Computing IS ratio diagnostic for "
                                  f"{int(hint_replaced.sum())} hint-replaced trajectories")

                            # 1. Filter to replaced trajectories with stored hinted texts
                            replaced_idxs = torch.where(hint_replaced)[0]
                            valid_replaced = [i for i in replaced_idxs
                                              if hinted_prompt_texts[i.item()]]
                            if valid_replaced:
                                replaced_idxs = torch.tensor(valid_replaced)
                                N_replaced = len(replaced_idxs)

                                # 2. Get response tokens (same y for both prompts)
                                replaced_responses = batch.batch['responses'][replaced_idxs]
                                replaced_resp_mask = batch.batch['response_mask'][replaced_idxs]

                                # 3. Tokenize hinted prompts x̃
                                replaced_hinted_texts = [
                                    hinted_prompt_texts[idx.item()]
                                    for idx in replaced_idxs
                                ]
                                hinted_prompt_proto = self.prompt_builder.tokenize_prompts(
                                    replaced_hinted_texts)

                                # 4. Build input_ids = [x̃ | y] (different prompt, same response)
                                hinted_input_ids = torch.cat([
                                    hinted_prompt_proto.batch['input_ids'],
                                    replaced_responses], dim=1)
                                hinted_attn_mask = torch.cat([
                                    hinted_prompt_proto.batch['attention_mask'],
                                    replaced_resp_mask], dim=1)
                                hinted_pos_ids = compute_position_id_with_mask(
                                    hinted_attn_mask)

                                # 5. Build DataProto for compute_log_prob
                                from tensordict import TensorDict as _TensorDict
                                hinted_batch = DataProto(batch=_TensorDict({
                                    'input_ids': hinted_input_ids,
                                    'attention_mask': hinted_attn_mask,
                                    'position_ids': hinted_pos_ids,
                                    'responses': replaced_responses,
                                }, batch_size=N_replaced))

                                # 6. Forward pass: log π_θ_old(y|x̃)
                                hinted_batch, _is_pad = pad_dataproto_to_divisor(
                                    hinted_batch, n_gpus)
                                hinted_lp_output = self.actor_rollout_wg.compute_log_prob(
                                    hinted_batch)
                                hinted_old_lps = hinted_lp_output.batch[
                                    'old_log_probs'][:N_replaced]

                                # 7. Existing old_log_probs: log π_θ_old(y|x)
                                orig_old_lps = batch.batch['old_log_probs'][replaced_idxs]

                                # 8. Per-token and per-trajectory IS diagnostic
                                # is_log_diff = log π_θ_old(y|x̃) - log π_θ_old(y|x)
                                is_log_diff_token = hinted_old_lps - orig_old_lps
                                is_log_diff_sum = (
                                    is_log_diff_token * replaced_resp_mask
                                ).sum(-1)  # [N_replaced]
                                is_diff_ratio = torch.exp(
                                    is_log_diff_sum.clamp(max=20.0))  # [N_replaced]
                                is_log_diff_mean = (
                                    is_log_diff_token * replaced_resp_mask
                                ).sum(-1) / replaced_resp_mask.sum(-1).clamp(min=1)

                                # ===== Aggregated wandb metrics =====
                                metrics['is_ratio/log_diff_sum_mean'] = \
                                    is_log_diff_sum.mean().item()
                                metrics['is_ratio/log_diff_sum_std'] = \
                                    is_log_diff_sum.std().item() if N_replaced > 1 else 0.0
                                metrics['is_ratio/diff_ratio_mean'] = \
                                    is_diff_ratio.mean().item()
                                metrics['is_ratio/diff_ratio_min'] = \
                                    is_diff_ratio.min().item()
                                metrics['is_ratio/diff_ratio_max'] = \
                                    is_diff_ratio.max().item()
                                metrics['is_ratio/mean_token_log_diff'] = \
                                    is_log_diff_mean.mean().item()
                                metrics['is_ratio/frac_positive'] = \
                                    (is_log_diff_sum > 0).float().mean().item()
                                metrics['is_ratio/n_replaced'] = N_replaced

                                metrics['is_ratio/hinted_old_lp_mean'] = (
                                    (hinted_old_lps * replaced_resp_mask).sum(-1) /
                                    replaced_resp_mask.sum(-1).clamp(min=1)
                                ).mean().item()
                                metrics['is_ratio/orig_old_lp_mean'] = (
                                    (orig_old_lps * replaced_resp_mask).sum(-1) /
                                    replaced_resp_mask.sum(-1).clamp(min=1)
                                ).mean().item()

                                # Per-level aggregated breakdown
                                for _lvl in set(hint_level_map):
                                    if not _lvl:
                                        continue
                                    _lm = torch.tensor(
                                        [hint_level_map[i.item()] == _lvl
                                         for i in replaced_idxs],
                                        dtype=torch.bool)
                                    if _lm.any():
                                        metrics[f'is_ratio/diff_ratio_mean_{_lvl}'] = \
                                            is_diff_ratio[_lm].mean().item()
                                        metrics[f'is_ratio/log_diff_mean_{_lvl}'] = \
                                            is_log_diff_mean[_lm].mean().item()

                                # Per-correctness aggregated breakdown
                                _r_sums_is = reward_tensor.sum(-1)[replaced_idxs]
                                if (_r_sums_is > 0).any():
                                    metrics['is_ratio/diff_ratio_correct'] = \
                                        is_diff_ratio[_r_sums_is > 0].mean().item()
                                if (_r_sums_is == 0).any():
                                    metrics['is_ratio/diff_ratio_wrong'] = \
                                        is_diff_ratio[_r_sums_is == 0].mean().item()

                                # ===== Per-problem JSON log =====
                                _is_log_dir = self.config.trainer.get(
                                    "diagnostic_log_dir", None)
                                if _is_log_dir:
                                    _is_log = {
                                        'step': self.global_steps,
                                        'n_replaced': int(N_replaced),
                                        'per_problem': [],
                                    }
                                    for _ii, _ridx in enumerate(replaced_idxs):
                                        _ri = _ridx.item()
                                        _uid = str(uids[_ri])
                                        _pid_arr = batch.non_tensor_batch.get(
                                            'problem_id',
                                            [None] * _diag_batch_size)
                                        _pid = _pid_arr[_ri]
                                        _h_lp = (
                                            (hinted_old_lps[_ii] * replaced_resp_mask[_ii]).sum() /
                                            replaced_resp_mask[_ii].sum().clamp(min=1)
                                        ).item()
                                        _o_lp = (
                                            (orig_old_lps[_ii] * replaced_resp_mask[_ii]).sum() /
                                            replaced_resp_mask[_ii].sum().clamp(min=1)
                                        ).item()
                                        _is_log['per_problem'].append({
                                            'batch_idx': _ri,
                                            'uid': _uid,
                                            'problem_id': str(_pid) if _pid else None,
                                            'hint_level': hint_level_map[_ri],
                                            'correct': bool(_r_sums_is[_ii] > 0),
                                            'is_log_diff_sum': round(
                                                is_log_diff_sum[_ii].item(), 4),
                                            'is_diff_ratio': round(
                                                is_diff_ratio[_ii].item(), 4),
                                            'is_mean_token_log_diff': round(
                                                is_log_diff_mean[_ii].item(), 4),
                                            'hinted_mean_lp': round(_h_lp, 4),
                                            'orig_mean_lp': round(_o_lp, 4),
                                            'response_len': int(
                                                replaced_resp_mask[_ii].sum().item()),
                                        })
                                    _is_path = os.path.join(
                                        _is_log_dir,
                                        f"step_{self.global_steps:04d}_is_ratio.json")
                                    os.makedirs(_is_log_dir, exist_ok=True)
                                    with open(_is_path, 'w') as _isf:
                                        json.dump(_is_log, _isf, indent=2)

                                # ===== wandb Table (per-problem, interactive) =====
                                try:
                                    import wandb
                                    if wandb.run is not None:
                                        _is_table = wandb.Table(columns=[
                                            "step", "uid", "problem_id",
                                            "hint_level", "correct",
                                            "is_diff_ratio", "is_log_diff_sum",
                                            "is_mean_token_log_diff",
                                            "hinted_mean_lp", "orig_mean_lp",
                                            "response_len",
                                        ])
                                        for _ii, _ridx in enumerate(replaced_idxs):
                                            _ri = _ridx.item()
                                            _uid = str(uids[_ri])
                                            _pid_arr = batch.non_tensor_batch.get(
                                                'problem_id',
                                                [None] * _diag_batch_size)
                                            _pid = _pid_arr[_ri]
                                            _h_lp = (
                                                (hinted_old_lps[_ii] * replaced_resp_mask[_ii]).sum() /
                                                replaced_resp_mask[_ii].sum().clamp(min=1)
                                            ).item()
                                            _o_lp = (
                                                (orig_old_lps[_ii] * replaced_resp_mask[_ii]).sum() /
                                                replaced_resp_mask[_ii].sum().clamp(min=1)
                                            ).item()
                                            _is_table.add_data(
                                                self.global_steps,
                                                _uid,
                                                str(_pid) if _pid else "",
                                                hint_level_map[_ri],
                                                bool(_r_sums_is[_ii] > 0),
                                                round(is_diff_ratio[_ii].item(), 4),
                                                round(is_log_diff_sum[_ii].item(), 4),
                                                round(is_log_diff_mean[_ii].item(), 4),
                                                round(_h_lp, 4),
                                                round(_o_lp, 4),
                                                int(replaced_resp_mask[_ii].sum().item()),
                                            )
                                        wandb.log(
                                            {"is_ratio/per_problem_table": _is_table},
                                            step=self.global_steps)
                                except (ImportError, Exception):
                                    pass  # wandb not available or not initialized

                                print(f"[IS Ratio] Done. N_replaced={N_replaced}, "
                                      f"ratio_mean={is_diff_ratio.mean().item():.3f}, "
                                      f"mean_token_log_diff={is_log_diff_mean.mean().item():.4f}")

                            metrics['custom/is_ratio_seconds'] = time.time() - _is_start
                        # --- End IS Ratio Diagnostic ---

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        # Use n_train for GRPO group size when subsampled, else use rollout.n
                        _grpo_group_size = n_train if n_train < n_explore else self.config.actor_rollout_ref.rollout.n
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=_grpo_group_size,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )

                    # ===== CUSTOM HOOK 2: Post-advantage, pre-actor-update =====
                    # Access available here:
                    #   - batch: now has advantages, returns, values, old_log_probs
                    #   - All non_tensor_batch fields still accessible
                    # Can do here:
                    #   - Filter/reweight samples based on advantages
                    #   - Modify advantages (e.g., clip, normalize differently)
                    #   - Apply curriculum-based sample weighting
                    _hook2_start = time.time()

                    advantages = batch.batch['advantages']
                    metrics["custom/advantage_mean"] = advantages.mean().item()
                    metrics["custom/advantage_std"] = advantages.std().item()
                    metrics["custom/advantage_min"] = advantages.min().item()
                    metrics["custom/advantage_max"] = advantages.max().item()

                    # --- Diagnostic: Split advantages by hint status ---
                    if hint_replaced is not None and hint_replaced.any():
                        _resp_mask_h2 = batch.batch['response_mask']
                        # Per-trajectory mean advantage (response-masked)
                        _adv_per_traj = (advantages * _resp_mask_h2).sum(-1) / \
                                        _resp_mask_h2.sum(-1).clamp(min=1)

                        # Group A: Advantage splits (hinted vs own)
                        metrics['diagnostics/hinted_advantage_mean'] = \
                            _adv_per_traj[hint_replaced].mean().item()
                        if (~hint_replaced).any():
                            metrics['diagnostics/own_advantage_mean'] = \
                                _adv_per_traj[~hint_replaced].mean().item()

                        # Group C: Fraction of hinted with positive advantage
                        metrics['diagnostics/hinted_frac_positive_adv'] = \
                            (_adv_per_traj[hint_replaced] > 0).float().mean().item()

                        # Group D: Per-level advantage
                        for _lvl in set(hint_level_map):
                            if not _lvl:
                                continue
                            _lm = torch.tensor(
                                [h == _lvl for h in hint_level_map],
                                dtype=torch.bool)
                            if _lm.any():
                                metrics[f'diagnostics/advantage_mean_{_lvl}'] = \
                                    _adv_per_traj[_lm].mean().item()

                        # Group D: Per-problem own-correct count
                        _reward_sums_h2 = reward_tensor.sum(-1)
                        _ocp = []
                        for uid in unique_uids:
                            _ui = np.where(uids == uid)[0]
                            _own = ~hint_replaced[_ui]
                            if _own.any():
                                _ocp.append(
                                    (_reward_sums_h2[_ui][_own] > 0).sum().item())
                        if _ocp:
                            metrics['diagnostics/n_correct_own_per_problem_mean'] = \
                                np.mean(_ocp)

                    # ===== IS WEIGHTING INSERTION POINT =====
                    # Off-policy correction for hint-replaced trajectories.
                    # See doc/plan_IS_implementation.md for full design.
                    # See ~/LUFFY/ExGRPO/exgrpo/verl/verl/mix_src/ for reference impl.
                    #
                    # LOCAL VARIABLES AVAILABLE HERE:
                    #   hint_replaced          [B] bool    — which trajectories were hint-replaced (line 1266)
                    #   hint_level_map         [B] str[]   — hint level that solved each (line 1267)
                    #   saved_rollout_lp       [B, resp_len] or None — original rollout log-probs before replacement (line 1272-1274)
                    #   saved_response_mask    [B, resp_len] — original response mask before replacement (line 1268)
                    #   skip_indices           set of ints — indices to skip in actor update
                    #   all_correct_indices    set of ints — all-correct group indices
                    #
                    # BATCH FIELDS AVAILABLE HERE:
                    #   batch.batch['advantages']       [B, resp_len] — GRPO-normalized advantages
                    #   batch.batch['old_log_probs']    [B, resp_len] — log π_θ(y|x_original) for ALL trajectories
                    #   batch.batch['response_mask']    [B, resp_len] — current response mask
                    #   batch.batch['token_level_rewards'] [B, resp_len]
                    #
                    # STRATEGY A (zero veRL changes):
                    #   Compute is_weights [B] and pre-multiply advantages:
                    #     batch.batch['advantages'] *= is_weights.unsqueeze(1)
                    #   advantages flows through select_keys in dp_actor.py (line 383) naturally.
                    #   See LUFFY mix_trainer.py:441-444 for identical pattern (prefix_weight * advantages).
                    #
                    # STRATEGY B (requires dp_actor.py changes):
                    #   Add per-trajectory tensors to batch.batch:
                    #     batch.batch['is_weights'] = is_weights        # [B] or [B, resp_len]
                    #     batch.batch['hint_clip_ratio_high'] = clips   # [B]
                    #   Then add them to select_keys in dp_actor.py (see comment there).
                    #   These survive curriculum filtering (batch[keep_mask] slices all keys).
                    #   See LUFFY mix_actor.py:69 for how they pass prefix_mask through select_keys.
                    # ===== END IS INSERTION POINT =====

                    # ===== IS CORRECTION: per_token_uniform (Approach C / B2) =====
                    _is_mode = self.config.hints.get('is_correction_mode', 'none')
                    if (_is_mode == 'per_token_uniform'
                            and hint_replaced is not None and hint_replaced.any()):
                        _clamp_min = self.config.hints.get('is_correction_clamp_min', 0.3)
                        # is_log_diff_mean [N_replaced] is computed in the IS ratio block above
                        # (line ~3545). It's only defined when compute_is_ratio=true AND
                        # hint replacements occurred this step.
                        try:
                            _has_is_data = (is_log_diff_mean is not None
                                            and replaced_idxs is not None)
                        except NameError:
                            _has_is_data = False

                        if not _has_is_data:
                            logger.warning(
                                "is_correction_mode=per_token_uniform requires "
                                "compute_is_ratio=true; skipping IS correction this step")
                        else:
                            # w_i = clamp(exp(-mean_per_token_log_diff_i)^gamma, clamp_min, 1.0)
                            _is_gamma = self.config.hints.get('is_weight_power', 1.0)
                            _is_w = torch.exp(-is_log_diff_mean).pow(_is_gamma).clamp(
                                min=_clamp_min, max=1.0)  # [N_replaced]
                            # Full-batch tensor (1.0 for non-replaced trajectories)
                            _is_full = torch.ones(
                                batch.batch['advantages'].shape[0],
                                device=batch.batch['advantages'].device)
                            _is_full[replaced_idxs] = _is_w
                            # Scale advantages (broadcast across token dim)
                            batch.batch['advantages'] = (
                                batch.batch['advantages'] * _is_full.unsqueeze(1))
                            metrics['is_correction/mean_weight'] = _is_w.mean().item()
                            metrics['is_correction/min_weight'] = _is_w.min().item()
                            metrics['is_correction/frac_clamped'] = (
                                _is_w <= _clamp_min + 1e-6).float().mean().item()
                            metrics['is_correction/n_corrected'] = len(_is_w)
                    elif (_is_mode == 'per_token'
                            and hint_replaced is not None and hint_replaced.any()):
                        _clamp_min = self.config.hints.get('is_correction_clamp_min', 0.3)
                        try:
                            _has_is_data = (is_log_diff_token is not None
                                            and replaced_idxs is not None
                                            and replaced_resp_mask is not None)
                        except NameError:
                            _has_is_data = False

                        if not _has_is_data:
                            logger.warning(
                                "is_correction_mode=per_token requires "
                                "compute_is_ratio=true; skipping IS correction this step")
                        else:
                            # Per-token IS weight: w_t = clamp(exp(-diff_t)^gamma, clamp_min, 1.0)
                            # diff_t = log π_old(y_t|x̃,y_{<t}) - log π_old(y_t|x,y_{<t})
                            _is_gamma = self.config.hints.get('is_weight_power', 1.0)
                            _is_token_w = torch.exp(-is_log_diff_token).pow(_is_gamma).clamp(
                                min=_clamp_min, max=1.0)  # [N_replaced, resp_len]
                            # For padding tokens, set weight to 1.0
                            _is_token_w = (_is_token_w * replaced_resp_mask
                                           + (1.0 - replaced_resp_mask))

                            # Shape assertion
                            assert _is_token_w.shape[-1] == batch.batch['advantages'].shape[-1], \
                                (f"IS token weights resp_len {_is_token_w.shape[-1]} != "
                                 f"advantages {batch.batch['advantages'].shape[-1]}")

                            # Build full-batch [B, resp_len] (1.0 for non-replaced)
                            _is_full = torch.ones_like(batch.batch['advantages'])
                            _is_full[replaced_idxs] = _is_token_w

                            # Apply to advantages
                            batch.batch['advantages'] = (
                                batch.batch['advantages'] * _is_full)

                            # Metrics: per-token weight statistics (response tokens only)
                            _resp_w = _is_token_w[replaced_resp_mask.bool()]
                            _resp_counts = replaced_resp_mask.sum(-1).clamp(min=1)
                            _mean_w_per_traj = (
                                (_is_token_w * replaced_resp_mask).sum(-1) / _resp_counts)

                            metrics['is_correction/per_token_weight_mean'] = (
                                _resp_w.mean().item())
                            metrics['is_correction/per_token_weight_std'] = (
                                _resp_w.std().item() if len(_resp_w) > 1 else 0.0)
                            metrics['is_correction/per_token_weight_min'] = (
                                _resp_w.min().item())
                            metrics['is_correction/intra_traj_std'] = torch.sqrt(
                                (((_is_token_w - _mean_w_per_traj.unsqueeze(1)) ** 2)
                                 * replaced_resp_mask).sum(-1) / _resp_counts
                            ).mean().item()
                            metrics['is_correction/frac_below_half'] = (
                                (_resp_w < 0.5).float().mean().item())
                            metrics['is_correction/frac_clamped'] = (
                                (_resp_w <= _clamp_min + 1e-6).float().mean().item())
                            metrics['is_correction/n_corrected'] = len(replaced_idxs)
                    # ===== END IS CORRECTION =====

                    # Pass hint_replaced mask to actor for per-status ratio/clipfrac diagnostics
                    # Also sets up infrastructure for IS Strategy B (is_weights, hint_clip_ratio_high)
                    if hint_replaced is not None and hint_replaced.any():
                        _resp_len_h2 = batch.batch['advantages'].shape[1]
                        batch.batch['hint_replaced_mask'] = hint_replaced.unsqueeze(1).expand(
                            -1, _resp_len_h2).float().to(batch.batch['advantages'].device)

                    _hook2_elapsed = time.time() - _hook2_start
                    metrics["custom/hook2_pre_update_seconds"] = _hook2_elapsed
                    # ===== END HOOK 2 =====

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # --- Curriculum: skip groups from actor update ---
                        # Combine all skip sources: all_correct + unhintable + mastered
                        actor_batch = batch
                        skip_actor = False
                        combined_skip = set(all_correct_indices) | set(skip_indices)
                        if combined_skip:
                            keep_mask = torch.ones(len(batch), dtype=torch.bool)
                            for idx in combined_skip:
                                keep_mask[idx] = False
                            n_total_skipped = len(combined_skip)
                            if keep_mask.any():
                                actor_batch = batch[keep_mask]
                                metrics["custom/n_all_correct_skipped"] = len(all_correct_indices)
                                metrics["custom/n_total_skipped_samples"] = n_total_skipped
                                metrics["custom/batch_size_after_skip"] = len(actor_batch)
                                print(f"[Curriculum] Skipping {n_total_skipped} samples "
                                      f"(all-correct={len(all_correct_indices)}, "
                                      f"unhintable+mastered={len(skip_indices)}), "
                                      f"actor batch: {len(actor_batch)}/{len(batch)}")
                            else:
                                # ALL groups skipped — skip actor update entirely
                                skip_actor = True
                                metrics["custom/n_all_correct_skipped"] = len(all_correct_indices)
                                metrics["custom/n_total_skipped_samples"] = n_total_skipped
                                metrics["custom/skip_actor_update"] = 1
                                print(f"[Curriculum] ALL groups skipped — skipping actor update")

                        # update actor
                        if not skip_actor:
                            with marked_timer("update_actor", timing_raw, color="red"):
                                actor_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                actor_output = self.actor_rollout_wg.update_actor(actor_batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                        # ===== CUSTOM HOOK 3: Post-actor-update =====
                        # Access available here:
                        #   - actor_output: dict with training loss, grad_norm, etc.
                        #   - batch: full batch that was used for the update
                        #   - All metrics from this step
                        # Can do here:
                        #   - Log custom training stats
                        #   - Track per-problem reward progression (curriculum)
                        #   - Save trajectories for analysis
                        _hook3_start = time.time()

                        metrics["custom/phase_generation_seconds"] = _generation_end - _generation_start
                        metrics["custom/phase_total_step_seconds"] = time.time() - _step_start

                        _hook3_elapsed = time.time() - _hook3_start
                        metrics["custom/hook3_post_update_seconds"] = _hook3_elapsed
                        # ===== END HOOK 3 =====

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # === hint_SFT: Epoch boundary SFT phase ===
                    hints_cfg_sft = self.config.get('hints', {})
                    if (hints_cfg_sft.get('enabled', False)
                            and hints_cfg_sft.get('hint_mode', '') == 'hint_SFT'
                            and self.global_steps % self._steps_per_epoch == 0
                            and self._sft_gold_buffer):
                        _sft_epoch = self.global_steps // self._steps_per_epoch
                        _sft_start_time = time.time()
                        sft_cfg = hints_cfg_sft.get('sft_config', {})
                        sft_epochs = sft_cfg.get('sft_epochs', 1)

                        # Build SFT batch from gold buffer
                        sft_batch = self._build_sft_batch(self._sft_gold_buffer)

                        if sft_batch is not None:
                            n_gold = len(self._sft_gold_buffer)
                            level_counts = {}
                            unique_pids = set()
                            for item in self._sft_gold_buffer:
                                level_counts[item['hint_level']] = level_counts.get(item['hint_level'], 0) + 1
                                if item['problem_id']:
                                    unique_pids.add(item['problem_id'])

                            print(f"[HintSFT] Epoch {_sft_epoch} SFT phase: "
                                  f"{n_gold} gold samples, "
                                  f"{len(unique_pids)} unique problems, "
                                  f"levels: {level_counts}")

                            # sft_update handles mini-batching and sft_epochs internally
                            # via InlineSFTTrainer.fit_gold()
                            sft_output = self.actor_rollout_wg.sft_update_actor(sft_batch)
                            sft_metrics = reduce_metrics(sft_output.meta_info["metrics"])

                            # Log SFT metrics
                            sft_log = {}
                            for k, v in sft_metrics.items():
                                sft_log[k if k.startswith('sft/') else f"sft/{k}"] = v
                            sft_log['sft/n_gold_samples'] = n_gold
                            sft_log['sft/unique_problems'] = len(unique_pids)
                            sft_log['sft/epoch'] = _sft_epoch
                            sft_log['sft/sft_epochs'] = sft_epochs
                            for level, count in level_counts.items():
                                sft_log[f'sft/n_gold_{level}'] = count
                            logger.log(data=sft_log, step=self.global_steps)

                            print(f"[HintSFT] SFT completed: "
                                  f"loss={sft_metrics.get('sft/loss', 'N/A'):.4f}, "
                                  f"n_steps={sft_metrics.get('sft/n_steps', 'N/A')}, "
                                  f"n_samples={sft_metrics.get('sft/n_samples', 'N/A')}")

                            _sft_elapsed = time.time() - _sft_start_time
                            print(f"[HintSFT] SFT phase completed in {_sft_elapsed:.1f}s")

                            # Save gold data diagnostics
                            diag_dir = self.config.trainer.get("diagnostic_log_dir", None)
                            if diag_dir:
                                os.makedirs(diag_dir, exist_ok=True)
                                gold_diag = {
                                    'epoch': _sft_epoch,
                                    'n_gold': n_gold,
                                    'unique_problems': len(unique_pids),
                                    'level_distribution': level_counts,
                                    'sft_elapsed_seconds': _sft_elapsed,
                                    'samples': [
                                        {
                                            'problem_id': item['problem_id'],
                                            'hint_level': item['hint_level'],
                                            'step': item['step'],
                                            'response_length': len(item['gold_response_text']),
                                        }
                                        for item in self._sft_gold_buffer
                                    ],
                                }
                                gold_path = os.path.join(diag_dir, f"epoch_{_sft_epoch:04d}_sft_gold.json")
                                with open(gold_path, 'w') as f:
                                    json.dump(gold_diag, f, indent=2)
                                print(f"[HintSFT] Saved gold diagnostics to {gold_path}")

                        # Clear buffer for next epoch
                        self._sft_gold_buffer = []

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return


# ===== Helper functions (from Scaf-GRPO, for future use) =====

def remove_uids_from_dataproto(data_proto, uids_to_remove):
    """Remove samples with specified UIDs from DataProto."""
    all_uids = data_proto.non_tensor_batch['uid']
    remaining_indices = [idx for idx, uid in enumerate(all_uids) if uid not in uids_to_remove]

    seen_uids = set()
    unique_indices = []
    for idx in remaining_indices:
        uid = all_uids[idx]
        if uid not in seen_uids:
            seen_uids.add(uid)
            unique_indices.append(idx)

    new_non_tensor_batch = {}
    for k, v in data_proto.non_tensor_batch.items():
        if isinstance(v, list):
            new_non_tensor_batch[k] = [v[i] for i in unique_indices]
        else:
            arr = np.array(v)
            new_non_tensor_batch[k] = arr[unique_indices]

    return DataProto(
        batch=None,
        non_tensor_batch=new_non_tensor_batch,
    )


def compute_position_id_with_mask(mask):
    """Recompute position IDs from attention mask."""
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)


def _expand_batch_prompt_dim(data, target_prompt_len, pad_token_id):
    """Left-pad prompts (and rebuild input_ids, attention_mask, position_ids) to target length.

    Used when hinted prompts are longer than original prompts and we need uniform
    tensor dimensions for row-level replacement in the batch.
    """
    current_len = data.batch['prompts'].shape[1]
    if target_prompt_len <= current_len:
        return

    pad_len = target_prompt_len - current_len
    B = data.batch['prompts'].shape[0]
    device = data.batch['prompts'].device
    dtype = data.batch['prompts'].dtype

    # Left-pad prompts
    prompt_pad = torch.full((B, pad_len), pad_token_id, dtype=dtype, device=device)
    data.batch['prompts'] = torch.cat([prompt_pad, data.batch['prompts']], dim=1)

    # Rebuild input_ids = cat(prompts, responses)
    data.batch['input_ids'] = torch.cat([data.batch['prompts'], data.batch['responses']], dim=1)

    # Expand attention_mask (0 for new padding)
    mask_pad = torch.zeros((B, pad_len), dtype=data.batch['attention_mask'].dtype, device=device)
    data.batch['attention_mask'] = torch.cat([mask_pad, data.batch['attention_mask']], dim=1)

    # Recompute position_ids
    data.batch['position_ids'] = compute_position_id_with_mask(data.batch['attention_mask'])
