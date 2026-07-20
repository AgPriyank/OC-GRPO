"""
Entry point for veRL-based GRPO training with custom hooks.
Adapted from Scaf-GRPO's main_ppo.py pattern.

Usage:
  python -m verl_grpo.main_ppo  [config overrides...]

Or via launch script:
  bash scripts/launch_train.sh
"""

import os
import random
import socket

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.dataset.sampler import AbstractSampler
from verl.utils.import_utils import load_extern_type


def set_global_seed(seed: int):
    """Set global random seed for reproducibility across all RNG sources.

    This seeds Python's random, numpy, PyTorch CPU, and all CUDA devices.
    Called early in TaskRunner.run() before model/data initialization.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Seed] Global seed set to {seed} (random, numpy, torch, cuda)")


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env=PPO_RAY_RUNTIME_ENV,
            num_cpus=config.ray_init.num_cpus,
        )

    if config.trainer.get("profile_steps") is not None and len(config.trainer.get("profile_steps", [])) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        # Set global seed before any initialization (model, data, vLLM)
        global_seed = config.trainer.get("seed", 0)
        if global_seed > 0:
            set_global_seed(global_seed)

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download model to local
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        # Tokenizer + processor
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # vLLM version check for LoRA
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # Worker classes (FSDP strategy)
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        # Import our custom trainer (not veRL's original)
        from verl_grpo.trainer.ray_trainer import ResourcePoolManager, Role

        # Role → worker mapping
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Resource pool
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # Reward model (if model-based)
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Reference policy (for KL)
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Load reward functions
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        # Create datasets using our custom dataset class (or default)
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize hint components if enabled
        hint_generator = None
        prompt_builder = None
        hints_config = config.get("hints", {})
        if hints_config.get("enabled", False):
            from verl_grpo.hints import HintGenerator, HintedPromptBuilder

            hint_mode = hints_config.get("hint_mode", "solution_aware")
            hint_generator = HintGenerator(tokenizer=tokenizer, hint_mode=hint_mode)
            prompt_builder = HintedPromptBuilder(
                tokenizer=tokenizer,
                max_prompt_length=config.data.max_prompt_length,
                enable_thinking=config.data.get("enable_thinking", None),
            )
            print(f"Adaptive hints enabled: mode={hint_mode}, "
                  f"prompt_mode={hints_config.get('prompt_mode', 'original')}")

        # Initialize external hint server client (3-GPU mode)
        hint_client = None
        if hints_config.get("use_external_server", False):
            from verl_grpo.hints import VLLMHintClient

            hint_client = VLLMHintClient(
                server_url=hints_config.get("hint_server_url", "http://localhost:8105"),
                lora_sync_path=hints_config.get("lora_sync_path", "/tmp/hint_lora_sync"),
            )
            hint_client.wait_until_ready(timeout=300)
            print(f"External hint server connected: {hints_config['hint_server_url']} "
                  f"(model_mode={hints_config.get('hint_model_mode', 'base')}, "
                  f"lora_sync={hints_config.get('lora_sync_frequency', 'never')})")

        # Initialize frontier hint client (API-based hints from Claude/GPT-4)
        frontier_client = None
        hint_source = hints_config.get("hint_source", "self")
        if hint_source == "frontier":
            from verl_grpo.hints import FrontierHintClient

            frontier_client = FrontierHintClient(
                api=hints_config.get("frontier_api", "anthropic"),
                model=hints_config.get("frontier_model", "claude-sonnet-4-20250514"),
                max_concurrency=hints_config.get("frontier_concurrency", 10),
                max_retries=hints_config.get("frontier_max_retries", 5),
            )
            print(f"Frontier hint client initialized: api={hints_config.get('frontier_api', 'anthropic')}, "
                  f"model={hints_config.get('frontier_model', 'claude-sonnet-4-20250514')}")

        # Import our custom RayPPOTrainer (with hooks)
        from verl_grpo.trainer.ray_trainer import RayPPOTrainer

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            hint_generator=hint_generator,
            prompt_builder=prompt_builder,
            hint_client=hint_client,
            frontier_client=frontier_client,
        )

        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create dataset, using custom class if specified in config."""
    from torch.utils.data import Dataset

    # Default to our HintRLHFDataset
    from verl_grpo.dataset.rl_dataset import HintRLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"Custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    else:
        dataset_cls = HintRLHFDataset

    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )
    return dataset


def create_rl_sampler(data_config, dataset):
    """Create sampler for the dataset."""
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(data_source=dataset, data_config=data_config)
        assert isinstance(sampler, AbstractSampler)
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
