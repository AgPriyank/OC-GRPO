import os
import argparse

# Import everything from baseline.py
from baseline import (
    evaluate_model, compute_statistics, print_summary, 
    save_results, log_to_wandb, load_gsm8k_problems,
    load_aime_problems, load_gsmplus_problems, load_math_problems
)


def merge_lora_and_save(base_model_path: str, lora_path: str, output_path: str) -> str:
    """
    Merge LoRA adapters with base model and save to disk.
    vLLM will load the merged model from this path.
    
    This function uses transformers ONLY for merging, not for inference.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print(f"\n{'='*60}")
    print("MERGING LORA ADAPTERS WITH BASE MODEL")
    print(f"{'='*60}")
    print(f"Base model: {base_model_path}")
    print(f"LoRA adapters: {lora_path}")
    print(f"Output path: {output_path}")
    
    # Check if already merged
    if os.path.exists(os.path.join(output_path, "config.json")):
        print(f"✓ Merged model already exists at {output_path}")
        print(f"{'='*60}\n")
        return output_path
    
    print("\n[1/4] Loading base model with transformers...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"  # Load to GPU for merging
    )
    
    print("[2/4] Loading LoRA adapters...")
    model = PeftModel.from_pretrained(model, lora_path)
    
    print("[3/4] Merging LoRA with base model...")
    model = model.merge_and_unload()
    
    print(f"[4/4] Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Free memory
    del model
    torch.cuda.empty_cache()
    
    print(f"✓ Merged model saved successfully!")
    print(f"{'='*60}\n")
    
    return output_path


def resolve_lora_path(path: str) -> str:
    """
    Resolve LoRA adapter path, handling both TRL and veRL checkpoint layouts.

    TRL layout:  runs/run_016/checkpoint-final/adapter_config.json
    veRL layout: checkpoints/.../global_step_42/actor/lora_adapter/adapter_config.json
    """
    path = path.rstrip('/')
    # Direct path: adapter_config.json exists here (TRL format)
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return path
    # veRL layout: path/actor/lora_adapter/
    verl_path = os.path.join(path, "actor", "lora_adapter")
    if os.path.exists(os.path.join(verl_path, "adapter_config.json")):
        return verl_path
    raise FileNotFoundError(
        f"adapter_config.json not found at {path} or {verl_path}"
    )


def get_merged_model_name(base_model: str, lora_path: str, original_lora_path: str) -> str:
    """
    Generate a meaningful merged model name from the checkpoint path.

    Handles both TRL paths (runs/run_016/checkpoint-final) and
    veRL paths (checkpoints/.../grpo_run016/global_step_42/actor/lora_adapter).
    """
    model_name = os.path.basename(base_model.rstrip('/'))
    lora_normalized = lora_path.rstrip('/')

    # veRL layout: path ends in .../actor/lora_adapter
    if lora_normalized.endswith(os.path.join("actor", "lora_adapter")):
        # Walk up to global_step_N level
        step_dir = os.path.dirname(os.path.dirname(lora_normalized))
        step_name = os.path.basename(step_dir)          # e.g., "global_step_14"
        run_name = os.path.basename(os.path.dirname(step_dir))  # e.g., "grpo_run016"
        return f"{model_name}_{run_name}_{step_name}_merged"

    # TRL layout: runs/run_016/checkpoint-final
    checkpoint_dir = os.path.basename(lora_normalized)
    run_dir = os.path.basename(os.path.dirname(lora_normalized))
    return f"{model_name}_{run_dir}_{checkpoint_dir}_merged"


def get_base_model_from_adapter_config(lora_path: str) -> str:
    """
    Extract base model name from adapter_config.json in the LoRA checkpoint.
    """
    import json
    adapter_config_path = os.path.join(lora_path, "adapter_config.json")

    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"adapter_config.json not found at {adapter_config_path}")

    with open(adapter_config_path, 'r') as f:
        config = json.load(f)

    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"base_model_name_or_path not found in {adapter_config_path}")

    return base_model


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GRPO-trained model (with LoRA)'
    )

    # Model paths
    parser.add_argument('--base_model', type=str, default=None,
                        help='Base model path (optional - auto-detected from adapter_config.json if not provided)')
    parser.add_argument('--lora_path', type=str, required=True,
                        help='Path to LoRA checkpoint (e.g., runs/run_004/checkpoint-final)')
    parser.add_argument('--merged_model_dir', type=str, 
                        default='models/merged_models',
                        help='Directory to save merged model (will be cached)')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='math',
                        choices=['gsm8k', 'aime', 'gsmplus', 'math'])
    parser.add_argument('--problems_file', type=str, default=None,
                        help='Path to local JSON problems file (overrides --dataset loading)')
    parser.add_argument('--n_problems', type=int, default=100)
    parser.add_argument('--gsm8k_split', type=str, default='test',
                        choices=['train', 'test'])
    parser.add_argument('--use_chat_template', action='store_true', default=True)
    parser.add_argument('--no_chat_template', dest='use_chat_template', 
                        action='store_false')
    
    # Generation parameters (same as baseline.py)
    parser.add_argument('--n_trajectories', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    
    # Infrastructure
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True)
    
    # W&B logging
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    
    args = parser.parse_args()

    # Resolve LoRA path (handles both TRL and veRL checkpoint layouts)
    original_lora_path = args.lora_path
    args.lora_path = resolve_lora_path(args.lora_path)
    if args.lora_path != original_lora_path:
        print(f"Resolved veRL checkpoint: {args.lora_path}")

    # Auto-detect base model from adapter_config.json if not provided
    if args.base_model is None:
        args.base_model = get_base_model_from_adapter_config(args.lora_path)
        print(f"Auto-detected base model: {args.base_model}")

    print("="*60)
    print("EVALUATING GRPO-TRAINED MODEL")
    print("="*60)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Base model: {args.base_model}")
    print(f"LoRA path: {args.lora_path}")
    print(f"Problems: {args.n_problems}")
    print("="*60)
    
    # =====================================================================
    # STEP 1: Merge LoRA + Save to Disk
    # =====================================================================
    # vLLM needs a model directory, so we merge and save first
    
    merged_name = get_merged_model_name(args.base_model, args.lora_path, original_lora_path)
    merged_path = os.path.join(args.merged_model_dir, merged_name)
    
    merged_model_path = merge_lora_and_save(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        output_path=merged_path
    )
    
    # =====================================================================
    # STEP 2: Load Problems (same as baseline.py)
    # =====================================================================
    
    if args.problems_file:
        # Load from local JSON file
        import json
        print(f"Loading problems from local file: {args.problems_file}")
        with open(args.problems_file, 'r') as f:
            data = json.load(f)
        problems = data['problems'] if 'problems' in data else data
        if args.n_problems and args.n_problems < len(problems):
            problems = problems[:args.n_problems]
        print(f"Loaded {len(problems)} problems from {args.problems_file}")
    elif args.dataset == 'gsm8k':
        problems = load_gsm8k_problems(
            n_problems=args.n_problems,
            split=args.gsm8k_split,
            seed=args.seed
        )
    elif args.dataset == 'aime':
        problems = load_aime_problems(
            n_problems=args.n_problems,
            split='train',
            seed=args.seed
        )
    elif args.dataset == 'gsmplus':
        problems = load_gsmplus_problems(
            n_problems=args.n_problems,
            split='test',
            seed=args.seed
        )
    elif args.dataset == 'math':
        problems = load_math_problems(
            n_problems=args.n_problems,
            levels=[3, 4, 5],
            subjects=None,
            split='train',
            seed=args.seed
        )
    
    # =====================================================================
    # STEP 3: Evaluate with vLLM (exactly like baseline.py)
    # =====================================================================
    # Now baseline.py's evaluate_model() will use vLLM to load merged_model_path
    
    print(f"\nvLLM will load model from: {merged_model_path}")
    
    results = evaluate_model(
        model_path=merged_model_path,  # ← vLLM loads from this path
        problems=problems,
        n_trajectories=args.n_trajectories,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        use_chat_template=args.use_chat_template,
        dataset=args.dataset
    )
    
    # =====================================================================
    # STEP 4-7: Statistics, Print, Save, Log (same as baseline.py)
    # =====================================================================
    
    stats = compute_statistics(results)
    print_summary(stats)


    args.model = f"{args.base_model} (LoRA: {os.path.basename(args.lora_path)})"
    
    save_results(results, stats, args, args.output_dir)
    log_to_wandb(results, stats, args)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Merged model: {merged_model_path}")
    print(f"Results: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()