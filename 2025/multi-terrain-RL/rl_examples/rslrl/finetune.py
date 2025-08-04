# finetune.py

import argparse
import os
import pickle
import torch

from rsl_rl.runners import OnPolicyRunner
from chrono_crmenv import ChronoQuadrupedEnv  # Use CRM granular terrain

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="chrono-quadruped")
    parser.add_argument("--ckpt", type=int, default=2000, help="Checkpoint number to load")
    parser.add_argument("-n", "--num_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=2000, help="Additional iterations for fine-tuning") # Reduced for fine-tuning
    parser.add_argument("--save_suffix", type=str, default="_crm_finetuned", help="Suffix for saving finetuned model")
    args = parser.parse_args()

    # --- Load existing configurations from original training ---
    original_log_dir = f"data/rl_models/rslrl"
    
    env_cfg, train_cfg = pickle.load(open(f"{original_log_dir}/cfgs.pkl", "rb"))
    print(f"Loaded configurations from {original_log_dir}/cfgs.pkl")
    
    # --- PROPOSED CHANGE: Create a specific configuration for fine-tuning ---
    print("Applying fine-tuning specific hyperparameters...")
    
    # 1. Drastically reduce the learning rate
    # The original is 0.001 (or 1e-3)
    train_cfg["algorithm"]["learning_rate"] = 1e-5  # A good starting point is 1/100th of the original
    train_cfg["algorithm"]["clip_param"] = 0.02
    train_cfg["algorithm"]["desired_kl"] = 0.001
    train_cfg["algorithm"]["max_grad_norm"] = 0.5
    train_cfg["num_steps_per_env"] = 500
    train_cfg["save_interval"] = 100

    env_cfg["max_steps"] = 500


    # --- Setup separate directory for finetuned models ---
    finetune_log_dir = f"data/rl_models/rslrl_finetune"
    os.makedirs(finetune_log_dir, exist_ok=True)
    
    # Update training config for fine-tuning
    train_cfg["runner"]["max_iterations"] = args.max_iterations
    train_cfg["runner"]["experiment_name"] = f"{args.exp_name}{args.save_suffix}"
    
    # Save NEW configurations to finetune directory
    pickle.dump(
        [env_cfg, train_cfg],
        open(f"{finetune_log_dir}/cfgs.pkl", "wb"),
    )
    print(f"Saved finetuning configurations to {finetune_log_dir}/cfgs.pkl")
    
    # --- Create CRM Environment ---
    device = 'cpu'
    
    env = ChronoQuadrupedEnv(num_envs=args.num_envs, env_cfg=env_cfg, device=device, render=False)
    
    # Create runner with the MODIFIED train_cfg and new log directory
    runner = OnPolicyRunner(env, train_cfg, finetune_log_dir, device=device)
    
    # --- Load checkpoint from original training ---
    resume_path = os.path.join(original_log_dir, f"model_{args.ckpt}.pt")
    
    try:
        runner.load(resume_path)
        print(f"Loaded checkpoint from {resume_path}")
    except FileNotFoundError:
        print(f"Checkpoint not found at {resume_path}")
        return
    
    # --- Start Fine-tuning ---
    print("Starting fine-tuning...")
    
    # Reset environment to start fresh
    obs, _ = env.reset()
    print(f"Environment reset. Starting fresh with observation shape: {obs.shape}")
    
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
    
    print(f"Fine-tuning completed! All models saved to {finetune_log_dir}")

if __name__ == "__main__":
    main()