# train_chrono.py

import argparse
import os
import pickle
import shutil
import torch

from rsl_rl.runners import OnPolicyRunner
from chrono_env import ChronoQuadrupedEnv # Import our new env

def get_train_cfg(exp_name, max_iterations):
    """Defines the training configuration dictionary for RSL-RL."""
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 125,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 5000,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

def get_env_cfg():
    """Defines the environment configuration dictionary."""
    return {
        'max_steps': 1000,
        'step_size': 0.001,
        'target_lin_vel': [0.5, 0.0],
        'termination_roll': 0.2,
        'termination_pitch': 0.2,
        'reward_scales': {
            'track_lin_vel': 1.0,
            'ang_vel': 0.2,
            'z_vel_penalty': 1.0,
            'height_penalty': 50.0,
            'action_rate_penalty': 0.005,
            'airtime': 0.0,
            'joint_pos_penalty': 0.1,
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="chrono-quadruped")
    parser.add_argument("-n", "--num_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=5000)
    args = parser.parse_args()

    # --- Setup logging ---
    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # --- Get configurations ---
    env_cfg = get_env_cfg()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # --- Save configurations ---
    pickle.dump(
        [env_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # --- Create Environment and Runner ---
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    env = ChronoQuadrupedEnv(num_envs=args.num_envs, env_cfg=env_cfg, device=device)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)

    # --- Start Training ---
    print(f"Starting training for {args.exp_name}...")
    print(f"Configuration:")
    print(f"  - Number of environments: {args.num_envs}")
    print(f"  - Max iterations: {args.max_iterations}")
    print(f"  - Steps per environment: {train_cfg['num_steps_per_env']}")
    print(f"  - Device: {device}")
    
    # Log initial episode statistics
    if hasattr(env, 'log_episode_statistics'):
        env.log_episode_statistics()
    
    runner.learn(num_learning_iterations=train_cfg["runner"]["max_iterations"], init_at_random_ep_len=True)

if __name__ == "__main__":
    main()