import argparse
import os
import pickle
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner
from chrono_env import ChronoQuadrupedEnv as RigidTerrainEnv
from chrono_crmenv import ChronoQuadrupedEnv as GranularTerrainEnv
# from chrono_crmenv import ChronoQuadrupedEnv
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="chrono-quadruped")
    parser.add_argument("--ckpt", type=int, default=2000, help="Checkpoint number to load")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments for evaluation")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true", help="Enable visualization")
    args = parser.parse_args()

    log_dir = f"./data/rl_models/rslrl"
    
    # Load configurations
    env_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    print(f"Loaded configurations from {log_dir}/cfgs.pkl")

    # Disable rewards for cleaner evaluation
    if 'reward_scales' in env_cfg:
        env_cfg["reward_scales"] = {}

    # Create environment
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # print(f"Using device: {device}")
    device = 'cpu'

    env_type = "rigid"

    if env_type == "granular":
        env = GranularTerrainEnv(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            device=device,
            render = True
        )
    else:
        env = RigidTerrainEnv(
            num_envs=args.num_envs,
            env_cfg=env_cfg,
            device=device,
            render = True
        )
    
    print(f"Created environment with {args.num_envs} parallel environments")

    # Create runner and load model
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    # resume_path = os.path.join(log_dir, f"ckpt_genesis.pt")
    
    try:
        runner.load(resume_path)
        print(f"Loaded model from {resume_path}")
    except FileNotFoundError:
        print(f"Model checkpoint not found at {resume_path}")
        print("Available checkpoints:")
        for file in os.listdir(log_dir):
            if file.startswith("model_") and file.endswith(".pt"):
                print(f"  - {file}")
        return
    
    # Get inference policy
    policy = runner.get_inference_policy(device=device)
    print("Policy loaded successfully")

    # Reset environment
    obs, _ = env.reset()
    print(f"Environment reset. Observation shape: {obs.shape}")
    
    # Evaluation statistics
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = torch.zeros(args.num_envs, device=device)
    current_episode_length = torch.zeros(args.num_envs, device=device, dtype=torch.long)
    
    print(f"\nStarting evaluation for {args.max_steps} steps...")
    print("=" * 50)
    
    step_count = 0
    with torch.no_grad():
        while step_count < args.max_steps:
            # Get actions from policy
            actions = policy(obs)

            # Step environment
            obs, rewards, dones, extras = env.step(actions)
            #print(f"rewards: {rewards}")

            # Update statistics
            current_episode_reward += rewards
            current_episode_length += 1
            
            # Check for episode completion
            if torch.any(dones):
                completed_envs = dones.nonzero(as_tuple=False).flatten()
                for env_idx in completed_envs:
                    episode_rewards.append(current_episode_reward[env_idx].item())
                    episode_lengths.append(current_episode_length[env_idx].item())
                    
                    print(f"Episode completed in env {env_idx}: "
                          f"Reward = {current_episode_reward[env_idx]:.3f}, "
                          f"Length = {current_episode_length[env_idx]}")
                    
                    # Reset counters for completed environments
                    current_episode_reward[env_idx] = 0.0
                    current_episode_length[env_idx] = 0
            
            step_count += 1
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"Step {step_count}/{args.max_steps}")
                
                # Print episode statistics from environment if available
                if hasattr(env, 'get_episode_statistics'):
                    stats = env.get_episode_statistics()
                    if stats['total_episodes'] > 0:
                        print(f"  Environment stats - Episodes: {stats['total_episodes']}, "
                              f"Mean Reward: {stats['mean_episode_reward']:.3f}, "
                              f"Mean Length: {stats['mean_episode_length']:.1f}")

    # Final statistics
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    
    if episode_rewards:
        print(f"Completed Episodes: {len(episode_rewards)}")
        print(f"Mean Episode Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
        print(f"Mean Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"Best Episode Reward: {np.max(episode_rewards):.3f}")
        print(f"Worst Episode Reward: {np.min(episode_rewards):.3f}")
    else:
        print("No episodes completed during evaluation")
    
    # Print environment statistics if available
    if hasattr(env, 'get_episode_statistics'):
        stats = env.get_episode_statistics()
        print(f"\nEnvironment Total Statistics:")
        print(f"  Total Episodes: {stats['total_episodes']}")
        print(f"  Mean Reward: {stats['mean_episode_reward']:.3f}")
        print(f"  Mean Length: {stats['mean_episode_length']:.1f}")
    
    print(f"\nEvaluation completed after {step_count} steps")


if __name__ == "__main__":
    main()

"""
Usage examples:

# Basic evaluation with default settings
python eval.py -e chrono-quadruped --ckpt 100

# Evaluation with specific parameters
python eval.py -e chrono-quadruped --ckpt 500 --num_envs 4 --max_steps 2000

# List available checkpoints if model not found
python eval.py -e chrono-quadruped --ckpt 999
"""
