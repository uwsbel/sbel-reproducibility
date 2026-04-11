import numpy as np
import matplotlib.pyplot as plt
import re

# Read the log file
log_file = './rl_log.txt'

steps = []
rewards = []

with open(log_file, 'r') as f:
    for line in f:
        # Match lines that start with "Step:"
        match = re.match(r'Step:\s+(\d+),\s+Average Reward:\s+([-\d.]+)', line)
        if match:
            step = int(match.group(1))
            reward = float(match.group(2))
            steps.append(step)
            rewards.append(reward)

# Convert to numpy arrays
steps = np.array(steps)
rewards = np.array(rewards)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(steps, rewards, linewidth=2.5)
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Reward', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./reward_curve.png', dpi=300)

print(f"Parsed {len(steps)} data points")
print(f"Initial reward: {rewards[0]:.2f}")
print(f"Final reward: {rewards[-1]:.2f}")
print(f"Best reward: {rewards.max():.2f} at step {steps[rewards.argmax()]}")

