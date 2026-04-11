import numpy as np
import matplotlib.pyplot as plt

# exp_index = np.random.choice(200, size=10, replace=False).tolist()
# print(f"Selected experiment indices: {exp_index}")
# exp_index = [91, 188, 22, 133, 76, 116, 72, 199, 176, 62]
exp_index = [91, 72]
policy_index = [1, 2, 3, 4]
policy_colors = {1: 'r', 2: 'g', 3: 'b', 4: 'm'}

traj_data = {}
chrono_traj_data = {}
for idx in exp_index:
    traj_data[idx] = {}
    chrono_traj_data[idx] = {}
    for p_idx in policy_index:
        filename = f"./data/traj/l2_{p_idx}_{idx}.csv"
        traj = np.loadtxt(filename, delimiter=',')  # Skip header
        
        traj_data[idx][p_idx] = traj
        chrono_traj_filename = f"./data/traj/{p_idx}_{idx}.csv"
        chrono_traj = np.loadtxt(chrono_traj_filename, delimiter=',')
        if idx == 72 and p_idx == 1:
            print("use less data")
            print(f"Original trajectory shape: {chrono_traj.shape}")
            chrono_traj= chrono_traj[:150,:]  # truncate to avoid long tail
            print(f"Truncated trajectory shape: {chrono_traj.shape}")
        chrono_traj_data[idx][p_idx] = chrono_traj

print("Loaded trajectory data for experiments and policies.")
# Plotting trajectories
plt.figure(figsize=(10, 8))
for idx in exp_index:
    for p_idx in policy_index:

        traj = traj_data[idx][p_idx]

        if p_idx == 1 and idx == exp_index[0]:
            plt.scatter(traj[0, 1], traj[0, 2], color='k', alpha=1.0, marker='X', s=300, label='Start' if idx == exp_index[0] else "", zorder=5, edgecolors='none')
            plt.scatter(traj[-1, 1], traj[-1, 2], color='g', alpha=1.0, marker='o', s=2000, label='Goal' if idx == exp_index[0] else "", zorder=5, edgecolors='none')
        
        if p_idx == 1 and idx == exp_index[1]:
            plt.scatter(traj[0, 1], traj[0, 2], color='k', alpha=1.0, marker='X', s=300, label='Start' if idx == exp_index[0] else "", zorder=5, edgecolors='none')
            plt.scatter(traj[-1, 1]-0.5, traj[-1, 2]+0.6, color='g', alpha=1.0, marker='o', s=2000, label='Goal' if idx == exp_index[0] else "", zorder=5, edgecolors='none')
        # plt.plot(traj[:, 1], traj[:, 2], label=f'Exp {idx} - Policy {p_idx}')
        plt.plot(traj[:, 1], traj[:, 2], color=policy_colors[p_idx], alpha=0.7, label=f'ROM Trajectory from Policy {p_idx}' if idx == exp_index[0] else "")
        plt.plot(chrono_traj_data[idx][p_idx][:, 1], chrono_traj_data[idx][p_idx][:, 2], linestyle='--', color=policy_colors[p_idx], alpha=0.7, label=f'Chrono Trajectory from Policy {p_idx}' if idx == exp_index[0] else "")

plt.xlim(-25,40)
plt.ylim(-2,35)
plt.xlabel('x (m)', fontsize=14)
plt.ylabel('y (m)', fontsize=14)
plt.grid(True)
plt.legend(fontsize=14, ncol=2, markerscale=0.4)
plt.tight_layout()
plt.savefig('./trajectories_comparison.png',dpi=300)