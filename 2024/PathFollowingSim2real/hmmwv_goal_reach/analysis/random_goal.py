import numpy as np
import csv
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Generate 100 random goal points
num_goals = 100
goal_points = []

for i in range(num_goals):
    r = np.random.uniform(10.0, 20.0)
    theta = np.random.uniform(0, 2 * np.pi)
    goal_x = r * np.cos(theta)
    goal_y = r * np.sin(theta)
    goal_points.append([goal_x, goal_y])

# Convert to numpy array for easier handling
goal_points = np.array(goal_points)

# Save to text file
np.savetxt(project_root + '/data/traj/goal_points_l2.txt', goal_points, fmt='%.6f', header='goal_x goal_y', comments='')

print(f"Generated {num_goals} goal points and saved to goal_points_l2.txt")