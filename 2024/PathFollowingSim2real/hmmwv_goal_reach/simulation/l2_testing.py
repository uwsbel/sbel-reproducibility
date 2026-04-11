import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import torch as th
from stable_baselines3.common.callbacks import BaseCallback

import time
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_script.rl_train_goal_reach import GoalEnv
import argparse
import csv

parser = argparse.ArgumentParser(description='ART Simulation')
parser.add_argument('ModelType',default='1', type=str, help='Type of Model Used for Training [1, 2, 3]')
parser.add_argument('exp_ind', type=int, default=0, help='index of the experiment')

args = parser.parse_args()
modeltype = args.ModelType
exp_ind = args.exp_ind

sim_env = GoalEnv()
# print(sim_env.goal)
goal_list = np.loadtxt(project_root + '/data/traj/goal_points_l2.txt', skiprows=1, delimiter=" ")
goal_x, goal_y = goal_list[exp_ind]

sim_env.goal = [goal_x, goal_y]
print(f"Goal point: {sim_env.goal}")

obs = sim_env.relative_state_to_goal()

# Load the trained model
model_path = project_root + '/data/rl_model/ppo_' + modeltype
# model_path = project_root + '/data/rl_models/ppo_M2_2'
model = PPO.load(model_path)

traj_save_file = project_root+'/data/traj/l2_'+modeltype+'_'+str(exp_ind+100)+'.csv'
max_steps = 400

for i in range(max_steps):
    controls, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = sim_env.step(controls)
    veh_x, veh_y, veh_theta, veh_v = sim_env.state[0], sim_env.state[1], sim_env.state[2], sim_env.state[3]
    time = i * sim_env.dt
    # log data:
    with open (traj_save_file,'a', encoding='UTF8') as savefile:
            save_writer = csv.writer(savefile, quoting=csv.QUOTE_NONE, escapechar=' ')
            save_writer.writerow([time, veh_x, veh_y, veh_theta, veh_v, goal_x, goal_y])
            savefile.close()
    if sim_env.old_distance < 3.0:
        print(f"Goal reached in {i} steps!")
        break