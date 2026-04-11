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
class GoalEnv(gym.Env):
    def __init__(self):
        super(GoalEnv, self).__init__()
        # Action = (alpha, beta)
        self.action_space = spaces.Box(
            low=np.array([0.2, -1.0]), 
            high=np.array([0.5,  1.0]),
            shape=(2,),
            dtype=np.float32
        )
        # Observation = (x, y, theta, v)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        # Generate random goals
        # self.goals = self.generate_random_goals(num_goals)  # Remove this line

        # Vehicle parameters
        self.r_wheel = 0.47
        self.i_wheel = 6.69
        self.gamma = 1/3
        self.tau_0 = 1000 + np.random.normal(0, 50)  # Adding some randomness
        self.omega_0 = 2600 + np.random.normal(0, 100)  # Adding some randomness
        self.c1 = 1e-4
        self.c0 = 0.02
        self.car_length = 1.0
        

        self.dt = 0.1
        self.max_steps = 400

        self.old_distance = 0.0

        self.reset()

    def f_vdot(self, alpha, v):
        omega_m = v / (self.r_wheel * self.gamma)
        f1_m = -self.tau_0 * omega_m / self.omega_0 + self.tau_0
        T_fun = alpha * f1_m - self.c1 * omega_m - self.c0
        return T_fun * self.gamma / self.i_wheel * self.r_wheel
    
    def relative_state_to_goal(self):
        x, y, theta, v = self.state[:4]
        rot_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        relative_pos = np.dot(rot_mat, np.array(self.goal) - np.array([x, y]))
        relative_theta = np.arctan2(relative_pos[1], relative_pos[0])
        return np.array([relative_pos[0], relative_pos[1], relative_theta, v], dtype=np.float32)

    def step(self, action):
        #start_t = time.time()
        alpha, beta = action
        alpha = alpha + np.random.normal(0, 0.05)
        beta = beta + np.random.normal(0, 0.05)
        x, y, theta, v = self.state
        x_dot = np.cos(theta) * v
        y_dot = np.sin(theta) * v
        theta_dot = v * np.tan(beta * 0.7) / self.car_length
        v_dot = self.f_vdot(alpha, v)

        x += x_dot * self.dt
        y += y_dot * self.dt
        theta += theta_dot * self.dt
        v += v_dot * self.dt
        #end_t = time.time()
        #print(f"Step time: {end_t - start_t}")        
        #theta = np.clip(theta, -np.pi, np.pi)  # Clip angle to [-pi, pi]
        v = np.clip(v, 0.0, 5.0) # Clip velocity to 5 m/s
        #print(f"v: {v}")

        self.state = np.array([x, y, theta, v], dtype=np.float32)
        self.steps += 1

        dist2 = (x - self.goal[0])**2 + (y - self.goal[1])**2
        distance = np.sqrt(dist2)

        # Reward shaping
        ## first part of reward is based on the progress towards the goal
        progress = (self.old_distance - distance) #goal reaching progress
        progress_scale = 50
        reward = 0.0
        if progress > 0:
            reward = progress * progress_scale
        # If we have not moved even by 1 cm in 0.1 seconds give a penalty
        if progress < 0.1:
            reward -= 100
        # done condition
        done = False
        if distance < 1.0:  # goal success
            print(f"Goal reached in {self.steps} steps!")
            print(f"reward: {reward}")
            reward += self.max_steps * 15
            done = True

        if self.steps >= self.max_steps:
            reward -= 1000*distance
            #print("Max steps reached!")
            done = True

        self.old_distance = distance  # Update old_distance for the next step

        relative_state = self.relative_state_to_goal()
        observation = relative_state
        #print(f"reward: {reward}")
        return observation, reward, done, {'velocity': v}

    def reset(self):
        # Generate a new random goal
        r = np.random.uniform(10.0, 20.0)
        theta = np.random.uniform(0, 2 * np.pi)
        self.goal = [r * np.cos(theta), r * np.sin(theta)]

        # x0 = np.random.uniform(-1.0, 1.0)
        # y0 = np.random.uniform(-1.0, 1.0)
        # theta0 = np.random.uniform(-np.pi, np.pi)
        x0 = 0
        y0 = 0
        theta0 = 0
        v0 = 0.0

        self.state = np.array([x0, y0, theta0, v0], dtype=np.float32)
        self.steps = 0
        self.old_distance = np.sqrt((x0 - self.goal[0])**2 + (y0 - self.goal[1])**2)

        relative_state = self.relative_state_to_goal()
        observation = relative_state
        return observation

    def render(self, mode='human'):
        pass

class RewardLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_freq, save_path, verbose=1):
        super(RewardLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_freq = save_freq
        self.save_path = save_path
        self.rewards = []
        self.velocities = []
        self.model_save_counter = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.rewards[-self.check_freq:])
            mean_velocity = np.mean(self.velocities[-self.check_freq:])
            if np.isnan(mean_reward) or np.isnan(mean_velocity):
                print(f"NaN detected! Rewards: {self.rewards[-self.check_freq:]}, Velocities: {self.velocities[-self.check_freq:]}")
            print(f"Step: {self.n_calls}, Average Reward: {mean_reward:.2f}, Average Velocity: {mean_velocity:.2f}")
        
        if self.n_calls % self.save_freq == 0:
            # Save the model every save_freq steps
            self.model_save_counter += 1
            model_save_path = f"{self.save_path}_{self.model_save_counter}.zip"
            self.model.save(model_save_path)
            print(f"Model saved to {model_save_path}")
        return True

    def _on_rollout_end(self) -> None:
        self.rewards.extend(self.locals['rewards'])
        for info in self.locals['infos']:
            if 'velocity' in info:
                self.velocities.append(info['velocity'])

    def _on_training_end(self) -> None:
        self.model.save(self.save_path)
        print(f"Model saved to {self.save_path}")

if __name__ == "__main__":
    # 1) Create environment and wrap with DummyVecEnv
    env = GoalEnv()
    vec_env = DummyVecEnv([lambda: env])

    # 2) Create and train the PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=env.max_steps*3, # update rate for nn
        batch_size=64, # batch size for training depends on the size of the network
        n_epochs=20,
        gamma=0.99,
        clip_range=0.2,
        verbose=0,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 128,128,64], vf=[64, 128,128, 64]),  # FCC with 4 layers, 128 neurons each
            activation_fn=th.nn.ReLU   # Use ReLU activation function
        ),
        tensorboard_log=project_root + "/data/rl_tb"
    )

    save_freq = 500000
    total_timesteps = 2000000
    # Define the callback
    callback = RewardLoggingCallback(check_freq=1000, save_freq=save_freq, save_path=project_root+'/data/rl_model/ppo')

    # Train the model with the callback
    model.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name=project_root+"/data/rl_model/PPO_Goal_Reach")

    # 3) Evaluate the trained policy
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=400)
    print(f"Evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # 4) Optional: test the policy manually
    obs = vec_env.reset()
    for step in range(400):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        if dones[0]:
            print("Goal reached or max steps used up!")
            obs = vec_env.reset()


