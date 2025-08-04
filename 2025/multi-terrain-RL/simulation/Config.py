import pychrono as chrono
import numpy as np
import simulation.Robots as Robots


class A1Config:
    """Configuration for A1 robot simulation"""
    
    def __init__(self):
        # System parameters
        self.step_size = 1e-2
        self.control_frequency = 50.0  # Hz
        
        # Robot setup
        self.robot_class = Robots.A1Robot
        self.initial_position = chrono.ChVector3d(0, 0, 0.5)
        self.initial_heading = 0.0
        self.initial_motor_config = np.array([0, -1.0, 2.0] * 4, dtype=np.float32)
        self.target_velocity = np.array([0.5, 0.0])
        
        # Policy settings
        self.model_path = "/home/harry/euler_log/logs/checkpoints/checkpoint_2"
        self.warm_up_time = 0.0
        
        # Visualization
        self.window_title = "A1 Robot RL Simulation"
        self.enable_vsg = False

        # Reward
        self.track_lin_vel_reward_param = 1.0
        self.ang_vel_reward_param = 0.2
        self.z_vel_penalty_param = 0.1
        self.height_penalty_param = 50.0
        self.action_rate_penalty_param = 0.1
        self.reward_airtime_param = 1.0
    
    @property
    def control_step_size(self):
        return 1.0 / self.control_frequency
    
    @property
    def control_steps(self):
        return int(self.control_step_size / self.step_size)
    
    @property
    def initial_state(self):
        return chrono.ChFramed(self.initial_position, chrono.QuatFromAngleZ(self.initial_heading))


class Go2Config:
    """Configuration for Go2 robot simulation"""
    
    def __init__(self):
        # System parameters
        self.step_size = 5e-3
        self.control_frequency = 50.0  # Hz
        
        # Robot setup
        self.robot_class = Robots.Go2Robot
        self.initial_position = chrono.ChVector3d(-4, 2.5, 0.5)
        self.initial_heading = 0.0
        self.initial_motor_config = np.array([0,-1.0,1.5, 0,-1.0,1.5, 0,-0.8,1.5, 0,-0.8,1.5])
        #self.initial_motor_config = np.array([0.3, -1.0, 2.0, -0.3, -1.0, 2.0, 0.3, -1.0, 2.0, -0.3, -1.0, 2.0], dtype=np.float32)
        self.target_velocity = np.array([0.5, 0.0])
        
        # Policy settings
        # self.model_path = "data/rl_models/a1_locomotion"
        self.model_path = "data/logs/checkpoints/checkpoint_9"

        self.warm_up_time = 2.0
        
        # Visualization
        self.window_title = "Go2 Robot RL Simulation"
        self.enable_vsg = False

        # Reward
        self.track_lin_vel_reward_param = 1.0
        self.ang_vel_reward_param = 0.2
        self.z_vel_penalty_param = 1.0
        self.height_penalty_param = 50.0
        self.action_rate_penalty_param = 0.005
        self.reward_airtime_param = 0.0
        self.joint_pos_penalty_param = 0.1

        # config
        self.dof_vel_scale = 0.05
        self.dof_pos_scale = 1.0
        self.lin_vel_scale = 2.0
        self.ang_vel_scale = 0.25

    @property
    def control_step_size(self):
        return 1.0 / self.control_frequency
    
    @property
    def control_steps(self):
        return int(self.control_step_size / self.step_size)
    
    @property
    def initial_state(self):
        return chrono.ChFramed(self.initial_position, chrono.QuatFromAngleZ(self.initial_heading))
