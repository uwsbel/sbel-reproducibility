# chrono_env.py

import torch
import numpy as np
import pychrono as chrono
import pychrono.vehicle as veh
import os
import sys

# Add your project root if necessary
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
import simulation.Robots as Robots
from simulation.Config import Go2Config
import pychrono.irrlicht as irr
import pychrono.vsg as vsg
class ChronoQuadrupedEnv:
    def __init__(self, num_envs, env_cfg, device='cuda:0', render=False):
        self.num_envs = num_envs
        self.env_cfg = env_cfg
        self.device = device
        self.render = render
        self.num_actions = 12
        self.num_obs = 45
        self.max_episode_length = self.env_cfg['max_steps']

        # Create batched simulations
        self.systems = [self._create_single_simulation() for _ in range(self.num_envs)]
        #print(len(self.systems))
        self.robots = [system_tuple[1] for system_tuple in self.systems]
        self.vis = [system_tuple[2] for system_tuple in self.systems]
        # if self.render:
        #     self.vis = irr.ChVisualSystemIrrlicht()
        #     self.vis.AttachSystem(self.systems[0][0])
        #     self.vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
        #     self.vis.SetWindowSize(1024, 768)
        #     self.vis.Initialize()
        #     self.vis.SetWindowTitle("Quadruped RL Environment")
        #     self.vis.AddLogo(chrono.GetChronoDataFile("logo_pychrono_alpha.png"))
        #     self.vis.AddCamera(chrono.ChVector3d(1, 1, 2))
        #     self.vis.AddTypicalLights()
        #     print("Visualization initialized")

        # --- Initialize all buffers as torch tensors ---
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # Start with False
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
        
        self.base_quat = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float)
        self.base_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.dof_pos = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
        self.dof_vel = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float)
        self.contact_forces = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float)

        self.target_lin_vel = torch.tensor(self.env_cfg['target_lin_vel'], device=self.device, dtype=torch.float).repeat(self.num_envs, 1)
        self.initial_config = torch.tensor(Go2Config().initial_motor_config, device=self.device, dtype=torch.float)

        # Reward tracking for airtime
        self.last_contact = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.bool)
        self.feet_air_time = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float)

        # Episode statistics tracking
        self.episode_cumulative_rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.episode_sums = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        # Running statistics for completed episodes
        self.total_episodes = 0
        self.sum_episode_rewards = 0.0
        self.sum_episode_lengths = 0.0

        # Scale factors from Config
        config = Go2Config()
        self.lin_vel_scale = config.lin_vel_scale
        self.ang_vel_scale = config.ang_vel_scale
        self.dof_pos_scale = config.dof_pos_scale
        self.dof_vel_scale = config.dof_vel_scale

        self.extras = {}

        # Initialize the environment
        self.reset()

    def _create_single_simulation(self):
        """Creates and returns a single Chrono system and the robot within it."""
        system = chrono.ChSystemSMC()
        system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
        system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        system.GetSolver().AsIterative().SetMaxIterations(60)
        
        # Create robot using actual Go2Robot class
        robot = Robots.Go2Robot(system, Go2Config().initial_state)
        robot.actuate(Go2Config().initial_motor_config + np.random.normal(0, 0.1, size=12))

        # Create floor
        mat = chrono.ChContactMaterialSMC()
        mat.SetFriction(0.9)
        mat.SetRestitution(0.1)
        mat.SetGn(60.0)
        mat.SetKn(2e5)
        
        terrain_type = "rigid"
        if terrain_type == "rigid":
            floor = chrono.ChBodyEasyBox(50, 50, 0.1, 1000, True, True, mat)
            floor.GetVisualShape(0).SetTexture(chrono.GetChronoDataFile("textures/checker2.png"), 10, 10)
            floor.SetPos(chrono.ChVector3d(0, 0, 0.0))
            floor.SetFixed(True)
            system.AddBody(floor)
        elif terrain_type == "deformable":
            # Create the SCM deformable terrain patch
            terrain = veh.SCMTerrain(system)
            terrain.SetSoilParameters(3e6,   # Bekker Kphi
                                    0,     # Bekker Kc
                                    1.1,   # Bekker n exponent
                                    0,     # Mohr cohesive limit (Pa)
                                    30,    # Mohr friction limit (degrees)
                                    0.0,  # Janosi shear coefficient (m)
                                    2e9,   # Elastic stiffness (Pa/m), before plastic yield
                                    3e4    # Damping (Pa s/m), proportional to negative vertical speed (optional)
            )
            # Set plot type for SCM (false color plotting)
            terrain.SetPlotType(veh.SCMTerrain.PLOT_SINKAGE, 0, 0.01);
            # Initialize the SCM terrain, specifying the initial mesh grid
            terrain.Initialize(20, 20, 0.02);

        vis = None
        if self.render:
            vis = irr.ChVisualSystemIrrlicht()
            vis.AttachSystem(system)
            vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
            vis.SetWindowSize(1024, 768)
            vis.Initialize()
            vis.SetWindowTitle("Quadruped RL Environment")
            vis.AddLogo(chrono.GetChronoDataFile("logo_pychrono_alpha.png"))
            vis.AddCamera(chrono.ChVector3d(8, -2, 1),chrono.ChVector3d(0, 2, 1))
            vis.AddLight(chrono.ChVector3d(10, 0, 20), 50)
        return system, robot, vis

    def step(self, actions):
        self.actions = actions.clone()
        
        # --- Step all simulations (CPU-bound) ---
        for i in range(self.num_envs):
            # Convert Genesis actions to Chrono format
            action_np = self.actions[i].cpu().numpy()

            # Genesis default joint angles (from Genesis config)
            genesis_defaults = np.array([0.0, 0.8, -1.5,  # FR: hip, thigh, calf
                                        0.0, 0.8, -1.5,   # FL: hip, thigh, calf  
                                        0.0, 1.0, -1.5,   # RR: hip, thigh, calf
                                        0.0, 1.0, -1.5])  # RL: hip, thigh, calf

            # Convert Genesis actions to actual joint angles
            target_angles_genesis_order = action_np * 0.25 + genesis_defaults

            # Reorder from Genesis [FR, FL, RR, RL] to Chrono [RR, RL, FR, FL]
            # Each leg has 3 joints (hip, thigh, calf)
            genesis_to_chrono_mapping = [6, 7, 8,    # RR: indices 6,7,8 from Genesis
                                        9, 10, 11,   # RL: indices 9,10,11 from Genesis  
                                        0, 1, 2,     # FR: indices 0,1,2 from Genesis
                                        3, 4, 5]     # FL: indices 3,4,5 from Genesis

            target_angles_chrono_order = -target_angles_genesis_order[genesis_to_chrono_mapping]

            # if self.systems[i].GetChTime() > 1.5:
            self.robots[i].actuate(target_angles_chrono_order)
            
            # Step multiple times for control frequency
            control_steps = int(Go2Config().control_step_size / Go2Config().step_size)
            for _ in range(control_steps):
                self.systems[i].DoStepDynamics(Go2Config().step_size)
        
            if self.render:
                view_dir = "static" # side, front, back, top
                if_save = False
                #while self.vis[i].Run():
                robot_loc = self.robots[i].get_base_pos()
                heading = self.robots[i].get_base_heading()
                if view_dir == "side":
                    cam_pos = chrono.ChVector3d(robot_loc.x+np.sin(heading), robot_loc.y-np.cos(heading), 0.4)
                    target_pos = chrono.ChVector3d(robot_loc.x, robot_loc.y, 0.4)
                elif view_dir == "front":
                    cam_pos = chrono.ChVector3d(robot_loc.x+np.cos(heading), robot_loc.y+np.sin(heading), 0.4)
                    target_pos = chrono.ChVector3d(robot_loc.x, robot_loc.y, 0.4)
                elif view_dir == "back":
                    cam_pos = chrono.ChVector3d(robot_loc.x-np.cos(heading), robot_loc.y-np.sin(heading), 0.4)
                    target_pos = chrono.ChVector3d(robot_loc.x, robot_loc.y, 0.4)
                elif view_dir == "top":
                    cam_pos = chrono.ChVector3d(robot_loc.x, robot_loc.y, 1.0)
                    target_pos = chrono.ChVector3d(robot_loc.x, robot_loc.y, 0.0)
                elif view_dir == "static":
                    cam_pos = chrono.ChVector3d(8, -2, 2)
                    target_pos = chrono.ChVector3d(0, 0, 1)
                self.vis[i].UpdateCamera(cam_pos, target_pos)
                self.vis[i].BeginScene()
                if if_save:
                    self.vis[i].WriteImageToFile('./demo/img'+str(self.episode_length_buf[i])+'.png')
                self.vis[i].Render()
                self.vis[i].EndScene()

        # --- Gather data from simulations and update torch buffers ---
        self._update_buffers()
        
        # --- Check for termination and reset environments that are done ---
        self._check_termination()
        
        # --- Compute rewards and observations on the GPU ---
        self._compute_reward()
        
        # Update episode statistics
        self.episode_cumulative_rew += self.rew_buf
        
        # Track completed episodes for statistics
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            # Update statistics for completed episodes
            for i in reset_env_ids:
                self.total_episodes += 1
                self.sum_episode_rewards += self.episode_cumulative_rew[i].item()
                self.sum_episode_lengths += self.episode_length_buf[i].item()
            
            # Log episode statistics whenever episodes complete
            if self.total_episodes > 0:
                mean_reward = self.sum_episode_rewards / self.total_episodes
                mean_length = self.sum_episode_lengths / self.total_episodes
                print(f"[Episode Stats] Episodes: {self.total_episodes}, "
                      f"Mean Reward: {mean_reward:.3f}, Mean Length: {mean_length:.1f}")
            
            self.reset_idx(reset_env_ids)

        self._compute_observations()

        # Compute mean statistics
        if self.total_episodes > 0:
            mean_reward = self.sum_episode_rewards / self.total_episodes
            mean_ep_length = self.sum_episode_lengths / self.total_episodes
        else:
            mean_reward = 0.0
            mean_ep_length = 0.0

        # Store critic observations and episode statistics in extras
        self.extras["observations"] = {"critic": self.obs_buf}
        self.extras["esp"] = {
            "mean_reward": mean_reward,
            "mean_ep_length": mean_ep_length
        }

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset_idx(self, env_ids):
        for i in env_ids:
            # Simple reset: destroy and recreate the simulation
            print("Resetting environment", i)
            self.systems[i], self.robots[i], self.vis[i] = self._create_single_simulation()
            self.episode_length_buf[i] = 0
            self.episode_cumulative_rew[i] = 0.0
            self.last_actions[i] = 0.0
            self.feet_air_time[i] = 0.0
            self.last_contact[i] = False
        # Reset the reset buffer for these environments
        self.reset_buf[env_ids] = 0

    def reset(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)
        self._update_buffers()
        self._compute_observations()
        # Genesis format - simple tensor return
        return self.obs_buf, None

    def get_observations(self):
        # Store critic observations in extras (Genesis format)
        self.extras["observations"] = {"critic": self.obs_buf}
        # Genesis format - return tensor and extras
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def _update_buffers(self):
        """Gather data from each Chrono system into the batched torch tensors."""
        for i in range(self.num_envs):
            robot = self.robots[i]
            #print(f"system time: {self.systems[i].GetChTime()}")
            # Base position and orientation
            pos = robot.get_base_pos()
            self.base_pos[i, :] = torch.tensor([pos.x, pos.y, pos.z], device=self.device)
            
            q = robot.get_base_quat()
            self.base_quat[i, :] = torch.tensor([q.e0, q.e1, q.e2, q.e3], device=self.device)
            
            # Base velocities in local frame
            lin_vel = robot.get_base_vel_local()
            self.base_lin_vel[i, :] = torch.tensor([lin_vel.x, lin_vel.y, lin_vel.z], device=self.device)
            # print(f"lin_vel: {self.base_lin_vel[i, :]}")
            # if not hasattr(self, 'vel_log_file'):
            #         self.vel_log_file = open('velocity_rigid_3000.csv', 'w')
            #         self.vel_log_file.write('time,vel_x,vel_y,vel_z\n')
            # self.vel_log_file.write(f"{self.systems[i].GetChTime()},{lin_vel.x},{lin_vel.y},{lin_vel.z}\n")
            # self.vel_log_file.flush()
            
            ang_vel = robot.get_base_angvel_local()
            self.base_ang_vel[i, :] = torch.tensor([ang_vel.x, ang_vel.y, ang_vel.z], device=self.device)
            
            # Joint positions and velocities
            joint_pos = robot.get_joint_pos()
            self.dof_pos[i, :] = torch.tensor(joint_pos, device=self.device)
            
            joint_vel = robot.get_joint_speed()
            self.dof_vel[i, :] = torch.tensor(joint_vel, device=self.device)
            
            # Contact forces
            contact_force = robot.get_contact_force()
            self.contact_forces[i, :] = torch.tensor(contact_force, device=self.device)

        # Compute projected gravity
        # Convert quaternion to rotation and project gravity
        for i in range(self.num_envs):
            # Create rotation from quaternion
            q = self.base_quat[i]  # [w, x, y, z]
            
            # Convert to rotation matrix approach - simplified
            # For now, let's use a simple approximation
            gravity_world = torch.tensor([0.0, 0.0, -1.0], device=self.device)
            
            # Simple projection using quaternion rotation
            # This is a simplified version - for production, use proper quaternion rotation
            qw, qx, qy, qz = q[0], q[1], q[2], q[3]
            
            # Rotation matrix elements we need for z-axis rotation
            R33 = 1 - 2*(qx*qx + qy*qy)
            R13 = 2*(qx*qz - qw*qy)
            R23 = 2*(qy*qz + qw*qx)
            
            # Project gravity into body frame
            self.projected_gravity[i, 0] = -R13
            self.projected_gravity[i, 1] = -R23  
            self.projected_gravity[i, 2] = -R33

        self.episode_length_buf += 1

    def _check_termination(self):
        """Check termination conditions based on robot orientation."""
        # Convert quaternion to roll/pitch for termination check
        # Extract roll and pitch from quaternion
        for i in range(self.num_envs):
            q = self.base_quat[i]  # [w, x, y, z]
            qw, qx, qy, qz = q[0], q[1], q[2], q[3]
            
            # Convert to roll/pitch
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            roll = torch.atan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (qw * qy - qz * qx)
            sinp = torch.clamp(sinp, -1.0, 1.0)  # Clamp for numerical stability
            pitch = torch.asin(sinp)
            
            # Check termination conditions
            if torch.abs(roll) > self.env_cfg.get('termination_roll', 0.2) or \
               torch.abs(pitch) > self.env_cfg.get('termination_pitch', 0.2):
                self.reset_buf[i] = 1
            
        # Check episode length
        self.reset_buf = torch.logical_or(
            self.reset_buf.bool(),
            (self.episode_length_buf >= self.max_episode_length)
        ).long()

    def _compute_observations(self):
        """Compute observations using torch operations on the batched buffers."""
        # Convert joint positions and velocities from Chrono ordering to Genesis ordering
        # Chrono ordering: [RR, RL, FR, FL] (each with hip, thigh, calf)
        # Genesis ordering: [FR, FL, RR, RL] (each with hip, thigh, calf)
        
        # Mapping from Chrono indices to Genesis indices
        chrono_to_genesis_mapping = [6, 7, 8,    # FR: indices 6,7,8 from Chrono -> indices 0,1,2 in Genesis
                                    9, 10, 11,   # FL: indices 9,10,11 from Chrono -> indices 3,4,5 in Genesis  
                                    0, 1, 2,     # RR: indices 0,1,2 from Chrono -> indices 6,7,8 in Genesis
                                    3, 4, 5]     # RL: indices 3,4,5 from Chrono -> indices 9,10,11 in Genesis
        
        # Reorder joint positions and velocities to Genesis format
        dof_pos_genesis_order = -self.dof_pos[:, chrono_to_genesis_mapping]
        dof_vel_genesis_order = -self.dof_vel[:, chrono_to_genesis_mapping]
        
        # Genesis default joint angles (in Genesis ordering)
        genesis_defaults = torch.tensor([0.0, 0.8, -1.5,  # FR: hip, thigh, calf
                                        0.0, 0.8, -1.5,   # FL: hip, thigh, calf  
                                        0.0, 1.0, -1.5,   # RR: hip, thigh, calf
                                        0.0, 1.0, -1.5],  # RL: hip, thigh, calf
                                       device=self.device, dtype=torch.float)
        
        # Scale the observations according to the Genesis format
        scaled_lin_vel = self.base_lin_vel * self.lin_vel_scale
        scaled_commands = torch.tensor([[0.5, 0.0, 0.0]], device=self.device).repeat(self.num_envs, 1) * self.lin_vel_scale
        scaled_ang_vel = self.base_ang_vel * self.ang_vel_scale
        scaled_dof_pos = (dof_pos_genesis_order - genesis_defaults) * self.dof_pos_scale
        #scaled_dof_pos = (dof_pos_genesis_order) * self.dof_pos_scale
        scaled_dof_vel = dof_vel_genesis_order * self.dof_vel_scale
        
        self.obs_buf = torch.cat((
            scaled_ang_vel,           # 3  
            self.projected_gravity,   # 3
            scaled_commands,          # 3
            scaled_dof_pos,          # 12
            scaled_dof_vel,          # 12
            self.actions             # 12
        ), dim=-1)

    def _compute_reward(self):
        """Compute rewards using torch operations."""
        # Get config for reward parameters
        config = Go2Config()
        
        # Tracking of linear velocity commands
        lin_vel_error = torch.sum(torch.square(self.target_lin_vel - self.base_lin_vel[:, :2]), dim=1)
        track_lin_vel_reward = torch.exp(-lin_vel_error / 0.25)
        
        # Angular velocity tracking (assume target is 0)
        target_ang_vel = 0.0
        ang_vel_error = torch.square(target_ang_vel - self.base_ang_vel[:, 2])
        ang_vel_reward = torch.exp(-ang_vel_error / 0.25)
        
        # Z velocity penalty
        z_vel_penalty = -torch.square(self.base_lin_vel[:, 2])
        
        # Height penalty
        target_height = 0.3
        height_penalty = -torch.square(self.base_pos[:, 2] - target_height)
        
        # Action rate penalty
        action_rate_penalty = -torch.sum(torch.square(self.last_actions - self.actions), dim=1)
        
        # Joint position penalty (regularization)
        joint_pos_penalty = -torch.sum(torch.abs(self.dof_pos - self.initial_config), dim=1)
        
        # Feet air time reward
        if_contact = self.contact_forces > 10
        contact_filt = torch.logical_or(if_contact, self.last_contact)
        self.last_contact = if_contact.clone()
        
        # Compute feet air time
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += config.control_step_size
        
        # Reward for feet in the air time
        reward_airtime = torch.sum((self.feet_air_time - 0.1) * first_contact.float(), dim=1)
        
        # Nullify feet air time after contact
        self.feet_air_time *= (~contact_filt).float()
        
        # Combine all rewards
        self.rew_buf = (
            track_lin_vel_reward * config.track_lin_vel_reward_param +
            ang_vel_reward * config.ang_vel_reward_param +
            z_vel_penalty * config.z_vel_penalty_param +
            height_penalty * config.height_penalty_param +
            action_rate_penalty * config.action_rate_penalty_param +
            reward_airtime * config.reward_airtime_param +
            joint_pos_penalty * config.joint_pos_penalty_param
        )
        
        # Store last actions for next iteration
        self.last_actions = self.actions.clone()

    def get_episode_statistics(self):
        """Method for RSL-RL to access episode statistics."""
        if self.total_episodes > 0:
            return {
                "mean_episode_reward": self.sum_episode_rewards / self.total_episodes,
                "mean_episode_length": self.sum_episode_lengths / self.total_episodes,
                "total_episodes": self.total_episodes
            }
        else:
            return {
                "mean_episode_reward": 0.0,
                "mean_episode_length": 0.0,
                "total_episodes": 0
            }

    def log_episode_statistics(self):
        """Print episode statistics to console for visibility."""
        if self.total_episodes > 0:
            mean_reward = self.sum_episode_rewards / self.total_episodes
            mean_length = self.sum_episode_lengths / self.total_episodes
            print(f"[Episode Stats] Episodes: {self.total_episodes}, "
                  f"Mean Reward: {mean_reward:.3f}, Mean Length: {mean_length:.1f}")