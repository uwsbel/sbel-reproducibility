import time
import vlfm.policy.chrono_policies_multiple
import math
import numpy as np
import pychrono.sensor as sens
import pychrono.irrlicht as chronoirr
import pychrono as chrono
import sys
import os
import torch

# Assuming the script is located in the 'experiments/apartment' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)
# Add the parent directory of 'models' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ChronoEnv:
    def __init__(self, target_object: str = "toilet", num_agents: int = 1):
        self.num_agents = num_agents
        self.my_system = None
        self.out_dir = "SENSOR_OUTPUT/"
        self.lens_model = getattr(sens, "PINHOLE", sens.CameraLensModelType_PINHOLE)
        self.update_rate = 30
        self.image_width = 640
        self.image_height = 480
        self.fov = 1.408
        self.lag = 0
        self.exposure_time = 0
        self.manager = None
        self.vis = None
        self.rt_timer = None
        self.timestep = 0.001
        self.control_frequency = 10
        self.steps_per_control = round(1 / (self.timestep * self.control_frequency))
        self.step_number = 0
        self.render_frame = 0
        self.observations = None
        self.target_object = target_object
        self.virtual_robots = []
        self.lidar_list = []
        self.cam_list = []

    def reset(self):
        self.my_system = chrono.ChSystemSMC()
        self.my_system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        patch_mat = chrono.ChContactMaterialSMC()

        # Specify starting positions for 1 or 2 agents
        if self.num_agents == 1:
            start_positions = [chrono.ChVector3d(-1.25, -1.25, 0.25)]
        else:
            start_positions = [
                chrono.ChVector3d(-1.25, -1.25, 0.25),
                chrono.ChVector3d(1.25, 1.25, 0.25),
            ]

        self.manager = sens.ChSensorManager(self.my_system)
        intensity_moderate = 1.0
        if hasattr(self.manager.scene, "AddAreaLight"):
            self.manager.scene.AddAreaLight(
                chrono.ChVector3f(0, 0, 1),
                chrono.ChColor(intensity_moderate, intensity_moderate, intensity_moderate),
                500.0,
                chrono.ChVector3f(1, 0, 0),
                chrono.ChVector3f(0, -1, 0)
            )
        else:
            self.manager.scene.AddPointLight(
                chrono.ChVector3f(0, 0, 1),
                chrono.ChColor(intensity_moderate, intensity_moderate, intensity_moderate),
                500.0
            )

        for pos in start_positions:
            robot = chrono.ChBodyEasyBox(0.25, 0.25, 0.5, 100, True, True, patch_mat)
            robot.SetPos(pos)
            robot.SetFixed(True)
            self.my_system.Add(robot)
            self.virtual_robots.append(robot)
            offset_pose = chrono.ChFramed(chrono.ChVector3d(0.3, 0, 0.25), chrono.QUNIT)

            lidar = sens.ChLidarSensor(
                robot, 30, offset_pose, self.image_width, self.image_height, self.fov,
                chrono.CH_PI/6, -chrono.CH_PI/6, 3.66,
                sens.LidarBeamShape_RECTANGULAR, 1, 0, 0,
                sens.LidarReturnMode_STRONGEST_RETURN
            )
            lidar.SetName("Lidar Sensor")
            lidar.SetLag(0)
            lidar.SetCollectionWindow(1/20)
            lidar.PushFilter(sens.ChFilterVisualize(self.image_width, self.image_height, "depth camera"))
            lidar.PushFilter(sens.ChFilterDIAccess())
            self.manager.AddSensor(lidar)
            self.lidar_list.append(lidar)

            cam = sens.ChCameraSensor(
                robot, self.update_rate, offset_pose,
                self.image_width, self.image_height, self.fov
            )
            cam.SetName("Camera Sensor")
            cam.SetLag(self.lag)
            cam.SetCollectionWindow(self.exposure_time)
            cam.PushFilter(sens.ChFilterVisualize(self.image_width, self.image_height, "rgb camera"))
            cam.PushFilter(sens.ChFilterRGBA8Access())
            self.manager.AddSensor(cam)
            self.cam_list.append(cam)

        mmesh = chrono.ChTriangleMeshConnected()
        mmesh.LoadWavefrontMesh(project_root + '/data/chrono_environment/new_flat_3.obj', False, True)
        trimesh_shape = chrono.ChVisualShapeTriangleMesh()
        trimesh_shape.SetMesh(mmesh)
        trimesh_shape.SetName("ENV MESH")
        trimesh_shape.SetMutable(False)
        mesh_body = chrono.ChBody()
        mesh_body.SetPos(chrono.ChVector3d(0, 0, 0))
        mesh_body.SetRot(chrono.Q_ROTATE_Y_TO_Z)
        mesh_body.AddVisualShape(trimesh_shape)
        mesh_body.SetFixed(True)
        self.my_system.Add(mesh_body)

        self.vis = chronoirr.ChVisualSystemIrrlicht(self.my_system)
        self.vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
        self.vis.AddLightWithShadow(chrono.ChVector3d(2, 2, 2), chrono.ChVector3d(0, 0, 0), 5, 1, 11, 55)
        self.vis.EnableAbsCoordsysDrawing(True)
        self.vis.Initialize()
        self.vis.AddSkyBox()
        self.vis.AddCamera(chrono.ChVector3d(-7/3, 0, 4.5/3), chrono.ChVector3d(0, 0, 0))
        self.observations = [self._get_observations(i) for i in range(len(self.virtual_robots))]
        return self.observations

    def step(self, actions):
        for i, action in enumerate(actions):
            self._do_action(action, self.virtual_robots[i])
        for _ in range(self.steps_per_control):
            self.manager.Update()
            self.my_system.DoStepDynamics(self.timestep)

        self.vis.BeginScene()
        self.vis.Render()
        self.vis.EndScene()

        new_observations = [self._get_observations(i) for i in range(len(self.virtual_robots))]
        stop = any(self._get_stop(action) for action in actions)
        return new_observations, stop

    def _get_stop(self, action):
        if isinstance(action, torch.Tensor) and action.numel() == 1 and action.item() == 0:
            print("STOPPED")
            time.sleep(5)
            return True
        return False

    def _get_observations(self, idx):
        lidar = self.lidar_list[idx]
        cam = self.cam_list[idx]
        robot = self.virtual_robots[idx]

        depth_buffer = lidar.GetMostRecentDIBuffer()
        if depth_buffer.HasData():
            depth_data = torch.tensor(depth_buffer.GetDIData()[:, :, 0], dtype=torch.float32)
            depth_data = torch.flip(depth_data, dims=[0, 1])
            MIN_DEPTH = 0
            MAX_DEPTH = 5.5
            depth_data = np.clip((depth_data - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH), 0, 1)
            depth_data[depth_data == 0] = 1
        else:
            depth_data = torch.zeros(self.image_height, self.image_width, dtype=torch.float32)

        camera_buffer = cam.GetMostRecentRGBA8Buffer()
        if camera_buffer.HasData():
            camera_data = torch.tensor(camera_buffer.GetRGBA8Data(), dtype=torch.uint8)[:, :, :3]
            camera_data = torch.flip(camera_data, dims=[0])
        else:
            camera_data = torch.zeros(self.image_height, self.image_width, 3, dtype=torch.uint8)

        robot_x = torch.tensor(robot.GetPos().x, dtype=torch.float32)
        robot_y = torch.tensor(robot.GetPos().y, dtype=torch.float32)
        quat_list = [robot.GetRot().e0, robot.GetRot().e1, robot.GetRot().e2, robot.GetRot().e3]
        yaw = self.quaternion_to_yaw(quat_list)
        robot_yaw = torch.tensor(yaw, dtype=torch.float32)

        obs_dict = {
            "rgb": camera_data,
            "depth": depth_data,
            "gps": torch.stack((robot_x, robot_y)),
            "compass": robot_yaw,
            "objectgoal": self.target_object
        }
        return obs_dict

    def _do_action(self, action, robot):
        if len(action[0]) > 1:
            target_x = float(action[0][0])
            target_y = float(action[0][1])
            cur_pos = robot.GetPos()
            heading = math.atan2(target_y - cur_pos.y, target_x - cur_pos.x)
            heading = (heading + math.pi) % (2 * math.pi) - math.pi
            current_heading = robot.GetRot().GetCardanAnglesXYZ().z
            turn_angle = heading - current_heading
            turn_angle = (turn_angle + math.pi) % (2 * math.pi) - math.pi
            print("TURN ANGLE: ", turn_angle)
            new_rotation = chrono.QuatFromAngleZ(turn_angle) * robot.GetRot()
            robot.SetRot(new_rotation)
            robot.SetPos(chrono.ChVector3d(target_x, target_y, 0.25))
            print("Moved Pos: ", robot.GetPos())
        else:
            action_id = action.item()
            if action_id == 1:
                rot_state = robot.GetRot().GetCardanAnglesXYZ()
                robot.SetPos(robot.GetPos() + chrono.ChVector3d(0.01 * np.cos(rot_state.z), 0.01 * np.sin(rot_state.z), 0))
            elif action_id == 2:
                robot.SetRot(chrono.QuatFromAngleZ(np.pi/12) * robot.GetRot())
            elif action_id == 3:
                robot.SetRot(chrono.QuatFromAngleZ(-np.pi/12) * robot.GetRot())

    def quaternion_to_yaw(self, quaternion):
        w, x, y, z = quaternion
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return yaw


if __name__ == "__main__":
    # Initialize environment with two agents.
    env = ChronoEnv(target_object='chair', num_agents=2)
    obs = env.reset()
    
    # Use an initial action for both agents.
    initial_action = torch.tensor([[5]], dtype=torch.long)
    obs, stop = env.step([initial_action for _ in range(env.num_agents)])

    # Sensor and policy parameters.
    camera_height = 0.5
    min_depth = 0
    max_depth = 5.5
    camera_fov = 80.67  # in degrees
    image_width = 640

    text_prompt = "Seems like there is a target_object ahead."
    use_max_confidence = False
    pointnav_policy_path = "data/pointnav_weights.pth"
    depth_image_shape = (480, 640)
    pointnav_stop_radius = 0.5
    object_map_erosion_size = 5
    exploration_thresh = 0.7
    obstacle_map_area_threshold = 1.5
    min_obstacle_height = 0.3
    max_obstacle_height = 0.5
    hole_area_thresh = 100000
    use_vqa = False
    vqa_prompt = "Is this "
    coco_threshold = 0.8
    non_coco_threshold = 0.4
    agent_radius = 0.15

    from vlfm.mapping.obstacle_map import ObstacleMap
    from vlfm.mapping.value_map import ValueMap

    # Create shared maps.
    shared_map = ObstacleMap(
        min_height=min_obstacle_height,
        max_height=max_obstacle_height,
        area_thresh=obstacle_map_area_threshold,
        agent_radius=agent_radius,
        hole_area_thresh=hole_area_thresh,
    )

    shared_value_map = ValueMap(
        value_channels=1,
        use_max_confidence=False,
        obstacle_map=None  # Optionally, link obstacle map for fusion
    )

    # Instantiate the policy with both shared maps.
    policy = vlfm.policy.chrono_policies_multiple.ChronoITMPolicyV2(
        camera_height=camera_height,
        min_depth=min_depth,
        max_depth=max_depth,
        camera_fov=camera_fov,
        image_width=image_width,
        text_prompt=text_prompt,
        use_max_confidence=use_max_confidence,
        pointnav_policy_path=pointnav_policy_path,
        depth_image_shape=depth_image_shape,
        pointnav_stop_radius=pointnav_stop_radius,
        object_map_erosion_size=object_map_erosion_size,
        obstacle_map_area_threshold=obstacle_map_area_threshold,
        min_obstacle_height=min_obstacle_height,
        max_obstacle_height=max_obstacle_height,
        hole_area_thresh=hole_area_thresh,
        use_vqa=use_vqa,
        vqa_prompt=vqa_prompt,
        coco_threshold=coco_threshold,
        non_coco_threshold=non_coco_threshold,
        agent_radius=agent_radius,
        shared_map=shared_map,
        shared_value_map=shared_value_map
    )

    end_time = 10
    control_timestep = 0.1
    time_count = 0
    # Use a mask with one entry per agent.
    masks = torch.zeros(env.num_agents, 1)
    
    while time_count < end_time:
        actions = []
        # Process each agent's observation.
        for agent_obs in obs:
            action, _ = policy.act(agent_obs, None, None, masks)
            actions.append(action)

        # NEW: Update the shared maps using all accumulated sensor info from both agents.
        policy._update_shared_maps()

        # Update the masks for all agents.
        masks = torch.ones(env.num_agents, 1)
        obs, stop = env.step(actions)
        if stop:
            break

        # (Optional) Uncomment the following code block to visualize RGB and depth images.
        # import matplotlib.pyplot as plt
        # for i, agent_obs in enumerate(obs):
        #     rgb_image = agent_obs["rgb"].numpy()
        #     annotated_depth = agent_obs["depth"].numpy()
        #     plt.figure(figsize=(15, 5))
        #     plt.subplot(1, 2, 1)
        #     plt.title(f"Agent {i} RGB")
        #     plt.imshow(rgb_image)
        #     plt.axis('off')
        #     plt.subplot(1, 2, 2)
        #     plt.title(f"Agent {i} Depth")
        #     plt.imshow(annotated_depth, cmap='gray')
        #     plt.axis('off')
        #     plt.tight_layout()
        #     timestamp = time.strftime("%Y%m%d-%H%M%S")
        #     plt.savefig(f"tmp_vis_2/policy_info_visualization_agent{i}_{timestamp}.png")
        #     plt.close()

        time_count += control_timestep
