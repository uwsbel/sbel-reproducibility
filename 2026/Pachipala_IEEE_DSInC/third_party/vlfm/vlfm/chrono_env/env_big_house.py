import time
import vlfm.policy.chrono_policies
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
    def __init__(self, target_object: str = "box"):
        self.my_system = None

        # Output directory
        self.out_dir = "SENSOR_OUTPUT/"

        # Camera lens model
        self.lens_model = getattr(sens, "PINHOLE", sens.CameraLensModelType_PINHOLE)

        # Update rate in Hz
        self.update_rate = 30

        # Image width and height
        self.image_width = 640 #213  (160, 213)
        self.image_height = 480 #160

        # Camera's horizontal field of view
        self.fov = 1.408

        # Lag (in seconds) between sensing and when data becomes accessible
        self.lag = 0

        # Exposure (in seconds) of each image
        self.exposure_time = 0

        self.manager = None

        self.lidar = None
        self.cam = None

        self.vis = None
        self.rt_timer = None
        self.timestep = 0.001
        self.control_frequency = 10  # Control frequency of the simulation
        self.steps_per_control = round(
            1 / (self.timestep * self.control_frequency))
        self.step_number = 0
        self.render_frame = 0
        self.observations = None
        self.target_object = target_object  # Target object
        self.virtual_robot = None

    def reset(self):
        self.my_system = chrono.ChSystemSMC()
        self.my_system.SetCollisionSystemType(
            chrono.ChCollisionSystem.Type_BULLET)

        patch_mat = chrono.ChContactMaterialSMC()

        # patch_mat.SetFriction(0.1)
        # patch_mat.SetRollingFriction(0.001)
        # terrain = veh.RigidTerrain(my_system)
        # patch = terrain.AddPatch(patch_mat,
        #     chrono.ChCoordsysd(chrono.ChVector3d(0, -0.09, 0), chrono.Q_ROTATE_Z_TO_Y),
        #     100, 100)
        # patch.SetColor(chrono.ChColor(0.0, 0.0, 0.0))
        # terrain.Initialize()
        self.virtual_robot = chrono.ChBodyEasyBox(
            0.25, 0.25, 0.5, 100, True, True, patch_mat)
        self.virtual_robot.SetPos(chrono.ChVector3d(0, -12, 0.25))
        self.virtual_robot.SetFixed(True)
        self.my_system.Add(self.virtual_robot)
        mmesh = chrono.ChTriangleMeshConnected()
        mmesh.LoadWavefrontMesh(
            project_root + '/data/patrick_env/bigHouseOnly.obj', False, True)

        # scale to a different size
        # mmesh.Transform(chrono.ChVector3d(0, 0, 0), chrono.ChMatrix33d(2))

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

        # ---------------------------------------

        # Add camera sensor

        # -----------------
        # Camera parameters
        # -----------------

        self.manager = sens.ChSensorManager(self.my_system)
        intensity_low = 0.01
        intensity_moderate = 1.0
        intensity_high = 50.0
        # self.manager.scene.AddAreaLight(chrono.ChVector3f(0, 0, 4), chrono.ChColor(
        #     intensity, intensity, intensity), 500.0, chrono.ChVector3f(1, 0, 0), chrono.ChVector3f(0, -1, 0))
        if hasattr(self.manager.scene, "AddAreaLight"):
            self.manager.scene.AddAreaLight(chrono.ChVector3f(0, 0, 1), chrono.ChColor(
                intensity_moderate, intensity_moderate, intensity_moderate), 500.0, chrono.ChVector3f(1, 0, 0), chrono.ChVector3f(0, -1, 0))
        else:
            self.manager.scene.AddPointLight(chrono.ChVector3f(0, 0, 1), chrono.ChColor(
                intensity_moderate, intensity_moderate, intensity_moderate), 500.0)

        offset_pose = chrono.ChFramed(
            chrono.ChVector3d(0.3, 0, 0.25), chrono.QUNIT)

        self.lidar = sens.ChLidarSensor(
            self.virtual_robot,             # body lidar is attached to
            30,   # was previously 20                    # scanning rate in Hz
            offset_pose,            # offset pose
            self.image_width,                   # number of horizontal samples
            self.image_height,                    # number of vertical channels
            self.fov,                    # horizontal field of view
            chrono.CH_PI/6,         # vertical field of view
            -chrono.CH_PI/6,
            3.66,                  # max lidar range
            sens.LidarBeamShape_RECTANGULAR,
            1,          # sample radius
            0,       # divergence angle
            0,       # divergence angle
            sens.LidarReturnMode_STRONGEST_RETURN)

        self.lidar.SetName("Lidar Sensor")
        self.lidar.SetLag(0)
        self.lidar.SetCollectionWindow(1/20)
        self.lidar.PushFilter(sens.ChFilterVisualize(
            self.image_width, self.image_height, "depth camera"))
        self.lidar.PushFilter(sens.ChFilterDIAccess())
        self.manager.AddSensor(self.lidar)

        self.cam = sens.ChCameraSensor(
            self.virtual_robot,              # body camera is attached to
            self.update_rate,            # update rate in Hz
            offset_pose,            # offset pose
            self.image_width,            # image width
            self.image_height,           # image height
            self.fov                    # camera's horizontal field of view
        )

        self.cam.SetName("Camera Sensor")
        self.cam.SetLag(self.lag)
        self.cam.SetCollectionWindow(self.exposure_time)
        self.cam.PushFilter(sens.ChFilterVisualize(
            self.image_width, self.image_height, "rgb camera"))
        # Provides the host access to this RGBA8 buffer
        self.cam.PushFilter(sens.ChFilterRGBA8Access())

        # add sensor to manager
        self.manager.AddSensor(self.cam)

        # ---------------------------------------

        # Create visualization
        self.vis = chronoirr.ChVisualSystemIrrlicht(self.my_system)
        self.vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
        self.vis.AddLightWithShadow(chrono.ChVector3d(2, 2, 2),  # point
                                    chrono.ChVector3d(0, 0, 0),  # aimpoint
                                    5,                       # radius (power)
                                    1, 11,                     # near, far
                                    55)                       # angle of FOV

        # vis.EnableShadows()
        self.vis.EnableAbsCoordsysDrawing(True)
        self.vis.Initialize()
        self.vis.AddSkyBox()
        self.vis.AddCamera(chrono.ChVector3d(-7/3, 0, 4.5/3),
                           chrono.ChVector3d(0, 0, 0))

        self.observations = self._get_observations()

        return self.observations

    def step(self, action):
        self._do_action(action, self.virtual_robot)
        for i in range(0, self.steps_per_control):
            self.manager.Update()
            sim_time = self.my_system.GetChTime()
            self.my_system.DoStepDynamics(self.timestep)

        self.vis.BeginScene()
        self.vis.Render()
        self.vis.EndScene()
        self.observations = self._get_observations()

        self.stop = self._get_stop(action)

        return self.observations, self.stop

    def _get_stop(self, action):
        if action[0][0] == 0:
            print("STOPPED")
            time.sleep(5)
            return True
        return False

    def _get_observations(self):
        depth_buffer = self.lidar.GetMostRecentDIBuffer()
        if depth_buffer.HasData():
            depth_data = depth_buffer.GetDIData()
            # Removes the 2nd column which is intensity
            depth_data = torch.tensor(
                depth_data[:, :, 0], dtype=torch.float32)

            # Flip vertically and horizontally
            depth_data = torch.flip(depth_data, dims=[0, 1])

            MIN_DEPTH = 0
            MAX_DEPTH = 5.5
            depth_data = np.clip(
                (depth_data - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH), 0, 1)

            # Set pixels to white for depth values greater than MAX_DEPTH
            depth_data[depth_data == 0] = 1  # Set all zero values to 1
        else:
            depth_data = torch.zeros(
                self.image_height, self.image_width, dtype=torch.float32)

        camera_buffer = self.cam.GetMostRecentRGBA8Buffer()
        if camera_buffer.HasData():
            camera_data = camera_buffer.GetRGBA8Data()
            camera_data = torch.tensor(camera_data, dtype=torch.uint8)
            # Remove the 4th column which is transparency
            camera_data = camera_data[:, :, :3]
            camera_data = torch.flip(camera_data, dims=[0])  # Flip vertically
        else:
            camera_data = torch.zeros(
                self.image_height, self.image_width, 3, dtype=torch.uint8)

        robot_x = torch.tensor(
            self.virtual_robot.GetPos().x, dtype=torch.float32)
        robot_y = torch.tensor(
            self.virtual_robot.GetPos().y, dtype=torch.float32)

        quat_list = [self.virtual_robot.GetRot().e0, self.virtual_robot.GetRot().e1,
                     self.virtual_robot.GetRot().e2, self.virtual_robot.GetRot().e3]
        yaw = self.quaternion_to_yaw(quat_list)
        robot_yaw = torch.tensor(yaw, dtype=torch.float32)

        obs_dict = {
            "rgb": camera_data,
            "depth": depth_data,
            "gps": torch.stack((robot_x, robot_y)),
            "compass": robot_yaw,
            "objectgoal": self.target_object  # Target object
        }

        return obs_dict

    def _do_action(self, action, robot):
        # Convert action tensor to integer
        if len(action[0]) > 1:
            print("target pos: ", float(action[0][0]), float(action[0][1]))
            print("CUR POS: ", robot.GetPos())
            print('size of target pos: ', action.shape)
            cur_pos = robot.GetPos()
            target_x = float(action[0][0])
            target_y = float(action[0][1])

            # Calculate heading angle to the target
            heading = math.atan2(target_y - cur_pos.y, target_x - cur_pos.x)
            heading = (heading + math.pi) % (2 * math.pi) - \
                math.pi  # Normalize to [-π, π]

            # Get the robot's current heading
            current_heading = robot.GetRot().GetCardanAnglesXYZ().z
            turn_angle = heading - current_heading
            turn_angle = (turn_angle + math.pi) % (2 * math.pi) - \
                math.pi  # Normalize to [-π, π]

            print("TURN ANGLE: ", turn_angle)

            # Update the robot's rotation
            new_rotation = chrono.QuatFromAngleZ(turn_angle) * robot.GetRot()
            robot.SetRot(new_rotation)

            # heading = math.atan2(
            #     float(action[0][1]) - cur_pos.y, float(action[0][0]) - cur_pos.x)
            # quat_list = [self.virtual_robot.GetRot().e0, self.virtual_robot.GetRot().e1,
            #              self.virtual_robot.GetRot().e2, self.virtual_robot.GetRot().e3]
            # current_heading = robot.GetRot().GetCardanAnglesXYZ().z
            # turn_angle = heading  # - current_heading
            # print("TURN ANGLE: ", turn_angle)
            # robot.SetRot(chrono.QuatFromAngleZ(turn_angle)*robot.GetRot())

            robot.SetPos(chrono.ChVector3d(
                float(action[0][0]), float(action[0][1]), 0.25))
            print("Moved Pos: ", robot.GetPos())
        else:
            action_id = action.item()

            if action_id == 1:  # MOVE_FORWARD
                rot_state = robot.GetRot().GetCardanAnglesXYZ()
                robot.SetPos(robot.GetPos()+chrono.ChVector3d(0.01 *
                             np.cos(rot_state.z), 0.01*np.sin(rot_state.z), 0))

            elif action_id == 2:  # TURN_LEFT
                robot.SetRot(chrono.QuatFromAngleZ(np.pi/12)*robot.GetRot())

            elif action_id == 3:  # TURN_RIGHT
                robot.SetRot(chrono.QuatFromAngleZ(-np.pi/12)*robot.GetRot())

    def quaternion_to_yaw(self, quaternion):
        # Unpack quaternion
        w, x, y, z = quaternion

        # Calculate yaw (angle with respect to the x-axis)
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return yaw


if __name__ == "__main__":
    env = ChronoEnv()
    obs = env.reset()
    # Take fake step
    # obs, stop = env.step(torch.tensor([[0]])) 
    # sensor params
    camera_height = 0.5
    min_depth = 0
    max_depth = 5.5
    camera_fov = 80.67 #1.408 # in deg 80.67
    image_width = 640 #213

    # kwargs for itm policy
    # name = "ChronoITMPolicy"
    text_prompt = "Seems like there is a target_object ahead."
    # text_prompt = "Find a target_object"
    use_max_confidence = False
    pointnav_policy_path = "data/pointnav_weights.pth"
    depth_image_shape = (480, 640) # (160, 213)
    pointnav_stop_radius = 0.5
    object_map_erosion_size = 5
    exploration_thresh = 0.7
    obstacle_map_area_threshold = 1.5  # in square meters
    min_obstacle_height = 0.3
    max_obstacle_height = 0.5
    hole_area_thresh = 100000
    use_vqa = False
    vqa_prompt = "Is this "
    coco_threshold = 0.8
    non_coco_threshold = 0.4
    agent_radius = 0.15

    vlfm_policy = vlfm.policy.chrono_policies.ChronoITMPolicyV2(
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
        agent_radius=agent_radius
    )

    end_time = 10
    control_timestep = 0.1
    time_count = 0
    masks = torch.zeros(1, 1)
    obs, stop = env.step(torch.tensor([[5]], dtype=torch.long))
    while time_count < end_time:
        action, _ = vlfm_policy.act(obs, None, None, masks)
        masks = torch.ones(1, 1)
        obs, stop = env.step(action)
        if stop:
            break

        # Visualize the depth and RGB images
        import matplotlib.pyplot as plt
        # Convert depth and RGB observations to numpy arrays
        # annotated_depth = torch.flip(obs["depth"], dims=[0, 1]).numpy()
        annotated_depth = obs["depth"].numpy()

        # annotated_depth = torch.flip(annotated_depth, dims=[0, 1]).numpy()e
        rgb_image = obs["rgb"].numpy()
        # Plot the depth and RGB images
        plt.figure(figsize=(15, 5))
        # Annotated Depth Image
        plt.subplot(1, 2, 1)
        plt.title("Annotated Depth")
        # Use grayscale for depth visualization
        plt.imshow(annotated_depth, cmap='gray')
        plt.axis('off')
        # RGB Image
        plt.subplot(1, 2, 2)
        plt.title("RGB Image")
        plt.imshow(rgb_image)  # No need for cmap, as it's an RGB image
        plt.axis('off')
        # Display the plot
        plt.tight_layout()
        # ---------------------------------------

        # Save the figure to a file with a unique name
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(f"tmp_vis_2/policy_info_visualization_{timestamp}.png")
        plt.close()

        # obs, stop = env.step(torch.tensor(0))
        # print(vlfm_policy._policy_info)

        time_count += control_timestep
