import argparse
import time
import vlfm.policy.chrono_policy_DSInC as DSInC
from vlfm.chrono_env.communication_manager import FusedFeatureExchangeManager
from vlfm.utils.feature_fusion import overlay_robot_maps
import math
from math import atan, tan
import numpy as np
import pychrono.sensor as sens
import pychrono.irrlicht as chronoirr
import pychrono as chrono
import torch
import sys
import shutil
import os
import cv2
import matplotlib.pyplot as plt
import csv

vis_dirs = ["tmp_vis", "tmp_vis_2"]
for d in vis_dirs:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

# Assuming the script is located in the 'experiments/apartment' directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(project_root)
# Add the parent directory of 'models' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

FRAME_WIDTH = 640   
FRAME_HEIGHT = 480

class ChronoEnv:
    def __init__(self, target_objects, num_agents: int = 2):
        self.num_agents = num_agents
        self.headless = os.environ.get("VLFM_HEADLESS", "0") == "1" or not os.environ.get("DISPLAY")
        self.my_system = None
        self.out_dir = "SENSOR_OUTPUT/"
        self.lens_model = getattr(sens, "PINHOLE", sens.CameraLensModelType_PINHOLE)
        self.update_rate = 30
        self.image_width = FRAME_WIDTH
        self.image_height = FRAME_HEIGHT
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
        if isinstance(target_objects, str):
            target_objects = [target_objects] * num_agents
        self.target_objects = target_objects
        self.virtual_robots = []
        self.lidar_list = []
        self.cam_list = []
        self._object_found = False

    def reset(self):
        self.my_system = chrono.ChSystemSMC()
        self.my_system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        patch_mat = chrono.ChContactMaterialSMC()

        if self.num_agents == 1:
            start_positions = [chrono.ChVector3d(-1.25, -1.25, 0.25)]
        else:
            start_positions = [
                chrono.ChVector3d(0, 0, 0.25), # Robot 1
                chrono.ChVector3d(2.5, 1.5, 0.25), # Robot 2
            ]

        self.manager = sens.ChSensorManager(self.my_system)
        intensity_moderate = 1.0
        if hasattr(self.manager.scene, "AddAreaLight"):
            self.manager.scene.AddAreaLight(chrono.ChVector3f(0, 0, 1), chrono.ChColor(intensity_moderate, intensity_moderate, intensity_moderate), 500.0, chrono.ChVector3f(1, 0, 0), chrono.ChVector3f(0, -1, 0))
        else:
            self.manager.scene.AddPointLight(chrono.ChVector3f(0, 0, 1), chrono.ChColor(intensity_moderate, intensity_moderate, intensity_moderate), 500.0)

        robot_id = 1

        for pos in start_positions:
            robot = chrono.ChBodyEasyBox(0.25, 0.25, 0.5, 100, True, True, patch_mat)
            robot.SetPos(pos)
            robot.SetFixed(True)
            self.my_system.Add(robot)
            self.virtual_robots.append(robot)
            offset_pose = chrono.ChFramed(chrono.ChVector3d(0.3, 0, 0.75), chrono.QUNIT)
            vfov = self.fov
            hfov = 2*atan((FRAME_WIDTH/FRAME_HEIGHT)*tan(vfov/2))


            lidar = sens.ChLidarSensor(
                robot,
                30,
                offset_pose,
                self.image_width,
                self.image_height,
                hfov,
                vfov/2,
                -vfov/2,
                3.66,
                sens.LidarBeamShape_RECTANGULAR,
                1,
                0, 0,
                sens.LidarReturnMode_STRONGEST_RETURN
            )
            lidar.SetName("Lidar Sensor")
            lidar.SetLag(0)
            lidar.SetCollectionWindow(1/20)
            if not self.headless:
                lidar.PushFilter(sens.ChFilterVisualize(self.image_width, self.image_height, "depth camera"))
            lidar.PushFilter(sens.ChFilterDIAccess())
            self.manager.AddSensor(lidar)
            self.lidar_list.append(lidar)

            cam = sens.ChCameraSensor(
                robot,
                self.update_rate,
                offset_pose,
                self.image_width,
                self.image_height,
                self.fov
            )
            cam.SetName("Camera Sensor")
            cam.SetLag(self.lag)
            cam.SetCollectionWindow(self.exposure_time)
            if not self.headless:
                cam.PushFilter(sens.ChFilterVisualize(self.image_width, self.image_height, f"rgb camera {robot_id}"))
            cam.PushFilter(sens.ChFilterRGBA8Access())
            self.manager.AddSensor(cam)
            self.cam_list.append(cam)
            robot_id += 1

        mmesh = chrono.ChTriangleMeshConnected()
        mmesh.LoadWavefrontMesh(project_root + '/data/chrono_environment/patricks_home/ph_wingra.obj', False, True)
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

        # # ---------------------------------------
        # # Add a target object to the environment

        # add_obj_path = os.path.join(project_root, "data/chrono_environment/target_object/no_tv.obj")  # Adjust as needed
        # add_obj_mat = chrono.ChContactMaterialSMC()

        # add_obj = chrono.ChBodyEasyMesh(
        #     add_obj_path,
        #     100,        # density
        #     False,      # no collision
        #     True,       # visualize
        #     True,       # smooth shading
        #     add_obj_mat,
        #     0.05       # envelope
        # )
        # # add_obj.SetRot(chrono.Q_ROTATE_Y_TO_Z)
        # add_obj.SetPos(chrono.ChVector3d(0.0, 0.0, 0.0))
        # # add_obj.SetPos(chrono.ChVector3d(-12.5, 1.5, 0.50))  
        # add_obj.SetFixed(True)
        # self.my_system.Add(add_obj)

        # # ---------------------------------------

        if not self.headless:
            self.vis = chronoirr.ChVisualSystemIrrlicht(self.my_system)
            self.vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
            self.vis.AddLightWithShadow(chrono.ChVector3d(2, 2, 2), chrono.ChVector3d(0, 0, 0), 5, 1, 11, 55)
            self.vis.AddLightWithShadow(chrono.ChVector3d(-2, 1.25, 2),  # point
                                        chrono.ChVector3d(0, 0, 0),  # aimpoint
                                        3,                       # radius (power)
                                        1, 11,                     # near, far
                                        55)                       # angle of FOV

            self.vis.AddLightWithShadow(chrono.ChVector3d(-2, -2, 2),  # point
                                        chrono.ChVector3d(0, 0, 0),  # aimpoint
                                        3,                       # radius (power)
                                        1, 11,                     # near, far
                                        55)                       # angle of FOV
            self.vis.EnableAbsCoordsysDrawing(True)
            self.vis.Initialize()
            self.vis.AddSkyBox()
            self.vis.AddCamera(chrono.ChVector3d(-7/3, 0, 4.5/3), chrono.ChVector3d(0, 0, 0))
        self.observations = [self._get_observations(i) for i in range(len(self.virtual_robots))]
        return self.observations

    # def step(self, actions):
    #     for i, action in enumerate(actions):
    #         self._do_action(action, self.virtual_robots[i])
    #     for _ in range(self.steps_per_control):
    #         self.manager.Update()
    #         self.my_system.DoStepDynamics(self.timestep)

    #     self.vis.BeginScene()
    #     self.vis.Render()
    #     self.vis.EndScene()

    #     new_observations = [self._get_observations(i) for i in range(len(self.virtual_robots))]
    #     stop = all(self._get_stop(action) for action in actions)
    #     return new_observations, stop

    ######################################## Common stop (Logical OR) (Preferred)
    def step(self, actions):
        if not self._object_found:
            for i, action in enumerate(actions):
                self._do_action(action, self.virtual_robots[i])
            for _ in range(self.steps_per_control):
                self.manager.Update()
                self.my_system.DoStepDynamics(self.timestep)

        if self.vis is not None:
            self.vis.BeginScene()
            self.vis.Render()
            self.vis.EndScene()

        new_observations = [self._get_observations(i) for i in range(len(self.virtual_robots))]
        for i, action in enumerate(actions):
            if self._get_stop(action):
                self._object_found = True
                target = self.target_objects[i]
                print(f"Agent {i+1} found the {target}")
                break

        return new_observations, self._object_found

    ########################################  Global stop (Logical AND) (Testing Only) 
    def _get_stop(self, action):
        if isinstance(action, torch.Tensor) and action.numel() == 1 and action.item() == 0:
            print("STOPPED")
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
        target_object = self.target_objects[idx]
        obs_dict = {
            "rgb": camera_data,
            "depth": depth_data,
            "gps": torch.stack((robot_x, robot_y)),
            "compass": robot_yaw,
            "objectgoal": target_object
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

    parser = argparse.ArgumentParser( description="Run ChronoEnv DSInC with specified targets and number of agents")
    parser.add_argument( "--targets", required=True, help="One or more object names seperated by '|' -- e.g. microwave|chair|tv")
    parser.add_argument( "--num_agents", type=int, default=2, help="Number of collabrative agents to spawn")
    args = parser.parse_args()

    env = ChronoEnv(args.targets, num_agents=args.num_agents)
    obs = env.reset()
    camera_height = 0.5
    min_depth = 0
    max_depth = 5.5
    camera_fov = 80.67 #1.408 # in deg 80.67
    image_width = FRAME_WIDTH 
    text_prompt = "Seems like there is a target_object ahead."
    use_max_confidence = False
    pointnav_policy_path = "data/pointnav_weights.pth"
    depth_image_shape = (FRAME_HEIGHT, FRAME_WIDTH) 
    pointnav_stop_radius = 0.5
    object_map_erosion_size = 5
    exploration_thresh = 0.7
    obstacle_map_area_threshold = 1.5  # in square meters
    min_obstacle_height = 0.3
    max_obstacle_height = 0.5
    hole_area_thresh = 100000
    use_vqa = False
    vqa_prompt = "Is this "
    coco_threshold = 0.65
    non_coco_threshold = 0.3
    agent_radius = 0.15
    comms = FusedFeatureExchangeManager()

    policy_1 = DSInC.ChronoITMPolicyV2(
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
        robot_id=1,
        comms_manager=comms
    )

    policy_2 = DSInC.ChronoITMPolicyV2(
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
        robot_id=2,
        comms_manager=comms
    )

    end_time = 20
    control_timestep = 0.1
    time_count = 0
    masks = torch.zeros(1, 1)
    obs, stop = env.step([torch.tensor([[5]], dtype=torch.long)])

    log_path = os.path.join("tmp_vis", "pose_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["sim_time", "robot_id", "x", "y", "yaw"])

    while time_count < end_time:
        action_1, _ = policy_1.act(obs[0], None, None, masks)
        action_2, _ = policy_2.act(obs[1], None, None, masks)
        actions = [action_1, action_2]
        masks = torch.ones(1, 1)
        obs, stop = env.step(actions)

        for idx, ob in enumerate(obs, start=1):
            x = ob["gps"][0].item()
            y = ob["gps"][1].item()
            yaw = ob["compass"].item()
            sim_time = time_count
            log_writer.writerow([sim_time, idx, x, y, yaw])

        # vm1 = policy_1.get_value_map()
        # vm2 = policy_2.get_value_map()

        # combined = overlay_robot_maps(vm1, vm2, alpha=0.5)
        # cv2.imwrite(f"tmp_vis/combined_{int(time_count*10):04d}.png",
        #             cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        # plt.figure(figsize=(6, 6))
        # plt.imshow(combined)
        # plt.axis("off")
        # plt.pause(0.001)


        if stop:
            log_file.close()
            break

        time_count += control_timestep
