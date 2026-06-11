# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch import Tensor
from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import get_fov, rho_theta
from vlfm.vlm.blip2 import BLIP2Client
from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.grounding_dino import GroundingDINOClient, ObjectDetections
from vlfm.vlm.sam import MobileSAMClient
from vlfm.vlm.yolov7 import YOLOv7Client

try:
    from habitat_baselines.common.tensor_dict import TensorDict

    from vlfm.policy.base_policy import BasePolicy
except Exception:

    class BasePolicy:  # type: ignore
        pass


class BaseObjectNavPolicy(BasePolicy):
    _target_object: str = ""
    _policy_info: Dict[str, Any] = {}
    # set by ._update_object_map()
    _object_masks: Union[np.ndarray, Any] = None
    _stop_action: Union[Tensor, Any] = None  # MUST BE SET BY SUBCLASS
    _observations_cache: Dict[str, Any] = {}
    _non_coco_caption = ""
    _load_yolo: bool = True
    _detector_class_aliases: Dict[str, str] = {
        "tv_monitor": "tv",
        "sofa": "couch",
        "plant": "potted plant",
    }

    def __init__(
        self,
        pointnav_policy_path: str,
        depth_image_shape: Tuple[int, int],
        pointnav_stop_radius: float,
        object_map_erosion_size: float,
        pointnav_stop_enable_radius: float = 1.0,
        pointnav_stop_consecutive_required: int = 3,
        pointnav_stop_close_tracking_radius: float = 2.0,
        pointnav_stop_overshoot_radius: float = 3.0,
        pointnav_stagnation_radius: float = 3.0,
        pointnav_stagnation_steps: int = 10,
        visualize: bool = True,
        compute_frontiers: bool = True,
        min_obstacle_height: float = 0.15,
        max_obstacle_height: float = 0.88,
        agent_radius: float = 0.18,
        obstacle_map_area_threshold: float = 1.5,
        hole_area_thresh: int = 100000,
        use_vqa: bool = False,
        vqa_prompt: str = "Is this ",
        coco_threshold: float = 0.8,
        non_coco_threshold: float = 0.4,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._object_detector = GroundingDINOClient(
            port=int(os.environ.get("GROUNDING_DINO_PORT", "12181")))
        self._coco_object_detector = YOLOv7Client(
            port=int(os.environ.get("YOLOV7_PORT", "12184")))
        self._mobile_sam = MobileSAMClient(
            port=int(os.environ.get("SAM_PORT", "12183")))
        self._use_vqa = use_vqa
        if use_vqa:
            self._vqa = BLIP2Client(
                port=int(os.environ.get("BLIP2_PORT", "12185")))
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(
            erosion_size=object_map_erosion_size)
        self._depth_image_shape = tuple(depth_image_shape)
        self._pointnav_stop_radius = pointnav_stop_radius
        self._pointnav_stop_enable_radius = pointnav_stop_enable_radius
        self._pointnav_stop_consecutive_required = max(
            int(pointnav_stop_consecutive_required), 1
        )
        self._pointnav_stop_close_tracking_radius = pointnav_stop_close_tracking_radius
        self._pointnav_stop_overshoot_radius = pointnav_stop_overshoot_radius
        self._pointnav_stagnation_radius = pointnav_stagnation_radius
        self._pointnav_stagnation_steps = max(int(pointnav_stagnation_steps), 1)
        self._visualize = visualize
        self._vqa_prompt = vqa_prompt
        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold

        self._num_steps = 0
        self._did_reset = False
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._called_stop = False
        self._compute_frontiers = compute_frontiers
        self._navigate_stop_rejection_active = False
        self._navigate_consecutive_stop_predictions = 0
        self._navigate_best_rho_after_rejection: Optional[float] = None
        self._navigate_seen_rho_below_close_tracking = False
        self._navigate_best_rho_recent: Optional[float] = None
        self._navigate_steps_since_rho_improvement = 0
        if compute_frontiers:
            self._obstacle_map = ObstacleMap(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                area_thresh=obstacle_map_area_threshold,
                agent_radius=agent_radius,
                hole_area_thresh=hole_area_thresh,
            )

    def _reset(self) -> None:
        self._target_object = ""
        self._pointnav_policy.reset()
        self._object_map.reset()
        self._last_goal = np.zeros(2)
        self._num_steps = 0
        self._done_initializing = False
        self._called_stop = False
        if self._compute_frontiers:
            self._obstacle_map.reset()
        self._did_reset = True
        self._reset_navigate_stop_gate()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        """
        Starts the episode by 'initializing' and allowing robot to get its bearings
        (e.g., spinning in place to get a good view of the scene).
        Then, explores the scene until it finds the target object.
        Once the target object is found, it navigates to the object.
        """
        self._pre_step(observations, masks)

        object_map_rgbd = self._observations_cache["object_map_rgbd"]
        detections = [
            self._update_object_map(
                rgb, depth, tf, min_depth, max_depth, fx, fy)
            for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
        ]
        robot_xy = self._observations_cache["robot_xy"]
        goal = self._get_target_object_location(robot_xy)

        if not self._done_initializing:  # Initialize
            mode = "initialize"
            pointnav_action = self._initialize()
        elif goal is None:  # Haven't found target object yet
            mode = "explore"
            pointnav_action = self._explore(observations)
        else:
            mode = "navigate"
            pointnav_action = self._pointnav(goal[:2], stop=True)

        action_id = int(torch.as_tensor(pointnav_action).view(-1)[0].item())
        action_label = {
            0: "STOP",
            1: "MOVE_FORWARD",
            2: "TURN_LEFT",
            3: "TURN_RIGHT",
        }.get(action_id, f"ACTION_{action_id}")
        self._policy_info["mode"] = mode
        self._policy_info["step"] = int(self._num_steps)
        self._policy_info["action"] = action_label
        self._policy_info["blocked_by_obstacle"] = "pointnav"
        print(
            f"Step: {self._num_steps} | Mode: {mode} | Action: {action_id}")
        self._policy_info.update(self._get_policy_info(detections[0]))
        self._num_steps += 1

        self._observations_cache = {}
        self._did_reset = False

        return pointnav_action, rnn_hidden_states

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if not self._did_reset and masks[0] == 0:
            self._reset()
            self._target_object = observations["objectgoal"]
        try:
            self._cache_observations(observations)
        except IndexError as e:
            print(e)
            print("Reached edge of map, stopping.")
            raise StopIteration
        self._policy_info = {}

    def _initialize(self) -> Tensor:
        raise NotImplementedError

    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        if self._object_map.has_object(self._target_object):
            return self._object_map.get_best_object(self._target_object, position)
        else:
            return None

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        if self._object_map.has_object(self._target_object):
            target_point_cloud = self._object_map.get_target_cloud(
                self._target_object)
        else:
            target_point_cloud = np.array([])
        policy_info = {
            "target_object": self._target_object.split("|")[0],
            "gps": str(self._observations_cache["robot_xy"] * np.array([1, -1])),
            "yaw": np.rad2deg(self._observations_cache["robot_heading"]),
            "target_detected": self._object_map.has_object(self._target_object),
            "target_point_cloud": target_point_cloud,
            "nav_goal": self._last_goal,
            "stop_called": self._called_stop,
            # don't render these on egocentric images when making videos:
            "render_below_images": [
                "target_object",
            ],
        }
        # Carry runtime step metadata produced in act()/pointnav into visualization info.
        for key in ("mode", "step", "action", "blocked_by_obstacle"):
            if key in self._policy_info:
                policy_info[key] = self._policy_info[key]

        if not self._visualize:
            return policy_info
        # Save depth data to CSV
        rgb_data = self._observations_cache["object_map_rgbd"][0][0]
        depth_data = self._observations_cache["object_map_rgbd"][0][1]
        # Normalize depth using fixed min/max values and scale to 255
        # MIN_DEPTH = 0.1
        # MAX_DEPTH = 7.5
        # normalized_depth = np.clip(
        #     (depth_data - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH), 0, 1)
        # annotated_depth = normalized_depth * 255
        annotated_depth = depth_data * 255
        annotated_depth = cv2.cvtColor(
            annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if self._object_masks.sum() > 0:
            # If self._object_masks isn't all zero, get the object segmentations and
            # draw them on the rgb and depth images
            contours, _ = cv2.findContours(
                self._object_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(
                detections.annotated_frame, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(
                annotated_depth, contours, -1, (255, 0, 0), 2)
        else:
            annotated_rgb = self._observations_cache["object_map_rgbd"][0][0]
        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        # Visualize the depth map
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(16, 8))
        # plt.subplot(1, 3, 1)
        # plt.title("RGB")
        # plt.imshow(rgb_data)
        # plt.axis('off')


        # plt.subplot(1, 3, 2)
        # plt.title("Annotated Depth")
        # plt.imshow(annotated_depth)
        # plt.axis('off')
        

        if self._compute_frontiers:
            obstacle_map_rgb = cv2.cvtColor(
                self._obstacle_map.visualize(), cv2.COLOR_BGR2RGB
            )
            policy_info["obstacle_map"] = obstacle_map_rgb

            # Zoom in on a portion of the obstacle map (crop the image)
            height, width, _ = obstacle_map_rgb.shape
            cropped_map = obstacle_map_rgb[
                height // 4: 2 * height // 3,  # Vertical range
                width // 3: 2 * width // 3    # Horizontal range
            ]

        #     # Visualize the zoomed-in obstacle map
        #     plt.subplot(1, 3, 3)
        #     plt.title("Obstacle Map (Zoomed In)")
        #     plt.imshow(cropped_map)
        #     plt.axis('off')

        # # Save the figure to a file with a unique name
        # timestamp = time.strftime("%Y%m%d-%H%M%S")
        # plt.savefig(f"tmp_vis/policy_info_visualization_{timestamp}.png")
        # plt.close()

        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        return policy_info

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        raw_target_classes = self._target_object.split("|")
        target_classes = [
            self._detector_class_aliases.get(cls_name, cls_name)
            for cls_name in raw_target_classes
        ]
        # Preserve order while removing duplicates after aliasing.
        target_classes = list(dict.fromkeys(target_classes))
        has_coco = any(
            c in COCO_CLASSES for c in target_classes) and self._load_yolo
        has_non_coco = any(c not in COCO_CLASSES for c in target_classes)

        detections = (
            self._coco_object_detector.predict(img)
            if has_coco
            else self._object_detector.predict(img, caption=self._non_coco_caption)
        )
        detections.filter_by_class(target_classes)
        det_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold
        detections.filter_by_conf(det_conf_threshold)

        if has_coco and has_non_coco and detections.num_detections == 0:
            # Retry with non-coco object detector
            detections = self._object_detector.predict(
                img, caption=self._non_coco_caption)
            detections.filter_by_class(target_classes)
            detections.filter_by_conf(self._non_coco_threshold)

        return detections

    def _pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        device = self._pointnav_policy.device
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device=device)
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                masks = torch.zeros_like(masks)
                self._reset_navigate_stop_gate()
            self._last_goal = goal

        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device=device, dtype=torch.float32)
        depth_for_pointnav = image_resize(
            self._observations_cache["nav_depth"],
            (self._depth_image_shape[0], self._depth_image_shape[1]),
            channels_last=True,
            interpolation_mode="area",
        ).to(device=device, dtype=torch.float32)
        obs_pointnav = {
            "depth": depth_for_pointnav,
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info["rho_theta"] = np.array([rho, theta])
        self._policy_info["navigate_stop_gate_active"] = self._navigate_stop_rejection_active
        self._policy_info["navigate_best_rho_after_rejection"] = self._navigate_best_rho_after_rejection
        self._policy_info["navigate_steps_since_rho_improvement"] = (
            self._navigate_steps_since_rho_improvement
        )
        if rho < self._pointnav_stop_radius and stop:
            self._called_stop = True
            self._policy_info["navigate_stop_reason"] = "force_stop_radius"
            return self._stop_action
        action, action_preferences, next_rnn_hidden_states = (
            self._pointnav_policy.act_with_action_preferences(
                obs_pointnav,
                masks,
                deterministic=True,
            )
        )
        chosen_action = action
        stop_action_id = int(torch.as_tensor(self._stop_action).view(-1)[0].item())
        chosen_action_id = int(torch.as_tensor(chosen_action).view(-1)[0].item())

        if stop:
            if rho < self._pointnav_stagnation_radius:
                if (
                    self._navigate_best_rho_recent is None
                    or rho < self._navigate_best_rho_recent - 1e-4
                ):
                    self._navigate_best_rho_recent = float(rho)
                    self._navigate_steps_since_rho_improvement = 0
                else:
                    self._navigate_steps_since_rho_improvement += 1
                if (
                    self._navigate_steps_since_rho_improvement
                    >= self._pointnav_stagnation_steps
                ):
                    self._called_stop = True
                    self._policy_info["navigate_stop_reason"] = "stagnation_within_radius"
                    self._pointnav_policy.commit_action(self._stop_action, next_rnn_hidden_states)
                    return self._stop_action
            else:
                self._navigate_best_rho_recent = None
                self._navigate_steps_since_rho_improvement = 0
            if self._navigate_stop_rejection_active:
                if (
                    self._navigate_best_rho_after_rejection is None
                    or rho < self._navigate_best_rho_after_rejection
                ):
                    self._navigate_best_rho_after_rejection = float(rho)
            if rho < self._pointnav_stop_close_tracking_radius:
                self._navigate_seen_rho_below_close_tracking = True
            if (
                self._navigate_stop_rejection_active
                and self._navigate_seen_rho_below_close_tracking
                and rho > self._pointnav_stop_overshoot_radius
            ):
                self._called_stop = True
                self._policy_info["navigate_stop_reason"] = "overshoot_fallback"
                self._pointnav_policy.commit_action(self._stop_action, next_rnn_hidden_states)
                return self._stop_action
            if chosen_action_id == stop_action_id:
                if rho < self._pointnav_stop_enable_radius:
                    self._navigate_consecutive_stop_predictions += 1
                    if (
                        self._navigate_consecutive_stop_predictions
                        >= self._pointnav_stop_consecutive_required
                    ):
                        self._called_stop = True
                        self._policy_info["navigate_stop_reason"] = "consecutive_learned_stop"
                        self._pointnav_policy.commit_action(chosen_action, next_rnn_hidden_states)
                        return chosen_action
                else:
                    self._navigate_stop_rejection_active = True
                    if (
                        self._navigate_best_rho_after_rejection is None
                        or rho < self._navigate_best_rho_after_rejection
                    ):
                        self._navigate_best_rho_after_rejection = float(rho)
                self._navigate_consecutive_stop_predictions = 0
                chosen_action = self._select_best_non_stop_action(
                    action_preferences,
                    fallback_action=action,
                )
                chosen_action_id = int(torch.as_tensor(chosen_action).view(-1)[0].item())
                self._policy_info["navigate_stop_reason"] = "learned_stop_rejected"
            else:
                self._navigate_consecutive_stop_predictions = 0

        self._pointnav_policy.commit_action(chosen_action, next_rnn_hidden_states)
        self._policy_info["navigate_selected_action"] = int(chosen_action_id)
        return chosen_action

    def _pointnav_navigate(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        return self._pointnav(goal, stop)

    def _reset_navigate_stop_gate(self) -> None:
        self._navigate_stop_rejection_active = False
        self._navigate_consecutive_stop_predictions = 0
        self._navigate_best_rho_after_rejection = None
        self._navigate_seen_rho_below_close_tracking = False
        self._navigate_best_rho_recent = None
        self._navigate_steps_since_rho_improvement = 0

    def _select_best_non_stop_action(
        self,
        action_preferences: Optional[Tensor],
        fallback_action: Tensor,
    ) -> Tensor:
        stop_action_id = int(torch.as_tensor(self._stop_action).view(-1)[0].item())
        if action_preferences is not None:
            preferences = action_preferences.detach().float().view(-1)
            ranked_actions = torch.argsort(preferences, descending=True)
            for action_idx in ranked_actions.tolist():
                if int(action_idx) == stop_action_id:
                    continue
                return torch.tensor(
                    [[int(action_idx)]],
                    dtype=torch.long,
                    device=preferences.device,
                )
        fallback_action_id = int(torch.as_tensor(fallback_action).view(-1)[0].item())
        if fallback_action_id != stop_action_id:
            return fallback_action
        return torch.tensor([[1]], dtype=torch.long, device=fallback_action.device)

    def _update_object_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections:
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        Args:
            rgb (np.ndarray): The rgb image to use for updating the object map. Used for
                object detection and Mobile SAM segmentation to extract better object
                point clouds.
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.

        Returns:
            ObjectDetections: The object detections from the object detector.
        """
        detections = self._get_object_detections(rgb)
        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8)
        if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0:
            depth = self._infer_depth(rgb, min_depth, max_depth)
            obs = list(self._observations_cache["object_map_rgbd"][0])
            obs[1] = depth
            self._observations_cache["object_map_rgbd"][0] = tuple(obs)
        for idx in range(len(detections.logits)):
            bbox_denorm = detections.boxes[idx] * \
                np.array([width, height, width, height])
            object_mask = self._mobile_sam.segment_bbox(
                rgb, bbox_denorm.tolist())

            # If we are using vqa, then use the BLIP2 model to visually confirm whether
            # the contours are actually correct.

            if self._use_vqa:
                contours, _ = cv2.findContours(
                    object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                annotated_rgb = cv2.drawContours(
                    rgb.copy(), contours, -1, (255, 0, 0), 2)
                question = f"Question: {self._vqa_prompt}"
                if not detections.phrases[idx].endswith("ing"):
                    question += "a "
                question += detections.phrases[idx] + "? Answer:"
                answer = self._vqa.ask(annotated_rgb, question)
                if not answer.lower().startswith("yes"):
                    continue

            self._object_masks[object_mask > 0] = 1
            self._object_map.update_map(
                self._target_object,
                depth,
                object_mask,
                tf_camera_to_episodic,
                min_depth,
                max_depth,
                fx,
                fy,
            )

        cone_fov = get_fov(fx, depth.shape[1])
        self._object_map.update_explored(
            tf_camera_to_episodic, max_depth, cone_fov)

        return detections

    def _cache_observations(self, observations: "TensorDict") -> None:
        """Extracts the rgb, depth, and camera transform from the observations.

        Args:
            observations ("TensorDict"): The observations from the current timestep.
        """
        raise NotImplementedError

    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """Infers the depth image from the rgb image.

        Args:
            rgb (np.ndarray): The rgb image to infer the depth from.

        Returns:
            np.ndarray: The inferred depth image.
        """
        raise NotImplementedError


@dataclass
class VLFMConfig:
    name: str = "HabitatITMPolicy"
    text_prompt: str = "Seems like there is a target_object ahead."
    pointnav_policy_path: str = "data/pointnav_weights.pth"
    depth_image_shape: Tuple[int, int] = (224, 224)
    pointnav_stop_radius: float = 0.9
    use_max_confidence: bool = False
    object_map_erosion_size: int = 5
    exploration_thresh: float = 0.0
    obstacle_map_area_threshold: float = 1.5  # in square meters
    min_obstacle_height: float = 0.61
    max_obstacle_height: float = 0.88
    hole_area_thresh: int = 100000
    use_vqa: bool = False
    vqa_prompt: str = "Is this "
    coco_threshold: float = 0.8
    non_coco_threshold: float = 0.4
    agent_radius: float = 0.18

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        # This returns all the fields listed above, except the name field
        return [f.name for f in fields(VLFMConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="vlfm_config_base", node=VLFMConfig())
