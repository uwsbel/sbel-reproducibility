from dataclasses import dataclass
from typing import Any, Dict, List, Union
import cv2

import numpy as np
import torch
from depth_camera_filtering import filter_depth
from frontier_exploration.base_explorer import BaseExplorer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from torch import Tensor

from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
from vlfm.vlm.grounding_dino import ObjectDetections

from ..mapping.obstacle_map import ObstacleMap
from .base_objectnav_policy_DSInC import BaseObjectNavPolicy, VLFMConfig
from .itm_policy_DSInC import ITMPolicy, ITMPolicyV2, ITMPolicyV3
import os
from typing import Any, Dict, Tuple, Union

from vlfm.chrono_env.communication_manager import FusedFeatureExchangeManager

class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)
    GOAL = torch.tensor([[4]], dtype=torch.float32)


class ChronoMixin:
    """Class for running VLFM in a Chrono environment."""
    _stop_action: Tensor = TorchActionIDs.STOP
    _start_yaw: Union[float, None] = None  # must be set by _reset() method
    _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}
    _compute_frontiers: bool = True
    _visualize: bool = True

    def __init__(self, camera_height: float, min_depth: float, max_depth: float, camera_fov: float, robot_id: int, comms_manager: FusedFeatureExchangeManager, image_width: int, comm_feature_fifo_size: int = 500, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._camera_height = camera_height
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._camera_fov = np.deg2rad(camera_fov)
        self._image_width = image_width
        self._robot_id = robot_id
        self._comm_manager = comms_manager
        self._comm_manager.register_agent(self._robot_id)
        self._peer_feats_index: List[Tensor] = []
        self._comm_feature_fifo_size = max(int(comm_feature_fifo_size), 1)
        # Convert to radians if in degrees
        camera_fov_rad = np.radians(camera_fov)
        self._fx = self._fy = image_width / (2 * np.tan(camera_fov_rad / 2))

    def act(self, observations, rnn_hidden_states: Any, prev_actions: Any, masks: Tensor, deterministic: bool = False) -> Tuple[Tensor, Any]:
        parent_cls: BaseObjectNavPolicy = super()

        try:
            action, rnn_hidden_states = parent_cls.act(
                observations, rnn_hidden_states, prev_actions, masks, deterministic)
        except StopIteration:
            action = self._stop_action

        return action, rnn_hidden_states

    def _initialize(self) -> Tensor:
        """Turn left 30 degrees 12 times to get a 360 view at the beginning"""
        self._done_initializing = not self._num_steps < 24  # type: ignore
        return TorchActionIDs.TURN_LEFT

    def _reset(self) -> None:
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        parent_cls._reset()
        self._start_yaw = None
        self._peer_feats_index = []
        self._same_start_origin = None
        self._same_start_partner_frontier = None
        self._same_start_last_selected_frontier = None
        self._same_start_split_applied = False

    def _sharing_enabled(self) -> bool:
        if not bool(getattr(self, "_same_start_mode", False)):
            return True
        return int(getattr(self, "_num_steps", 0)) >= int(
            getattr(self, "_same_start_warmup_steps", 0)
        )

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        """Get policy info for logging"""
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        info = parent_cls._get_policy_info(detections)

        if not self._visualize:  # type: ignore
            return info

        if self._start_yaw is None:
            self._start_yaw = self._observations_cache["habitat_start_yaw"]
        info["start_yaw"] = self._start_yaw
        return info

    def _cache_observations(self: Union["ChronoMixin", BaseObjectNavPolicy], observations) -> None:
        """Caches the rgb, depth, and camera transform from the observations.

        Args:
           observations: The observations from the current timestep.
        """
        if len(self._observations_cache) > 0:
            return
        rgb = observations["rgb"].cpu().numpy()
        depth_raw = observations["depth"].cpu().numpy()
        if depth_raw.ndim == 3 and depth_raw.shape[-1] == 1:
            depth_2d = depth_raw[..., 0]
        elif depth_raw.ndim == 2:
            depth_2d = depth_raw
        else:
            depth_2d = depth_raw.reshape(depth_raw.shape[:2])
        nav_depth = torch.from_numpy(np.ascontiguousarray(depth_2d.astype(np.float32))).unsqueeze(0).unsqueeze(-1)
        depth = filter_depth(depth_2d.reshape(depth_2d.shape[:2]), blur_type=None)
        x, y = observations["gps"].cpu().numpy()
        camera_yaw = observations["compass"].cpu().item()

        camera_position = np.array([x, y, self._camera_height])
        robot_xy = camera_position[:2]
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(
            camera_position, camera_yaw)

        self._obstacle_map: ObstacleMap
        if self._compute_frontiers:
            self._obstacle_map.update_map(
                depth,
                tf_camera_to_episodic,
                self._min_depth,
                self._max_depth,
                self._fx,
                self._fy,
                self._camera_fov,
            )
            frontiers = self._obstacle_map.frontiers
            self._obstacle_map.update_agent_traj(robot_xy, camera_yaw)
        else:
            if "frontier_sensor" in observations:
                frontiers = observations["frontier_sensor"][0].cpu().numpy()
            else:
                frontiers = np.array([])

        self._observations_cache = {
            "frontier_sensor": frontiers,
            "nav_depth": nav_depth,  # PointNav expects (N, H, W, 1)
            "robot_xy": robot_xy,
            "robot_heading": camera_yaw,
            "object_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._fx,
                    self._fy,
                )
            ],
            "value_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._camera_fov,
                )
            ],
            "habitat_start_yaw": observations["compass"].item(),
        }
    
    def retrieve_peer_features(self):
        if not self._sharing_enabled():
            return
        new_feats = self._comm_manager.retrieve(self._robot_id)
        self._peer_feats_index.extend(new_feats)
        if len(self._peer_feats_index) > self._comm_feature_fifo_size:
            self._peer_feats_index = self._peer_feats_index[-self._comm_feature_fifo_size:]
    
    def get_value_map(self) -> np.ndarray:
        """
        Quickly grab the latest H×W×3 RGB heatmap for this agent,
        without all the extra plotting and file I/O in _get_policy_info.
        """
        # no frontier markers, just the raw fused value channels
        raw_bgr = self._value_map.visualize(markers=[], reduce_fn=self._vis_reduce_fn)
        return cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)


class ChronoITMPolicy(ChronoMixin, ITMPolicy):
    pass


class ChronoITMPolicyV2(ChronoMixin, ITMPolicyV2):
    pass


class ChronoITMPolicyV3(ChronoMixin, ITMPolicyV3):
    pass
