from dataclasses import dataclass
from typing import Any, Dict, Union, Tuple
import numpy as np
import torch
from torch import Tensor

# Import helper functions and mapping modules.
from depth_camera_filtering import filter_depth
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.vlm.grounding_dino import ObjectDetections

# Import base policies (assumed to be available in the package structure)
from .base_objectnav_policy_multiple import BaseObjectNavPolicy, VLFMConfig
from .itm_policy_multiple import ITMPolicy, ITMPolicyV2, ITMPolicyV3

# Define a simple enumeration for action IDs.
class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)
    GOAL = torch.tensor([[4]], dtype=torch.float32)

class ChronoMixin(BaseObjectNavPolicy):
    """
    Mixin for running VLFM in a Chrono environment.
    Expects a shared mapping object to be provided.
    """
    _stop_action: Tensor = TorchActionIDs.STOP
    _start_yaw: Union[float, None] = None  # Must be set in reset
    # We now use an observation list for multi-agent fusion.
    _observations_cache_list: list = []
    _policy_info: Dict[str, Any] = {}
    _compute_frontiers: bool = True
    _visualize: bool = True

    def __init__(self, camera_height: float, min_depth: float, max_depth: float,
                 camera_fov: float, image_width: int, shared_map: ObstacleMap, *args: Any, **kwargs: Any) -> None:
        # Pass any extra arguments to the parent.
        kwargs.setdefault("shared_map", shared_map)
        super().__init__(camera_height=camera_height, min_depth=min_depth, max_depth=max_depth,
                         camera_fov=camera_fov, image_width=image_width, *args, **kwargs)
        self.shared_map = shared_map  # Shared obstacle map instance
        # Also store the shared value map if provided.
        self.shared_value_map = kwargs.get("shared_value_map", None)
        self._camera_height = camera_height
        self._min_depth = min_depth
        self._max_depth = max_depth
        # Convert provided field of view to radians.
        self._camera_fov = np.deg2rad(camera_fov)
        self._image_width = image_width
        # Calculate focal lengths (fx and fy) from camera intrinsics.
        camera_fov_rad = np.radians(camera_fov)
        self._fx = self._fy = image_width / (2 * np.tan(camera_fov_rad / 2))
        # Initialize our observation list.
        self._observations_cache_list = []

    def act(self, observations, rnn_hidden_states: Any, prev_actions: Any, masks: Tensor,
            deterministic: bool = False) -> Tuple[Tensor, Any]:
        parent_cls: BaseObjectNavPolicy = super()
        # Cache this observation before taking action.
        # self._cache_observations(observations)

        # Pre-process observations: cache and aggregate them.
        self._pre_step(observations, masks)
        try:
            action, rnn_hidden_states = parent_cls.act(
                observations, rnn_hidden_states, prev_actions, masks, deterministic)
        except StopIteration:
            action = self._stop_action
        return action, rnn_hidden_states

    def _initialize(self) -> Tensor:
        """Spin (turn left) repeatedly at reset for an initial 360° scan."""
        # Turning left 30° 12 times gives a full 360° rotation.
        self._done_initializing = not self._num_steps < 24  # type: ignore
        return TorchActionIDs.TURN_LEFT

    def _reset(self) -> None:
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        parent_cls._reset()
        self._start_yaw = None
        self._observations_cache_list = []  # Clear previous observations

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        """Get policy info for logging."""
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        info = parent_cls._get_policy_info(detections)
        if not self._visualize:
            return info
        if self._start_yaw is None:
            # Use the first cached observation's compass value if available.
            if self._observations_cache_list:
                self._start_yaw = self._observations_cache_list[0].get("habitat_start_yaw", 0)
            else:
                self._start_yaw = 0
        info["start_yaw"] = self._start_yaw
        return info
    
    def _aggregate_observations(self) -> None:
        """
        Aggregate the multi-agent observation cache (a list of dictionaries) into a single
        dictionary in the format expected by the ITM policy.
        
        This implementation loops through the cached observations and builds lists for keys
        that are expected to be in list format, then for keys expected to be single values (like
        robot_xy) it picks the first element.
        """
        # Define default keys with empty or default values.
        default = {
            "frontier_sensor": [],
            "nav_depth": None,
            "robot_xy": [],
            "robot_heading": 0,
            "object_map_rgbd": [],
            "value_map_rgbd": [],
            "habitat_start_yaw": 0,
            "rgb": [],
            "depth": [],
            "tf": []
        }
        # If nothing was cached, set _observations_cache to default.
        if not self._observations_cache_list:
            self._observations_cache = default
            return
        
        # Initialize aggregation dictionary.
        agg = {key: [] for key in default.keys()}
        
        # Loop over each cached observation and append values.
        for obs in self._observations_cache_list:
            for key in default.keys():
                if key in obs:
                    agg[key].append(obs[key])
                else:
                    agg[key].append(default[key])
        
        # For keys that are expected to be singular (like robot_xy, robot_heading, habitat_start_yaw),
        # pick the first value. For keys that ITM expects to be provided as a list (e.g. value_map_rgbd),
        # you might choose to pick the first one if that is what the ITM policy expects.
        aggregated = {
            "frontier_sensor": agg["frontier_sensor"][0] if agg["frontier_sensor"] else np.array([]),
            "nav_depth": agg["nav_depth"][0] if agg["nav_depth"] is not None and len(agg["nav_depth"])>0 else None,
            "robot_xy": agg["robot_xy"][0] if agg["robot_xy"] else np.array([0, 0]),
            "robot_heading": agg["robot_heading"][0] if agg["robot_heading"] else 0,
            "object_map_rgbd": agg["object_map_rgbd"][0] if agg["object_map_rgbd"] else [],
            "value_map_rgbd": agg["value_map_rgbd"][0] if agg["value_map_rgbd"] else [],
            "habitat_start_yaw": agg["habitat_start_yaw"][0] if agg["habitat_start_yaw"] else 0,
            "rgb": agg["rgb"][0] if agg["rgb"] else None,
            "depth": agg["depth"][0] if agg["depth"] else None,
            "tf": agg["tf"][0] if agg["tf"] else None,
        }
        
        self._observations_cache = aggregated




    def _cache_observations(self, observations) -> None:
        """
        Cache observations from an agent. Instead of overwriting a single dictionary,
        we append the processed observation to the internal list for multi-agent fusion.
        Also, update the shared obstacle map with the current observation.
        """
        # Process the current observation.
        rgb = observations["rgb"].cpu().numpy()
        depth = observations["depth"].cpu().numpy()
        x, y = observations["gps"].cpu().numpy()
        camera_yaw = observations["compass"].cpu().item()
        
        # Optionally filter or pre-process the depth map.
        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
        
        camera_position = np.array([x, y, self._camera_height])
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)
        robot_xy = camera_position[:2]

        # Immediately update the shared obstacle map.
        self.shared_map.update_map(
            depth,
            tf_camera_to_episodic,
            self._min_depth,
            self._max_depth,
            self._fx,
            self._fy,
            self._camera_fov,
        )
        self.shared_map.update_agent_traj(robot_xy, camera_yaw)
        
        # Append the processed observation to the cache list.
        self._observations_cache_list.append({
            "frontier_sensor": self.shared_map.frontiers,
            "nav_depth": observations["depth"],
            "robot_xy": robot_xy,
            "robot_heading": camera_yaw,
            "object_map_rgbd": [
                (rgb, depth, tf_camera_to_episodic, self._min_depth, self._max_depth, self._fx, self._fy)
            ],
            "value_map_rgbd": [
                (rgb, depth, tf_camera_to_episodic, self._min_depth, self._max_depth, self._camera_fov)
            ],
            "habitat_start_yaw": observations["compass"].item(),
            "rgb": rgb,
            "depth": depth,
            "tf": tf_camera_to_episodic
        })

    def _update_shared_maps(self) -> None:
        """
        Fuse the sensor updates from all cached observations across agents.
        For each cached observation:
          - Compute a local confidence (or observation) map using _process_local_data.
          - Compute a semantic value using your _itm.cosine function.
        Then call the shared value map's update_map (which accepts a list of updates) to fuse them.
        Finally, clear the observation cache.
        """
        if not self._observations_cache_list:
            return

        new_maps_list = []
        new_values_list = []
        # Iterate through the collected observations from all agents.
        for obs in self._observations_cache_list:
            # Compute the local confidence map from depth.
            curr_map = self.shared_map._process_local_data(
                obs["depth"],
                self._camera_fov,
                self._min_depth,
                self._max_depth
            )
            # Compute a semantic value using the cosine function.
            # Here, we assume self._itm is defined in your ITM policy (or inherited),
            # and that the text prompt is available as self._text_prompt.
            cosine_val = self._itm.cosine(
                obs["rgb"],
                self._text_prompt.replace("target_object", obs["object_map_rgbd"][0][0])
            )
            values = np.array(cosine_val)
            # If values is 2D and only one channel is expected, expand dims.
            if values.ndim == 2 and self.shared_value_map._value_channels == 1:
                values = values[..., np.newaxis]
            new_maps_list.append(curr_map)
            new_values_list.append(values)
        
        # Call the shared value map's update_map with lists of new updates.
        self.shared_value_map.update_map(new_maps_list, new_values_list, self._min_depth, self._max_depth, self._camera_fov)
        
        # After fusing, aggregate the per-agent observations into a single dictionary.
        self._aggregate_observations()

        # Clear the cache for the next time step.
        self._observations_cache_list = []

    def _pre_step(self, observations: Any, masks: Tensor) -> None:
        """
        Pre-processing before taking an action. This method caches the new observation
        and aggregates the multi-agent observation cache so that downstream routines (such as
        _update_value_map) find the expected keys.
        """
        # Reset if needed (example: if not _did_reset and mask[0] == 0)
        if not self._did_reset and masks[0] == 0:
            self._reset()
            # Assuming the target object is stored in observations["objectgoal"]
            self._target_object = observations["objectgoal"]
        
        # Cache the new observation from this agent.
        self._cache_observations(observations)
        
        # Aggregate the cached observations into a dictionary with expected keys.
        self._aggregate_observations()
        
        # Clear any temporary policy information.
        self._policy_info = {}


# Define multi-agent policy classes by mixing in ChronoMixin with your ITM variants.
class ChronoITMPolicy(ChronoMixin, ITMPolicy):
    pass

class ChronoITMPolicyV2(ChronoMixin, ITMPolicyV2):
    pass

class ChronoITMPolicyV3(ChronoMixin, ITMPolicyV3):
    pass

# Optionally, if using dataclass-based configuration, you might add a VLFMPolicyConfig here.
# @dataclass
# class VLFMPolicyConfig(VLFMConfig, PolicyConfig):
#     pass
