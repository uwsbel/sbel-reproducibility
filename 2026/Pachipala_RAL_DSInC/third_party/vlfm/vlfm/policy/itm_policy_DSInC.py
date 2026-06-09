# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import math
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from vlfm.mapping.frontier_map import FrontierMap
from vlfm.mapping.value_map import ValueMap
from vlfm.policy.base_objectnav_policy_DSInC import BaseObjectNavPolicy
from vlfm.policy.utils.acyclic_enforcer import AcyclicEnforcer
from vlfm.utils.geometry_utils import closest_point_within_threshold
from vlfm.utils.feature_fusion import per_query_gated_fusion, compute_cosine_similarity
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.detections import ObjectDetections
import matplotlib.pyplot as plt
import time

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except Exception:
    pass

PROMPT_SEPARATOR = "|"


class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    @staticmethod
    def _vis_reduce_fn(i: np.ndarray) -> np.ndarray:
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        sync_explored_areas: bool = False,
        comm_fuse_window: int = 8,
        comm_novelty_threshold: float = 0.85,
        comm_novelty_beta: float = 0.003,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._itm = BLIP2ITMClient(port=int(os.environ.get("BLIP2ITM_PORT", "12182")))
        self._comm_fuse_window = max(int(comm_fuse_window), 1)
        self._comm_novelty_threshold = float(comm_novelty_threshold)
        self._comm_novelty_beta = max(float(comm_novelty_beta), 1e-12)
        self._text_prompt = text_prompt
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split(PROMPT_SEPARATOR)),
            use_max_confidence=use_max_confidence,
            obstacle_map=self._obstacle_map if sync_explored_areas else None,
        )
        self._acyclic_enforcer = AcyclicEnforcer()

    @staticmethod
    def _sigmoid_scalar(value: float) -> float:
        if value >= 0.0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def _reset(self) -> None:
        super()._reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)
        self._fuse_state = {}
        self._comm_redundant_frame_limit = 5
        self._same_start_split_applied = False
        self._same_start_last_selected_frontier = None
        self._same_start_origin = None

    def _ensure_same_start_origin(self, robot_xy: np.ndarray) -> np.ndarray:
        origin = getattr(self, "_same_start_origin", None)
        if origin is None:
            origin = np.asarray(robot_xy, dtype=np.float32).copy()
            self._same_start_origin = origin
        return np.asarray(origin, dtype=np.float32)

    def _select_same_start_split_frontier(
        self,
        sorted_pts: np.ndarray,
        sorted_values: List[float],
        robot_xy: np.ndarray,
    ) -> Union[int, None]:
        if not bool(getattr(self, "_same_start_mode", False)):
            return None
        if not bool(getattr(self, "_same_start_first_split_pending", False)):
            return None
        if int(getattr(self, "_robot_id", 0)) != 2:
            return None

        partner_frontier = getattr(self, "_same_start_partner_frontier", None)
        if partner_frontier is None or len(sorted_pts) < 2:
            return None

        origin = self._ensure_same_start_origin(robot_xy)
        primary = np.asarray(partner_frontier, dtype=np.float32)[:2]
        primary_vec = primary - origin
        primary_norm = float(np.linalg.norm(primary_vec))
        if primary_norm <= 1e-6:
            return None

        best_idx = None
        best_angle = -1.0
        best_value = float("-inf")
        best_dist = float("-inf")
        for idx, frontier in enumerate(sorted_pts):
            frontier_xy = np.asarray(frontier, dtype=np.float32)[:2]
            if float(np.linalg.norm(frontier_xy - primary)) <= 0.25:
                continue
            frontier_vec = frontier_xy - origin
            frontier_norm = float(np.linalg.norm(frontier_vec))
            if frontier_norm <= 1e-6:
                continue
            cosine = float(
                np.clip(
                    np.dot(primary_vec, frontier_vec) / (primary_norm * frontier_norm),
                    -1.0,
                    1.0,
                )
            )
            angle = float(np.degrees(np.arccos(cosine)))
            if angle <= 0.0:
                continue
            frontier_value = float(sorted_values[idx])
            if (
                angle > best_angle + 1e-6
                or (
                    abs(angle - best_angle) <= 1e-6
                    and (
                        frontier_value > best_value + 1e-9
                        or (
                            abs(frontier_value - best_value) <= 1e-9
                            and frontier_norm > best_dist
                        )
                    )
                )
            ):
                best_idx = idx
                best_angle = angle
                best_value = frontier_value
                best_dist = frontier_norm
        return best_idx

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("Robot",self._robot_id,": No frontiers found during exploration, stopping.")
            return self._stop_action
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Robot{self._robot_id}: Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, stop=False)

        return pointnav_action

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        # The points and values will be sorted in descending order
        sorted_pts, sorted_values = self._sort_frontiers_by_value(observations, frontiers)
        robot_xy = self._observations_cache["robot_xy"]
        self._ensure_same_start_origin(robot_xy)
        best_frontier_idx = None
        top_two_values = tuple(sorted_values[:2])

        os.environ["DEBUG_INFO"] = ""
        split_frontier_idx = self._select_same_start_split_frontier(
            sorted_pts, sorted_values, robot_xy
        )
        if split_frontier_idx is not None:
            best_frontier_idx = split_frontier_idx
            self._same_start_split_applied = True
            self._same_start_first_split_pending = False
            self._same_start_partner_frontier = None
            os.environ["DEBUG_INFO"] += "Same-start split. "
        elif (
            bool(getattr(self, "_same_start_mode", False))
            and bool(getattr(self, "_same_start_first_split_pending", False))
            and int(getattr(self, "_robot_id", 0)) == 2
            and getattr(self, "_same_start_partner_frontier", None) is not None
        ):
            self._same_start_split_applied = False
            self._same_start_first_split_pending = False
            self._same_start_partner_frontier = None

        # If there is a last point pursued, then we consider sticking to pursuing it
        # if it is still in the list of frontiers and its current value is not much
        # worse than self._last_value.
        if best_frontier_idx is None and not np.array_equal(self._last_frontier, np.zeros(2)):
            curr_index = None

            for idx, p in enumerate(sorted_pts):
                if np.array_equal(p, self._last_frontier):
                    # Last point is still in the list of frontiers
                    curr_index = idx
                    break

            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_pts, self._last_frontier, threshold=0.5)

                if closest_index != -1:
                    # There is a point close to the last point pursued
                    curr_index = closest_index

            if curr_index is not None:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    # The last point pursued is still in the list of frontiers and its
                    # value is not much worse than self._last_value
                    print("Sticking to last point.")
                    os.environ["DEBUG_INFO"] += "Sticking to last point. "
                    best_frontier_idx = curr_index

        # If there is no last point pursued, then just take the best point, given that
        # it is not cyclic.
        if best_frontier_idx is None:
            for idx, frontier in enumerate(sorted_pts):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier, top_two_values)
                if cyclic:
                    print("Suppressed cyclic frontier.")
                    continue
                best_frontier_idx = idx
                break

        if best_frontier_idx is None:
            print("Robot ",self._robot_id,": All frontiers are cyclic. Just choosing the closest one.")
            os.environ["DEBUG_INFO"] += "All frontiers are cyclic. "
            best_frontier_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
            )

        best_frontier = sorted_pts[best_frontier_idx]
        best_value = sorted_values[best_frontier_idx]
        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier, top_two_values)
        self._last_value = best_value
        self._last_frontier = best_frontier
        self._same_start_last_selected_frontier = np.asarray(best_frontier, dtype=np.float32).copy()
        os.environ["DEBUG_INFO"] += f" Best value: {best_value*100:.2f}%"

        return best_frontier, best_value

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 4, 1)
        plt.title("Annotated RGB")
        plt.imshow(policy_info["annotated_rgb"])
        plt.axis('off')

        # depth image
        plt.subplot(1, 4, 2)
        plt.title("Annotated Depth")
        plt.imshow(policy_info["annotated_depth"])
        plt.axis('off')

        # Visualize the zoomed-in obstacle map
        obstacle_map_rgb = policy_info["obstacle_map"]
        # Zoom in on a portion of the obstacle map (crop the image)
        height, width, _ = obstacle_map_rgb.shape
        # Define a smaller cropping range to zoom in further on the center
        cropped_map = obstacle_map_rgb[
            height // 2 - height // 5: height // 2 + height // 5,  # Center vertical range
            width // 2 - width // 5: width // 2 + width // 5       # Center horizontal range
        ]

        plt.subplot(1, 4, 3)
        plt.title("Obstacle Map")
        plt.imshow(cropped_map)
        plt.axis('off')

        # value map
        value_map = policy_info["value_map"]
        height, width, _ = value_map.shape
        # cropped_value_map = value_map[
        #     height // 4: 2 * height // 3,  # Vertical range
        #     width // 3: 2 * width // 3    # Horizontal range
        # ]
        cropped_value_map = value_map[
            height // 2 - height // 5: height // 2 + height // 5,  # Center vertical range
            width // 2 - width // 5: width // 2 + width // 5       # Center horizontal range
        ]
        plt.subplot(1, 4, 4)
        plt.title("Value Map")
        plt.imshow(cropped_value_map)
        plt.axis('off')

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        target_name = self._target_object.split(PROMPT_SEPARATOR)[0] if self._target_object else "unknown"
        episode_num = os.environ.get("VLFM_EPISODE_NUM", "na")
        episode_dir = os.path.join("tmp_viz", f"episode_{episode_num}")
        os.makedirs(episode_dir, exist_ok=True)
        step_num = policy_info.get("step", "na")
        mode = policy_info.get("mode", "na")
        action = policy_info.get("action", "na")
        target_found = policy_info.get("target_detected", "na")
        blocked = policy_info.get("blocked_by_obstacle", "na")
        plt.suptitle(
            (
                f"Episode: {episode_num} | Robot: {self._robot_id} | Target: {target_name}\n"
                f"Step: {step_num} | Mode: {mode} | Action: {action} | "
                f"Target Found: {target_found} | Blocked: {blocked}"
            ),
            fontsize=12,
            y=0.99,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.savefig(os.path.join(episode_dir, f"R{self._robot_id}_{step_num}_{timestamp}.png"))
        plt.close()

        return policy_info

    def _fuse_image_features(self, batch_results):
        """
        Streaming fusion: for each (view, prompt) pair we fuse one image feature embedding at a time.
        Once 8 have been accumulated for that (view,prompt), append the fused
        embedding to self._fused_feats and reset.
        """
        # Initialize state on first call
        if not hasattr(self, '_fuse_state'):
            # maps view_idx -> prompt_idx -> {'fused': Tensor, 'count': int}
            self._fuse_state = {}
            # flat list of fused embeddings
            # self._fused_feats = []

        for view_idx, infer_out in enumerate(batch_results):
            view_state = self._fuse_state.setdefault(view_idx, {})
            for prompt_idx, result in enumerate(infer_out):
                if len(result) == 3:
                    _cosine, feat, novelty = result
                else:
                    _cosine, feat = result
                    novelty = 1.0
                state = view_state.setdefault(prompt_idx, {'fused': None, 'count': 0})
                redundant_frames = int(state.get('redundant_frames', 0))
                if float(novelty) < self._comm_novelty_threshold:
                    redundant_frames += 1
                    state['redundant_frames'] = redundant_frames
                    if redundant_frames >= self._comm_redundant_frame_limit:
                        # Pause outgoing fusion only after sustained peer overlap.
                        # Local value-map updates still use the novelty score below.
                        continue
                else:
                    state['redundant_frames'] = 0

                # accumulate or start
                if state['fused'] is None:
                    state['fused'] = feat
                else:
                    state['fused'] = per_query_gated_fusion(state['fused'], feat)

                state['count'] += 1

                # Once the configured window is full, broadcast and reset.
                if state['count'] >= self._comm_fuse_window:
                    if self._sharing_enabled():
                        self._comm_manager.push(self._robot_id, state['fused'])
                    state['fused'] = None
                    state['count'] = 0

    def _attach_peer_scores(self, batch_results):
        """
        For each (view, prompt) entry we compute the highest cosine similarity
        against peer features. If it’s at least 0.8 we keep it, otherwise we use 0 and treat the current view as unexplored
        Returns:
        [view_idx][prompt_idx] = (local_cosine, feat, peer_score)
        """
        peer_feats = getattr(self, "_peer_feats_index", [])
        new_batch = []
        for view_results in batch_results:
            new_view = []
            for local_cos, feat in view_results:
                # find max similarity across all peer features
                max_score = 0.0
                for peer_feat in peer_feats:
                    score = compute_cosine_similarity(feat, peer_feat)
                    if score > max_score:
                        max_score = score
                novelty_activation = self._sigmoid_scalar(
                    (max_score - self._comm_novelty_threshold)
                    / self._comm_novelty_beta
                )
                novelty_score = float(
                    np.clip(1.0 - max_score * novelty_activation, 0.0, 1.0)
                )
                if (novelty_score < 0.2):
                    print("Robot",self._robot_id,": Match found, Area was explored by a peer, novelty score:", novelty_score)
                new_view.append((local_cos, feat, novelty_score))
            new_batch.append(new_view)
        return new_batch

    def _update_value_map(self) -> None:
        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        # for each rgb view, call infer() on each prompt -> returns (cosine, feats_tensor)
        batch_results = [
            [
                self._itm.infer(
                    rgb,
                    p.replace("target_object", self._target_object.replace("|", "/")),
                )
                for p in self._text_prompt.split(PROMPT_SEPARATOR)
            ]
            for rgb in all_rgb
        ]

        self.retrieve_peer_features()
        batch_results = self._attach_peer_scores(batch_results)
        self._fuse_image_features(batch_results)

        # print("Robot",self._robot_id,": Length of the peer features received by Agent", self._robot_id, ":", len(self._peer_feats_index))
        
        for infer_out, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            batch_results, self._observations_cache["value_map_rgbd"]
        ):
            cosine, feats, novelty = zip(*[(c, f, n) for c, f, n in infer_out])
            updated_cosine = np.array(cosine) * np.array(novelty)
            self._value_map.update_map(np.array(updated_cosine), depth, tf, min_depth, max_depth, fov)

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError


class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        if self._visualize:
            self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _reset(self) -> None:
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        rgb = self._observations_cache["object_map_rgbd"][0][0]
        text = self._text_prompt.replace("target_object", self._target_object)
        self._frontier_map.update(frontiers, rgb, text)  # type: ignore
        return self._frontier_map.sort_waypoints()


class ITMPolicyV2(BaseITMPolicy):
    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        self._pre_step(observations, masks)
        self._update_value_map()
        return super().act(observations, rnn_hidden_states, prev_actions, masks, deterministic)

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)
        return sorted_frontiers, sorted_values


class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exploration_thresh = exploration_thresh

        def visualize_value_map(arr: np.ndarray) -> np.ndarray:
            # Get the values in the first channel
            first_channel = arr[:, :, 0]
            # Get the max values across the two channels
            max_values = np.max(arr, axis=2)
            # Create a boolean mask where the first channel is above the threshold
            mask = first_channel > exploration_thresh
            # Use the mask to select from the first channel or max values
            result = np.where(mask, first_channel, max_values)

            return result

        self._vis_reduce_fn = visualize_value_map  # type: ignore

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5, reduce_fn=self._reduce_values)

        return sorted_frontiers, sorted_values

    def _reduce_values(self, values: List[Tuple[float, float]]) -> List[float]:
        """
        Reduce the values to a single value per frontier

        Args:
            values: A list of tuples of the form (target_value, exploration_value). If
                the highest target_value of all the value tuples is below the threshold,
                then we return the second element (exploration_value) of each tuple.
                Otherwise, we return the first element (target_value) of each tuple.

        Returns:
            A list of values, one per frontier.
        """
        target_values = [v[0] for v in values]
        max_target_value = max(target_values)

        if max_target_value < self._exploration_thresh:
            explore_values = [v[1] for v in values]
            return explore_values
        else:
            return [v[0] for v in values]
