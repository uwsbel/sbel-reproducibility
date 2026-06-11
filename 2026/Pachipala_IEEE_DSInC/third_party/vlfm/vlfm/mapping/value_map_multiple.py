# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import glob
import json
import os
import os.path as osp
import shutil
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from vlfm.mapping.base_map import BaseMap
from vlfm.utils.geometry_utils import extract_yaw, get_rotation_matrix
from vlfm.utils.img_utils import (
    monochannel_to_inferno_rgb,
    pixel_value_within_radius,
    place_img_in_img,
    rotate_image,
)

DEBUG = False
SAVE_VISUALIZATIONS = False
RECORDING = os.environ.get("RECORD_VALUE_MAP", "0") == "1"
PLAYING = os.environ.get("PLAY_VALUE_MAP", "0") == "1"
RECORDING_DIR = "value_map_recordings"
JSON_PATH = osp.join(RECORDING_DIR, "data.json")
KWARGS_JSON = osp.join(RECORDING_DIR, "kwargs.json")


class ValueMap(BaseMap):
    """Generates a map representing how valuable explored regions of the environment
    are with respect to finding and navigating to the target object."""

    _confidence_masks: Dict[Tuple[float, float], np.ndarray] = {}
    _camera_positions: List[np.ndarray] = []
    _last_camera_yaw: float = 0.0
    _min_confidence: float = 0.25
    _decision_threshold: float = 0.35
    _map: np.ndarray

    def __init__(
        self,
        value_channels: int,
        size: int = 1000,
        use_max_confidence: bool = True,
        fusion_type: str = "default",
        obstacle_map: Optional["ObstacleMap"] = None,  # type: ignore # noqa: F821
    ) -> None:
        """
        Args:
            value_channels: The number of channels in the value map.
            size: The size of the value map in pixels.
            use_max_confidence: Whether to use the maximum confidence value in the value
                map or a weighted average confidence value.
            fusion_type: The type of fusion to use when combining the value map with the
                obstacle map.
            obstacle_map: An optional obstacle map to use for overriding the occluded
                areas of the FOV
        """
        if PLAYING:
            size = 2000
        super().__init__(size)
        self._value_map = np.zeros((size, size, value_channels), np.float32)
        self._value_channels = value_channels
        self._use_max_confidence = use_max_confidence
        self._fusion_type = fusion_type
        self._obstacle_map = obstacle_map
        if self._obstacle_map is not None:
            assert self._obstacle_map.pixels_per_meter == self.pixels_per_meter
            assert self._obstacle_map.size == self.size
        if os.environ.get("MAP_FUSION_TYPE", "") != "":
            self._fusion_type = os.environ["MAP_FUSION_TYPE"]

        if RECORDING:
            if osp.isdir(RECORDING_DIR):
                warnings.warn(f"Recording directory {RECORDING_DIR} already exists. Deleting it.")
                shutil.rmtree(RECORDING_DIR)
            os.mkdir(RECORDING_DIR)
            # Dump all args to a file
            with open(KWARGS_JSON, "w") as f:
                json.dump(
                    {
                        "value_channels": value_channels,
                        "size": size,
                        "use_max_confidence": use_max_confidence,
                    },
                    f,
                )
            # Create a blank .json file inside for now
            with open(JSON_PATH, "w") as f:
                f.write("{}")

    def reset(self) -> None:
        super().reset()
        self._value_map.fill(0)

    def update_map(
        self,
        new_maps_list: List[np.ndarray],
        new_values_list: List[np.ndarray],
        min_depth: float,
        max_depth: float,
        fov: float,
    ) -> None:
        """
        Updates the value map using a list of new updates (one per agent).

        Args:
            new_maps_list: A list of numpy arrays of shape (H, W), each a new confidence map.
            new_values_list: A list of numpy arrays, each with shape (H, W) or (H, W, value_channels),
                             representing new value updates.
            min_depth: The minimum depth value (in meters).
            max_depth: The maximum depth value (in meters).
            fov: The field of view of the camera in radians.
        """
        self._fuse_new_data(new_maps_list, new_values_list)

        if RECORDING:
            idx = len(glob.glob(osp.join(RECORDING_DIR, "*.png")))
            # For recording purposes, record the first agent's depth (for example)
            img_path = osp.join(RECORDING_DIR, f"{idx:04d}.png")
            cv2.imwrite(img_path, (new_maps_list[0] * 255).astype(np.uint8))
            with open(JSON_PATH, "r") as f:
                data = json.load(f)
            data[img_path] = {
                "values": new_values_list[0].tolist(),
                "min_depth": min_depth,
                "max_depth": max_depth,
                "fov": fov,
            }
            with open(JSON_PATH, "w") as f:
                json.dump(data, f)

    def sort_waypoints(
        self, waypoints: np.ndarray, radius: float, reduce_fn: Optional[Callable] = None
    ) -> Tuple[np.ndarray, List[float]]:
        """Selects the best waypoint from the given list of waypoints.

        Args:
            waypoints (np.ndarray): An array of 2D waypoints to choose from.
            radius (float): The radius in meters to use for selecting the best waypoint.
            reduce_fn (Callable, optional): The function to use for reducing the values
                within the given radius. Defaults to np.max.

        Returns:
            Tuple[np.ndarray, List[float]]: A tuple of the sorted waypoints and
                their corresponding values.
        """
        radius_px = int(radius * self.pixels_per_meter)

        def get_value(point: np.ndarray) -> Union[float, Tuple[float, ...]]:
            x, y = point
            px = int(-x * self.pixels_per_meter) + self._episode_pixel_origin[0]
            py = int(-y * self.pixels_per_meter) + self._episode_pixel_origin[1]
            point_px = (self._value_map.shape[0] - px, py)
            all_values = [
                pixel_value_within_radius(self._value_map[..., c], point_px, radius_px)
                for c in range(self._value_channels)
            ]
            if len(all_values) == 1:
                return all_values[0]
            return tuple(all_values)

        values = [get_value(point) for point in waypoints]

        if self._value_channels > 1:
            assert reduce_fn is not None, "Must provide a reduction function when using multiple value channels."
            values = reduce_fn(values)

        # Use np.argsort to get the indices of the sorted values
        sorted_inds = np.argsort([-v for v in values])  # type: ignore
        sorted_values = [values[i] for i in sorted_inds]
        sorted_frontiers = np.array([waypoints[i] for i in sorted_inds])

        return sorted_frontiers, sorted_values

    def visualize(
        self,
        markers: Optional[List[Tuple[np.ndarray, Dict[str, Any]]]] = None,
        reduce_fn: Callable = lambda i: np.max(i, axis=-1),
        obstacle_map: Optional["ObstacleMap"] = None,  # type: ignore # noqa: F821
    ) -> np.ndarray:
        """Return an image representation of the map"""
        reduced_map = reduce_fn(self._value_map).copy()
        if obstacle_map is not None:
            reduced_map[obstacle_map.explored_area == 0] = 0
        map_img = np.flipud(reduced_map)
        zero_mask = map_img == 0
        map_img[zero_mask] = np.max(map_img)
        map_img = monochannel_to_inferno_rgb(map_img)
        map_img[zero_mask] = (255, 255, 255)
        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                map_img,
                self._camera_positions,
                self._last_camera_yaw,
            )
            if markers is not None:
                for pos, marker_kwargs in markers:
                    map_img = self._traj_vis.draw_circle(map_img, pos, **marker_kwargs)

        return map_img

    def _process_local_data(self, depth: np.ndarray, fov: float, min_depth: float, max_depth: float) -> np.ndarray:
        if len(depth.shape) == 3:
            depth = depth.squeeze(2)
        depth_row = np.max(depth, axis=0) * (max_depth - min_depth) + min_depth
        angles = np.linspace(-fov / 2, fov / 2, len(depth_row))
        x = depth_row
        y = depth_row * np.tan(angles)
        cone_mask = self._get_confidence_mask(fov, max_depth)
        x = (x * self.pixels_per_meter + cone_mask.shape[0] / 2).astype(int)
        y = (y * self.pixels_per_meter + cone_mask.shape[1] / 2).astype(int)
        last_row = cone_mask.shape[0] - 1
        last_col = cone_mask.shape[1] - 1
        start = np.array([[0, last_col]])
        end = np.array([[last_row, last_col]])
        contour = np.concatenate((start, np.stack((y, x), axis=1), end), axis=0)
        visible_mask = cv2.drawContours(cone_mask, [contour], -1, 0, -1)  # type: ignore
        if DEBUG:
            vis = cv2.cvtColor((cone_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            cv2.drawContours(vis, [contour], -1, (0, 0, 255), -1)
            for point in contour:
                vis[point[1], point[0]] = (0, 255, 0)
            if SAVE_VISUALIZATIONS:
                if not os.path.exists("visualizations"):
                    os.makedirs("visualizations")
                depth_row_full = np.repeat(depth_row.reshape(1, -1), depth.shape[0], axis=0)
                depth_rgb = cv2.cvtColor((depth * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                depth_row_full = cv2.cvtColor((depth_row_full * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                vis = np.flipud(vis)
                new_width = int(vis.shape[1] * (depth_rgb.shape[0] / vis.shape[0]))
                vis_resized = cv2.resize(vis, (new_width, depth_rgb.shape[0]))
                vis = np.hstack((depth_rgb, depth_row_full, vis_resized))
                time_id = int(time.time() * 1000)
                cv2.imwrite(f"visualizations/{time_id}.png", vis)
            else:
                cv2.imshow("obstacle mask", vis)
                cv2.waitKey(0)
        return visible_mask

    def _localize_new_data(
        self,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fov: float,
    ) -> np.ndarray:
        curr_data = self._process_local_data(depth, fov, min_depth, max_depth)
        yaw = extract_yaw(tf_camera_to_episodic)
        if PLAYING:
            if yaw > 0:
                yaw = 0
            else:
                yaw = np.deg2rad(30)
        curr_data = rotate_image(curr_data, -yaw)
        cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]
        px = int(cam_x * self.pixels_per_meter) + self._episode_pixel_origin[0]
        py = int(-cam_y * self.pixels_per_meter) + self._episode_pixel_origin[1]
        curr_map = np.zeros_like(self._map)
        curr_map = place_img_in_img(curr_map, curr_data, px, py)
        return curr_map

    def _get_blank_cone_mask(self, fov: float, max_depth: float) -> np.ndarray:
        size = int(max_depth * self.pixels_per_meter)
        cone_mask = np.zeros((size * 2 + 1, size * 2 + 1))
        cone_mask = cv2.ellipse(
            cone_mask,
            (size, size),
            (size, size),
            0,
            -np.rad2deg(fov) / 2 + 90,
            np.rad2deg(fov) / 2 + 90,
            1,
            -1,
        )
        return cone_mask

    def _get_confidence_mask(self, fov: float, max_depth: float) -> np.ndarray:
        if (fov, max_depth) in self._confidence_masks:
            return self._confidence_masks[(fov, max_depth)].copy()
        cone_mask = self._get_blank_cone_mask(fov, max_depth)
        adjusted_mask = np.zeros_like(cone_mask).astype(np.float32)
        for row in range(adjusted_mask.shape[0]):
            for col in range(adjusted_mask.shape[1]):
                horizontal = abs(row - adjusted_mask.shape[0] // 2)
                vertical = abs(col - adjusted_mask.shape[1] // 2)
                angle = np.arctan2(vertical, horizontal)
                angle = remap(angle, 0, fov / 2, 0, np.pi / 2)
                confidence = np.cos(angle) ** 2
                confidence = remap(confidence, 0, 1, self._min_confidence, 1)
                adjusted_mask[row, col] = confidence
        adjusted_mask = adjusted_mask * cone_mask
        self._confidence_masks[(fov, max_depth)] = adjusted_mask.copy()
        return adjusted_mask

    def _fuse_new_data(self, new_maps_list: list, new_values_list: list) -> None:
        """
        Fuse multiple new map updates (from different agents) with the existing
        confidence map (self._map) and value map (self._value_map).

        Args:
            new_maps_list: A list of numpy arrays of shape (H, W), containing the new
                           confidence/observation maps from each agent.
            new_values_list: A list of numpy arrays, each of shape (H, W, value_channels) or (H, W)
                             for a single channel, containing the new value estimates from the agent.
        """
        for idx, values in enumerate(new_values_list):
            if values.ndim < 2:
                raise ValueError(f"Expected values array with at least 2 dimensions, got shape {values.shape}.")
            if values.ndim == 2:
                if self._value_channels == 1:
                    new_values_list[idx] = values[..., np.newaxis]
                else:
                    raise ValueError(f"Expected values with {self._value_channels} channels, but got a 2D array.")
            if new_values_list[idx].shape[-1] != self._value_channels:
                raise ValueError(f"Expected {self._value_channels} channels, but got {new_values_list[idx].shape[-1]}.")

        if self._obstacle_map is not None:
            explored_area = self._obstacle_map.explored_area
            for i in range(len(new_maps_list)):
                new_maps_list[i] = new_maps_list[i].copy()
                new_values_list[i] = new_values_list[i].copy()
                new_maps_list[i][explored_area == 0] = 0
                new_values_list[i][explored_area == 0] = 0

        if self._fusion_type == "replace":
            print("VALUE MAP ABLATION: replace")
            for new_map, values in zip(new_maps_list, new_values_list):
                mask = new_map > 0
                self._map[mask] = new_map[mask]
                self._value_map[mask] = values[mask]
            return
        elif self._fusion_type == "equal_weighting":
            print("VALUE MAP ABLATION: equal_weighting")
            self._map[self._map > 0] = 1
            for i in range(len(new_maps_list)):
                new_maps_list[i][new_maps_list[i] > 0] = 1

        cumulative_confidence = self._map.copy()
        cumulative_value = self._value_map.copy()

        for new_map, values in zip(new_maps_list, new_values_list):
            new_map_mask = np.logical_and(new_map < self._decision_threshold, new_map < cumulative_confidence)
            new_map[new_map_mask] = 0
            confidence_denominator = cumulative_confidence + new_map + 1e-6
            weight_existing = cumulative_confidence / confidence_denominator
            weight_new = new_map / confidence_denominator
            weight_existing_channeled = np.repeat(np.expand_dims(weight_existing, axis=2), self._value_channels, axis=2)
            weight_new_channeled = np.repeat(np.expand_dims(weight_new, axis=2), self._value_channels, axis=2)
            cumulative_value = cumulative_value * weight_existing_channeled + values * weight_new_channeled
            cumulative_confidence = cumulative_confidence + new_map

        cumulative_value = np.nan_to_num(cumulative_value)
        cumulative_confidence = np.nan_to_num(cumulative_confidence)
        self._value_map = cumulative_value
        self._map = cumulative_confidence


def remap(value: float, from_low: float, from_high: float, to_low: float, to_high: float) -> float:
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low


def replay_from_dir() -> None:
    with open(KWARGS_JSON, "r") as f:
        kwargs = json.load(f)
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    v = ValueMap(**kwargs)
    sorted_keys = sorted(list(data.keys()))
    for img_path in sorted_keys:
        tf_camera_to_episodic = np.array(data[img_path]["tf_camera_to_episodic"])
        values = np.array(data[img_path]["values"])
        depth = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        v.update_map([v._localize_new_data(depth, tf_camera_to_episodic, float(data[img_path]["min_depth"]), float(data[img_path]["max_depth"]), float(data[img_path]["fov"]))],
                     [values],
                     float(data[img_path]["min_depth"]),
                     float(data[img_path]["max_depth"]),
                     float(data[img_path]["fov"]))
        img = v.visualize()
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break


if __name__ == "__main__":
    if PLAYING:
        replay_from_dir()
        quit()
    v = ValueMap(value_channels=1)
    depth = cv2.imread("depth.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    img = v._process_local_data(
        depth=depth,
        fov=np.deg2rad(79),
        min_depth=0.5,
        max_depth=5.0,
    )
    cv2.imshow("img", (img * 255).astype(np.uint8))
    cv2.waitKey(0)
    num_points = 20
    x = [0, 10, 10, 0]
    y = [0, 0, 10, 10]
    angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    points = np.stack((x, y), axis=1)
    for pt, angle in zip(points, angles):
        tf = np.eye(4)
        tf[:2, 3] = pt
        tf[:2, :2] = get_rotation_matrix(angle)
        v.update_map([np.array([1])],
                     [depth],
                     tf,
                     min_depth=0.5,
                     max_depth=5.0,
                     fov=np.deg2rad(79))
        img = v.visualize()
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
