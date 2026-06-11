import os
import sys
import time
import csv
import json
import logging
import hashlib
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import types

# Import action classes to register them - must happen before habitat is imported
# Add multi-robot-setting to path and import the nav module
multi_robot_path = os.path.join(os.path.dirname(__file__), 'multi-robot-setting')
sys.path.insert(0, multi_robot_path)
# Import the module to trigger the @registry.register_task_action decorators
import importlib
importlib.import_module('habitat.tasks.nav.nav')

script_path = Path(__file__).resolve()
repo_root = script_path.parent
vendored_vlfm_root = repo_root / "third_party" / "vlfm"
path_candidates = [
    repo_root,
    vendored_vlfm_root,
]
for path_entry in path_candidates:
    if path_entry.is_dir():
        entry_str = str(path_entry)
        if entry_str not in sys.path:
            sys.path.insert(0, entry_str)

try:
    import depth_camera_filtering  # type: ignore
except ImportError:  # pragma: no cover - runtime fallback
    fallback_module = types.ModuleType("depth_camera_filtering")

    def _fallback_filter_depth(depth, blur_type=None):
        arr = np.asarray(depth)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr.astype(np.float32, copy=False)

    fallback_module.filter_depth = _fallback_filter_depth  # type: ignore[attr-defined]
    sys.modules["depth_camera_filtering"] = fallback_module
    warnings.warn(
        "depth_camera_filtering package not found; using identity fallback. "
        "Install the upstream module for proper depth denoising.",
        RuntimeWarning,
    )

# Import frontier_exploration submodules individually so that a failure in
# base_explorer (which may depend on full Habitat) does not clobber the
# working frontier_detection / fog_of_war modules with no-op stubs.
try:
    from frontier_exploration.frontier_detection import detect_frontier_waypoints as _fe_check  # type: ignore  # noqa: F401
    from frontier_exploration.utils.fog_of_war import reveal_fog_of_war as _fow_check  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - runtime fallback
    frontier_pkg = types.ModuleType("frontier_exploration")
    frontier_detection_mod = types.ModuleType("frontier_exploration.frontier_detection")
    utils_pkg = types.ModuleType("frontier_exploration.utils")
    fog_of_war_mod = types.ModuleType("frontier_exploration.utils.fog_of_war")

    def _fallback_detect_frontier_waypoints(*args, **kwargs):
        return np.zeros((0, 2), dtype=np.int32)

    def _fallback_reveal_fog_of_war(top_down_map, *args, **kwargs):
        return np.zeros_like(top_down_map, dtype=np.uint8)

    frontier_detection_mod.detect_frontier_waypoints = _fallback_detect_frontier_waypoints  # type: ignore[attr-defined]
    fog_of_war_mod.reveal_fog_of_war = _fallback_reveal_fog_of_war  # type: ignore[attr-defined]
    utils_pkg.fog_of_war = fog_of_war_mod  # type: ignore[attr-defined]
    frontier_pkg.frontier_detection = frontier_detection_mod  # type: ignore[attr-defined]
    frontier_pkg.utils = utils_pkg  # type: ignore[attr-defined]
    sys.modules.setdefault("frontier_exploration", frontier_pkg)
    sys.modules.setdefault("frontier_exploration.frontier_detection", frontier_detection_mod)
    sys.modules.setdefault("frontier_exploration.utils", utils_pkg)
    sys.modules.setdefault("frontier_exploration.utils.fog_of_war", fog_of_war_mod)
    warnings.warn(
        "frontier_exploration package not found; using minimal stubs. "
        "Install the upstream module for frontier-based exploration features.",
        RuntimeWarning,
    )

try:
    import frontier_exploration.base_explorer  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - base_explorer may need full Habitat
    base_explorer_mod = types.ModuleType("frontier_exploration.base_explorer")

    class _FallbackBaseExplorer:
        cls_uuid: str = "frontier_explorer"

    base_explorer_mod.BaseExplorer = _FallbackBaseExplorer  # type: ignore[attr-defined]
    sys.modules.setdefault("frontier_exploration.base_explorer", base_explorer_mod)

try:
    import open3d  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover - runtime fallback
    open3d_pkg = types.ModuleType("open3d")
    geometry_mod = types.ModuleType("open3d.geometry")
    utility_mod = types.ModuleType("open3d.utility")

    class _StubPointCloud:
        def __init__(self):
            self.points = None

        def cluster_dbscan(self, eps, min_points):
            if self.points is None:
                return np.array([], dtype=int)
            return np.zeros(len(self.points), dtype=int)

    def _vector3d_vector(arr):
        return np.asarray(arr, dtype=float)

    geometry_mod.PointCloud = _StubPointCloud  # type: ignore[attr-defined]
    utility_mod.Vector3dVector = _vector3d_vector  # type: ignore[attr-defined]
    open3d_pkg.geometry = geometry_mod  # type: ignore[attr-defined]
    open3d_pkg.utility = utility_mod  # type: ignore[attr-defined]
    sys.modules["open3d"] = open3d_pkg
    sys.modules["open3d.geometry"] = geometry_mod
    sys.modules["open3d.utility"] = utility_mod
    warnings.warn(
        "open3d package not found; using stub implementation. "
        "Install Open3D for point-cloud clustering support.",
        RuntimeWarning,
    )

import habitat
import habitat_sim
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)


from envs.habitat.multi_agent_env_vlm import Multi_Agent_Env
from arguments import get_args

from vlfm.chrono_env.communication_manager import FusedFeatureExchangeManager
from vlfm.policy.chrono_policy_DSInC import (
    ChronoITMPolicyV2,
    TorchActionIDs,
)
from constants import hm3d_category, category_to_id, object_category

from utils.debug_data import (
    log_spawn_states,
)
from utils.spawn_additional_agent import (
    randomize_agent_after_reset,
    get_scene_and_episode,
)


class NoOpFeatureExchangeManager:
    def register_agent(self, agent_id: int):
        pass

    def push(self, sender_id: int, feat):
        pass

    def retrieve(self, receiver_id: int):
        return []

    def reset(self):
        pass


@habitat.registry.register_action_space_configuration
class PreciseTurn(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()
        config[HabitatSimActions.TURN_LEFT_S] = habitat_sim.ActionSpec(
            "turn_left", habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE_S)
        )
        config[HabitatSimActions.TURN_RIGHT_S] = habitat_sim.ActionSpec(
            "turn_right", habitat_sim.ActuationSpec(amount=self.config.TURN_ANGLE_S)
        )
        # In the multi-agent path Habitat calls sim.step([...]) directly, so a
        # WAIT action must be a valid simulator motion primitive. Habitat-Sim's
        # "stop" action is handled at the task layer and has no move_fn here.
        config[HabitatSimActions.WAIT] = habitat_sim.ActionSpec(
            "move_forward", habitat_sim.ActuationSpec(amount=0.0)
        )
        return config


GOAL_NAME_PRIORITY = [
    hm3d_category,
    category_to_id,
    object_category,
]


class HabitatDSInCPolicy(ChronoITMPolicyV2):
    pass


def goal_index_to_name(goal_idx: int) -> str:
    for names in GOAL_NAME_PRIORITY:
        if 0 <= goal_idx < len(names):
            return names[goal_idx]
    return str(goal_idx)


def resolve_goal_name(env: Any, agent_idx: int, goal_idx: int) -> str:
    base_env = getattr(env, "_env", env)
    candidate = None
    if hasattr(base_env, "current_episodes"):
        episodes = getattr(base_env, "current_episodes")
        if isinstance(episodes, (list, tuple)) and len(episodes) > 0:
            candidate = episodes[min(agent_idx, len(episodes) - 1)]
    if candidate is None and hasattr(base_env, "current_episode"):
        candidate = getattr(base_env, "current_episode")
    if candidate is not None:
        category = getattr(candidate, "object_category", None)
        if isinstance(category, (list, tuple)) and len(category) > 0:
            category = category[min(agent_idx, len(category) - 1)]
        if isinstance(category, str):
            return category
    return goal_index_to_name(goal_idx)


def get_episode_for_agent(env: Any, agent_idx: int) -> Any:
    base_env = getattr(env, "_env", env)
    candidate = None
    if hasattr(base_env, "current_episodes"):
        episodes = getattr(base_env, "current_episodes")
        if isinstance(episodes, (list, tuple)) and len(episodes) > 0:
            candidate = episodes[min(agent_idx, len(episodes) - 1)]
    if candidate is None and hasattr(base_env, "current_episode"):
        candidate = getattr(base_env, "current_episode")
    return candidate


def extract_goal_view_points(env: Any, agent_idx: int) -> List[np.ndarray]:
    episode = get_episode_for_agent(env, agent_idx)
    if episode is None:
        return []
    points: List[np.ndarray] = []
    goals = getattr(episode, "goals", None)
    if isinstance(goals, (list, tuple)):
        for goal in goals:
            view_points = getattr(goal, "view_points", None)
            if not isinstance(view_points, (list, tuple)):
                continue
            for view_point in view_points:
                agent_state = getattr(view_point, "agent_state", None)
                position = getattr(agent_state, "position", None)
                if position is not None:
                    points.append(np.asarray(position, dtype=np.float32))
    return points


def extract_goal_centers(env: Any, agent_idx: int) -> List[np.ndarray]:
    episode = get_episode_for_agent(env, agent_idx)
    if episode is None:
        return []
    centers: List[np.ndarray] = []
    goals = getattr(episode, "goals", None)
    if isinstance(goals, (list, tuple)):
        for goal in goals:
            position = getattr(goal, "position", None)
            if position is None:
                continue
            center = np.asarray(position, dtype=np.float32).reshape(-1)
            if center.size < 3:
                continue
            center = center[:3]
            if np.all(np.isfinite(center)):
                centers.append(center)
    return centers


def ensure_agent_obs_list(obs: Any) -> List[Dict[str, Any]]:
    if isinstance(obs, dict):
        agent_keys = [k for k in obs.keys() if isinstance(k, str) and k.startswith("agent_")]
        if agent_keys:
            return [obs[k] for k in sorted(agent_keys)]
        return [obs]
    if isinstance(obs, (list, tuple)):
        return list(obs)
    return [obs]


def prepare_policy_obs(
    raw_obs: Dict[str, Any],
    args,
    env: Any,
    agent_idx: int,
) -> Dict[str, Any]:
    rgb = raw_obs.get("rgb")
    if rgb is None:
        raise KeyError("RGB sensor missing from observation.")
    rgb_np = np.asarray(rgb)
    if rgb_np.dtype != np.uint8:
        rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)

    depth = raw_obs.get("depth")
    if depth is None:
        raise KeyError("Depth sensor missing from observation.")
    depth_np = np.asarray(depth, dtype=np.float32)
    if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
        depth_np = depth_np[..., 0]
    # Habitat's depth sensor already normalizes to [0, 1] when
    # NORMALIZE_DEPTH=True (the default).  Only clean up NaN/inf values;
    # do NOT re-normalize or the signal collapses to near-zero.
    depth_norm = np.nan_to_num(depth_np, nan=1.0, posinf=1.0)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)

    gps = raw_obs.get("gps")
    if gps is None:
        raise KeyError("GPS sensor missing from observation.")
    gps_np = np.asarray(gps, dtype=np.float32).reshape(-1)
    # Habitat's 2D EpisodicGPSSensor returns (forward, right) in the episode
    # start frame.  The VLFM policy (written for Chrono) expects (forward, LEFT)
    # so that atan2(y, x) is consistent with the compass convention (positive =
    # turned left / CCW).  We negate the second component to convert.
    # Ref: DSInC agents do the same: curr_sim_pose = [gps[0], -gps[1], compass[0]]
    if gps_np.size == 3:
        # 3D GPS: [X_start(right), Y_start(up), Z_start(backward)]
        # → (forward, left) = (-Z_start, -X_start)
        gps_xy = np.array([-gps_np[2], -gps_np[0]], dtype=np.float32)
    else:
        # 2D GPS: [forward, right] → negate right to get left
        if gps_np.size < 2:
            gps_np = np.pad(gps_np, (0, 2 - gps_np.size), mode="constant")
        gps_xy = np.array([gps_np[0], -gps_np[1]], dtype=np.float32)

    compass = raw_obs.get("compass")
    compass_val = float(np.asarray(compass).reshape(-1)[0]) if compass is not None else 0.0

    objectgoal = raw_obs.get("objectgoal")
    if objectgoal is None:
        raise KeyError("objectgoal sensor missing from observation.")
    goal_idx = int(np.asarray(objectgoal).reshape(-1)[0])
    goal_name = resolve_goal_name(env, agent_idx, goal_idx)

    obs_dict: Dict[str, Any] = {
        "rgb": torch.from_numpy(np.ascontiguousarray(rgb_np)),
        "depth": torch.from_numpy(np.ascontiguousarray(depth_norm.astype(np.float32))),
        "gps": torch.tensor(gps_xy, dtype=torch.float32),
        "compass": torch.tensor(compass_val, dtype=torch.float32),
        "objectgoal": goal_name,
    }
    return obs_dict


def get_eval_metrics(env: Any) -> Dict[str, Any]:
    get_metrics_fn = getattr(env, "get_metrics", None)
    if callable(get_metrics_fn):
        try:
            metrics = get_metrics_fn()
            if isinstance(metrics, dict):
                return metrics
        except Exception:
            pass
    base_env = getattr(env, "_env", None)
    get_metrics_fn = getattr(base_env, "get_metrics", None)
    if callable(get_metrics_fn):
        try:
            metrics = get_metrics_fn()
            if isinstance(metrics, dict):
                return metrics
        except Exception:
            pass
    return {}


def collect_metric_values(metrics: Any, metric_name: str) -> List[float]:
    values: List[float] = []
    target = metric_name.lower()
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if isinstance(key, str) and key.lower() == target and isinstance(value, (int, float, bool, np.number)):
                values.append(float(value))
            values.extend(collect_metric_values(value, metric_name))
    elif isinstance(metrics, (list, tuple)):
        for item in metrics:
            values.extend(collect_metric_values(item, metric_name))
    return values


def is_no_frontier_signal(policy: Any, action_value: int) -> bool:
    if action_value != int(TorchActionIDs.STOP.item()):
        return False
    policy_info = getattr(policy, "_policy_info", {})
    if not isinstance(policy_info, dict):
        return False
    mode = str(policy_info.get("mode", "")).lower()
    action = str(policy_info.get("action", "")).upper()
    return mode == "explore" and action == "STOP"


def get_sim(env: Any) -> Any:
    sim = getattr(env, "sim", None)
    if sim is None and hasattr(env, "_env"):
        sim = getattr(env._env, "sim", None)
    return sim


def get_agent_position(sim: Any, agent_idx: int = 0) -> Any:
    if sim is None:
        return None
    try:
        if hasattr(sim, "agents") and len(sim.agents) > agent_idx:
            state = sim.agents[agent_idx].get_state()
            return np.asarray(state.position, dtype=np.float32)
    except Exception:
        return None
    return None


def compute_geodesic_distance(
    sim: Any,
    position_a: Any,
    goal_positions: List[np.ndarray],
    episode: Any,
) -> Optional[float]:
    if sim is None or position_a is None or len(goal_positions) == 0:
        return None
    try:
        source = np.asarray(position_a, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if source.size < 3:
        return None
    source = source[:3]
    sanitized_goals: List[np.ndarray] = []
    for goal_position in goal_positions:
        try:
            goal = np.asarray(goal_position, dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if goal.size < 3:
            continue
        goal = goal[:3]
        if np.all(np.isfinite(goal)):
            sanitized_goals.append(goal)
    if len(sanitized_goals) == 0:
        return None
    try:
        dist = sim.geodesic_distance(source, sanitized_goals, episode)
    except Exception:
        try:
            dist = sim.geodesic_distance(source, sanitized_goals)
        except Exception:
            return None
    dist_f = float(dist)
    if not np.isfinite(dist_f):
        return None
    return dist_f


def compute_agent_goal_distance(
    sim: Any,
    agent_idx: int,
    goal_view_points: List[np.ndarray],
    episode: Any,
) -> Optional[float]:
    if sim is None or len(goal_view_points) == 0:
        return None
    try:
        curr_pos = sim.get_agent_state(agent_id=agent_idx).position
    except Exception:
        try:
            curr_pos = sim.agents[agent_idx].get_state().position
        except Exception:
            return None
    return compute_geodesic_distance(sim, curr_pos, goal_view_points, episode)


def precompute_agent_start_geodesics(
    sim: Any,
    start_positions: List[Optional[np.ndarray]],
    goal_centers: List[np.ndarray],
    episode: Any,
) -> List[List[Optional[float]]]:
    geodesics: List[List[Optional[float]]] = []
    for start_position in start_positions:
        row: List[Optional[float]] = []
        for goal_center in goal_centers:
            row.append(
                compute_geodesic_distance(
                    sim,
                    start_position,
                    [goal_center],
                    episode,
                )
            )
        geodesics.append(row)
    return geodesics


def compute_xy_distance_to_goal_center(
    agent_position: Any,
    goal_center: Any,
) -> Optional[float]:
    if agent_position is None or goal_center is None:
        return None
    try:
        agent = np.asarray(agent_position, dtype=np.float32).reshape(-1)
        goal = np.asarray(goal_center, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if agent.size < 2 or goal.size < 2:
        return None
    # The requested metric projects to XY and ignores Z.
    delta_xy = agent[:2] - goal[:2]
    dist_multi_robot = float(np.linalg.norm(delta_xy))
    dist = dist_multi_robot/3
    if not np.isfinite(dist):
        return None
    return dist


def match_nearest_goal_center_xy(
    agent_position: Any,
    goal_centers: List[np.ndarray],
) -> Optional[tuple[int, np.ndarray, float]]:
    best_idx: Optional[int] = None
    best_center: Optional[np.ndarray] = None
    best_distance: Optional[float] = None
    for idx, goal_center in enumerate(goal_centers):
        dist = compute_xy_distance_to_goal_center(agent_position, goal_center)
        if dist is None:
            continue
        if best_distance is None or dist < best_distance:
            best_idx = idx
            best_center = np.asarray(goal_center, dtype=np.float32)
            best_distance = dist
    if best_idx is None or best_center is None or best_distance is None:
        return None
    return best_idx, best_center, best_distance


def get_precomputed_goal_geodesic(
    agent_goal_geodesics: List[Optional[float]],
    goal_idx: Optional[int],
) -> Optional[float]:
    if goal_idx is None or goal_idx < 0 or goal_idx >= len(agent_goal_geodesics):
        return None
    value = agent_goal_geodesics[goal_idx]
    if value is None:
        return None
    value_f = float(value)
    if not np.isfinite(value_f):
        return None
    return value_f


def compute_event_spl(
    success_value: int,
    min_path_length: Optional[float],
    path_length: Optional[float],
) -> float:
    if success_value <= 0:
        return 0.0
    if min_path_length is None or path_length is None or path_length <= 0.0:
        return 0.0
    min_path_length = float(min_path_length)
    path_length = float(path_length)
    if min_path_length <= 0.0:
        return 0.0
    return min_path_length / max(path_length, min_path_length)


def format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


def format_optional_int(value: Optional[int]) -> str:
    if value is None:
        return "NA"
    return str(int(value))


def update_agent_path_lengths(
    sim: Any,
    prev_positions: List[Optional[np.ndarray]],
    path_lengths: List[float],
) -> None:
    for agent_idx in range(len(path_lengths)):
        curr_pos = get_agent_position(sim, agent_idx=agent_idx)
        if prev_positions[agent_idx] is not None and curr_pos is not None:
            path_lengths[agent_idx] += float(
                np.linalg.norm(curr_pos - prev_positions[agent_idx])
            )
        prev_positions[agent_idx] = curr_pos


def resolve_pointnav_weights_path(repo_root: Path) -> Path:
    """Resolve the PointNav checkpoint from the vendored VLFM runtime."""
    candidates = [
        repo_root / "third_party" / "vlfm" / "data" / "pointnav_weights.pth",
    ]
    for weights_path in candidates:
        if weights_path.is_file():
            return weights_path
    raise FileNotFoundError(
        "Expected PointNav weights at one of:\n"
        + "\n".join(f"  - {path}" for path in candidates)
        + "\n"
        "File not found. Place pointnav_weights.pth in third_party/vlfm/data/."
    )


def select_episode_shard(
    episodes: List[Any], num_shards: int, shard_index: int
) -> tuple[List[Any], int, int]:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards - 1}], got {shard_index}."
        )
    total = len(episodes)
    if total == 0:
        return [], 0, 0
    if num_shards > total:
        raise ValueError(
            f"Cannot split {total} episodes into {num_shards} non-empty shards."
        )

    base = total // num_shards
    remainder = total % num_shards
    start = shard_index * base + min(shard_index, remainder)
    shard_len = base + (1 if shard_index < remainder else 0)
    end = start + shard_len
    return list(episodes[start:end]), start, end


def resolve_manifest_path(manifest_arg: str) -> Path:
    manifest_path = Path(manifest_arg)
    if manifest_path.is_absolute():
        return manifest_path
    candidates = [
        Path.cwd() / manifest_path,
        script_path.parent / manifest_path,
        repo_root / manifest_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_episode_manifest(manifest_path: Path) -> List[tuple[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Episode manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(
            f"Episode manifest must be a JSON list of objects with scene_id/episode_id: {manifest_path}"
        )

    manifest_entries: List[tuple[str, str]] = []
    seen = set()
    for idx, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry #{idx} is not an object: {entry!r}")
        scene_id = entry.get("scene_id")
        episode_id = entry.get("episode_id")
        if not isinstance(scene_id, str) or not isinstance(episode_id, str):
            raise ValueError(
                f"Manifest entry #{idx} must contain string scene_id and episode_id fields."
            )
        key = (scene_id, episode_id)
        if key in seen:
            raise ValueError(
                f"Duplicate manifest entry for scene_id={scene_id!r}, episode_id={episode_id!r}."
            )
        seen.add(key)
        manifest_entries.append(key)
    return manifest_entries


def filter_episodes_by_manifest(
    episodes: List[Any], manifest_entries: List[tuple[str, str]]
) -> List[Any]:
    episode_lookup: Dict[tuple[str, str], Any] = {}
    for episode in episodes:
        key = (
            str(getattr(episode, "scene_id", "")),
            str(getattr(episode, "episode_id", "")),
        )
        if key in episode_lookup:
            raise ValueError(
                "Dataset contains duplicate episode keys for "
                f"scene_id={key[0]!r}, episode_id={key[1]!r}."
            )
        episode_lookup[key] = episode

    missing = [key for key in manifest_entries if key not in episode_lookup]
    if missing:
        details = "\n".join(
            f"  - scene_id={scene_id}, episode_id={episode_id}"
            for scene_id, episode_id in missing[:20]
        )
        if len(missing) > 20:
            details += f"\n  ... and {len(missing) - 20} more"
        raise ValueError(
            f"Episode manifest references {len(missing)} entries not found in dataset:\n{details}"
        )

    return [episode_lookup[key] for key in manifest_entries]


def make_stable_episode_seed(global_seed: int, scene_id: str, episode_id: str) -> int:
    key = f"{scene_id}::{episode_id}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=4).digest()
    episode_hash = int.from_bytes(digest, byteorder="little", signed=False)
    return (int(global_seed) ^ episode_hash) & 0xFFFFFFFF


def main():
    args = get_args()
    habitat_success_distance_xy_m = float(args.success_dist)
    if not np.isfinite(habitat_success_distance_xy_m) or habitat_success_distance_xy_m <= 0.0:
        raise ValueError(
            "--success_dist must be a positive finite Habitat XY success threshold, "
            f"got {args.success_dist!r}."
        )

    # ------------------------------------------------------------------
    # Config / Env setup 
    # ------------------------------------------------------------------
    HabitatSimActions.extend_action_space("TURN_LEFT_S")
    HabitatSimActions.extend_action_space("TURN_RIGHT_S")
    HabitatSimActions.extend_action_space("WAIT")

    config_env = habitat.get_config(
        config_paths=["envs/habitat/configs/" + args.task_config]
    )
    config_env.defrost()

    agent_sensors = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
    simulator_agent_names = list(
        getattr(config_env.SIMULATOR, "AGENTS", ["AGENT_0"])
    )
    for agent_name in simulator_agent_names:
        getattr(config_env.SIMULATOR, agent_name).SENSORS = agent_sensors
    config_env.SIMULATOR.SEMANTIC_SENSOR.WIDTH = args.env_frame_width
    config_env.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = args.env_frame_height
    config_env.SIMULATOR.SEMANTIC_SENSOR.HFOV = args.hfov
    config_env.SIMULATOR.SEMANTIC_SENSOR.POSITION = [0, args.camera_height, 0]

    # Add precise turn actions & bind action-space config
    config_env.TASK.POSSIBLE_ACTIONS = (
        config_env.TASK.POSSIBLE_ACTIONS + ["TURN_LEFT_S", "TURN_RIGHT_S", "WAIT"]
    )
    config_env.TASK.ACTIONS.TURN_LEFT_S = habitat.config.Config()
    config_env.TASK.ACTIONS.TURN_LEFT_S.TYPE = "TurnLeftAction_S"
    config_env.TASK.ACTIONS.TURN_RIGHT_S = habitat.config.Config()
    config_env.TASK.ACTIONS.TURN_RIGHT_S.TYPE = "TurnRightAction_S"
    config_env.TASK.ACTIONS.WAIT = habitat.config.Config()
    config_env.TASK.ACTIONS.WAIT.TYPE = "WaitAction"
    config_env.SIMULATOR.ACTION_SPACE_CONFIG = "PreciseTurn"

    config_env.freeze()

    # Instantiate multi-agent env
    env = Multi_Agent_Env(config_env=config_env)

    dataset_num_episodes = len(env.episodes)
    manifest_path: Optional[Path] = None
    if args.episode_manifest:
        manifest_path = resolve_manifest_path(args.episode_manifest)
        manifest_entries = load_episode_manifest(manifest_path)
        env.episodes = filter_episodes_by_manifest(env.episodes, manifest_entries)
        env.number_of_episodes = len(env.episodes)
    selected_dataset_num_episodes = len(env.episodes)
    if args.num_shards > 1:
        shard_episodes, shard_start, shard_end = select_episode_shard(
            env.episodes, args.num_shards, args.shard_index
        )
        env.episodes = shard_episodes
        env.number_of_episodes = len(shard_episodes)
    else:
        shard_start = 0
        shard_end = selected_dataset_num_episodes

    total_selected_episodes = len(env.episodes)
    resume_episode_index = int(args.resume_episode_index)
    if resume_episode_index < 1 or resume_episode_index > total_selected_episodes + 1:
        raise ValueError(
            "resume_episode_index must be in "
            f"[1, {total_selected_episodes + 1}], got {resume_episode_index}."
        )
    resume_episode_offset = resume_episode_index - 1
    if resume_episode_offset > 0:
        env.episodes = list(env.episodes[resume_episode_offset:])
        env.number_of_episodes = len(env.episodes)

    num_episodes = len(env.episodes)
    assert num_episodes > 0, (
        "No episodes found in the dataset/split after applying resume_episode_index."
    )
    num_agents = config_env.SIMULATOR.NUM_AGENTS

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_dir = f"{args.dump_location}/logs/{args.exp_name}/"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "output.log")
    episode_csv_path = os.path.join(log_dir, "episode_metrics.csv")
    csv_header_written = os.path.exists(episode_csv_path) and os.path.getsize(episode_csv_path) > 0
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    if args.num_shards > 1:
        start_msg = (
            f"[multi-eval] shard={args.shard_index + 1}/{args.num_shards} "
            f"episodes={num_episodes} selected_range=[{shard_start}, {shard_end}) "
            f"resume_episode_index={resume_episode_index} "
            f"dataset_total={dataset_num_episodes} agents={num_agents} "
            f"comms={not args.no_comm} "
            f"selected_total={selected_dataset_num_episodes} "
            f"shard_total={total_selected_episodes} "
            f"manifest={manifest_path if manifest_path else 'none'} "
            f"comm_fuse_window={args.comm_fuse_window} "
            f"comm_novelty_threshold={args.comm_novelty_threshold:.3f} "
            f"log_file={log_file_path}"
        )
    else:
        start_msg = (
            f"[multi-eval] episodes={num_episodes}, agents={num_agents}, "
            f"resume_episode_index={resume_episode_index}, "
            f"comms={not args.no_comm}, "
            f"dataset_total={dataset_num_episodes}, "
            f"selected_total={total_selected_episodes}, "
            f"manifest={manifest_path if manifest_path else 'none'}, "
            f"comm_fuse_window={args.comm_fuse_window}, "
            f"comm_novelty_threshold={args.comm_novelty_threshold:.3f}, "
            f"log_file={log_file_path}"
        )
    print(start_msg)
    logging.info(start_msg)
    if manifest_path is not None:
        filter_msg = (
            f"[multi-eval] filtered episodes via manifest={manifest_path} "
            f"from dataset_total={dataset_num_episodes} down to selected={selected_dataset_num_episodes}"
        )
        print(filter_msg)
        logging.info(filter_msg)

    # ------------------------------------------------------------------
    # Action helpers: map names -> enum ids (for your policy convenience)
    # ------------------------------------------------------------------
    A = HabitatSimActions
    # Typical discrete set; add/remove as your policy needs
    ACTION_SET = {
        "STOP": int(A.STOP),
        "FORWARD": int(A.MOVE_FORWARD),
        "LEFT": int(A.TURN_LEFT),
        "RIGHT": int(A.TURN_RIGHT),
        "LEFT_S": int(A.TURN_LEFT_S),
        "RIGHT_S": int(A.TURN_RIGHT_S),
        "WAIT": int(A.WAIT),
    }
    stop_action = ACTION_SET["STOP"]
    torch_to_habitat = {
        int(TorchActionIDs.STOP.item()): stop_action,
        int(TorchActionIDs.MOVE_FORWARD.item()): ACTION_SET["FORWARD"],
        int(TorchActionIDs.TURN_LEFT.item()): ACTION_SET["LEFT"],
        int(TorchActionIDs.TURN_RIGHT.item()): ACTION_SET["RIGHT"],
    }

    # Keep communication enabled by default so existing commands still behave the same.
    comms_enabled = not args.no_comm
    comms_manager = (
        FusedFeatureExchangeManager()
        if comms_enabled
        else NoOpFeatureExchangeManager()
    )
    policy_weights_path = resolve_pointnav_weights_path(repo_root)
    policies: List[HabitatDSInCPolicy] = []
    for agent_idx in range(num_agents):
        policies.append(
            HabitatDSInCPolicy(
                camera_height=args.camera_height,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                camera_fov=args.hfov,
                image_width=args.env_frame_width,
                text_prompt="Seems like there is a target_object ahead.",
                use_max_confidence=False,
                pointnav_policy_path=str(policy_weights_path),
                depth_image_shape=(args.env_frame_height, args.env_frame_width),
                pointnav_stop_radius=0.5,
                pointnav_stop_enable_radius=1.0,
                pointnav_stop_consecutive_required=3,
                pointnav_stop_close_tracking_radius=2.0,
                pointnav_stop_overshoot_radius=3.0,
                pointnav_stagnation_radius=3.0,
                pointnav_stagnation_steps=10,
                object_map_erosion_size=5,
                visualize=False,
                obstacle_map_area_threshold=1.5,
                min_obstacle_height=0.3,
                max_obstacle_height=0.5,
                hole_area_thresh=100000,
                use_vqa=False,
                vqa_prompt="Is this ",
                coco_threshold=0.65,
                non_coco_threshold=0.3,
                agent_radius=0.15,
                robot_id=agent_idx + 1,
                comms_manager=comms_manager,
                comm_feature_fifo_size=args.max_episode_length,
                comm_fuse_window=args.comm_fuse_window,
                comm_novelty_threshold=args.comm_novelty_threshold,
                comm_novelty_beta=args.comm_novelty_beta,
            )
        )

    # ------------------------------------------------------------------
    # Minimal eval loop (plug your policy where indicated)
    # ------------------------------------------------------------------
    t0 = time.time()
    steps_total = 0
    episodes_success = 0
    episodes_with_spl = 0
    spl_total = 0.0
    habitat_successes = 0
    habitat_spl_total = 0.0
    mcoconav_successes = 0
    mcoconav_spl_total = 0.0

    for ep in range(num_episodes):
        display_episode_index = resume_episode_offset + ep + 1
        os.environ["VLFM_EPISODE_NUM"] = str(display_episode_index)
        obs = env.reset()
        scene_id, episode_id = get_scene_and_episode(env)
        comms_manager.reset()
        obs_list = ensure_agent_obs_list(obs)
        reset_metrics = get_eval_metrics(env)
        min_path_values = collect_metric_values(reset_metrics, "distance_to_goal")
        episode_min_path_length = max(min_path_values) if min_path_values else None

        # Derive a deterministic per-episode seed from the scene+episode id.
        # Python's built-in hash() is process-randomized, which makes split and
        # non-split runs disagree on agent spawn randomization.
        global_seed = int(config_env.SEED)
        per_ep_seed = make_stable_episode_seed(global_seed, str(scene_id), str(episode_id))

        if num_agents > 1:
            for agent_idx in range(1, num_agents):
                spawn_seed = make_stable_episode_seed(
                    global_seed,
                    str(scene_id),
                    f"{episode_id}::spawn:{agent_idx}",
                )
                randomize_agent_after_reset(
                    env,
                    agent_index=agent_idx,
                    anchor_agent_index=0,
                    min_geodesic=7.0,
                    same_floor_eps=0.35,
                    max_tries=15,
                    episode_seed=spawn_seed,
                )

        same_start_mode = False
        if num_agents == 2:
            spawn_info = getattr(env, "_last_spawn_randomization", {})
            same_start_mode = bool(spawn_info.get("same_location", False))
            sim_after_spawn = get_sim(env)
            pos0 = get_agent_position(sim_after_spawn, agent_idx=0)
            pos1 = get_agent_position(sim_after_spawn, agent_idx=1)
            if pos0 is not None and pos1 is not None:
                same_start_mode = same_start_mode or (
                    float(np.linalg.norm(pos0 - pos1)) <= 0.25
                )

        hidden_states: List[Any] = [None] * num_agents
        prev_actions: List[Any] = [None] * num_agents
        masks: List[torch.Tensor] = [torch.zeros((1, 1), dtype=torch.float32) for _ in range(num_agents)]
        for policy in policies:
            setattr(policy, "_same_start_mode", same_start_mode)
            setattr(policy, "_same_start_warmup_steps", 15)
            setattr(policy, "_same_start_first_split_pending", same_start_mode)
            setattr(policy, "_same_start_partner_frontier", None)
            setattr(policy, "_same_start_last_selected_frontier", None)
            setattr(policy, "_same_start_split_applied", False)

        # Example: if you want per-episode policy state, init it here
        # my_policy_state = [None for _ in range(num_agents)]
        log_spawn_states(env, logging)
        episode_steps = 0
        episode_success = False
        termination_reason = "unknown"
        success_distance_m = 0.5
        episode_object_found = False
        agent_object_found = [False] * num_agents
        agent_no_frontiers = [False] * num_agents
        agent_policy_stop_called = [False] * num_agents
        agent_geodesic_success = [False] * num_agents
        accepted_navigate_stop_flags = [False] * num_agents
        winner_agent_idx: Optional[int] = None
        sim = get_sim(env)
        agent_episodes = [get_episode_for_agent(env, i) for i in range(num_agents)]
        agent_goal_view_points = [extract_goal_view_points(env, i) for i in range(num_agents)]
        agent_goal_centers = [extract_goal_centers(env, i) for i in range(num_agents)]
        agent_start_distances = [
            compute_agent_goal_distance(
                sim,
                i,
                agent_goal_view_points[i],
                agent_episodes[i],
            )
            for i in range(num_agents)
        ]
        if episode_min_path_length is None:
            valid_start_distances = [d for d in agent_start_distances if d is not None]
            if valid_start_distances:
                episode_min_path_length = max(valid_start_distances)
        agent_path_lengths = [0.0] * num_agents
        agent_prev_positions = [get_agent_position(sim, agent_idx=i) for i in range(num_agents)]
        agent_start_goal_geodesics: List[List[Optional[float]]] = []
        for agent_idx in range(num_agents):
            geodesic_rows = precompute_agent_start_geodesics(
                sim,
                [agent_prev_positions[agent_idx]],
                agent_goal_centers[agent_idx],
                agent_episodes[agent_idx],
            )
            if geodesic_rows:
                agent_start_goal_geodesics.append(geodesic_rows[0])
            else:
                agent_start_goal_geodesics.append(
                    [None] * len(agent_goal_centers[agent_idx])
                )
        habitat_success = 0
        habitat_spl = 0.0
        habitat_stop_distance_xy: Optional[float] = None
        habitat_success_agent: Optional[int] = None
        habitat_success_step: Optional[int] = None
        habitat_path_length: Optional[float] = None
        habitat_min_path_length: Optional[float] = None
        mcoconav_success = 0
        mcoconav_spl = 0.0
        mcoconav_find_agent: Optional[int] = None
        mcoconav_find_step: Optional[int] = None
        mcoconav_path_length: Optional[float] = None
        mcoconav_min_path_length: Optional[float] = None
        mcoconav_frozen = False
        while True:
            agent_positions = [get_agent_position(sim, agent_idx=i) for i in range(num_agents)]
            agent_distances = [
                compute_agent_goal_distance(
                    sim,
                    i,
                    agent_goal_view_points[i],
                    agent_episodes[i],
                )
                for i in range(num_agents)
            ]
            if all(agent_no_frontiers):
                termination_reason = "all_agents_no_frontier"
                break
            if episode_steps >= args.max_episode_length:
                termination_reason = f"max_steps_{args.max_episode_length}"
                break
            if env.episode_over:
                termination_reason = "env_episode_over"
                break

            # ---- Build one action per agent ----
            actions: List[int] = []
            # Ensure we have observations for all agents to avoid crossing logic
            if len(obs_list) < num_agents:
                print(f"WARNING: Expected {num_agents} obs but got {len(obs_list)}. Duplicating last obs.")
                while len(obs_list) < num_agents:
                    obs_list.append(obs_list[-1])

            success_stop_emitted = False
            for agent_idx in range(num_agents):
                if agent_no_frontiers[agent_idx]:
                    actions.append(ACTION_SET["WAIT"])
                    prev_actions[agent_idx] = torch.tensor(
                        [[int(TorchActionIDs.TURN_LEFT.item())]], dtype=torch.long
                    )
                    masks[agent_idx] = torch.ones_like(masks[agent_idx])
                    continue

                agent_obs = obs_list[agent_idx]
                policy_input = prepare_policy_obs(agent_obs, args, env, agent_idx)
                with torch.no_grad():
                    action_tensor, hidden_states[agent_idx] = policies[agent_idx].act(
                        policy_input,
                        hidden_states[agent_idx],
                        prev_actions[agent_idx],
                        masks[agent_idx],
                    )
                policy_info = getattr(policies[agent_idx], "_policy_info", {})
                policy_mode = (
                    str(policy_info.get("mode", "")).lower()
                    if isinstance(policy_info, dict)
                    else ""
                )
                if isinstance(policy_info, dict) and bool(policy_info.get("target_detected", False)):
                    agent_object_found[agent_idx] = True
                    if not mcoconav_frozen and policy_mode != "initialize":
                        matched_goal = match_nearest_goal_center_xy(
                            agent_positions[agent_idx],
                            agent_goal_centers[agent_idx],
                        )
                        mcoconav_success = 1
                        mcoconav_find_agent = agent_idx + 1
                        mcoconav_find_step = episode_steps
                        mcoconav_path_length = float(agent_path_lengths[agent_idx])
                        if matched_goal is not None:
                            matched_goal_idx, _, _ = matched_goal
                            mcoconav_min_path_length = get_precomputed_goal_geodesic(
                                agent_start_goal_geodesics[agent_idx],
                                matched_goal_idx,
                            )
                        mcoconav_spl = compute_event_spl(
                            mcoconav_success,
                            mcoconav_min_path_length,
                            mcoconav_path_length,
                        )
                        mcoconav_frozen = True
                action_flat = torch.as_tensor(action_tensor).view(-1)
                if action_flat.numel() != 1:
                    raise ValueError(
                        "Expected PointNav to output one discrete action id, "
                        f"but got shape {tuple(action_tensor.shape)} with {action_flat.numel()} values."
                    )
                action_value = int(action_flat[0].item())
                is_stop_action = action_value == int(TorchActionIDs.STOP.item())
                distance_success = (
                    agent_distances[agent_idx] is not None
                    and float(agent_distances[agent_idx]) <= success_distance_m
                )
                agent_geodesic_success[agent_idx] = distance_success
                agent_policy_stop_called[agent_idx] = bool(
                    isinstance(policy_info, dict) and policy_info.get("stop_called", False)
                )
                frontier_stop = is_stop_action and is_no_frontier_signal(
                    policies[agent_idx], action_value
                )
                success_stop = (
                    is_stop_action
                    and policy_mode == "navigate"
                )
                if success_stop:
                    sim_action = stop_action
                    accepted_navigate_stop_flags[agent_idx] = True
                    if winner_agent_idx is None:
                        winner_agent_idx = agent_idx
                    success_stop_emitted = True
                elif frontier_stop:
                    agent_no_frontiers[agent_idx] = True
                    sim_action = ACTION_SET["WAIT"]
                else:
                    if is_stop_action:
                        sim_action = ACTION_SET["WAIT"]
                    else:
                        sim_action = torch_to_habitat.get(action_value, ACTION_SET["WAIT"])
                actions.append(sim_action)
                prev_actions[agent_idx] = torch.tensor([[action_value]], dtype=torch.long)
                masks[agent_idx] = torch.ones_like(masks[agent_idx])
                if (
                    comms_enabled
                    and same_start_mode
                    and num_agents == 2
                    and agent_idx == 0
                    and getattr(policies[0], "_same_start_first_split_pending", False)
                ):
                    primary_frontier = getattr(
                        policies[0], "_same_start_last_selected_frontier", None
                    )
                    if primary_frontier is not None:
                        setattr(
                            policies[1],
                            "_same_start_partner_frontier",
                            np.asarray(primary_frontier, dtype=np.float32).copy(),
                        )
                if (
                    comms_enabled
                    and same_start_mode
                    and num_agents == 2
                    and agent_idx == 1
                ):
                    if not bool(getattr(policies[1], "_same_start_first_split_pending", False)):
                        setattr(policies[0], "_same_start_first_split_pending", False)

            if any(agent_object_found):
                episode_object_found = True

            if success_stop_emitted:
                obs = env.step(actions)
                obs_list = ensure_agent_obs_list(obs)
                steps_total += 1
                episode_steps += 1
                update_agent_path_lengths(sim, agent_prev_positions, agent_path_lengths)
                episode_success = True
                winning_agent = winner_agent_idx if winner_agent_idx is not None else 0
                matched_goal = match_nearest_goal_center_xy(
                    agent_prev_positions[winning_agent],
                    agent_goal_centers[winning_agent],
                )
                if matched_goal is not None:
                    matched_goal_idx, _, matched_goal_dist_xy = matched_goal
                    matched_goal_dist_xy = matched_goal_dist_xy
                    habitat_stop_distance_xy = matched_goal_dist_xy
                    if matched_goal_dist_xy <= habitat_success_distance_xy_m:
                        habitat_success = 1
                        habitat_success_agent = winning_agent + 1
                        habitat_success_step = episode_steps
                        habitat_path_length = float(agent_path_lengths[winning_agent])
                        habitat_min_path_length = get_precomputed_goal_geodesic(
                            agent_start_goal_geodesics[winning_agent],
                            matched_goal_idx,
                        )
                        habitat_spl = compute_event_spl(
                            habitat_success,
                            habitat_min_path_length,
                            habitat_path_length,
                        )
                termination_reason = (
                    f"navigate_stop_agent_{winning_agent+1}"
                )
                break

            if all(agent_no_frontiers):
                termination_reason = "all_agents_no_frontier"
                break

            # Step the env for all agents at once
            obs = env.step(actions)
            obs_list = ensure_agent_obs_list(obs)
            steps_total += 1
            episode_steps += 1
            update_agent_path_lengths(sim, agent_prev_positions, agent_path_lengths)

        metrics = get_eval_metrics(env)
        spl_values = collect_metric_values(metrics, "spl")
        native_success_values = collect_metric_values(metrics, "success")
        if any(agent_object_found):
            episode_object_found = True
        for policy in policies:
            try:
                target_name = getattr(policy, "_target_object", "")
                obj_map = getattr(policy, "_object_map", None)
                if target_name and obj_map is not None and bool(obj_map.has_object(target_name)):
                    episode_object_found = True
                    break
            except Exception:
                continue
        if episode_success:
            episodes_success += 1
        habitat_successes += habitat_success
        habitat_spl_total += habitat_spl
        mcoconav_successes += mcoconav_success
        mcoconav_spl_total += mcoconav_spl
        if winner_agent_idx is not None:
            episode_path_length = agent_path_lengths[winner_agent_idx]
            winner_min_path = agent_start_distances[winner_agent_idx]
            if winner_min_path is not None:
                episode_min_path_length = winner_min_path
        else:
            episode_path_length = agent_path_lengths[0] if agent_path_lengths else 0.0
        episode_spl = max(spl_values) if spl_values else None
        habitat_success_metric = max(native_success_values) if native_success_values else None
        if episode_spl is not None:
            episodes_with_spl += 1
            spl_total += float(episode_spl)

        sr_pass_by_steps = int(episode_success and episode_steps <= args.max_episode_length)
        outcome = "success" if episode_success else "failure"
        steps_to_target = episode_steps if episode_success else "NA"
        spl_str = f"{episode_spl:.3f}" if episode_spl is not None else "NA"
        habitat_success_str = (
            f"{habitat_success_metric:.3f}" if habitat_success_metric is not None else "NA"
        )
        min_path_str = f"{episode_min_path_length:.3f}" if episode_min_path_length is not None else "NA"
        object_found_str = "yes" if episode_object_found else "no"
        no_frontier_agents = sum(agent_no_frontiers)
        msg = (
            f"[episode {display_episode_index}/{total_selected_episodes}] "
            f"outcome={outcome} episode_steps={episode_steps} "
            f"path_length={episode_path_length:.3f} min_path_length={min_path_str} "
            f"steps_to_target={steps_to_target} spl={spl_str} sr_pass_by_steps={sr_pass_by_steps} "
            f"accepted_navigate_stop={accepted_navigate_stop_flags} "
            f"policy_stop_called={agent_policy_stop_called} "
            f"geodesic_success={agent_geodesic_success} "
            f"habitat_success={habitat_success_str} "
            f"same_start_mode={same_start_mode} "
            f"object_found={object_found_str} "
            f"termination={termination_reason} no_frontier_agents={no_frontier_agents}/{num_agents} "
            f"habitat_explicit_success={habitat_success} "
            f"habitat_explicit_spl={habitat_spl:.3f} "
            f"habitat_stop_distance_xy={format_optional_float(habitat_stop_distance_xy)} "
            f"habitat_success_agent={format_optional_int(habitat_success_agent)} "
            f"habitat_success_step={format_optional_int(habitat_success_step)} "
            f"habitat_path_length={format_optional_float(habitat_path_length)} "
            f"habitat_min_path_length={format_optional_float(habitat_min_path_length)} "
            f"mcoconav_success={mcoconav_success} "
            f"mcoconav_spl={mcoconav_spl:.3f} "
            f"mcoconav_find_agent={format_optional_int(mcoconav_find_agent)} "
            f"mcoconav_find_step={format_optional_int(mcoconav_find_step)} "
            f"mcoconav_path_length={format_optional_float(mcoconav_path_length)} "
            f"mcoconav_min_path_length={format_optional_float(mcoconav_min_path_length)} "
            f"scene={scene_id} episode_id={episode_id}"
        )
        print(msg)
        logging.info(msg)
        with open(episode_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not csv_header_written:
                writer.writerow(
                    [
                        "episode_index",
                        "num_episodes",
                        "scene",
                        "episode_id",
                        "outcome",
                        "success_metric",
                        "sr_pass_by_steps",
                        "episode_steps",
                        "steps_to_target",
                        "path_length",
                        "min_path_length",
                        "spl_metric",
                        "accepted_navigate_stop",
                        "policy_stop_called",
                        "geodesic_success",
                        "habitat_success_metric",
                        "same_start_mode",
                        "object_found",
                        "habitat_success",
                        "habitat_spl",
                        "habitat_stop_distance_xy",
                        "habitat_success_agent",
                        "habitat_success_step",
                        "habitat_path_length",
                        "habitat_min_path_length",
                        "mcoconav_success",
                        "mcoconav_spl",
                        "mcoconav_find_agent",
                        "mcoconav_find_step",
                        "mcoconav_path_length",
                        "mcoconav_min_path_length",
                        "termination_reason",
                        "no_frontier_agents",
                    ]
                )
                csv_header_written = True
            row = [
                display_episode_index,
                total_selected_episodes,
                scene_id,
                episode_id,
                outcome,
                float(episode_success),
                sr_pass_by_steps,
                episode_steps,
                steps_to_target,
                f"{episode_path_length:.6f}",
                "" if episode_min_path_length is None else f"{float(episode_min_path_length):.6f}",
                "" if episode_spl is None else f"{float(episode_spl):.6f}",
                "|".join(str(int(flag)) for flag in accepted_navigate_stop_flags),
                "|".join(str(int(flag)) for flag in agent_policy_stop_called),
                "|".join(str(int(flag)) for flag in agent_geodesic_success),
                "" if habitat_success_metric is None else f"{float(habitat_success_metric):.6f}",
                int(same_start_mode),
                object_found_str,
                int(habitat_success),
                f"{habitat_spl:.6f}",
                "" if habitat_stop_distance_xy is None else f"{float(habitat_stop_distance_xy):.6f}",
                "" if habitat_success_agent is None else str(int(habitat_success_agent)),
                "" if habitat_success_step is None else str(int(habitat_success_step)),
                "" if habitat_path_length is None else f"{float(habitat_path_length):.6f}",
                "" if habitat_min_path_length is None else f"{float(habitat_min_path_length):.6f}",
                int(mcoconav_success),
                f"{mcoconav_spl:.6f}",
                "" if mcoconav_find_agent is None else str(int(mcoconav_find_agent)),
                "" if mcoconav_find_step is None else str(int(mcoconav_find_step)),
                "" if mcoconav_path_length is None else f"{float(mcoconav_path_length):.6f}",
                "" if mcoconav_min_path_length is None else f"{float(mcoconav_min_path_length):.6f}",
                termination_reason,
                no_frontier_agents,
            ]
            writer.writerow(row)

    # Done
    dt = time.time() - t0
    sr = episodes_success / max(num_episodes, 1)
    avg_spl = spl_total / max(episodes_with_spl, 1)
    habitat_sr = habitat_successes / max(num_episodes, 1)
    habitat_avg_spl = habitat_spl_total / max(num_episodes, 1)
    mcoconav_sr = mcoconav_successes / max(num_episodes, 1)
    mcoconav_avg_spl = mcoconav_spl_total / max(num_episodes, 1)
    summary = (
        f"[multi-eval] Done in {dt:.1f}s, FPS ~ {steps_total/max(dt,1e-6):.2f} "
        f"SR={sr:.3f} avg_SPL={avg_spl:.3f} "
        f"Habitat_SR={habitat_sr:.3f} Habitat_SPL={habitat_avg_spl:.3f} "
        f"MCoCoNav_SR={mcoconav_sr:.3f} MCoCoNav_SPL={mcoconav_avg_spl:.3f}"
    )
    print(summary)
    logging.info(summary)


if __name__ == "__main__":
    main()
