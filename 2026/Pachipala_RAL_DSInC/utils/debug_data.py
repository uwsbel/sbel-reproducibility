import logging
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

__all__ = [
    "get_sim_from_env",
    "log_spawn_states",
    "get_scene_and_episode",
    "find_rgb",
    "normalize_rgb",
    "save_rgb_first_frame_per_episode",
]

def get_sim_from_env(env: Any):
    candidates = [getattr(env, "_env", None), env]
    for base in candidates:
        if base is None:
            continue
        for attr in ("sim", "_sim"):
            if hasattr(base, attr):
                return getattr(base, attr)
    raise RuntimeError("Could not find simulator handle on env")

def _quat_to_str(q):
    try:
        return f"(w={q.w:.4f}, x={q.x:.4f}, y={q.y:.4f}, z={q.z:.4f})"
    except Exception:
        return str(q)

def log_spawn_states(env, logger=logging):
    sim = get_sim_from_env(env)
    ep = None
    for attr in ("current_episode", "episode", "current_episodes"):
        if hasattr(getattr(env, "_env", env), attr):
            ep = getattr(getattr(env, "_env", env), attr)
            break

    ep_start_positions = getattr(ep, "start_positions", None) if ep is not None else None
    ep_start_rotations = getattr(ep, "start_rotations", None) if ep is not None else None

    num_agents = getattr(env, "num_agents", None) or getattr(env, "number_of_agents", None)
    if num_agents is None:
        try:
            num_agents = env.config.SIMULATOR.NUM_AGENTS
        except Exception:
            num_agents = len(getattr(sim, "agents", [None]))

    print("==== Agent spawn states ====")
    logger.info("==== Agent spawn states ====")
    for aid in range(num_agents):
        st = sim.get_agent_state(aid)
        pos, rot = st.position, st.rotation
        msg_now = f"agent_id={aid} :: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  rot={_quat_to_str(rot)}"
        print(msg_now); logger.info(msg_now)

        if ep_start_positions is not None and aid < len(ep_start_positions):
            spos = ep_start_positions[aid]
            srot = ep_start_rotations[aid] if (ep_start_rotations is not None and aid < len(ep_start_rotations)) else None
            msg_ep = f"agent_id={aid} :: ep_start_pos=({spos[0]:.3f}, {spos[1]:.3f}, {spos[2]:.3f})" + \
                     (f"  ep_start_rot={_quat_to_str(srot)}" if srot is not None else "")
            print(msg_ep); logger.info(msg_ep)

def get_scene_and_episode(env):
    base = getattr(env, "_env", env)
    ep = getattr(base, "current_episode", None) or getattr(base, "episode", None)
    if ep is None:
        return "unknown_scene", "unknown_ep"
    return getattr(ep, "scene_id", "unknown_scene"), str(getattr(ep, "episode_id", "unknown_ep"))

def normalize_rgb(arr):
    if hasattr(arr, "detach"):  # torch.Tensor
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 1) if np.issubdtype(arr.dtype, np.floating) else np.clip(arr, 0, 255)
        arr = (arr * 255.0).round().astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return arr

def find_rgb(obs_dict):
    for k in ("rgb", "rgb_sensor", "RGB_SENSOR"):
        if k in obs_dict:
            return obs_dict[k]
    for k in obs_dict.keys():
        if "rgb" in k.lower():
            return obs_dict[k]
    return None

def save_rgb_first_frame_per_episode(env, obs, out_root):
    scene_id, episode_id = get_scene_and_episode(env)
    ob_list = obs if isinstance(obs, (list, tuple)) else [obs]
    safe_scene = str(scene_id).replace("/", "_").replace("\\", "_")
    out_dir = Path(out_root) / safe_scene / f"ep-{episode_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for aid, ob in enumerate(ob_list):
        rgb = find_rgb(ob)
        if rgb is None:
            logging.warning(f"No RGB found in obs for agent {aid}; keys={list(ob.keys())}")
            continue
        try:
            img = Image.fromarray(normalize_rgb(rgb))
        except Exception as e:
            logging.exception(f"Failed to convert RGB for agent {aid}: {e}")
            continue
        fname = out_dir / f"first_rgb_agent-{aid}.png"
        img.save(fname)
        logging.info(f"Saved first RGB for episode {episode_id}, agent {aid} -> {fname}")
        print(f"[save] {fname}")
