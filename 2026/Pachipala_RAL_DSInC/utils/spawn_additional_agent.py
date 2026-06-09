from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import habitat_sim


def get_sim(env):
    """Return the habitat-sim Simulator handle from a possibly-wrapped env."""
    base = getattr(env, "_env", env)
    sim = getattr(base, "sim", None) or getattr(base, "_sim", None)
    if sim is None:
        raise RuntimeError("Simulator handle not found on env")
    return sim


def get_scene_and_episode(env) -> Tuple[str, str]:
    base = getattr(env, "_env", env)
    ep = getattr(base, "current_episode", None) or getattr(base, "episode", None)
    if ep is None:
        return "unknown_scene", "unknown_ep"
    return getattr(ep, "scene_id", "unknown_scene"), str(getattr(ep, "episode_id", "unknown_ep"))


def maybe_reseed_sim(sim, seed: Optional[int]) -> None:
    """
    Reseed sim's internal RNG if the build exposes sim.seed(int).
    If not available, silently no-op.
    """
    if seed is None:
        return
    if hasattr(sim, "seed"):
        try:
            sim.seed(int(seed))
        except Exception:
            # Some builds might expose pathfinder.seed instead
            pf = getattr(sim, "pathfinder", None)
            if pf is not None and hasattr(pf, "seed"):
                pf.seed(int(seed))

def _as_wxyz_vec1d(rot) -> np.ndarray:
    import numpy as np
    try:
        import quaternion as np_quat
        if isinstance(rot, np_quat.quaternion):
            wxyz = np.array([rot.w, rot.x, rot.y, rot.z], dtype=np.float64)
            wxyz /= max(np.linalg.norm(wxyz), 1e-12)
            return wxyz.astype(np.float32)
    except Exception:
        pass
    if all(hasattr(rot, a) for a in ("w", "x", "y", "z")):
        wxyz = np.array([float(rot.w), float(rot.x), float(rot.y), float(rot.z)], dtype=np.float64)
        wxyz /= max(np.linalg.norm(wxyz), 1e-12)
        return wxyz.astype(np.float32)
    arr = np.asarray(rot, dtype=np.float64).reshape(-1)
    if arr.size == 4:
        arr /= max(np.linalg.norm(arr), 1e-12)
        return arr.astype(np.float32)
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # identity fallback



def randomize_agent_after_reset(
    env,
    agent_index: int = 1,
    anchor_agent_index: int = 0,
    min_geodesic: float = 15.0,
    min_euclidean: float = 7.0,
    same_floor_eps: float = 0.35,
    max_tries: int = 100,
    episode_seed: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    import numpy as np
    import habitat_sim
    import logging

    sim = get_sim(env)
    pf = sim.pathfinder
    maybe_reseed_sim(sim, episode_seed)

    num_agents = len(getattr(sim, "agents", [])) or getattr(env, "num_agents", 0)
    if agent_index >= num_agents:
        logging.error(f"Agent index {agent_index} >= num_agents={num_agents}")
        return None, None

    s_anchor = sim.get_agent_state(anchor_agent_index)
    y_anchor = float(s_anchor.position[1])

    spath = habitat_sim.ShortestPath()
    spath.requested_start = np.array(s_anchor.position, dtype=np.float32)

    setattr(
        env,
        "_last_spawn_randomization",
        {
            "used_tier": None,
            "used_relaxed_best": False,
            "same_location": False,
            "geodesic": None,
            "euclidean": None,
        },
    )

    def sample_point():
        if hasattr(pf, "get_random_navigable_point"):
            return pf.get_random_navigable_point()
        return sim.sample_navigable_point()

    def far_from_all_agents(p: np.ndarray, min_euclidean_req: float) -> bool:
        for i in range(num_agents):
            if i == agent_index:
                continue
            pos_i = np.asarray(sim.get_agent_state(i).position, dtype=np.float32)
            if float(np.linalg.norm(p - pos_i)) < float(min_euclidean_req):
                return False
        return True

    tiers = []
    for geo_req, euc_req in (
        (float(min_geodesic), float(min_euclidean)),
        (7.0, 7.0),
        (5.0, 5.0),
        (3.0, 3.0),
    ):
        tier = (geo_req, euc_req)
        if geo_req > 0 and euc_req > 0 and tier not in tiers:
            tiers.append(tier)

    best_p: Optional[np.ndarray] = None
    best_d = -1.0
    best_euc = -1.0

    for geo_req, euc_req in tiers:
        for _ in range(max_tries):
            p = np.asarray(sample_point(), dtype=np.float32)

            if abs(float(p[1]) - y_anchor) > same_floor_eps:
                continue
            if hasattr(pf, "is_navigable") and not pf.is_navigable(p):
                continue

            euclid = float(np.linalg.norm(p - np.asarray(s_anchor.position, dtype=np.float32)))
            if not far_from_all_agents(p, euc_req):
                continue

            spath.requested_end = p
            if not pf.find_path(spath):
                continue

            d = float(spath.geodesic_distance)
            if d > best_d:
                best_p, best_d, best_euc = p, d, euclid

            if d >= geo_req and euclid >= euc_req:
                state = sim.get_agent_state(agent_index)
                state.position = p
                sim.get_agent(agent_index).set_state(state, True)
                setattr(
                    env,
                    "_last_spawn_randomization",
                    {
                        "used_tier": (geo_req, euc_req),
                        "used_relaxed_best": False,
                        "same_location": False,
                        "geodesic": d,
                        "euclidean": euclid,
                    },
                )
                logging.info(
                    f"agent{agent_index} randomized to {p.tolist()} (geo={d:.2f}m, euclid={euclid:.2f}m, tier=({geo_req:.1f},{euc_req:.1f}))"
                )
                print(
                    f"[spawn] agent{agent_index} -> pos={p.tolist()}  geodesic={d:.2f}m  euclid={euclid:.2f}m  tier=({geo_req:.1f},{euc_req:.1f})"
                )
                return p, d

    if best_p is not None:
        state = sim.get_agent_state(agent_index)
        state.position = best_p
        sim.get_agent(agent_index).set_state(state, True)
        achieved_euclid = float(
            np.linalg.norm(best_p - np.asarray(s_anchor.position, dtype=np.float32))
        )
        setattr(
            env,
            "_last_spawn_randomization",
            {
                "used_tier": None,
                "used_relaxed_best": True,
                "same_location": False,
                "geodesic": best_d if best_d >= 0 else None,
                "euclidean": achieved_euclid,
            },
        )
        logging.warning(
            f"Relaxed spawn for agent{agent_index}: best geodesic {best_d:.2f}m "
            f"(< {min_geodesic}m), euclid={achieved_euclid:.2f}m."
        )
        print(
            f"[spawn-relaxed] agent{agent_index} -> pos={best_p.tolist()}  geodesic={best_d:.2f}m  euclid={achieved_euclid:.2f}m"
        )
        return best_p, best_d

    state = sim.get_agent_state(agent_index)
    state.position = np.asarray(s_anchor.position, dtype=np.float32)
    sim.get_agent(agent_index).set_state(state, True)
    setattr(
        env,
        "_last_spawn_randomization",
        {
            "used_tier": None,
            "used_relaxed_best": False,
            "same_location": True,
            "geodesic": 0.0,
            "euclidean": 0.0,
        },
    )
    logging.warning(
        f"Failed to find separated spawn for agent{agent_index}; falling back to anchor start."
    )
    print(f"[spawn-same-location] agent{agent_index} -> pos={np.asarray(s_anchor.position).tolist()}")
    return np.asarray(s_anchor.position, dtype=np.float32), 0.0
