# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import sys
import types
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete
from torch import Tensor

habitat_version = ""
DISCRETE_ONLY_ERROR = (
    "DSInC eval expects discrete PointNav checkpoint. "
    "Please convert/export a discrete-compatible state_dict from your 0.24.1 environment."
)

try:
    import habitat
    from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy

    habitat_version = habitat.__version__

    if habitat_version == "0.1.5":
        print("Using habitat 0.1.5; assuming SemExp code is being used")

        class PointNavResNetTensorOutputPolicy(PointNavResNetPolicy):
            def act(self, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]:
                value, action, action_log_probs, rnn_hidden_states = super().act(*args, **kwargs)
                return action, rnn_hidden_states

    else:
        from habitat_baselines.common.tensor_dict import TensorDict
        from habitat_baselines.rl.ppo.policy import PolicyActionData

        class PointNavResNetTensorOutputPolicy(PointNavResNetPolicy):  # type: ignore
            def act(self, *args: Any, **kwargs: Any) -> Tuple[Tensor, Tensor]:
                policy_actions: "PolicyActionData" = super().act(*args, **kwargs)
                return policy_actions.actions, policy_actions.rnn_hidden_states

    HABITAT_BASELINES_AVAILABLE = True
except Exception as exc:
    print(
        "Warning: habitat_baselines PointNav import failed "
        f"({type(exc).__name__}: {exc}). Falling back to non-habitat PointNav policy."
    )
    from vlfm.policy.utils.non_habitat_policy.nh_pointnav_policy import (
        PointNavResNetPolicy,
    )

    class PointNavResNetTensorOutputPolicy(PointNavResNetPolicy):  # type: ignore
        """Already outputs a tensor, so no need to convert."""

        pass

    HABITAT_BASELINES_AVAILABLE = False


class WrappedPointNavResNetPolicy:
    """
    Wrapper for the PointNavResNetPolicy that allows for easier usage, however it can
    only handle one environment at a time. Automatically updates the hidden state
    and previous action for the policy.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: Union[str, torch.device] = "cuda",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.policy = load_pointnav_policy(ckpt_path)
        self.policy.to(device)
        self.policy.eval()
        discrete_actions = not hasattr(self.policy.action_distribution, "mu_maybe_std")
        self.pointnav_test_recurrent_hidden_states = torch.zeros(
            1,  # The number of environments.
            self.policy.net.num_recurrent_layers,
            512,  # hidden state size
            device=device,
        )
        if discrete_actions:
            num_actions = 1
            action_dtype = torch.long
        else:
            num_actions = 2
            action_dtype = torch.float32
        self.pointnav_prev_actions = torch.zeros(
            1,  # number of environments
            num_actions,
            device=device,
            dtype=action_dtype,
        )
        self.device = device

    def act(
        self,
        observations: Union["TensorDict", Dict],
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tensor:
        """Infers action to take towards the given (rho, theta) based on depth vision.

        Args:
            observations (Union["TensorDict", Dict]): A dictionary containing (at least)
                the following:
                    - "depth" (torch.float32): Depth image tensor (N, H, W, 1).
                    - "pointgoal_with_gps_compass" (torch.float32):
                        PointGoalWithGPSCompassSensor tensor representing a rho and
                        theta w.r.t. to the agent's current pose (N, 2).
            masks (torch.bool): Tensor of masks, with a value of 1 for any step after
                the first in an episode; has 0 for first step.
            deterministic (bool): Whether to select a logit action deterministically.

        Returns:
            Tensor: A tensor denoting the action to take.
        """
        pointnav_action, _, rnn_hidden_states = self.act_with_action_preferences(
            observations,
            masks,
            deterministic=deterministic,
        )
        self.commit_action(pointnav_action, rnn_hidden_states)
        return pointnav_action

    def _forward_distribution(
        self,
        observations: Union["TensorDict", Dict],
        masks: Tensor,
    ) -> Tuple[Any, Tensor]:
        observations = move_obs_to_device(observations, self.device)
        masks = masks.to(device=self.device)
        with torch.inference_mode():
            net_out = self.policy.net(
                observations,
                self.pointnav_test_recurrent_hidden_states,
                self.pointnav_prev_actions,
                masks,
            )
        if not isinstance(net_out, tuple) or len(net_out) < 2:
            raise RuntimeError("Unexpected PointNav net output while computing action distribution.")
        features = net_out[0]
        rnn_hidden_states = net_out[1]
        distribution = self.policy.action_distribution(features)
        return distribution, rnn_hidden_states

    def _select_action_from_distribution(
        self,
        distribution: Any,
        deterministic: bool = False,
    ) -> Tensor:
        if hasattr(distribution, "logits"):
            if deterministic:
                return torch.argmax(distribution.logits, dim=-1, keepdim=True)
            sample = distribution.sample()
            if sample.ndim == 1:
                sample = sample.unsqueeze(-1)
            return sample
        if deterministic and hasattr(distribution, "mean"):
            return distribution.mean
        return distribution.sample()

    def act_with_action_preferences(
        self,
        observations: Union["TensorDict", Dict],
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Union[Tensor, None], Tensor]:
        distribution, rnn_hidden_states = self._forward_distribution(observations, masks)
        pointnav_action = self._select_action_from_distribution(
            distribution,
            deterministic=deterministic,
        )
        pointnav_action = pointnav_action.detach()
        rnn_hidden_states = rnn_hidden_states.detach()
        action_preferences = None
        if hasattr(distribution, "logits"):
            action_preferences = distribution.logits.detach().clone()
        return pointnav_action, action_preferences, rnn_hidden_states

    def commit_action(self, pointnav_action: Tensor, rnn_hidden_states: Tensor) -> None:
        pointnav_action = pointnav_action.detach().to(device=self.device)
        rnn_hidden_states = rnn_hidden_states.detach().to(device=self.device)
        self.pointnav_prev_actions = pointnav_action.clone()
        self.pointnav_test_recurrent_hidden_states = rnn_hidden_states

    def reset(self) -> None:
        """
        Resets the hidden state and previous action for the policy.
        """
        self.pointnav_test_recurrent_hidden_states = torch.zeros_like(self.pointnav_test_recurrent_hidden_states)
        self.pointnav_prev_actions = torch.zeros_like(self.pointnav_prev_actions)


def _ensure_module_stub(module_name: str) -> None:
    """Create a dynamic module stub that lazily materializes classes by name."""
    if module_name in sys.modules:
        return

    parts = module_name.split(".")
    for i in range(1, len(parts)):
        parent_name = ".".join(parts[:i])
        child_name = parts[i]
        parent_mod = sys.modules.get(parent_name)
        if parent_mod is None:
            parent_mod = types.ModuleType(parent_name)
            sys.modules[parent_name] = parent_mod
        full_child = ".".join(parts[: i + 1])
        child_mod = sys.modules.get(full_child)
        if child_mod is None:
            child_mod = types.ModuleType(full_child)
            sys.modules[full_child] = child_mod
        setattr(parent_mod, child_name, child_mod)

    module = sys.modules[module_name]

    def _getattr(name: str) -> Any:
        stub_cls = type(name, (), {})
        setattr(module, name, stub_cls)
        return stub_cls

    setattr(module, "__getattr__", _getattr)


def _load_checkpoint_compat(file_path: str) -> Any:
    """
    Load a PointNav checkpoint while tolerating Habitat-version pickle drift.
    """
    # Prefer weights-only loading to avoid unpickling warnings/risk when possible.
    def _torch_load_with_optional_weights_only(weights_only: bool) -> Any:
        try:
            return torch.load(file_path, map_location="cpu", weights_only=weights_only)
        except TypeError:
            # Older torch versions do not accept `weights_only`.
            return torch.load(file_path, map_location="cpu")

    try:
        return _torch_load_with_optional_weights_only(weights_only=True)
    except Exception:
        pass

    try:
        return _torch_load_with_optional_weights_only(weights_only=False)
    except ModuleNotFoundError as exc:
        if exc.name == "habitat.config.default_structured_configs":
            _ensure_module_stub(exc.name)
            return _torch_load_with_optional_weights_only(weights_only=False)
        raise


def _is_discrete_pointnav_checkpoint(state_dict: Dict[str, Tensor]) -> bool:
    if "action_distribution.linear.weight" in state_dict:
        return True
    if "net.prev_action_embedding_discrete.weight" in state_dict:
        return True
    if "net.prev_action_embedding.weight" in state_dict:
        legacy_weight = state_dict["net.prev_action_embedding.weight"]
        # Legacy discrete checkpoints usually store embedding as (num_actions+1, emb_dim) = (5, 32).
        if legacy_weight.ndim == 2 and legacy_weight.shape[1] == 32:
            return True
    return False


def _patch_legacy_discrete_prev_action_key(state_dict: Dict[str, Tensor]) -> None:
    if "net.prev_action_embedding_discrete.weight" not in state_dict:
        if "net.prev_action_embedding.weight" in state_dict:
            state_dict["net.prev_action_embedding_discrete.weight"] = state_dict["net.prev_action_embedding.weight"]
        elif "net.prev_action_embedding_cont.weight" in state_dict:
            candidate = state_dict["net.prev_action_embedding_cont.weight"]
            if candidate.ndim == 2 and candidate.shape[1] == 32:
                state_dict["net.prev_action_embedding_discrete.weight"] = candidate


def _select_shape_compatible_tensors(
    src_state_dict: Dict[str, Tensor],
    dst_state_dict: Dict[str, Tensor],
) -> Tuple[Dict[str, Tensor], Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]]:
    compatible: Dict[str, Tensor] = {}
    mismatched: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = {}
    for key, tensor in src_state_dict.items():
        if key not in dst_state_dict:
            continue
        if tuple(tensor.shape) == tuple(dst_state_dict[key].shape):
            compatible[key] = tensor
        else:
            mismatched[key] = (tuple(tensor.shape), tuple(dst_state_dict[key].shape))
    return compatible, mismatched


def _normalize_state_dict_keys(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    normalized: Dict[str, Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("actor_critic."):
            new_key = new_key[len("actor_critic.") :]
        normalized[new_key] = value
    return normalized


def _load_non_habitat_discrete_policy_from_state_dict(
    ckpt_state_dict: Dict[str, Tensor],
) -> PointNavResNetTensorOutputPolicy:
    from vlfm.policy.utils.non_habitat_policy.nh_pointnav_policy import (
        PointNavResNetPolicy as NonHabitatPointNavResNetPolicy,
    )

    normalized = _normalize_state_dict_keys(ckpt_state_dict)
    _patch_legacy_discrete_prev_action_key(normalized)
    if not _is_discrete_pointnav_checkpoint(normalized):
        raise RuntimeError(DISCRETE_ONLY_ERROR)

    pointnav_policy = NonHabitatPointNavResNetPolicy(discrete_actions=True)
    current_state_dict = pointnav_policy.state_dict()

    compatible, mismatched = _select_shape_compatible_tensors(normalized, current_state_dict)
    if "action_distribution.linear.weight" not in compatible:
        raise RuntimeError(
            "DSInC eval expects discrete PointNav checkpoint with action_distribution.linear.* "
            "weights; checkpoint appears incompatible."
        )

    pointnav_policy.load_state_dict(compatible, strict=False)
    unused_keys = [k for k in normalized.keys() if k not in current_state_dict]
    if mismatched:
        print(
            "The following keys had shape mismatches and were skipped when loading the pointnav policy: "
            f"{mismatched}"
        )
    if unused_keys:
        print(f"The following unused keys were not loaded when loading the pointnav policy: {unused_keys}")
    return pointnav_policy  # type: ignore[return-value]


def load_pointnav_policy(file_path: str) -> PointNavResNetTensorOutputPolicy:
    """Loads a PointNavResNetPolicy policy from a .pth file.

    Args:
        file_path (str): The path to the trained weights of the pointnav policy.
    Returns:
        PointNavResNetTensorOutputPolicy: The policy.
    """
    if HABITAT_BASELINES_AVAILABLE:
        obs_space = SpaceDict(
            {
                "depth": spaces.Box(low=0.0, high=1.0, shape=(224, 224, 1), dtype=np.float32),
                "pointgoal_with_gps_compass": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        action_space = Discrete(4)
        if habitat_version == "0.1.5":
            pointnav_policy = PointNavResNetTensorOutputPolicy(
                obs_space,
                action_space,
                hidden_size=512,
                num_recurrent_layers=2,
                rnn_type="LSTM",
                resnet_baseplanes=32,
                backbone="resnet18",
                normalize_visual_inputs=False,
                obs_transform=None,
            )
            # Need to overwrite the visual encoder because it uses an older version of
            # ResNet that calculates the compression size differently
            from vlfm.policy.utils.non_habitat_policy.nh_pointnav_policy import (
                PointNavResNetNet,
            )

            # print(pointnav_policy)
            pointnav_policy.net = PointNavResNetNet(discrete_actions=True, no_fwd_dict=True)
            state_dict = torch.load(file_path + ".state_dict", map_location="cpu")
        else:
            ckpt_dict = _load_checkpoint_compat(file_path)
            if isinstance(ckpt_dict, dict) and "config" in ckpt_dict and "state_dict" in ckpt_dict:
                pointnav_policy = PointNavResNetTensorOutputPolicy.from_config(ckpt_dict["config"], obs_space, action_space)
                state_dict = ckpt_dict["state_dict"]
            else:
                return _load_non_habitat_discrete_policy_from_state_dict(ckpt_dict)
        state_dict = _normalize_state_dict_keys(state_dict)
        _patch_legacy_discrete_prev_action_key(state_dict)
        if not _is_discrete_pointnav_checkpoint(state_dict):
            raise RuntimeError(DISCRETE_ONLY_ERROR)
        pointnav_policy.load_state_dict(state_dict)
        return pointnav_policy

    else:
        ckpt_dict = _load_checkpoint_compat(file_path)
        if isinstance(ckpt_dict, dict) and "state_dict" in ckpt_dict:
            ckpt_dict = ckpt_dict["state_dict"]
        return _load_non_habitat_discrete_policy_from_state_dict(ckpt_dict)


def move_obs_to_device(
    observations: Dict[str, Any],
    device: torch.device,
    unsqueeze: bool = False,
) -> Dict[str, Tensor]:
    """Moves observations to the given device, converts numpy arrays to torch tensors.

    Args:
        observations (Dict[str, Union[Tensor, np.ndarray]]): The observations.
        device (torch.device): The device to move the observations to.
        unsqueeze (bool): Whether to unsqueeze the tensors or not.
    Returns:
        Dict[str, Tensor]: The observations on the given device as torch tensors.
    """
    # Convert numpy arrays to torch tensors for each dict value
    for k, v in observations.items():
        if isinstance(v, np.ndarray):
            tensor_dtype = torch.uint8 if v.dtype == np.uint8 else torch.float32
            observations[k] = torch.from_numpy(v).to(device=device, dtype=tensor_dtype)
            if unsqueeze:
                observations[k] = observations[k].unsqueeze(0)

    return observations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Load a checkpoint file for PointNavResNetPolicy")
    parser.add_argument("ckpt_path", help="path to checkpoint file")
    args = parser.parse_args()

    policy = load_pointnav_policy(args.ckpt_path)
    print("Loaded model from checkpoint successfully!")
    mask = torch.zeros(1, 1, device=torch.device("cuda"), dtype=torch.bool)
    observations = {
        "depth": torch.zeros(1, 224, 224, 1, device=torch.device("cuda")),
        "pointgoal_with_gps_compass": torch.zeros(1, 2, device=torch.device("cuda")),
    }
    policy.to(torch.device("cuda"))
    action = policy.act(
        observations,
        torch.zeros(1, 4, 512, device=torch.device("cuda"), dtype=torch.float32),
        torch.zeros(1, 1, device=torch.device("cuda"), dtype=torch.long),
        mask,
    )
    print("Forward pass successful!")
