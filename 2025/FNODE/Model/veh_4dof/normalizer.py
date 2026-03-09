"""
Data normalization utilities for vehicle parameter model
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional


class VehicleDataNormalizer:
    """
    Normalizes vehicle states, controls, and parameters to [-1, 1] or [0, 1] range
    """

    def __init__(self):
        # State ranges (based on typical values from 20-second trajectories)
        self.state_ranges = {
            'x': (-100, 100),        # Position x in meters
            'y': (-100, 100),        # Position y in meters
            'v': (0, 30),            # Velocity in m/s
            'theta': (-np.pi, np.pi) # Heading angle in radians
        }

        # Control ranges
        self.control_ranges = {
            'alpha': (0, 1),         # Throttle [0, 1]
            'beta': (-1, 1)          # Steering [-1, 1]
        }

        # Parameter ranges (from parameter_sampler.py - mid-size vehicles)
        self.param_ranges = {
            'l': (2.7, 2.9),           # wheelbase
            'r_wheel': (0.32, 0.35),   # wheel radius
            'i_wheel': (0.9, 1.3),     # wheel inertia
            'tau0': (250, 350),        # max torque
            'omega0': (1200, 1500),    # max engine speed
            'c0': (0.015, 0.025),      # drag coefficient 0
            'c1': (0.025, 0.045),      # drag coefficient 1
            'delta': (0.60, 0.75),     # steering ratio
            'gamma': (0.15, 0.25)      # transmission ratio
        }

        # Target ranges (derivatives)
        self.target_ranges = {
            'v_dot': (-10, 10),        # Acceleration in m/s²
            'theta_dot': (-2, 2)       # Angular velocity in rad/s
        }

    def normalize_value(self, value, min_val, max_val, target_min=-1, target_max=1):
        """Normalize a value from [min_val, max_val] to [target_min, target_max]"""
        if max_val == min_val:
            return 0.0
        normalized = (value - min_val) / (max_val - min_val)
        return target_min + normalized * (target_max - target_min)

    def denormalize_value(self, value, min_val, max_val, target_min=-1, target_max=1):
        """Denormalize a value from [target_min, target_max] to [min_val, max_val]"""
        if target_max == target_min:
            return min_val
        normalized = (value - target_min) / (target_max - target_min)
        return min_val + normalized * (max_val - min_val)

    def normalize_states(self, states: np.ndarray) -> np.ndarray:
        """
        Normalize state vector [x, y, v, theta]

        Args:
            states: Array of shape (..., 4) or (..., 6) if includes controls

        Returns:
            Normalized states in range [-1, 1]
        """
        normalized = states.copy()

        # Normalize position
        normalized[..., 0] = self.normalize_value(states[..., 0], *self.state_ranges['x'])
        normalized[..., 1] = self.normalize_value(states[..., 1], *self.state_ranges['y'])

        # Normalize velocity
        normalized[..., 2] = self.normalize_value(states[..., 2], *self.state_ranges['v'])

        # Normalize theta
        normalized[..., 3] = self.normalize_value(states[..., 3], *self.state_ranges['theta'])

        return normalized

    def normalize_controls(self, controls: np.ndarray) -> np.ndarray:
        """
        Normalize control inputs [alpha, beta]

        Args:
            controls: Array of shape (..., 2)

        Returns:
            Normalized controls in range [-1, 1]
        """
        normalized = controls.copy()

        # Alpha is already [0, 1], map to [-1, 1]
        normalized[..., 0] = self.normalize_value(controls[..., 0], *self.control_ranges['alpha'])

        # Beta is already [-1, 1], keep as is
        normalized[..., 1] = controls[..., 1]  # Already normalized

        return normalized

    def normalize_parameters(self, params: np.ndarray) -> np.ndarray:
        """
        Normalize parameter vector (9 parameters)

        Args:
            params: Array of shape (..., 9)

        Returns:
            Normalized parameters in range [-1, 1]
        """
        normalized = params.copy()
        param_order = ['l', 'r_wheel', 'i_wheel', 'tau0', 'omega0', 'c0', 'c1', 'delta', 'gamma']

        for i, param_name in enumerate(param_order):
            normalized[..., i] = self.normalize_value(
                params[..., i],
                *self.param_ranges[param_name]
            )

        return normalized

    def normalize_targets(self, targets: np.ndarray) -> np.ndarray:
        """
        Normalize target values [v_dot, theta_dot]

        Args:
            targets: Array of shape (..., 2)

        Returns:
            Normalized targets in range [-1, 1]
        """
        normalized = targets.copy()

        # Normalize v_dot (acceleration)
        normalized[..., 0] = self.normalize_value(targets[..., 0], *self.target_ranges['v_dot'])

        # Normalize theta_dot
        normalized[..., 1] = self.normalize_value(targets[..., 1], *self.target_ranges['theta_dot'])

        return normalized

    def denormalize_targets(self, normalized_targets: np.ndarray) -> np.ndarray:
        """
        Denormalize target values from [-1, 1] back to physical units

        Args:
            normalized_targets: Array of shape (..., 2) in range [-1, 1]

        Returns:
            Denormalized targets in physical units
        """
        denormalized = normalized_targets.copy()

        # Denormalize v_dot
        denormalized[..., 0] = self.denormalize_value(
            normalized_targets[..., 0],
            *self.target_ranges['v_dot']
        )

        # Denormalize theta_dot
        denormalized[..., 1] = self.denormalize_value(
            normalized_targets[..., 1],
            *self.target_ranges['theta_dot']
        )

        return denormalized

    def normalize_full_input(self, states: np.ndarray, controls: np.ndarray,
                            params: np.ndarray) -> np.ndarray:
        """
        Normalize full input vector [states(4), controls(2), params(9)]

        Args:
            states: Array of shape (..., 4)
            controls: Array of shape (..., 2)
            params: Array of shape (..., 9)

        Returns:
            Normalized input of shape (..., 15) in range [-1, 1]
        """
        norm_states = self.normalize_states(states)
        norm_controls = self.normalize_controls(controls)
        norm_params = self.normalize_parameters(params)

        # Concatenate along last dimension
        if len(states.shape) == 1:
            return np.concatenate([norm_states, norm_controls, norm_params])
        else:
            return np.concatenate([norm_states, norm_controls, norm_params], axis=-1)

    def to_torch(self, device='cuda'):
        """Convert normalizer to use PyTorch tensors on specified device"""
        # Convert ranges to torch tensors
        for key in ['state_ranges', 'control_ranges', 'param_ranges', 'target_ranges']:
            ranges = getattr(self, key)
            for name, (min_val, max_val) in ranges.items():
                ranges[name] = (
                    torch.tensor(min_val, device=device, dtype=torch.float32),
                    torch.tensor(max_val, device=device, dtype=torch.float32)
                )

    def get_normalization_stats(self):
        """Print normalization statistics"""
        print("\n=== Normalization Ranges ===")

        print("\nStates:")
        for name, (min_val, max_val) in self.state_ranges.items():
            print(f"  {name:8s}: [{min_val:7.2f}, {max_val:7.2f}]")

        print("\nControls:")
        for name, (min_val, max_val) in self.control_ranges.items():
            print(f"  {name:8s}: [{min_val:7.2f}, {max_val:7.2f}]")

        print("\nParameters:")
        for name, (min_val, max_val) in self.param_ranges.items():
            print(f"  {name:8s}: [{min_val:7.2f}, {max_val:7.2f}]")

        print("\nTargets:")
        for name, (min_val, max_val) in self.target_ranges.items():
            print(f"  {name:8s}: [{min_val:7.2f}, {max_val:7.2f}]")


if __name__ == "__main__":
    # Test the normalizer
    normalizer = VehicleDataNormalizer()
    normalizer.get_normalization_stats()

    # Test normalization
    print("\n=== Normalization Test ===")

    # Test state normalization
    test_states = np.array([50, -30, 15, 1.57])  # [x, y, v, theta]
    norm_states = normalizer.normalize_states(test_states)
    print(f"\nOriginal states: {test_states}")
    print(f"Normalized states: {norm_states}")

    # Test parameter normalization
    test_params = np.array([2.8, 0.34, 1.1, 300, 1400, 0.02, 0.035, 0.7, 0.2])
    norm_params = normalizer.normalize_parameters(test_params)
    print(f"\nOriginal params: {test_params}")
    print(f"Normalized params: {norm_params}")

    # Test full input normalization
    test_controls = np.array([0.5, 0.3])
    full_norm = normalizer.normalize_full_input(test_states, test_controls, test_params)
    print(f"\nFull normalized input shape: {full_norm.shape}")
    print(f"Full normalized input: {full_norm}")
    print(f"Range check: min={full_norm.min():.3f}, max={full_norm.max():.3f}")