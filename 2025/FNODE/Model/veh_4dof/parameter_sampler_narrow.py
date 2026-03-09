"""
Narrowed parameter sampler for 4DOF vehicle model - focusing on mid-size passenger vehicles
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import torch


class VehicleParameterSamplerNarrow:
    """
    Handles parameter sampling for 4DOF vehicle model with narrowed ranges
    Focused on mid-size passenger vehicles for better model convergence
    """

    # Option 1: Mid-size passenger car focus (e.g., Toyota Camry, Honda Accord)
    PARAM_RANGES_MIDSIZE = {
        'l': (2.7, 2.9),           # wheelbase (typical mid-size sedans)
        'r_wheel': (0.32, 0.35),   # wheel radius (15-17 inch wheels)
        'i_wheel': (0.9, 1.3),     # wheel inertia (mid-range)
        'tau0': (250, 350),        # max torque (typical 4-cyl to V6)
        'omega0': (1200, 1500),    # max engine speed (typical range)
        'c0': (0.015, 0.025),      # drag coefficient 0 (narrowed)
        'c1': (0.025, 0.045),      # drag coefficient 1 (narrowed)
        'delta': (0.60, 0.75),     # steering ratio (typical range)
        'gamma': (0.15, 0.25)      # power curve exponent (narrowed)
    }

    # Option 2: Compact car focus (e.g., Honda Civic, Toyota Corolla)
    PARAM_RANGES_COMPACT = {
        'l': (2.5, 2.7),           # wheelbase (compact cars)
        'r_wheel': (0.30, 0.33),   # wheel radius (14-16 inch wheels)
        'i_wheel': (0.7, 1.0),     # wheel inertia (lighter wheels)
        'tau0': (180, 250),        # max torque (smaller engines)
        'omega0': (1000, 1400),    # max engine speed
        'c0': (0.010, 0.020),      # drag coefficient 0
        'c1': (0.020, 0.035),      # drag coefficient 1
        'delta': (0.55, 0.70),     # steering ratio
        'gamma': (0.12, 0.20)      # power curve exponent
    }

    # Option 3: Performance car focus (e.g., BMW M3, Audi S4)
    PARAM_RANGES_PERFORMANCE = {
        'l': (2.8, 3.0),           # wheelbase (performance sedans)
        'r_wheel': (0.33, 0.37),   # wheel radius (17-19 inch wheels)
        'i_wheel': (1.1, 1.6),     # wheel inertia (heavier performance wheels)
        'tau0': (400, 550),        # max torque (turbo engines)
        'omega0': (1400, 1800),    # max engine speed (higher revving)
        'c0': (0.020, 0.030),      # drag coefficient 0
        'c1': (0.030, 0.050),      # drag coefficient 1
        'delta': (0.65, 0.80),     # steering ratio
        'gamma': (0.20, 0.30)      # power curve exponent
    }

    # Option 4: Very narrow range (±15% around nominal values)
    PARAM_RANGES_VERY_NARROW = {
        'l': (2.75, 2.85),         # wheelbase ±3.5% around 2.8m
        'r_wheel': (0.33, 0.35),   # wheel radius ±3% around 0.34m
        'i_wheel': (1.05, 1.15),   # wheel inertia ±4.5% around 1.1
        'tau0': (285, 315),        # max torque ±5% around 300
        'omega0': (1350, 1450),    # max engine speed ±3.5% around 1400
        'c0': (0.019, 0.021),      # drag coefficient 0 ±5% around 0.02
        'c1': (0.033, 0.037),      # drag coefficient 1 ±5% around 0.035
        'delta': (0.67, 0.73),     # steering ratio ±4% around 0.7
        'gamma': (0.19, 0.21)      # power curve exponent ±5% around 0.2
    }

    PARAM_ORDER = ['l', 'r_wheel', 'i_wheel', 'tau0', 'omega0', 'c0', 'c1', 'delta', 'gamma']

    def __init__(self, vehicle_type='midsize', num_samples_per_param=3):
        """
        Initialize parameter sampler with specified vehicle type

        Args:
            vehicle_type: 'midsize', 'compact', 'performance', or 'very_narrow'
            num_samples_per_param: Number of samples per parameter (default 3 for 3^9 = 19,683 combinations)
        """
        self.param_grid = None
        self.param_combinations = None
        self.num_samples_per_param = num_samples_per_param

        # Select parameter ranges based on vehicle type
        if vehicle_type == 'midsize':
            self.PARAM_RANGES = self.PARAM_RANGES_MIDSIZE
        elif vehicle_type == 'compact':
            self.PARAM_RANGES = self.PARAM_RANGES_COMPACT
        elif vehicle_type == 'performance':
            self.PARAM_RANGES = self.PARAM_RANGES_PERFORMANCE
        elif vehicle_type == 'very_narrow':
            self.PARAM_RANGES = self.PARAM_RANGES_VERY_NARROW
        else:
            raise ValueError(f"Unknown vehicle type: {vehicle_type}")

        self.vehicle_type = vehicle_type

    def generate_parameter_samples(self) -> Dict[str, np.ndarray]:
        """
        Generate uniformly spaced samples for each parameter

        Returns:
            Dictionary with parameter names as keys and sample arrays as values
        """
        param_samples = {}

        for param_name in self.PARAM_ORDER:
            min_val, max_val = self.PARAM_RANGES[param_name]
            # Generate uniformly spaced points
            samples = np.linspace(min_val, max_val, self.num_samples_per_param)
            param_samples[param_name] = samples

        return param_samples

    def generate_all_combinations(self) -> np.ndarray:
        """
        Generate all combinations of parameter samples

        Returns:
            Array of shape (N, 9) where N = num_samples_per_param^9
        """
        param_samples = self.generate_parameter_samples()

        # Create meshgrid for all combinations
        param_arrays = [param_samples[param] for param in self.PARAM_ORDER]
        param_grid = np.meshgrid(*param_arrays, indexing='ij')

        # Reshape to (N, 9) array
        param_combinations = np.stack([grid.flatten() for grid in param_grid], axis=1)

        self.param_grid = param_grid
        self.param_combinations = param_combinations

        print(f"Generated {len(param_combinations)} parameter combinations for {self.vehicle_type} vehicles")
        print(f"Shape: {param_combinations.shape}")

        return param_combinations

    def select_random_test_params(self, n_test: int, seed: Optional[int] = None) -> List[int]:
        """
        Select random parameter combinations for testing

        Args:
            n_test: Number of test combinations to select
            seed: Random seed for reproducibility

        Returns:
            List of indices for selected test parameter combinations
        """
        if self.param_combinations is None:
            raise ValueError("Must call generate_all_combinations() first")

        if seed is not None:
            np.random.seed(seed)

        n_total = len(self.param_combinations)
        test_indices = np.random.choice(n_total, size=min(n_test, n_total), replace=False)

        return test_indices.tolist()

    def get_param_combination(self, idx: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Get a specific parameter combination by index

        Args:
            idx: Index of the parameter combination

        Returns:
            Tuple of (parameter vector, parameter dictionary)
        """
        if self.param_combinations is None:
            raise ValueError("Must call generate_all_combinations() first")

        param_vec = self.param_combinations[idx]
        param_dict = {name: float(param_vec[i]) for i, name in enumerate(self.PARAM_ORDER)}

        return param_vec, param_dict

    def get_statistics(self):
        """
        Print statistics about the parameter ranges
        """
        print(f"\n=== Parameter Range Statistics for {self.vehicle_type.upper()} vehicles ===")
        print(f"Total combinations: {self.num_samples_per_param}^9 = {self.num_samples_per_param**9:,}")
        print("\nParameter ranges:")
        for param in self.PARAM_ORDER:
            min_val, max_val = self.PARAM_RANGES[param]
            range_width = max_val - min_val
            range_percent = (range_width / ((max_val + min_val) / 2)) * 100
            print(f"  {param:8s}: [{min_val:7.3f}, {max_val:7.3f}] (range: {range_width:.3f}, ±{range_percent/2:.1f}%)")


if __name__ == "__main__":
    # Example usage
    print("Comparing different parameter range options:\n")

    for vehicle_type in ['midsize', 'compact', 'performance', 'very_narrow']:
        sampler = VehicleParameterSamplerNarrow(vehicle_type=vehicle_type, num_samples_per_param=3)
        sampler.get_statistics()
        print()