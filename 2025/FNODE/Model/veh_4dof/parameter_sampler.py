"""
Parameter sampling utilities for 4DOF vehicle model
Handles generation of parameter grid and efficient data generation
"""

import numpy as np
import itertools
from typing import Dict, List, Tuple
import json
import os

class VehicleParameterSampler:
    """
    Handles parameter sampling for 4DOF vehicle model
    """

    # Parameter ranges focused on mid-size passenger vehicles
    # Narrowed ranges for better model convergence while keeping 4 samples per parameter
    PARAM_RANGES = {
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

    PARAM_ORDER = ['l', 'r_wheel', 'i_wheel', 'tau0', 'omega0', 'c0', 'c1', 'delta', 'gamma']
    NUM_SAMPLES_PER_PARAM = 4  # Changed from 5 to 4 for faster generation

    def __init__(self):
        self.param_grid = None
        self.param_combinations = None

    def generate_parameter_samples(self) -> Dict[str, np.ndarray]:
        """
        Generate uniformly spaced samples for each parameter

        Returns:
            Dictionary with parameter names as keys and sample arrays as values
        """
        param_samples = {}

        for param_name in self.PARAM_ORDER:
            min_val, max_val = self.PARAM_RANGES[param_name]
            # Generate 5 uniformly spaced points
            samples = np.linspace(min_val, max_val, self.NUM_SAMPLES_PER_PARAM)
            param_samples[param_name] = samples

        return param_samples

    def generate_all_combinations(self) -> np.ndarray:
        """
        Generate all possible parameter combinations using Cartesian product

        Returns:
            Array of shape (5^9, 9) containing all parameter combinations
        """
        param_samples = self.generate_parameter_samples()

        # Create list of sample arrays in the correct order
        sample_lists = [param_samples[param] for param in self.PARAM_ORDER]

        # Generate all combinations using itertools.product
        all_combinations = list(itertools.product(*sample_lists))

        # Convert to numpy array
        self.param_combinations = np.array(all_combinations)

        print(f"Generated {len(self.param_combinations)} parameter combinations")
        print(f"Shape: {self.param_combinations.shape}")

        return self.param_combinations

    def get_parameter_dict(self, param_vector: np.ndarray) -> Dict[str, float]:
        """
        Convert parameter vector to dictionary

        Args:
            param_vector: 9-element array of parameters

        Returns:
            Dictionary with parameter names and values
        """
        return {
            param_name: float(param_vector[i])
            for i, param_name in enumerate(self.PARAM_ORDER)
        }

    def save_parameter_info(self, save_dir: str):
        """
        Save parameter sampling information to JSON file

        Args:
            save_dir: Directory to save the parameter info
        """
        os.makedirs(save_dir, exist_ok=True)

        param_samples = self.generate_parameter_samples()

        # Convert numpy arrays to lists for JSON serialization
        param_info = {
            'param_ranges': self.PARAM_RANGES,
            'param_samples': {k: v.tolist() for k, v in param_samples.items()},
            'num_samples_per_param': self.NUM_SAMPLES_PER_PARAM,
            'total_combinations': self.NUM_SAMPLES_PER_PARAM ** len(self.PARAM_ORDER),
            'param_order': self.PARAM_ORDER
        }

        with open(os.path.join(save_dir, 'parameter_info.json'), 'w') as f:
            json.dump(param_info, f, indent=2)

    def select_random_test_params(self, n_test: int = 5, seed: int = 42) -> List[int]:
        """
        Randomly select parameter combination indices for testing

        Args:
            n_test: Number of test parameter combinations
            seed: Random seed for reproducibility

        Returns:
            List of indices for test parameter combinations
        """
        if self.param_combinations is None:
            self.generate_all_combinations()

        np.random.seed(seed)
        n_total = len(self.param_combinations)
        test_indices = np.random.choice(n_total, n_test, replace=False)

        return test_indices.tolist()

    def get_param_combination(self, index: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Get a specific parameter combination by index

        Args:
            index: Index of the parameter combination

        Returns:
            Tuple of (parameter vector, parameter dictionary)
        """
        if self.param_combinations is None:
            self.generate_all_combinations()

        param_vector = self.param_combinations[index]
        param_dict = self.get_parameter_dict(param_vector)

        return param_vector, param_dict


def test_sampler():
    """Test the parameter sampler"""
    sampler = VehicleParameterSampler()

    # Generate parameter samples
    param_samples = sampler.generate_parameter_samples()
    print("\nParameter samples:")
    for param, samples in param_samples.items():
        print(f"  {param}: {samples}")

    # Generate all combinations
    all_combos = sampler.generate_all_combinations()
    print(f"\nTotal combinations: {len(all_combos)}")
    print(f"Expected: {4**9} = {4**9}")

    # Test random selection
    test_indices = sampler.select_random_test_params(5)
    print(f"\nRandomly selected test indices: {test_indices}")

    # Show first test parameter combination
    param_vec, param_dict = sampler.get_param_combination(test_indices[0])
    print(f"\nFirst test parameter combination:")
    for param, value in param_dict.items():
        print(f"  {param}: {value:.4f}")


if __name__ == "__main__":
    test_sampler()