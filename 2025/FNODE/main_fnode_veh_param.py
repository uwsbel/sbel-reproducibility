#!/usr/bin/env python3
"""
Main training script for FNODE with parameterized vehicle control (veh_4dof)
This version handles 15D input (6 states/controls + 9 vehicle parameters)
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import sys
import logging
import h5py
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

# Setup logging
def setup_logging(log_file_path=None):
    """Setup logging with optional file output"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    if log_file_path:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file_path, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )

setup_logging()
logger = logging.getLogger("FNODE_VEH_PARAM")

# Ensure Model directory is in Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
model_package_dir = os.path.join(script_dir, 'Model')
if model_package_dir not in sys.path:
    logger.info(f"Adding Model package directory to sys.path: {model_package_dir}")
    sys.path.insert(0, model_package_dir)

# Import Model components
from Model.model import FNODE_CON
from Model.utils import save_model_state, load_model_state
from Model.veh_4dof.parameter_sampler import VehicleParameterSampler
from Model.veh_4dof.rom_vehicle_param import ParameterizedVehModel
from Model.veh_4dof.normalizer import VehicleDataNormalizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Online grouped normalization (states/controls/params/targets).
# We keep the HDF5 data in physical units to avoid any ambiguity, and normalize
# only at train/infer time.
normalizer = VehicleDataNormalizer()


class TorchVehicleDataNormalizer:
    """Torch/GPU implementation of VehicleDataNormalizer.

    Same semantics as VehicleDataNormalizer:
      - states: map each component to [-1, 1] using configured min/max
      - controls: alpha to [-1, 1], beta stays as-is
      - params: each to [-1, 1]
      - targets: each to [-1, 1]

    Designed to avoid per-batch CPU<->GPU copies.
    """

    def __init__(self, base: VehicleDataNormalizer, device: torch.device):
        self.device = device

        def _pair_to_tensor(r):
            return (
                torch.tensor(r[0], device=device, dtype=torch.float32),
                torch.tensor(r[1], device=device, dtype=torch.float32),
            )

        self.state_ranges = {k: _pair_to_tensor(v) for k, v in base.state_ranges.items()}
        self.control_ranges = {k: _pair_to_tensor(v) for k, v in base.control_ranges.items()}
        self.param_ranges = {k: _pair_to_tensor(v) for k, v in base.param_ranges.items()}
        self.target_ranges = {k: _pair_to_tensor(v) for k, v in base.target_ranges.items()}

        self.param_order = ['l', 'r_wheel', 'i_wheel', 'tau0', 'omega0', 'c0', 'c1', 'delta', 'gamma']

    @staticmethod
    def _normalize_value(x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor,
                         target_min: float = -1.0, target_max: float = 1.0) -> torch.Tensor:
        denom = (max_val - min_val)
        # avoid divide-by-zero
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        normalized_01 = (x - min_val) / denom
        return target_min + normalized_01 * (target_max - target_min)

    @staticmethod
    def _denormalize_value(x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor,
                           target_min: float = -1.0, target_max: float = 1.0) -> torch.Tensor:
        denom = (target_max - target_min)
        denom = denom if denom != 0 else 1.0
        normalized_01 = (x - target_min) / denom
        return min_val + normalized_01 * (max_val - min_val)

    def normalize_states(self, states: torch.Tensor) -> torch.Tensor:
        out = states.clone()
        out[:, 0] = self._normalize_value(states[:, 0], *self.state_ranges['x'])
        out[:, 1] = self._normalize_value(states[:, 1], *self.state_ranges['y'])
        out[:, 2] = self._normalize_value(states[:, 2], *self.state_ranges['v'])
        out[:, 3] = self._normalize_value(states[:, 3], *self.state_ranges['theta'])
        return out

    def normalize_controls(self, controls: torch.Tensor) -> torch.Tensor:
        out = controls.clone()
        out[:, 0] = self._normalize_value(controls[:, 0], *self.control_ranges['alpha'])
        out[:, 1] = controls[:, 1]
        return out

    def normalize_parameters(self, params: torch.Tensor) -> torch.Tensor:
        out = params.clone()
        for i, name in enumerate(self.param_order):
            out[:, i] = self._normalize_value(params[:, i], *self.param_ranges[name])
        return out

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        out = targets.clone()
        out[:, 0] = self._normalize_value(targets[:, 0], *self.target_ranges['v_dot'])
        out[:, 1] = self._normalize_value(targets[:, 1], *self.target_ranges['theta_dot'])
        return out

    def denormalize_targets(self, targets_n: torch.Tensor) -> torch.Tensor:
        out = targets_n.clone()
        out[:, 0] = self._denormalize_value(targets_n[:, 0], *self.target_ranges['v_dot'])
        out[:, 1] = self._denormalize_value(targets_n[:, 1], *self.target_ranges['theta_dot'])
        return out


def _adapt_state_dict_for_model(model: torch.nn.Module, state_dict: dict) -> dict:
    """Adapt checkpoint state_dict keys to match the current model.

    torch.compile wraps the model and prefixes all parameter/buffer keys with
    `_orig_mod.`. Depending on whether the checkpoint was saved from a compiled
    or non-compiled model, keys may or may not contain this prefix.

    This function adds/removes the prefix as needed by comparing against the
    current model's expected keys.
    """
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict

    try:
        model_keys = set(model.state_dict().keys())
    except Exception:
        return state_dict

    ckpt_keys = set(state_dict.keys())

    model_wants_orig = any(k.startswith('_orig_mod.') for k in model_keys)
    ckpt_has_orig = any(k.startswith('_orig_mod.') for k in ckpt_keys)

    # Case A: model is compiled (expects _orig_mod.*) but checkpoint is not.
    if model_wants_orig and not ckpt_has_orig:
        adapted = {f'_orig_mod.{k}': v for k, v in state_dict.items()}
        if set(adapted.keys()) & model_keys:
            return adapted
        return state_dict

    # Case B: model is not compiled but checkpoint is.
    if (not model_wants_orig) and ckpt_has_orig:
        adapted = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
        if set(adapted.keys()) & model_keys:
            return adapted
        return state_dict

    return state_dict


def _load_checkpoint_flexible(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    *,
    allow_partial: bool,
) -> bool:
    """Load checkpoint into model.

    - Handles compiled/non-compiled prefix via _adapt_state_dict_for_model.
    - If allow_partial=True, loads only keys that exist in the current model and
      have matching tensor shapes (useful when you changed architecture).

    Returns True if at least one tensor was loaded.
    Raises if checkpoint missing/invalid or if allow_partial=False and strict load fails.
    """

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Unwrap various checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        # very old format - serialized model object
        try:
            model.load_state_dict(checkpoint.state_dict())
            return True
        except Exception:
            raise RuntimeError("Unsupported checkpoint format (expected state_dict-like object)")

    # Back-compat: sometimes saved dict keys were stripped already
    if any(k.startswith('_orig_mod.') for k in list(state_dict.keys())[:5]):
        # don't eagerly strip; let the adapter decide
        pass

    state_dict = _adapt_state_dict_for_model(model, state_dict)

    if not allow_partial:
        # strict load (will raise on mismatch)
        model.load_state_dict(state_dict)
        return True

    # Partial load: keep only keys that exist and match shapes
    model_sd = model.state_dict()
    filtered = {}
    skipped_unexpected = 0
    skipped_mismatch = 0
    for k, v in state_dict.items():
        if k not in model_sd:
            skipped_unexpected += 1
            continue
        if hasattr(v, 'shape') and hasattr(model_sd[k], 'shape') and v.shape != model_sd[k].shape:
            skipped_mismatch += 1
            continue
        filtered[k] = v

    if not filtered:
        logger.warning(
            "Checkpoint '%s' did not contain any compatible tensors for the current model (unexpected=%d, shape_mismatch=%d).",
            checkpoint_path,
            skipped_unexpected,
            skipped_mismatch,
        )
        return False

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    logger.warning(
        "Loaded checkpoint partially: loaded=%d tensors, missing=%d, unexpected_filtered=%d, shape_mismatch_filtered=%d",
        len(filtered),
        len(missing) if isinstance(missing, (list, tuple)) else 0,
        skipped_unexpected,
        skipped_mismatch,
    )
    return True


class VehicleParameterDataset(Dataset):
    """Dataset for parameterized vehicle data stored in HDF5"""

    def __init__(self, h5_path, mode='train'):
        self.h5_path = h5_path
        self.mode = mode

        # Open HDF5 file to get metadata
        with h5py.File(h5_path, 'r') as h5f:
            self.n_samples = h5f[f'{mode}_data'].shape[0]
            self.n_params, self.steps_per_traj = _get_h5_param_metadata(h5f, mode, self.n_samples)

        logger.info(f"Loaded {mode} dataset: {self.n_samples:,} samples from {self.n_params} parameter combinations")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as h5f:
            # Get data and acceleration
            data = h5f[f'{self.mode}_data'][idx]  # Shape: (15,)
            accel = h5f[f'{self.mode}_accel'][idx]  # Shape: scalar

            # Split into components
            states_controls = data[:6]  # [x, y, v, theta, alpha, beta]
            parameters = data[6:]  # 9 vehicle parameters

            # Recompute targets (v_dot, theta_dot) using the model
            # For now, we'll use acceleration as v_dot
            # theta_dot needs to be computed from the vehicle dynamics
            v = states_controls[2]
            theta = states_controls[3]
            beta = states_controls[5]

            # Get parameters
            param_dict = {
                'l': parameters[0],
                'r_wheel': parameters[1],
                'i_wheel': parameters[2],
                'tau0': parameters[3],
                'omega0': parameters[4],
                'c0': parameters[5],
                'c1': parameters[6],
                'delta': parameters[7],
                'gamma': parameters[8]
            }

            # Compute theta_dot from vehicle kinematics
            theta_dot = v * np.tan(beta * param_dict['delta']) / param_dict['l']

            # NOTE: This dataset is stored in physical units (no normalization).
            # Create input tensor (15D) and target tensor (2D): [v_dot, theta_dot]
            input_tensor = torch.tensor(data, dtype=torch.float16)
            target_tensor = torch.tensor([accel, theta_dot], dtype=torch.float16)

            return input_tensor, target_tensor


class VehicleParameterDatasetGPU:
    """GPU-based dataset that loads ALL data to GPU memory for fast training"""

    def __init__(self, h5_path, mode='train', device='cuda', max_gb=12.0):
        self.device = device
        self.mode = mode

        logger.info(f"Loading {mode} dataset to GPU memory (max {max_gb} GB)...")

        with h5py.File(h5_path, 'r') as h5f:
            # Check total size and determine how much we can load
            total_samples = h5f[f'{mode}_data'].shape[0]

            # Calculate memory required per sample (15 inputs + 2 targets) * 4 bytes for float32
            bytes_per_sample = 17 * 4  # 68 bytes per sample
            max_samples = int((max_gb * 1024**3) / bytes_per_sample)

            # Load only what fits
            samples_to_load = min(total_samples, max_samples)

            if samples_to_load < total_samples:
                logger.warning(f"GPU memory limited: loading {samples_to_load:,} / {total_samples:,} samples")

            # Load data
            data = h5f[f'{mode}_data'][:samples_to_load]
            accel = h5f[f'{mode}_accel'][:samples_to_load]
            self.n_samples = samples_to_load

            logger.info(f"Processing {self.n_samples:,} samples...")

            # Convert to float16 numpy arrays for memory efficiency
            data = data.astype(np.float16)
            accel = accel.astype(np.float16)

            # Extract inputs (physical units)
            inputs = data[:, :15]

            # Compute theta_dot for all samples from PHYSICAL input columns
            # input layout: [x, y, v, theta, alpha, beta] + 9 params where
            # params[0]=l and params[7]=delta.
            v = data[:, 2]
            beta = data[:, 5]
            l = data[:, 6]       # param l
            delta = data[:, 13]  # param delta
            theta_dot = v * np.tan(beta * delta) / l

            # Create targets in physical units
            targets = np.stack([accel, theta_dot], axis=1).astype(np.float16)

            # Move everything to GPU as float16 to save memory
            logger.info(f"Moving data to {device}...")
            self.inputs = torch.from_numpy(inputs).to(device, dtype=torch.float16)
            self.targets = torch.from_numpy(targets).to(device, dtype=torch.float16)

            # Clear CPU memory
            del data, accel, inputs, targets

        if device == 'cuda':
            torch.cuda.synchronize()
            memory_gb = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"GPU memory used: {memory_gb:.2f} GB")
            logger.info(f"Loaded {self.n_samples:,} samples to GPU")

    def __len__(self):
        return self.n_samples

    def get_batch(self, indices):
        """Get batch by indices - already on GPU"""
        return self.inputs[indices], self.targets[indices]

    def random_batch(self, batch_size):
        """Get random batch - already on GPU"""
        # Generate random indices directly on GPU for better performance
        indices = torch.randint(0, self.n_samples, (batch_size,), device=self.device)
        return self.inputs[indices], self.targets[indices]

    def get_epoch_iterator(self, batch_size):
        """Get an iterator for shuffled batches for one epoch"""
        # Pre-shuffle indices for the entire epoch
        shuffled_indices = torch.randperm(self.n_samples, device=self.device)
        n_batches = (self.n_samples + batch_size - 1) // batch_size

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, self.n_samples)
            batch_indices = shuffled_indices[start_idx:end_idx]
            yield self.inputs[batch_indices], self.targets[batch_indices]


class VehicleParameterDatasetOptimized(Dataset):
    """Optimized CPU-based dataset for safe memory management"""

    def __init__(self, h5_path, mode='train', max_samples=None):
        self.device = device
        self.mode = mode

        # Check dataset size first
        with h5py.File(h5_path, 'r') as h5f:
            total_samples = h5f[f'{mode}_data'].shape[0]
            logger.info(f"{mode} dataset has {total_samples:,} samples total")

            # Determine how many samples to load
            if max_samples is not None and max_samples < total_samples:
                samples_to_load = max_samples
                logger.info(f"Loading {samples_to_load:,} samples (limited by memory)")
            else:
                samples_to_load = total_samples

            # Load data in chunks to avoid memory spikes
            chunk_size = min(500000, samples_to_load)  # Process 500k samples at a time

            logger.info(f"Loading {mode} dataset...")

            # Initialize lists to collect data
            data_chunks = []
            accel_chunks = []

            for start_idx in range(0, samples_to_load, chunk_size):
                end_idx = min(start_idx + chunk_size, samples_to_load)
                data_chunks.append(h5f[f'{mode}_data'][start_idx:end_idx])
                accel_chunks.append(h5f[f'{mode}_accel'][start_idx:end_idx])

                if len(data_chunks) % 5 == 0:  # Progress update every 5 chunks
                    logger.info(f"  Loaded {end_idx:,} / {samples_to_load:,} samples")

            # Concatenate all chunks
            data = np.concatenate(data_chunks, axis=0) if len(data_chunks) > 1 else data_chunks[0]
            accel = np.concatenate(accel_chunks, axis=0) if len(accel_chunks) > 1 else accel_chunks[0]

            self.n_samples = len(data)

            # Clear chunks from memory
            del data_chunks, accel_chunks

            # Process and move to device
            logger.info(f"Processing and moving to {device}...")

            # Extract inputs - move to device in smaller batches if needed
            if device.type == 'cuda' and self.n_samples > 1000000:
                # For large datasets, move to GPU in batches
                inputs_list = []
                targets_list = []

                batch_size = 500000
                for i in range(0, self.n_samples, batch_size):
                    end_i = min(i + batch_size, self.n_samples)
                    batch_data = data[i:end_i]
                    batch_accel = accel[i:end_i]

                    # Compute theta_dot
                    v = batch_data[:, 2]
                    beta = batch_data[:, 5]
                    l = batch_data[:, 6]
                    delta = batch_data[:, 13]
                    theta_dot = v * np.tan(beta * delta) / l

                    # Create tensors
                    inputs_list.append(torch.tensor(batch_data[:, :15], dtype=torch.float16, device=device))
                    targets_list.append(torch.tensor(np.stack([batch_accel, theta_dot], axis=1), dtype=torch.float16, device=device))

                self.inputs = torch.cat(inputs_list, dim=0)
                self.targets = torch.cat(targets_list, dim=0)
            else:
                # Small dataset - process all at once
                v = data[:, 2]
                beta = data[:, 5]
                l = data[:, 6]
                delta = data[:, 13]
                theta_dot = v * np.tan(beta * delta) / l

                self.inputs = torch.tensor(data[:, :15], dtype=torch.float16, device=device)
                targets = np.stack([accel, theta_dot], axis=1)
                self.targets = torch.tensor(targets, dtype=torch.float16, device=device)

            # Clear numpy arrays
            del data, accel

        logger.info(f"Loaded {self.n_samples:,} samples to {device}")
        if device.type == 'cuda':
            memory_gb = torch.cuda.memory_allocated(device) / 1024**3
            logger.info(f"GPU memory used: {memory_gb:.2f} GB")

    def get_batch(self, indices):
        """Get a batch of data by indices"""
        return self.inputs[indices], self.targets[indices]

    def random_batch(self, batch_size):
        """Get a random batch of data"""
        indices = torch.randperm(self.n_samples, device=self.device)[:batch_size]
        return self.get_batch(indices)

    def __len__(self):
        return self.n_samples


class VehicleParameterDatasetEfficient(Dataset):
    """Efficient dataset with optimized HDF5 reading and chunk caching"""

    def __init__(self, h5_path, mode='train', chunk_size=100000):
        self.h5_path = h5_path
        self.mode = mode
        self.chunk_size = chunk_size

        # Open HDF5 file once and keep it open
        self.h5_file = h5py.File(h5_path, 'r', swmr=True)  # swmr=True for thread safety
        self.data_dset = self.h5_file[f'{mode}_data']
        self.accel_dset = self.h5_file[f'{mode}_accel']

        # Get dataset size
        self.n_samples = self.data_dset.shape[0]

        # Get metadata for compatibility
        self.n_params, self.steps_per_traj = _get_h5_param_metadata(self.h5_file, mode, self.n_samples)

        # Initialize chunk cache
        self.current_chunk_start = -1
        self.current_chunk_data = None
        self.current_chunk_accel = None

        logger.info(f"Loaded {mode} dataset: {self.n_samples:,} samples, chunk_size={chunk_size:,}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Calculate which chunk this index belongs to
        chunk_start = (idx // self.chunk_size) * self.chunk_size

        # Load new chunk if necessary
        if chunk_start != self.current_chunk_start:
            chunk_end = min(chunk_start + self.chunk_size, self.n_samples)
            # Efficient contiguous read from HDF5
            self.current_chunk_data = self.data_dset[chunk_start:chunk_end]
            self.current_chunk_accel = self.accel_dset[chunk_start:chunk_end]
            self.current_chunk_start = chunk_start

        # Get data from cached chunk
        local_idx = idx - self.current_chunk_start
        data = self.current_chunk_data[local_idx]
        accel = self.current_chunk_accel[local_idx]

        # Extract inputs (first 15 dimensions)
        inputs = data[:15].astype(np.float16)

        # Compute theta_dot from vehicle kinematics
        v = data[2]  # velocity
        beta = data[5]  # steering angle
        l = data[6]  # wheelbase (first parameter)
        delta = data[13]  # steering ratio (8th parameter)
        theta_dot = v * np.tan(beta * delta) / l

        # Create targets
        targets = np.array([accel, theta_dot], dtype=np.float16)

        # Return as tensors (DataLoader will handle device transfer)
        return torch.from_numpy(inputs), torch.from_numpy(targets)

    def __del__(self):
        """Clean up HDF5 file handle"""
        if hasattr(self, 'h5_file') and self.h5_file:
            try:
                self.h5_file.close()
            except:
                pass


def _get_h5_param_metadata(h5f: h5py.File, mode: str, n_samples: int) -> tuple[int, int]:
    """
    Backward/forward-compatible metadata reader for parameterized vehicle HDF5 files.

    Older/newer dataset generators may not populate file-level attributes.
    We infer:
      - n_params from dataset 'parameter_combinations' if present, else from attrs.
      - steps_per_traj from dataset 'time_{mode}' if present, else from attrs.
    """
    # Prefer explicit attributes if they exist
    n_params = h5f.attrs.get('n_parameters', None)
    steps_per_traj = h5f.attrs.get(f'{mode}_steps', None)

    # Infer n_params from parameter_combinations dataset if needed
    if n_params is None:
        if 'parameter_combinations' in h5f:
            n_params = int(h5f['parameter_combinations'].shape[0])
            logger.warning("HDF5 missing attr 'n_parameters'; inferred n_params=%d from 'parameter_combinations'", n_params)
        else:
            n_params = 1
            logger.warning("HDF5 missing attr 'n_parameters' and dataset 'parameter_combinations'; defaulting n_params=1")
    else:
        n_params = int(n_params)

    # Infer steps_per_traj from time array dataset if needed
    if steps_per_traj is None:
        time_key = f'time_{mode}'
        if time_key in h5f:
            steps_per_traj = int(h5f[time_key].shape[0])
            logger.warning("HDF5 missing attr '%s'; inferred steps_per_traj=%d from '%s'", f'{mode}_steps', steps_per_traj, time_key)
        else:
            # As a last resort, infer from n_samples / n_params when divisible.
            if n_params > 0 and n_samples % n_params == 0:
                steps_per_traj = int(n_samples // n_params)
                logger.warning(
                    "HDF5 missing attr '%s' and dataset '%s'; inferred steps_per_traj=%d from n_samples/n_params",
                    f'{mode}_steps',
                    time_key,
                    steps_per_traj,
                )
            else:
                steps_per_traj = 0
                logger.warning("Could not infer steps_per_traj (n_samples=%d, n_params=%d); setting steps_per_traj=0", n_samples, n_params)
    else:
        steps_per_traj = int(steps_per_traj)

    return n_params, steps_per_traj


def parse_arguments():
    parser = argparse.ArgumentParser(description="FNODE training for parameterized vehicle")

    # Basic parameters
    parser.add_argument('--generate_data', action='store_true', default=False,
                       help='Generate new dataset instead of using existing')
    parser.add_argument('--data_path', type=str, default='dataset/veh_4dof_param/vehicle_data.h5',
                       help='Path to HDF5 data file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_gpu_generation', action='store_true', default=True,
                       help='Use GPU for data generation if available')
    parser.add_argument('--gpu_batch_size', type=int, default=100000,
                       help='Batch size for GPU data generation (adjust based on GPU memory)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help="Computation device")

    # Model architecture
    parser.add_argument('--layers', type=int, default=6,
                       help='Number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='Hidden layer width')
    parser.add_argument('--activation', type=str, default='tanh',
                       choices=['relu', 'tanh'],
                       help='Activation function')
    parser.add_argument('--initializer', type=str, default='xavier',
                       choices=['xavier', 'kaiming'],
                       help='Weight initialization')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')    
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Mini-batch size for training (default: 65536 for better GPU utilization)')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw'],
                       help='Optimizer type')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')

    # Learning rate scheduler
    parser.add_argument('--lr_scheduler', type=str, default='exponential',
                       choices=['none', 'exponential', 'step', 'cosine'],
                       help='Learning rate scheduler')
    parser.add_argument('--lr_decay_rate', type=float, default=0.98,
                       help='LR decay rate for exponential scheduler')

    # Other parameters
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--skip_train', action='store_true', default=False,
                       help='Skip training and only test')
    parser.add_argument('--model_load_filename', type=str, default=None,
                       help='Path to load model checkpoint')
    parser.add_argument('--gpu_memory_gb', type=float, default=20.0,
                       help='Maximum GPU memory to use for dataset (in GB)')

    return parser.parse_args()


def train_fnode_param(model, train_dataset, args):
    """Train FNODE_CON model with entire dataset loaded on GPU"""

    logger.info("Training with GPU dataset (all data in GPU memory)")

    # Fast GPU-side normalizer (no CPU/NumPy round-trips)
    torch_normalizer = TorchVehicleDataNormalizer(normalizer, device)

    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Setup scheduler
    if args.lr_scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    # Loss function
    # We train in normalized target space. This is equivalent to per-dimension
    # std/scale normalization in the loss and prevents one target component
    # from dominating the gradients.
    criterion = nn.MSELoss()

    # Setup GradScaler for mixed precision training
    scaler = GradScaler('cuda')
    logger.info("Using mixed precision training with GradScaler")

    # Training loop
    loss_history = []
    best_train_loss = float('inf')
    best_epoch = 0

    # Create save directory
    save_dir = os.path.join('saved_model', 'veh_4dof_param')
    os.makedirs(save_dir, exist_ok=True)

    # Calculate number of batches
    n_batches = (len(train_dataset) + args.batch_size - 1) // args.batch_size

    logger.info("Starting GPU-based training...")
    logger.info(f"Total samples: {len(train_dataset):,}, Batch size: {args.batch_size}, Batches per epoch: {n_batches}")
    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()

        # Training
        epoch_loss = 0.0

        # Create progress bar with less frequent updates
        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx in pbar:
            # Get random batch directly from GPU dataset (already in fp16)
            inputs, targets = train_dataset.random_batch(args.batch_size)

            # inputs are in PHYSICAL units: [x, y, v, theta, alpha, beta] + 9 params
            # targets are in PHYSICAL units: [v_dot, theta_dot]
            # Apply grouped normalization online on GPU.
            inputs_f = inputs.float()
            targets_f = targets.float()

            states_n = torch_normalizer.normalize_states(inputs_f[:, 0:4])
            controls_n = torch_normalizer.normalize_controls(inputs_f[:, 4:6])
            params_n = torch_normalizer.normalize_parameters(inputs_f[:, 6:15])
            inputs_n = torch.cat([states_n, controls_n, params_n], dim=1)

            targets_n = torch_normalizer.normalize_targets(targets_f)

            # Forward pass with mixed precision
            # autocast will handle fp16 conversion internally
            with autocast('cuda', dtype=torch.float16):
                predictions = model(inputs_n)  # model runs in fp32 under autocast
                loss = criterion(predictions, targets_n)

            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()

            # Update statistics
            batch_loss = loss.item()
            epoch_loss += batch_loss

            # Update progress bar less frequently (every 100 batches)
            if batch_idx % 100 == 0:
                pbar.set_postfix({'loss': batch_loss})

        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / n_batches
        loss_history.append(avg_epoch_loss)

        # Save best model based on training loss
        if avg_epoch_loss < best_train_loss:
            best_train_loss = avg_epoch_loss  # Using training loss as the criterion
            best_epoch = epoch
            save_model_state(model, save_dir, 'FNODE_best.pkl')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_epoch_loss
            }, os.path.join(save_dir, 'checkpoint_best.pth'))

        # Save checkpoint periodically
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_epoch_loss
            }, checkpoint_path)

        # Learning rate scheduling
        if scheduler:
            scheduler.step()

        # Print epoch summary
        logger.info(f"Epoch {epoch+1:3d}: Train Loss: {avg_epoch_loss:.6f}, "
                   f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save final model
    save_model_state(model, save_dir, 'FNODE_final.pkl')

    end_time = time.perf_counter()
    total_time = end_time - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"Best training loss: {best_train_loss:.6f} at epoch {best_epoch+1}")

    return loss_history


def test_parameter_combinations(model, h5_path, output_dir, dt=0.01, num_test=4):
    """Test model on selected parameter combinations.

    We use 4 test trajectories:
      - first 2: parameter combinations that are already in the training set
      - last  2: parameters within the allowed ranges but NOT on the training grid (interpolation)
    """

    # Initialize parameter sampler (defines ranges/order)
    sampler = VehicleParameterSampler()

    # Load parameter combinations that exist in the generated dataset (train/test split)
    with h5py.File(h5_path, 'r') as h5f:
        if 'parameter_combinations' not in h5f:
            raise KeyError(
                "HDF5 file does not contain 'parameter_combinations'. "
                "Please regenerate the dataset with the updated generator.")
        dataset_param_combos = np.array(h5f['parameter_combinations'][:])
        gen_train_num_steps = int(h5f.attrs.get('gen_train_num_steps', 1500))
        num_steps_in_h5 = int(h5f.attrs.get('num_steps', 2000))

    # Infer how many (param_combo, time_step) samples are in the train split.
    # Data_generator flattens trajectories by time; for each parameter combination we have gen_train_num_steps samples.
    with h5py.File(h5_path, 'r') as h5f:
        train_samples = int(h5f['train_data'].shape[0])
    combos_in_train = max(1, train_samples // max(1, gen_train_num_steps))

    # --- 1) pick two FROM training parameter combos (spread out for diversity) ---
    n_train_combos = min(combos_in_train, len(dataset_param_combos))
    if n_train_combos < 2:
        raise ValueError(
            f"Not enough parameter combinations in training set. "
            f"Need >=2, got {n_train_combos}")

    train_param_indices = [0, n_train_combos - 1]
    selected_train_params = [dataset_param_combos[idx] for idx in train_param_indices]

    # --- 2) make two off-grid params (within ranges, deliberately diverse) ---
    def _make_offgrid_diverse(which: int) -> np.ndarray:
        """Construct a param vector inside range but off the linspace grid.

        which=0 -> near lower corner, which=1 -> near upper corner, with
        alternating per-dimension jitter to avoid coinciding with the grid.
        """
        base_frac = 0.20 if which == 0 else 0.80
        jitter_frac = 0.07
        vec = np.zeros(len(sampler.PARAM_ORDER), dtype=np.float64)
        for j, name in enumerate(sampler.PARAM_ORDER):
            pmin, pmax = sampler.PARAM_RANGES[name]
            span = float(pmax - pmin)
            if span <= 0:
                vec[j] = float(pmin)
                continue
            sign = -1.0 if (j % 2 == 0) else 1.0
            v = float(pmin) + base_frac * span + sign * jitter_frac * span
            vec[j] = np.clip(v, pmin, pmax)
        return vec.astype(np.float32)

    interp_param_1 = _make_offgrid_diverse(0)
    interp_param_2 = _make_offgrid_diverse(1)

    # If by chance an interpolated point exactly matches an existing grid point, tweak one dimension.
    def _ensure_not_in_grid(p: np.ndarray) -> np.ndarray:
        if np.any(np.all(np.isclose(dataset_param_combos, p[None, :]), axis=1)):
            # nudge l slightly (still in range)
            pmin, pmax = sampler.PARAM_RANGES['l']
            p = p.copy()
            p[0] = np.clip(p[0] + 0.011 * (pmax - pmin), pmin, pmax)
        return p

    interp_param_1 = _ensure_not_in_grid(interp_param_1)
    interp_param_2 = _ensure_not_in_grid(interp_param_2)

    selected_params = selected_train_params + [interp_param_1, interp_param_2]

    # Keep the old signature behavior: num_test controls how many we run
    selected_params = selected_params[:num_test]

    # Create output directory
    results_dir = os.path.join(output_dir, 'test_results')
    os.makedirs(results_dir, exist_ok=True)

    # Store all test trajectories for combined plot
    all_gt_trajectories = []
    all_pred_trajectories = []
    all_param_dicts = []
    all_errors = []

    model.eval()

    for i, param_vec in enumerate(selected_params):
        # Determine whether this is from-training or interpolation.
        in_grid = np.any(np.all(np.isclose(dataset_param_combos, param_vec[None, :]), axis=1))
        src_tag = "train_grid" if (i < 2 and in_grid) else ("interp" if not in_grid else "grid")
        logger.info(f"Testing parameter set {i+1}/{num_test} ({src_tag})")

        # Build param_dict
        param_dict = sampler.get_parameter_dict(param_vec)

        # Create test directory
        test_dir = os.path.join(results_dir, f'param_set_{i+1}_{src_tag}')
        os.makedirs(test_dir, exist_ok=True)

        # Generate ground truth trajectory using ParameterizedVehModel
        from Model.veh_4dof.rom_vehicle_param import ParameterizedVehModel

        # Generate control inputs MATCHING the training data exactly
        num_steps = 2000
        t_array = np.linspace(0, num_steps * dt, num_steps)

        # Use EXACTLY the same control generation as training data
        alpha = np.zeros(num_steps)
        beta = np.zeros(num_steps)

        accel_time = 5.0
        for j, t in enumerate(t_array):  # Changed i to j to avoid variable name conflict
            if t <= accel_time:
                # Phase 1: Straight-line acceleration (same as training)
                ramp_time = 4.0
                s = np.clip(t / ramp_time, 0.0, 1.0)
                alpha[j] = 0.25 * (1.0 - np.cos(np.pi * s))  # Smooth ramp to 0.5
                beta[j] = 0.0  # No steering
            else:
                # Phase 2: S-curve with deceleration (same as training)
                t_s = t - accel_time
                alpha[j] = 0.5 * np.exp(-t_s / 10.0)  # Exponential decay
                # Steering: sinusoidal S-curve with 0.3 Hz frequency
                amplitude = 0.4
                frequency = 0.3
                beta[j] = amplitude * np.sin(2 * np.pi * frequency * t_s)

        # Initialize vehicle model
        initial_state = [0, 0, 0, 5]  # [x, y, theta, v]
        control = [alpha[0], beta[0]]
        vehicle = ParameterizedVehModel(None, initial_state, control, dt, False, param_dict)

        # Generate trajectory
        gt_trajectory = np.zeros((num_steps, 6))
        gt_accel = np.zeros(num_steps)

        for j in range(num_steps):  # Changed i to j to avoid variable name conflict
            control = [alpha[j], beta[j]]
            vehicle.update(control)
            state = vehicle.get_state()
            gt_trajectory[j] = [state[0], state[1], state[3], state[2], alpha[j], beta[j]]  # [x, y, v, theta, alpha, beta]
            gt_accel[j] = state[4]  # acceleration

        # Integrate trajectory using model
        initial_state = gt_trajectory[0, :4]  # [x, y, v, theta]
        controls = gt_trajectory[:, 4:6]  # [alpha, beta]

        pred_trajectory = integrate_trajectory_param(
            model, initial_state, controls, param_vec, dt, len(gt_trajectory)
        )

        # Debug: Check if predictions are different for different parameters
        logger.info(f"  Param l={param_dict['l']:.3f}, tau0={param_dict['tau0']:.1f}, gamma={param_dict['gamma']:.3f}")
        logger.info(f"  GT final pos: ({gt_trajectory[-1, 0]:.2f}, {gt_trajectory[-1, 1]:.2f})")
        logger.info(f"  Pred final pos: ({pred_trajectory[-1, 0]:.2f}, {pred_trajectory[-1, 1]:.2f})")

        # Save results
        np.save(os.path.join(test_dir, 'ground_truth.npy'), gt_trajectory)
        np.save(os.path.join(test_dir, 'prediction.npy'), pred_trajectory)

        # Save parameters
        with open(os.path.join(test_dir, 'parameters.json'), 'w') as f:
            json.dump(param_dict, f, indent=2)

        # Create comparison plot
        create_comparison_plot(gt_trajectory, pred_trajectory, param_dict, test_dir, dt=dt, id_num_steps=gen_train_num_steps)

        # Calculate errors
        position_error = np.linalg.norm(
            pred_trajectory[-1, :2] - gt_trajectory[-1, :2]
        )
        logger.info(f"  Final position error: {position_error:.3f} m")

        # Store for combined plot
        all_gt_trajectories.append(gt_trajectory)
        all_pred_trajectories.append(pred_trajectory)
        all_param_dicts.append(param_dict)
        all_errors.append(position_error)

    # Create combined plot with all 5 test trajectories
    create_all_trajectories_plot(all_gt_trajectories, all_pred_trajectories,
                                  all_param_dicts, all_errors, results_dir, dt=dt, id_num_steps=gen_train_num_steps)


def integrate_trajectory_param(model, initial_state, controls, param_vec, dt, num_steps):
    """
    Integrate vehicle trajectory using RK4 with parameterized model
    NOTE: HDF5 stores PHYSICAL units; model is trained in NORMALIZED input/target space.

    Args:
    model: Trained FNODE_CON model (15D input, normalized)
        initial_state: [x, y, v, theta] initial state (physical units)
        controls: Array of [alpha, beta] control inputs (physical units)
        param_vec: 9-element parameter vector (physical units)
        dt: Time step
        num_steps: Number of integration steps

    Returns:
        trajectory: Integrated state trajectory (physical units)
    """
    model.eval()

    # Fast GPU-side normalizer
    torch_normalizer = TorchVehicleDataNormalizer(normalizer, device)

    # Debug: Log parameters being used for this trajectory
    if not hasattr(integrate_trajectory_param, 'logged_params'):
        integrate_trajectory_param.logged_params = set()

    param_tuple = tuple(param_vec)
    if param_tuple not in integrate_trajectory_param.logged_params:
        logger.info(f"  integrate_trajectory_param using params: {param_vec[:3]}... (showing first 3)")
        integrate_trajectory_param.logged_params.add(param_tuple)

    def rk4_step(state, control, dt):
        """Single RK4 step for vehicle dynamics"""

        def get_derivative(s):
            # Build 15D PHYSICAL model input
            model_input_phys = np.concatenate([s, control, param_vec]).astype(np.float32)

            # Group-normalize to match training (torch, on GPU)
            s_t = torch.tensor(model_input_phys[0:4], dtype=torch.float32, device=device).unsqueeze(0)
            u_t = torch.tensor(model_input_phys[4:6], dtype=torch.float32, device=device).unsqueeze(0)
            p_t = torch.tensor(model_input_phys[6:15], dtype=torch.float32, device=device).unsqueeze(0)
            model_input_n = torch.cat([
                torch_normalizer.normalize_states(s_t),
                torch_normalizer.normalize_controls(u_t),
                torch_normalizer.normalize_parameters(p_t),
            ], dim=1)

            with torch.no_grad():
                pred_n = model(model_input_n).squeeze(0)

            # Model outputs NORMALIZED [v_dot, theta_dot] -> denormalize to physical for integration
            pred_phys = torch_normalizer.denormalize_targets(pred_n.unsqueeze(0)).squeeze(0)
            v_dot, theta_dot = float(pred_phys[0].item()), float(pred_phys[1].item())

            if not hasattr(get_derivative, 'debug_counter'):
                get_derivative.debug_counter = 0
            if get_derivative.debug_counter < 3:
                logger.debug(f"  Model input (norm): {model_input_n.squeeze(0)[:6].detach().cpu().numpy()}")  # Show first 6 values
                logger.debug(f"  Model output (phys): v_dot={v_dot:.3f}, theta_dot={theta_dot:.3f}")
                get_derivative.debug_counter += 1

            # Compute x_dot and y_dot from kinematics (physical units)
            x, y, v, theta = s
            x_dot = v * np.cos(theta)
            y_dot = v * np.sin(theta)

            return np.array([x_dot, y_dot, v_dot, theta_dot])

        # RK4 stages
        k1 = get_derivative(state)
        k2 = get_derivative(state + 0.5 * dt * k1)
        k3 = get_derivative(state + 0.5 * dt * k2)
        k4 = get_derivative(state + dt * k3)

        # Combine stages
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Integrate trajectory
    trajectory = np.zeros((num_steps, 6))
    current_state = initial_state

    for i in range(num_steps):
        control = controls[min(i, len(controls) - 1)]

        # Store current state and control (physical units)
        trajectory[i, :4] = current_state
        trajectory[i, 4:6] = control

        if i < num_steps - 1:
            current_state = rk4_step(current_state, control, dt)

    return trajectory


def create_comparison_plot(gt_trajectory, pred_trajectory, param_dict, save_dir, dt, id_num_steps=None):
    """Create comparison plot for a parameter combination"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Parameter Test - l={param_dict["l"]:.2f}, tau0={param_dict["tau0"]:.1f}, gamma={param_dict["gamma"]:.3f}')

    # Trajectory comparison
    time_steps = np.arange(len(gt_trajectory)) * float(dt)
    if id_num_steps is None:
        split_idx = len(time_steps)
    else:
        split_idx = max(0, min(int(id_num_steps), len(time_steps)))

    axes[0, 0].plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b-', label='Ground Truth', linewidth=2)
    if split_idx > 0:
        axes[0, 0].plot(pred_trajectory[:split_idx, 0], pred_trajectory[:split_idx, 1], 'r--', label='Pred (ID)', linewidth=2)
    if split_idx < len(pred_trajectory):
        start_idx = max(split_idx - 1, 0)
        axes[0, 0].plot(pred_trajectory[start_idx:, 0], pred_trajectory[start_idx:, 1], 'r:', label='Pred (OOD)', linewidth=2)
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].set_title('Vehicle Trajectory')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')

    # Velocity comparison
    axes[0, 1].plot(time_steps, gt_trajectory[:, 2], 'b-', label='Ground Truth')
    if split_idx > 0:
        axes[0, 1].plot(time_steps[:split_idx], pred_trajectory[:split_idx, 2], 'r--', label='Pred (ID)')
    if split_idx < len(pred_trajectory):
        start_idx = max(split_idx - 1, 0)
        axes[0, 1].plot(time_steps[start_idx:], pred_trajectory[start_idx:, 2], 'r:', label='Pred (OOD)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Velocity Profile')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Heading angle comparison
    axes[0, 2].plot(time_steps, gt_trajectory[:, 3], 'b-', label='Ground Truth')
    if split_idx > 0:
        axes[0, 2].plot(time_steps[:split_idx], pred_trajectory[:split_idx, 3], 'r--', label='Pred (ID)')
    if split_idx < len(pred_trajectory):
        start_idx = max(split_idx - 1, 0)
        axes[0, 2].plot(time_steps[start_idx:], pred_trajectory[start_idx:, 3], 'r:', label='Pred (OOD)')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Theta (rad)')
    axes[0, 2].set_title('Heading Angle')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # MSE (position) over time
    traj_mse = np.mean((pred_trajectory[:, :2] - gt_trajectory[:, :2]) ** 2, axis=1)
    axes[1, 0].plot(time_steps, traj_mse, 'k-')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('MSE Over Time')
    axes[1, 0].grid(True, alpha=0.3)

    # Velocity error
    velocity_error = np.abs(pred_trajectory[:, 2] - gt_trajectory[:, 2])
    axes[1, 1].plot(time_steps, velocity_error, 'k-')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Velocity Error (m/s)')
    axes[1, 1].set_title('Velocity Error Over Time')
    axes[1, 1].grid(True, alpha=0.3)

    # Control inputs
    axes[1, 2].plot(time_steps, gt_trajectory[:, 4], 'g-', label='Alpha (throttle)')
    axes[1, 2].plot(time_steps, gt_trajectory[:, 5], 'b-', label='Beta (steering)')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Control Input')
    axes[1, 2].set_title('Control Inputs')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_plot.png'), dpi=150)
    plt.close()


def create_all_trajectories_plot(all_gt_trajectories, all_pred_trajectories,
                                  all_param_dicts, all_errors, save_dir, dt, id_num_steps=None):
    """Create a combined plot showing all 5 test trajectories"""

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], wspace=0.25, hspace=0.30)
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_vel = fig.add_subplot(gs[0, 1])
    ax_head = fig.add_subplot(gs[1, 1])
    fig.suptitle('All Test Trajectories Comparison', fontsize=16, fontweight='bold')

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    time_steps = np.arange(len(all_gt_trajectories[0])) * float(dt)
    if id_num_steps is None:
        split_idx = len(time_steps)
    else:
        split_idx = max(0, min(int(id_num_steps), len(time_steps)))

    # Plot 1: All trajectories on same plot (left column)
    ax = ax_traj
    for i in range(len(all_gt_trajectories)):
        gt_traj = all_gt_trajectories[i]
        pred_traj = all_pred_trajectories[i]

        # Plot ground truth with solid line
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], '-', color=colors[i],
            label=f'Ground Truth {i+1}', linewidth=2, alpha=0.7)
        # Plot prediction split into ID (first 15s) and OOD (remaining)
        if split_idx > 0:
            ax.plot(pred_traj[:split_idx, 0], pred_traj[:split_idx, 1], '--', color=colors[i],
                label=f'Pred (ID) {i+1}', linewidth=2, alpha=0.7)
        if split_idx < len(pred_traj):
            start_idx = max(split_idx - 1, 0)
            ax.plot(pred_traj[start_idx:, 0], pred_traj[start_idx:, 1], ':', color=colors[i],
                label=f'Pred (OOD) {i+1}', linewidth=2, alpha=0.7)

    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Vehicle Trajectories - All Tests', fontsize=14)
    ax.legend(ncol=2, fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Plot 2: Velocity profiles (right/top)
    ax = ax_vel
    for i in range(len(all_gt_trajectories)):
        gt_traj = all_gt_trajectories[i]
        pred_traj = all_pred_trajectories[i]

        ax.plot(time_steps, gt_traj[:, 2], '-', color=colors[i],
            label=f'Ground Truth {i+1}', linewidth=1.5, alpha=0.7)
        if split_idx > 0:
            ax.plot(time_steps[:split_idx], pred_traj[:split_idx, 2], '--', color=colors[i],
                label=f'Pred (ID) {i+1}', linewidth=1.5, alpha=0.7)
        if split_idx < len(pred_traj):
            start_idx = max(split_idx - 1, 0)
            ax.plot(time_steps[start_idx:], pred_traj[start_idx:, 2], ':', color=colors[i],
                label=f'Pred (OOD) {i+1}', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Velocity (m/s)', fontsize=12)
    ax.set_title('Velocity Profiles - All Tests', fontsize=14)
    ax.legend(ncol=2, fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 3: Heading angle comparison (right/bottom)
    ax = ax_head
    for i in range(len(all_gt_trajectories)):
        gt_traj = all_gt_trajectories[i]
        pred_traj = all_pred_trajectories[i]

        ax.plot(time_steps, gt_traj[:, 3], '-', color=colors[i],
            label=f'Ground Truth {i+1}', linewidth=1.5, alpha=0.7)
        if split_idx > 0:
            ax.plot(time_steps[:split_idx], pred_traj[:split_idx, 3], '--', color=colors[i],
                label=f'Pred (ID) {i+1}', linewidth=1.5, alpha=0.7)
        if split_idx < len(pred_traj):
            start_idx = max(split_idx - 1, 0)
            ax.plot(time_steps[start_idx:], pred_traj[start_idx:, 3], ':', color=colors[i],
                label=f'Pred (OOD) {i+1}', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Heading Angle (rad)', fontsize=12)
    ax.set_title('Heading Angles - All Tests', fontsize=14)
    ax.legend(ncol=2, fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(save_dir, 'all_trajectories_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Combined trajectory plot saved to {os.path.join(save_dir, 'all_trajectories_comparison.png')}")

    # Create separate parameter table figure
    traj_mse_list = []
    for gt_traj, pred_traj in zip(all_gt_trajectories, all_pred_trajectories):
        mse = float(np.mean((pred_traj[:, :2] - gt_traj[:, :2]) ** 2))
        traj_mse_list.append(mse)

    fig_table, ax_table = plt.subplots(1, 1, figsize=(18, 4))
    ax_table.axis('off')

    param_keys = list(all_param_dicts[0].keys()) if len(all_param_dicts) > 0 else []
    param_names = ['Test'] + param_keys + ['MSE']
    table_data = []
    for i, (params, mse) in enumerate(zip(all_param_dicts, traj_mse_list)):
        row = [f'{i+1}']
        for k in param_keys:
            v = params.get(k, None)
            if v is None:
                row.append('')
            elif k in ['tau0', 'omega0']:
                row.append(f'{float(v):.0f}')
            else:
                row.append(f'{float(v):.4f}'.rstrip('0').rstrip('.'))
        row.append(f'{mse:.3e}')
        table_data.append(row)

    n_param_cols = len(param_keys)
    test_w = 0.06
    mse_w = 0.12
    param_w = (1.0 - test_w - mse_w) / max(1, n_param_cols)
    col_widths = [test_w] + [param_w] * n_param_cols + [mse_w]

    table = ax_table.table(cellText=table_data, colLabels=param_names,
                           cellLoc='center', loc='center',
                           colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for i in range(len(table_data)):
        for j in range(len(param_names)):
            cell = table[(i+1, j)]
            cell.set_facecolor(colors[i])
            cell.set_alpha(0.2)

    ax_table.set_title('Test Parameters and MSE', fontsize=14, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_parameters_mse.png'), dpi=200, bbox_inches='tight')
    plt.close(fig_table)

    logger.info(f"Parameter table plot saved to {os.path.join(save_dir, 'test_parameters_mse.png')}")

    # Also create individual trajectory plots in a grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Individual Test Trajectories', fontsize=16, fontweight='bold')

    # Flatten axes for easier indexing
    axes_flat = axes.flatten()

    for i in range(len(all_gt_trajectories)):
        ax = axes_flat[i]
        gt_traj = all_gt_trajectories[i]
        pred_traj = all_pred_trajectories[i]
        params = all_param_dicts[i]

        ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', label='Ground Truth', linewidth=2.5)
        if split_idx > 0:
            ax.plot(pred_traj[:split_idx, 0], pred_traj[:split_idx, 1], 'r--', label='Pred (ID)', linewidth=2.5)
        if split_idx < len(pred_traj):
            start_idx = max(split_idx - 1, 0)
            ax.plot(pred_traj[start_idx:, 0], pred_traj[start_idx:, 1], 'r:', label='Pred (OOD)', linewidth=2.5)

        ax.set_xlabel('X Position (m)', fontsize=11)
        ax.set_ylabel('Y Position (m)', fontsize=11)
        ax.set_title(f'Test {i+1}: l={params["l"]:.2f}, τ₀={params["tau0"]:.0f}, γ={params["gamma"]:.3f}\n'
                    f'Final Error: {all_errors[i]:.1f}m', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    # Hide the last subplot if we have 5 tests
    if len(all_gt_trajectories) < 6:
        axes_flat[-1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'individual_trajectories.png'), dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Individual trajectory plot saved to {os.path.join(save_dir, 'individual_trajectories.png')}")


def main():
    # Parse arguments
    args = parse_arguments()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("="*60)
    logger.info("FNODE Parameterized Vehicle Training")
    logger.info("="*60)
    logger.info(f"Configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info("="*60)

    # Generate data if requested or if file doesn't exist
    if args.generate_data or not os.path.exists(args.data_path):
        logger.info("Generating new parameterized vehicle dataset...")
        from Model.Data_generator import generate_dataset

        data_path = generate_dataset(
            test_case='veh_4dof_param',
            numerical_methods='rk4',
            dt=0.01,
            num_steps=2000,  # 20 seconds (reduced from 4000)
            if_noise=False,
            output_root_dir='.',
            seed=args.seed,
            gen_train_num_steps=1500,  # 75% for training
            use_gpu=args.use_gpu_generation,
            gpu_batch_size=args.gpu_batch_size,  # Pass GPU batch size
            verify_consistency=False
        )

        # Update data path if generation returned a path
        if isinstance(data_path, str):
            args.data_path = data_path
            logger.info(f"Using generated data from: {args.data_path}")

    # Check if data exists
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        logger.info("Data generation may have failed")
        return

    # Dataset format guard: this script assumes HDF5 stores PHYSICAL units and
    # applies grouped normalization online.
    with h5py.File(args.data_path, 'r') as h5f:
        normalized_flag = h5f.attrs.get('normalized', None)
    if normalized_flag is True:
        raise ValueError(
            "This run is configured for ONLINE grouped normalization with PHYSICAL HDF5 data, "
            "but the dataset indicates normalized=True. Please regenerate the dataset with the current "
            "generator (normalized=False), or switch the script to an offline-normalized dataset mode."
        )

    # Load datasets to GPU
    logger.info("Loading dataset to GPU memory...")

    if device.type != 'cuda':
        logger.error("GPU not available! This code requires CUDA.")
        return

    train_dataset = VehicleParameterDatasetGPU(args.data_path, mode='train', device=device, max_gb=args.gpu_memory_gb)
    test_dataset = VehicleParameterDatasetGPU(args.data_path, mode='test', device=device, max_gb=args.gpu_memory_gb * 0.3)

    logger.info(f"Train samples: {len(train_dataset):,}, Test samples: {len(test_dataset):,}")

    # Initialize model
    logger.info("Initializing FNODE_CON model...")
    model = FNODE_CON(
        num_bodies=2,  # We're predicting 2 derivatives (v_dot, theta_dot)
        dim_input=15,  # [x, y, v, theta, alpha, beta] + 9 parameters
        dim_output=2,  # [v_dot, theta_dot]
        layers=args.layers,
        width=args.hidden_size,
        activation=args.activation,
        initializer=args.initializer
    ).to(device)  # Keep model in fp32 for mixed precision training

    # Apply torch.compile for faster execution (PyTorch 2.0+) - only during training
    if hasattr(torch, 'compile') and not args.skip_train:
        logger.info("Compiling model with torch.compile for optimization...")
        model = torch.compile(model, mode='reduce-overhead')
        logger.info("Model compiled successfully")
    else:
        if args.skip_train:
            logger.info("Skipping torch.compile for testing phase")
        else:
            logger.info("torch.compile not available, using standard model")

    logger.info(f"Model architecture:")
    logger.info(f"  Input dimension: 15 (6 states/controls + 9 parameters)")
    logger.info(f"  Output dimension: 2 (v_dot, theta_dot)")
    logger.info(f"  Hidden layers: {args.layers}")
    logger.info(f"  Hidden size: {args.hidden_size}")
    logger.info(f"  Activation: {args.activation}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {total_params:,}")

    # Enforce: if we are going to run testing, we MUST use a trained model checkpoint.
    # - If user didn't provide --model_load_filename, we load the default best checkpoint produced by this script.
    # - If that file doesn't exist, we raise an error to avoid silently testing a random model.
    default_best_ckpt = os.path.join('saved_model', 'veh_4dof_param', 'checkpoint_best.pth')
    will_run_testing = True  # this script always tests after (optional) training
    if will_run_testing and not args.model_load_filename:
        if os.path.exists(default_best_ckpt):
            args.model_load_filename = default_best_ckpt
            logger.info(f"No --model_load_filename provided. Using default trained checkpoint: {default_best_ckpt}")
        else:
            raise FileNotFoundError(
                "Testing requires a trained model checkpoint, but none was provided and the default "
                f"checkpoint was not found: {default_best_ckpt}.\n"
                "Run training first (without --skip_train), or pass --model_load_filename to an existing checkpoint.")

    # Load checkpoint.
    # IMPORTANT behavior:
    # - If we are skipping training (testing-only), then a compatible checkpoint MUST load strictly.
    #   Otherwise we'd silently test a random/partially-initialized model.
    # - If we are training, and the architecture changed, we allow partial loading (or none).
    #   Training will then produce a new compatible checkpoint.
    loaded_any_weights = False
    if args.model_load_filename:
        logger.info(f"Loading checkpoint from {args.model_load_filename}...")
        try:
            loaded_any_weights = _load_checkpoint_flexible(
                model,
                args.model_load_filename,
                device,
                allow_partial=(not args.skip_train),
            )
        except RuntimeError as e:
            if args.skip_train:
                # testing-only must be strict and compatible
                raise
            logger.warning(
                "Checkpoint load failed (likely architecture mismatch). Will train from scratch. Error: %s",
                str(e),
            )
            loaded_any_weights = False

        if loaded_any_weights:
            logger.info("Checkpoint loaded successfully!")

    # Training
    if not args.skip_train:
        logger.info("="*60)
        logger.info("Starting Training")
        logger.info("="*60)

        train_history = train_fnode_param(model, train_dataset, args)

        # Plot loss history
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_history, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(train_history[-50:], label='Training Loss (last 50)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss History (Last 50 Epochs)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs('figures/veh_4dof_param', exist_ok=True)
        plt.savefig('figures/veh_4dof_param/loss_history.png', dpi=150)
        plt.close()

    # If we skipped training, we must have loaded a compatible checkpoint.
    if args.skip_train and not loaded_any_weights:
        raise RuntimeError(
            "Testing requires a compatible trained checkpoint, but no weights were loaded. "
            "This usually means you changed the model architecture. Train first (without --skip_train) "
            "to produce a new checkpoint, or point --model_load_filename to a compatible checkpoint."
        )

    # Testing on parameter combinations
    logger.info("="*60)
    logger.info("Testing on Parameter Combinations")
    logger.info("="*60)

    test_parameter_combinations(model, args.data_path, 'results/veh_4dof_param', num_test=4)

    # Save final model
    save_dir = os.path.join('saved_model', 'veh_4dof_param')
    os.makedirs(save_dir, exist_ok=True)
    save_model_state(model, save_dir, 'FNODE_final.pkl')
    logger.info(f"Final model saved to {os.path.join(save_dir, 'FNODE_final.pkl')}")

    logger.info("="*60)
    logger.info("Training and testing completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
