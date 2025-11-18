#!/usr/bin/env python3
"""
main_con.py - FNODE_CON training using MBDNODE-for-MBD2 approach
Trains neural network to learn cart-pole dynamics Swith control
"""
import os
import sys
import time
import torch
import torch.optim as optim
import logging
import argparse
from datetime import datetime

# Add Model directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
model_package_dir = os.path.join(script_dir, 'Model')
if model_package_dir not in sys.path:
    sys.path.insert(0, model_package_dir)

# Import from Model package
from Model.model import *
from Model.Data_generator import (generate_cartpole_data, load_cartpole_data, data_generator_cp_d, 
                                 load_cartdoublependulum_data)


def setup_logging(log_level='INFO', log_file=None):
    """Setup comprehensive logging configuration"""
    # Create formatter with module information
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Setup handlers list
    handlers = [console_handler]
    
    # Add file handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger to capture all module logs
    logging.root.setLevel(getattr(logging, log_level.upper()))
    logging.root.handlers = handlers
    
    # Ensure Model package logs are visible
    logging.getLogger('Model').setLevel(getattr(logging, log_level.upper()))
    
    return logging.getLogger(__name__)

def main():
    """Main training function using MBDNODE-for-MBD2 approach"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train FNODE_CON for controlled cart-pole systems')
    parser.add_argument('--test_case', type=str, default='Cart_Pole_Controlled',
                        choices=['Cart_Pole_Controlled', 'Cart_Pole_D_Controlled'],
                        help='Which system to train')
    parser.add_argument('--epochs', type=int, default=450, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512*8, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=1_000_000, help='Number of training samples')
    parser.add_argument('--save_dir', type=str, default=None, 
                        help='Directory to save models (defaults to saved_model/{test_case})')
    parser.add_argument('--use_existing_data', action='store_true', default=False,
                        help='Use existing dataset instead of generating new data')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to existing dataset file (for loading)')
    parser.add_argument('--dataset_seed', type=int, default=42,
                        help='Random seed for data generation')
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Optional log file path')
    parser.add_argument('--optimizer_type', type=str, default='adam',
                        choices=['adam', 'sgd', 'adamw', 'rmsprop'],
                        help='Type of optimizer to use')
    parser.add_argument('--scheduler_type', type=str, default='step',
                        choices=['none', 'step', 'exponential', 'cosine', 'reduce_on_plateau'],
                        help='Type of learning rate scheduler to use')
    parser.add_argument('--scheduler_step_interval', type=int, default=1,
                        help='Number of epochs between scheduler steps (default: 5, for step scheduler)')
    parser.add_argument('--scheduler_gamma', type=float, default=0.98,
                        help='Multiplicative factor for lr decay (for step/exponential schedulers)')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for schedulers')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--grad_clip', type=float, default=0.5,
                        help='Gradient clipping max norm')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision training')
    parser.add_argument('--compile_model', action='store_true', default=True,
                        help='Use torch.compile for model optimization')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    
    
    args = parser.parse_args()
    
    # Setup logging with new configuration
    logger = setup_logging(log_level=args.log_level, log_file=args.log_file)

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configure TensorFloat32 for better performance on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')  # Enable TF32 for matmul
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    logger.info(f"Starting FNODE_CON training for {args.test_case}")
    logger.info(f"Device: {device}")
    logger.info(f"Samples: {args.num_samples}, Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}, Learning rate: {args.lr}")
    logger.info(f"Optimizer: {args.optimizer_type.upper()}, Scheduler: {args.scheduler_type}")
    if args.scheduler_type == 'step':
        logger.info(f"Scheduler step interval: {args.scheduler_step_interval}, gamma: {args.scheduler_gamma}")

    # Data loading/generation
    dataset_dir = f'dataset/{args.test_case}'
    
    # Set save directory if not specified
    if args.save_dir is None:
        args.save_dir = f'saved_model/{args.test_case}'
    
    # Check if we should use existing data
    generate_new = not args.use_existing_data
    
    if args.test_case == 'Cart_Pole_Controlled':
        if generate_new:
            # Generate new data
            logger.info(f"Generating new dataset with seed {args.dataset_seed}")
            body_tensor, force_tensor, accel_tensor = generate_cartpole_data(
                num_steps=args.num_samples,
                seed=args.dataset_seed,
                save_to_dataset=True,
                dataset_dir=dataset_dir
            )
        else:
            # Load existing dataset
            if args.dataset_path:
                dataset_path = args.dataset_path
            else:
                # Default path based on seed and num_samples
                dataset_path = os.path.join(dataset_dir, 
                                          f'controlled_dataset_seed{args.dataset_seed}_n{args.num_samples}.npz')
            
            if not os.path.exists(dataset_path):
                logger.warning(f"Dataset not found at {dataset_path}, generating new data instead")
                body_tensor, force_tensor, accel_tensor = generate_cartpole_data(
                    num_steps=args.num_samples,
                    seed=args.dataset_seed,
                    save_to_dataset=True,
                    dataset_dir=dataset_dir
                )
            else:
                logger.info(f"Loading existing dataset from {dataset_path}")
                body_tensor, force_tensor, accel_tensor = load_cartpole_data(dataset_path)
    
    elif args.test_case == 'Cart_Pole_D_Controlled':
        if generate_new:
            # Generate new data
            logger.info(f"Generating new dataset with seed {args.dataset_seed}")
            states_tensor, force_tensor, accel_tensor = data_generator_cp_d(
                num_steps=args.num_samples,
                seed=args.dataset_seed,
                save_to_dataset=True,
                dataset_dir=dataset_dir
            )
            # Convert states to body format for compatibility
            body_tensor = states_tensor
        else:
            # Load existing dataset
            if args.dataset_path:
                dataset_path = args.dataset_path
            else:
                # Default path based on seed and num_samples
                dataset_path = os.path.join(dataset_dir, 
                                          f'controlled_dataset_seed{args.dataset_seed}_n{args.num_samples}.npz')
            
            if not os.path.exists(dataset_path):
                logger.warning(f"Dataset not found at {dataset_path}, generating new data instead")
                states_tensor, force_tensor, accel_tensor = data_generator_cp_d(
                    num_steps=args.num_samples,
                    seed=args.dataset_seed,
                    save_to_dataset=True,
                    dataset_dir=dataset_dir
                )
                body_tensor = states_tensor
            else:
                logger.info(f"Loading existing dataset from {dataset_path}")
                states_tensor, force_tensor, accel_tensor = load_cartdoublependulum_data(dataset_path)
                body_tensor = states_tensor

    # Create model based on variant selection
    if args.test_case == 'Cart_Pole_Controlled':
        num_bodies, dim_input, dim_output = 2, 5, 2
    elif args.test_case == 'Cart_Pole_D_Controlled':
        num_bodies, dim_input, dim_output = 3, 7, 3
    
    # Create FNODE_CON model with standard settings
    logger.info("Using FNODE_CON with standard settings")
    model = FNODE_CON(
        num_bodies=num_bodies,
        dim_input=dim_input,
        dim_output=dim_output,
        layers=3,
        width=256,
        activation='tanh'
    ).to(device)

    # Compile model if requested and PyTorch version supports it
    if args.compile_model and hasattr(torch, 'compile'):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,}")

    # Setup training - Create optimizer based on type
    if args.optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unknown optimizer type: {args.optimizer_type}")
    
    logger.info(f"Using optimizer: {args.optimizer_type.upper()}")
    if args.optimizer_type == 'adamw':
        logger.info(f"Weight decay: {args.weight_decay}")
    
    # Create scheduler based on type
    if args.scheduler_type == 'none':
        scheduler = None
        logger.info("No learning rate scheduler")
    elif args.scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                             step_size=args.scheduler_step_interval, 
                                             gamma=args.scheduler_gamma)
        logger.info(f"Using StepLR scheduler: step_size={args.scheduler_step_interval}, gamma={args.scheduler_gamma}")
    elif args.scheduler_type == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
        logger.info(f"Using ExponentialLR scheduler: gamma={args.scheduler_gamma}")
    elif args.scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                        T_max=args.epochs, 
                                                        eta_min=args.scheduler_min_lr)
        logger.info(f"Using CosineAnnealingLR scheduler: T_max={args.epochs}, eta_min={args.scheduler_min_lr}")
    elif args.scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        mode='min', 
                                                        factor=args.scheduler_gamma, 
                                                        patience=10,
                                                        min_lr=args.scheduler_min_lr)
        logger.info(f"Using ReduceLROnPlateau scheduler: factor={args.scheduler_gamma}, patience=10")
    else:
        raise ValueError(f"Unknown scheduler type: {args.scheduler_type}")

    # Create date-based folder for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"run_{timestamp}"
    model_save_path = os.path.join(args.save_dir, run_folder)
    
    # Training parameters
    train_params = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'save_dir': model_save_path,
        'scheduler_step_interval': args.scheduler_step_interval,
        'test_case': args.test_case,
        'grad_clip': args.grad_clip,
        'grad_accumulation_steps': args.grad_accumulation_steps,
        'use_amp': args.use_amp,
        'num_workers': args.num_workers
    }
    
    # Output paths
    output_paths = {
        'model': model_save_path,
        'results_dir': f'results/{args.test_case}',
        'figures_dir': f'figures/{args.test_case}'
    }
    
    # Create directories
    os.makedirs(model_save_path, exist_ok=True)
    
    logger.info(f"Models will be saved to: {model_save_path}")
    
    # Save run configuration
    config_path = os.path.join(model_save_path, 'run_config.txt')
    with open(config_path, 'w') as f:
        f.write(f"FNODE_CON Training Configuration\n")
        f.write(f"================================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Device: {device}\n\n")
        
        f.write(f"Training Parameters:\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Optimizer: {args.optimizer_type}\n")
        if args.optimizer_type == 'adamw':
            f.write(f"  Weight Decay: {args.weight_decay}\n")
        f.write(f"  Gradient Clipping: {args.grad_clip}\n")
        f.write(f"  Gradient Accumulation Steps: {args.grad_accumulation_steps}\n")
        f.write(f"  Mixed Precision: {args.use_amp}\n")
        f.write(f"  Scheduler: {args.scheduler_type}\n")
        if args.scheduler_type == 'step':
            f.write(f"  Scheduler Step Interval: {args.scheduler_step_interval}\n")
            f.write(f"  Scheduler Gamma: {args.scheduler_gamma}\n")
        elif args.scheduler_type == 'exponential':
            f.write(f"  Scheduler Gamma: {args.scheduler_gamma}\n")
        elif args.scheduler_type == 'cosine':
            f.write(f"  Scheduler T_max: {args.epochs}\n")
            f.write(f"  Scheduler Min LR: {args.scheduler_min_lr}\n")
        elif args.scheduler_type == 'reduce_on_plateau':
            f.write(f"  Scheduler Factor: {args.scheduler_gamma}\n")
            f.write(f"  Scheduler Patience: 10\n")
            f.write(f"  Scheduler Min LR: {args.scheduler_min_lr}\n")
        f.write(f"\n")
        
        f.write(f"Data Parameters:\n")
        f.write(f"  Number of Samples: {args.num_samples}\n")
        f.write(f"  Dataset Seed: {args.dataset_seed}\n")
        f.write(f"  Use Existing Data: {args.use_existing_data}\n\n")
        
        f.write(f"Model Architecture:\n")
        f.write(f"  Model Type: FNODE_CON\n")
        f.write(f"  Test Case: {args.test_case}\n")
        f.write(f"  Number of Bodies: {num_bodies}\n")
        f.write(f"  Input Dimension: {dim_input}\n")
        f.write(f"  Output Dimension: {dim_output}\n")
        f.write(f"  Layers: {model.layers if hasattr(model, 'layers') else 'N/A'}\n")
        f.write(f"  Width: {model.width if hasattr(model, 'width') else 'N/A'}\n")
        f.write(f"  Activation: {model.activation if hasattr(model, 'activation') else 'N/A'}\n")
        f.write(f"  Total Parameters: {total_params:,}\n")
        f.write(f"  Model Compilation: {args.compile_model}\n\n")
        
        f.write(f"Save Location:\n")
        f.write(f"  Model Directory: {model_save_path}\n")
    
    logger.info(f"Run configuration saved to: {config_path}")
    
    # Use the appropriate training function
    train_fnode_con_batch(
        model=model,
        body_tensor=body_tensor,
        force_tensor=force_tensor,
        accel_tensor=accel_tensor,
        train_params=train_params,
        optimizer=optimizer,
        scheduler=scheduler,
        output_paths=output_paths
    )

    logger.info("Training completed!")
    return True


if __name__ == "__main__":
    start_time = time.time()
    success = main()
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")
    sys.exit(0 if success else 1)