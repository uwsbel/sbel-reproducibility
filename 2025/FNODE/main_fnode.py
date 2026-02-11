# FNODE/main_fnode.py
# Enhanced version with prob parameter for FFT truncation
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import logging
import pandas as pd


# --- Simplified Logging Configuration ---
def setup_logging(log_file_path=None):
    """Setup logging with optional file output, avoiding duplicates"""
    # Remove existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure basic logging
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


# Initial basic logging setup
setup_logging()
logger = logging.getLogger("FNODE_Main")

# --- Ensure Model directory is in the Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_package_dir = os.path.join(script_dir, 'Model')
if model_package_dir not in sys.path:
    logger.info(f"Adding Model package directory to sys.path: {model_package_dir}")
    sys.path.insert(0, model_package_dir)

# --- Imports from custom Model package ---
try:
    from Model.Data_generator import *
    from Model.utils import *
    from Model.model import *
    from Model.force_fun import *

    logger.info("Successfully imported components from Model package/directory.")
except ImportError as e:
    logger.error(f"Failed to import from Model package/directory: {e}. Check PYTHONPATH.", exc_info=True)
    sys.exit(1)


# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Main Script for FNODE Model (State-to-Acceleration Mapping)")

    parser.add_argument('--test_case', type=str, default='Double_Pendulum',
                        choices=['Single_Mass_Spring_Damper', 'Double_Pendulum', 'Triple_Mass_Spring_Damper',
                                 'Slider_Crank', 'Cart_Pole'],
                        help="Dynamical system to simulate.")
    parser.add_argument('--seed', type=int, default=42, help='Global random seed.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Computation device.")
    parser.add_argument('--generate_new_data', action='store_true', default=False, help="Flag to generate new dataset.")
    parser.add_argument('--data_dt', type=float, default=0.01, help="Time step for data generation.")
    parser.add_argument('--data_total_steps', type=int, default=400,
                        help="Total steps for the FIRST generated trajectory (train+test). Additional trajectories use train portion length.")
    parser.add_argument('--data_train_split', type=float, default=3/4,
                        help="Fraction of FIRST trajectory for its training segment. Defines segment length.")

    # NEW: prob parameter for FFT truncation
    parser.add_argument('--prob', type=int, default=100,
                        help="Probability factor for FFT truncation. Used to calculate trunc = train_time_step//prob")

    parser.add_argument('--layers', type=int, default=3, help="Number of layers for FNODE (input + hidden + output).")
    parser.add_argument('--hidden_size', type=int, default=256, help="Hidden layer width for FNODE.")
    parser.add_argument('--activation', type=str, default='tanh', choices=['relu', 'tanh'], help="Activation function.")
    parser.add_argument('--initializer', type=str, default='xavier', choices=['xavier', 'kaiming'],
                        help="Weight initializer.")
    parser.add_argument('--d_interest', type=int, default=0,
                        help="Extra input features for FNODE. MUST BE 0 for state -> (a1,a2) mapping.")

    parser.add_argument('--epochs', type=int, default=400, help="Number of training epochs.")
    parser.add_argument('--early_stop', type=str, default='False',
                        help="Enable early stopping ('True' or 'False').")
    parser.add_argument('--patience', type=int, default=50,
                        help="Patience for early stopping (number of epochs without improvement).")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help="Optimizer.")
    parser.add_argument('--lr_scheduler', type=str, default='exponential', choices=['none', 'exponential', 'step', 'cosine'],
                        help="LR scheduler.")
    parser.add_argument('--lr_decay_rate', type=float, default=0.98, help="Decay rate for schedulers.")
    parser.add_argument('--lr_decay_steps', type=int, default=1, help="Step size or T_max for schedulers.")
    parser.add_argument('--out_time_log', type=int, default=10, help="Log training progress every N epochs.")
    parser.add_argument('--save_ckpt_freq', type=int, default=0, help="Save checkpoint every N epochs (0 to disable).")

    parser.add_argument('--fnode_loss_type', type=str, default='derivative', choices=['derivative'],
                        help="Loss type for FNODE. MUST be 'derivative' for state->acceleration mapping.")
    parser.add_argument('--fnode_use_hybrid_target', action='store_true', default=False,
                        help="Use hybrid FFT-FD target.")

    parser.add_argument('--ode_method', type=str, default="rk4", help="ODE solver for testing.")
    parser.add_argument('--ode_rtol', type=float, default=1e-7, help="Relative tolerance for ODE solver (testing).")
    parser.add_argument('--ode_atol', type=float, default=1e-9, help="Absolute tolerance for ODE solver (testing).")

    parser.add_argument('--plot_bodies', type=int, default=0, help="Max bodies for plots (0 for all).")
    parser.add_argument('--skip_train', action='store_true', default=False, help="Skip training, load model and test.")
    parser.add_argument('--model_load_filename', type=str, default="FNODE_best.pkl", help="Model filename to load.")

    # DataLoader and train/val/test split parameters
    parser.add_argument('--num_workers', type=int, default=0, help="Number of parallel workers for data loading.")
    parser.add_argument('--train_ratio', type=float, default=0.75, help="Training set ratio.")
    parser.add_argument('--val_ratio', type=float, default=0, help="Validation set ratio (0 = no validation).")


    args = parser.parse_args()
    args.model_type = 'FNODE'

    if args.d_interest != 0:
        logger.critical(
            f"CRITICAL: For state -> (a1,a2) mapping, --d_interest MUST be 0. Current value: {args.d_interest}. Please correct and rerun.")
        sys.exit(1)

    logger.info(f"Arguments parsed: {args}")
    return args


# --- Main Execution ---
def main():
    args = parse_arguments()

    # --- Setup File Logging ---
    base_log_dir = os.path.join(os.getcwd(), 'log')
    log_directory = os.path.join(base_log_dir, args.test_case)
    os.makedirs(log_directory, exist_ok=True)
    log_file_path = os.path.join(log_directory, 'fnode.log')
    # Reconfigure logging with file output
    setup_logging(log_file_path)
    logger = logging.getLogger("FNODE_Main")

    # Fixed parameters for each test case: (weight_decay, grad_clip)
    test_case_params = {
        'Single_Mass_Spring_Damper': (9e-5, 0.3),
        'Double_Pendulum': (5e-4, 1.0),
        'Triple_Mass_Spring_Damper': (5e-5, 1.0),
        'Slider_Crank': (1e-5, 1.0),
        'Cart_Pole': (1e-3, 1.0)
    }
 
    weight_decay, grad_clip = test_case_params[args.test_case]

    logger.info(f"--- Starting Main Execution for FNODE (Test Case: {args.test_case}, Model: {args.model_type}) ---")
    logger.info(f"Prob parameter value: {args.prob}")
    logger.info(f"Fixed parameters: weight_decay={weight_decay}, grad_clip={grad_clip}")

    set_seed(args.seed)
    current_device = torch.device(args.device)
    output_paths = get_output_paths(args.test_case, args.model_type)

    logger.info(f"Using device: {current_device}")
    logger.info(f"Output paths: {output_paths}")

    # Make sure all directories exist and are writable
    try:
        os.makedirs(output_paths["model"], exist_ok=True)
        os.makedirs(output_paths["results"], exist_ok=True)
        os.makedirs(output_paths["figures"], exist_ok=True)

        # Test write permissions
        test_file = os.path.join(output_paths["results"], "test_write.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        logger.info("All output directories created and writable")
    except Exception as dir_err:
        logger.error(f"Failed to create or write to output directories: {dir_err}")
        logger.error("Please check permissions and disk space")
        return

    best_model_path = os.path.join(output_paths["model"], "FNODE_best.pkl")

    # --- Data Generation and Loading ---
    logger.info("=== Data Generation and Loading ===")
    dataset_path = os.path.join('.', 'dataset', args.test_case)
    required_s_train_file = os.path.join(dataset_path, "s_train.csv")
    required_t_train_file = os.path.join(dataset_path, "t_train.csv")

    data_gen_needed = args.generate_new_data or \
                      not os.path.exists(required_s_train_file) or \
                      not os.path.exists(required_t_train_file)

    num_steps_for_train_segment = int(args.data_total_steps * args.train_ratio)

    # Data Generation
    if data_gen_needed:
        logger.info(f"Generating new dataset for {args.test_case}...")

        if args.test_case == "Slider_Crank":
            slider_crank_args = {
                'total_num_steps': args.data_total_steps * 10,
                'train_num_steps': num_steps_for_train_segment * 10
            }
            try:
                _ = generate_slider_crank_dataset(**slider_crank_args)
                logger.info(f"Slider-Crank dataset generation complete.")
            except Exception as gen_err:
                logger.error(f"Slider-Crank dataset generation failed: {gen_err}", exc_info=True)
                return
        else:
            # Use standard dataset generation for other test cases
            data_gen_kwargs_standard = {
                'gen_train_num_steps': num_steps_for_train_segment,
                'output_root_dir': '.',
                'save_to_file': True
            }
            try:
                generate_dataset(
                    test_case=args.test_case, numerical_methods="rk4", dt=args.data_dt,
                    num_steps=args.data_total_steps, seed=args.seed, **data_gen_kwargs_standard
                )
                logger.info(f"Standard dataset generation complete for {args.test_case}.")
            except Exception as gen_err:
                logger.error(f"Standard dataset generation failed: {gen_err}", exc_info=True)
                return
    else:
        logger.info(f"Using existing dataset found at {dataset_path}")

    # Data Loading
    try:
        s_train_df = pd.read_csv(required_s_train_file)
        t_train_df = pd.read_csv(required_t_train_file)
        s_test_df = pd.read_csv(os.path.join(dataset_path, "s_test.csv"))
        t_test_df = pd.read_csv(os.path.join(dataset_path, "t_test.csv"))

        # For Slider Crank, we only need theta and omega as FNODE inputs
        if args.test_case == "Slider_Crank":
            if 'theta_0_2pi' in s_train_df.columns and 'omega' in s_train_df.columns:
                s_train_np = s_train_df[['theta_0_2pi', 'omega']].values
                s_test_np = s_test_df[['theta_0_2pi', 'omega']].values
                logger.info(f"Loaded Slider-Crank data with columns: theta_0_2pi, omega")
            else:
                if 'Unnamed: 0' in s_train_df.columns:
                    s_train_np = s_train_df.iloc[:, 1:3].values
                    s_test_np = s_test_df.iloc[:, 1:3].values
                else:
                    s_train_np = s_train_df.iloc[:, 0:2].values
                    s_test_np = s_test_df.iloc[:, 0:2].values
                logger.info(f"Loaded Slider-Crank data using first two columns (assumed to be theta, omega)")
        else:
            # Standard data processing for other test cases
            s_train_np = s_train_df.values[:, 1:] if s_train_df.columns[0] in ["idx",
                                                                               "Unnamed: 0"] else s_train_df.values
            s_test_np = s_test_df.values[:, 1:] if s_test_df.columns[0] in ["idx", "Unnamed: 0"] else s_test_df.values

        # Load time data
        t_train_np = t_train_df['time'].values if 'time' in t_train_df.columns else t_train_df.values.flatten()
        t_test_np = t_test_df['time'].values if 'time' in t_test_df.columns else t_test_df.values.flatten()

        # Convert to tensors
        # Create tensors on CPU for DataLoader compatibility with num_workers > 0
        # They will be moved to CUDA in the training loop
        s_train_tensor = torch.tensor(s_train_np, dtype=torch.float32, device='cpu')
        t_train_tensor = torch.tensor(t_train_np, dtype=torch.float32, device='cpu')
        s_test_tensor = torch.tensor(s_test_np, dtype=torch.float32, device='cpu')
        t_test_tensor = torch.tensor(t_test_np, dtype=torch.float32, device='cpu')

        # Full trajectory
        s_full_tensor = torch.cat((s_train_tensor, s_test_tensor), dim=0)
        t_full_tensor = torch.cat((t_train_tensor, t_test_tensor), dim=0)

        # Store actual data sizes
        actual_train_size = len(s_train_tensor)
        actual_test_size = len(s_test_tensor)
        actual_total_size = len(s_full_tensor)

        logger.info(f"Data loaded successfully:")
        logger.info(f"  Train: {s_train_tensor.shape}, Test: {s_test_tensor.shape}")
        logger.info(f"  Actual sizes - Train: {actual_train_size}, Test: {actual_test_size}")

    except Exception as e:
        logger.error(f"Data loading failed: {e}", exc_info=True)
        return

    # --- Model Initialization ---
    logger.info("=== Model Initialization ===")
    if args.test_case == "Slider_Crank":
        num_bodies = 1  # Only one angular acceleration to predict
        raw_state_dim_per_step = 2  # [theta, omega]
    else:
        # Standard handling for other test cases
        raw_state_dim_per_step = s_train_tensor.shape[-1]
        if raw_state_dim_per_step % 2 != 0:
            logger.error(f"Raw state dimension ({raw_state_dim_per_step}) is odd. Cannot infer num_bodies.")
            return
        num_bodies = raw_state_dim_per_step // 2

    logger.info(f"Inferred num_bodies = {num_bodies} from state dimension {raw_state_dim_per_step}")

    try:
        fnode_model = FNODE(num_bodys=num_bodies, layers=args.layers, width=args.hidden_size,
                            d_interest=args.d_interest, activation=args.activation, initializer=args.initializer
                            ).to(current_device)
        logger.info(
            f"FNODE model initialized: InputDim={fnode_model.dim_input}, OutputDim(Accelerations)={fnode_model.output_dim}")
    except Exception as model_init_err:
        logger.error(f"Failed to initialize FNODE model: {model_init_err}", exc_info=True)
        return

    calculate_model_parameters(fnode_model)

    # --- Target Generation ---
    logger.info("=== Target Acceleration Generation ===")

    # Generate target accelerations with improved method selection
    selected_target_csv_path = generate_target_accelerations(
        s_full_tensor, t_full_tensor,
        args.test_case,
        num_bodies,
        args,
        output_paths["results"]
    )

    if selected_target_csv_path is None:
        logger.error("Failed to generate target accelerations. Training aborted.")
        return

    # --- Optimizer and Scheduler ---
    logger.info("=== Optimizer and Scheduler Setup ===")
    optimizer_class = optim.AdamW if args.optimizer == 'adamw' else optim.Adam
    optimizer = optimizer_class(fnode_model.parameters(), lr=args.lr, weight_decay=weight_decay)
    scheduler = None
    if args.lr_scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_rate)
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.lr_decay_steps, eta_min=args.lr * 0.0001)
    logger.info(f"Optimizer '{args.optimizer}' and Scheduler '{args.lr_scheduler}' configured.")

    # --- Training ---
    if not args.skip_train:
        logger.info("=== FNODE Training ===")
        training_start_time = time.time()

        train_params = {
            'epochs': args.epochs,
            'grad_clip': grad_clip,
            'outime_log': args.out_time_log,
            'save_ckpt_freq': args.save_ckpt_freq,
            'fnode_loss_type': args.fnode_loss_type,
            'fnode_use_hybrid_target': args.fnode_use_hybrid_target,
            'prob': args.prob,  # Pass prob parameter to training
            'num_workers': args.num_workers,  # Add batch size parameter
            'train_ratio': args.train_ratio,  # Add train ratio
            'val_ratio': 0,  # No validation set
            'early_stop': args.early_stop.lower() == 'true',  # Convert string to boolean
            'patience': args.patience
        }

        logger.info(f"Training parameters: {train_params}")
        logger.info(f"Selected target CSV: {selected_target_csv_path}")

        # Train with CSV targets
        try:
            trained_fnode_model, loss_history = train_fnode_with_csv_targets(
                model=fnode_model,
                s_train=s_train_tensor,
                t_train=t_train_tensor,
                train_params=train_params,
                optimizer=optimizer,
                scheduler=scheduler,
                output_paths=output_paths,
                target_csv_path=selected_target_csv_path
            )
        except Exception as train_err:
            logger.error(f"Exception during FNODE training: {train_err}", exc_info=True)
            return

        if trained_fnode_model is None:
            logger.error("FNODE training failed - function returned None.")
            return

        # Save the final trained model
        try:
            save_model_state(trained_fnode_model, output_paths["model"], model_filename="FNODE_final.pkl")
            logger.info("Final model saved successfully")
        except Exception as save_err:
            logger.warning(f"Failed to save final model: {save_err}")

        # Load best model for testing
        if os.path.exists(best_model_path):
            load_model_state(fnode_model, output_paths["model"], model_filename="FNODE_best.pkl",
                             current_device=current_device)
            logger.info("Best model loaded for testing")
        else:
            logger.warning("Best model not found, using final model")
            fnode_model = trained_fnode_model

        total_training_time = time.time() - training_start_time
        logger.info(f"--- FNODE Training Finished: Total Time {total_training_time:.2f}s ---")

        train_data_size = s_train_tensor.shape[0]
        test_data_size = s_test_tensor.shape[0]
        full_data_size = s_full_tensor.shape[0]
        # Save training time to CSV (will be updated with MSE results after testing)
        training_time_data = {
            'model': 'FNODE',
            'dataset': args.test_case,
            'training_time': total_training_time,
            'train_size': train_data_size,
            'test_size': test_data_size
        }

    else:
        logger.info("=== Skipping Training - Loading Pre-trained Model ===")
        if not os.path.exists(best_model_path):
            logger.error(f"Best model file {best_model_path} not found. Cannot proceed.")
            return
        if not load_model_state(fnode_model, output_paths["model"], model_filename="FNODE_best.pkl",
                                current_device=current_device):
            logger.error(f"Failed to load model from {best_model_path}.")
            return
        logger.info(f"Successfully loaded best model from {best_model_path}")

    # --- Testing ---
    logger.info("=== FNODE Testing ===")
    fnode_model.eval()

    # Define data sizes for metrics (needed regardless of training or loading)
    train_data_size = s_train_tensor.shape[0]
    test_data_size = s_test_tensor.shape[0] if s_test_tensor is not None else 0
    full_data_size = s_full_tensor.shape[0]

    # Get initial state for testing
    s0_for_test = s_train_tensor[0:1]
    logger.info(f"Initial state for testing: {s0_for_test.cpu().numpy()}")

    # Configure ODE parameters
    test_ode_params = {
        'ode_solver_params': {'method': args.ode_method, 'rtol': args.ode_rtol, 'atol': args.ode_atol}
    }

    # Generate trajectory predictions
    predictions_full = test_fnode(fnode_model, s0_for_test, t_full_tensor, test_ode_params, output_paths)

    if predictions_full is None:
        logger.error("FNODE testing failed (no predictions).")
        return

    logger.info(f"FNODE testing completed. Predictions shape: {predictions_full.shape}")

    # --- Evaluation and Metrics ---
    logger.info("=== Evaluation and Metrics ===")
    ground_truth_full = s_full_tensor

    # Align lengths for comparison
    min_len = min(predictions_full.shape[0], ground_truth_full.shape[0])
    if predictions_full.shape[0] != ground_truth_full.shape[0]:
        logger.warning(
            f"Shape mismatch: Predictions {predictions_full.shape[0]}, GT {ground_truth_full.shape[0]}. Truncating to {min_len}.")

    predictions_eval = predictions_full[:min_len]
    ground_truth_eval = ground_truth_full[:min_len]
    time_eval = t_full_tensor[:min_len]

    # Calculate MSE metrics - ensure tensors are on same device
    try:
        full_mse = torch.mean((predictions_eval - ground_truth_eval.to(predictions_eval.device)) ** 2).item()

        train_end_idx = min(actual_train_size, min_len)
        train_mse = torch.mean((predictions_eval[:train_end_idx] - ground_truth_eval[:train_end_idx].to(predictions_eval.device)) ** 2).item()

        if min_len > train_end_idx:
            test_mse = torch.mean((predictions_eval[train_end_idx:] - ground_truth_eval[train_end_idx:].to(predictions_eval.device)) ** 2).item()
        else:
            test_mse = float('nan')

        logger.info("=" * 70)
        logger.info("TEST MSE METRICS:")
        logger.info(f"Overall MSE (FNODE vs GT, {full_data_size} steps): {full_mse:.6e}")
        logger.info(f"Train Region MSE (first {train_data_size} steps): {train_mse:.6e}")
        logger.info(f"Extrapolation Region MSE (steps {train_data_size} to {full_data_size}): {test_mse:.6e}")
        logger.info("=" * 70)

        # Save metrics
        metrics_to_save = {
            'full_mse': [full_mse],
            'train_mse': [train_mse],
            'test_mse': [test_mse]
        }
        save_data_pd(metrics_to_save, output_paths["results"], f"{args.model_type}_test_metrics.csv")

        # Save training time and MSE results to CSV
        if 'training_time_data' in locals():
            training_results_csv = os.path.join(output_paths["results"], "training_results.csv")
            import csv
            with open(training_results_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Model', 'Dataset', 'Training_Time_Seconds', 'Train_Size', 'Test_Size',
                               'Overall_MSE', 'Train_MSE', 'Test_MSE'])
                writer.writerow(['FNODE', args.test_case, training_time_data['training_time'],
                               training_time_data['train_size'], training_time_data['test_size'],
                               full_mse, train_mse, test_mse])

    except Exception as eval_err:
        logger.error(f"Error during metrics calculation: {eval_err}", exc_info=True)

    # --- Save Prediction Results ---
    logger.info("=== Saving Results ===")
    try:
        pred_np = predictions_eval.detach().cpu().numpy()
        gt_np = ground_truth_eval.detach().cpu().numpy()
        time_np = time_eval.cpu().numpy()
        results_df = pd.DataFrame({'time': time_np})

        if args.test_case == "Slider_Crank":
            columns = ['theta_0_2pi', 'omega']
            for i, col in enumerate(columns):
                results_df[f'pred_{col}'] = pred_np[:, i]
                results_df[f'true_{col}'] = gt_np[:, i]
        else:
            for i in range(pred_np.shape[1]):
                results_df[f'pred_state_{i}'] = pred_np[:, i]
                results_df[f'true_state_{i}'] = gt_np[:, i]

        save_data_pd(results_df, output_paths["results"], f"{args.model_type}_predictions_vs_truth.csv")
        logger.info("Prediction results saved successfully")
    except Exception as save_err:
        logger.error(f"Error saving prediction results: {save_err}", exc_info=True)

    # --- Visualizations ---
    logger.info("=== Generating Visualizations ===")
    try:
        num_bodies_to_plot = args.plot_bodies if args.plot_bodies > 0 else num_bodies

        # Trajectory comparison plots
        if num_bodies > 0 and predictions_eval.shape[-1] == num_bodies * 2:
            pred_plot_reshaped = predictions_eval.reshape(min_len, num_bodies, 2)
            gt_plot_reshaped = ground_truth_eval.reshape(min_len, num_bodies, 2)

            plot_trajectory_comparison(
                test_case_name=args.test_case,
                model_predictions={args.model_type: pred_plot_reshaped},
                ground_truth_trajectory=gt_plot_reshaped,
                time_vector=time_eval,
                num_bodies_to_plot=num_bodies_to_plot,
                num_steps_train=actual_train_size,
                output_dir=output_paths["figures"],
                base_filename=f"{args.test_case}_{args.model_type}_traj_comparison",
                num_epochs=args.epochs
            )
            logger.info("Trajectory comparison plots generated.")

        # Acceleration comparison plots
        logger.info("Generating acceleration comparison plots...")
        with torch.no_grad():
            # For acceleration comparison, we should use the full trajectory to show model's performance
            # on both training and test data
            if fnode_model.d_interest == 0:
                fnode_input_full = s_full_tensor
            elif fnode_model.d_interest == 1:
                fnode_input_full = torch.cat([s_full_tensor, t_full_tensor.unsqueeze(-1)], dim=-1)
            else:
                logger.warning("d_interest > 1 not supported for acceleration comparison plots")
                fnode_input_full = s_full_tensor

            # Get predictions for full trajectory - ensure input is on same device as model
            predicted_accelerations_full = fnode_model(fnode_input_full.to(current_device))

        plot_acceleration_comparison(
            test_case_name=args.test_case,
            model_predictions_accel={args.model_type: predicted_accelerations_full},
            ground_truth_trajectory=s_full_tensor,
            time_vector=t_full_tensor,
            num_bodies_to_plot=num_bodies_to_plot,
            num_steps_train=actual_train_size,  # This should be 1500 for your case
            output_dir=output_paths["figures"],
            results_dir=output_paths["results"],
            selected_target_csv_path=selected_target_csv_path,
            num_epochs=args.epochs,
            base_filename=f"{args.test_case}_{args.model_type}_accel_comparison"
        )
        logger.info("Acceleration comparison plots generated.")

    except Exception as vis_err:
        logger.error(f"Error during visualization: {vis_err}", exc_info=True)

    # --- Save final results data ---
    try:
        save_data(ground_truth_full, args.test_case, "Ground_truth", actual_train_size, actual_test_size, args.data_dt)
        save_data(predictions_full, args.test_case, args.model_type, actual_train_size, actual_test_size, args.data_dt)
        logger.info("Final results data saved successfully")
    except Exception as save_err:
        logger.error(f"Error saving results data: {save_err}", exc_info=True)

    # Final summary
    logger.info("\n" + "=" * 50)
    logger.info("FNODE EXECUTION SUMMARY:")
    logger.info(f"Test Case: {args.test_case}")
    logger.info(f"Model Type: {args.model_type}")
    logger.info(f"Data Sizes - Train: {actual_train_size}, Test: {actual_test_size}")
    logger.info(f"Prob parameter: {args.prob}")

    # Determine which target method was used
    target_method = "Unknown"
    if "analytical_accelerations.csv" in selected_target_csv_path:
        target_method = "Analytical"
    elif "hybrid_target.csv" in selected_target_csv_path:
        target_method = "Hybrid FFT-FD"
    elif "fd_target.csv" in selected_target_csv_path:
        target_method = "Finite Difference"
    logger.info(f"Target Method Used: {target_method}")

    try:
        logger.info(f"Final Metrics - Full MSE: {full_mse:.6e}, Train MSE: {train_mse:.6e}, Test MSE: {test_mse:.6e}")
    except:
        logger.info("MSE metrics not available")
    logger.info("=" * 50)


if __name__ == '__main__':
    logger.info(f"Executing {__file__} as a script.")
    main()