# FNODE/main_mbdnode.py - Simplified to match MNODE-code repo
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import sys
import logging
import pandas as pd

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("MBDNODE_Main")

# --- Ensure Model directory is in the Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir_abs = os.path.join(script_dir, 'Model')
if model_dir_abs not in sys.path:
    sys.path.insert(0, model_dir_abs)

# --- Imports ---
from Model.Data_generator import generate_dataset, generate_slider_crank_dataset
from Model.utils import set_seed, get_output_paths, calculate_model_parameters, save_data, plot_trajectory_comparison
from Model.model import MBDNODE, train_trajectory_MBDNODE, test_MBDNODE

# --- Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Main Script for MBDNODE Model")

    # Experiment Setup
    parser.add_argument('--test_case', type=str, default='Cart_Pole',
                        choices=['Single_Mass_Spring_Damper', 'Slider_Crank',
                                 'Double_Pendulum', 'Triple_Mass_Spring_Damper', 'Cart_Pole'],
                        help="Dynamical system to simulate.")
    parser.add_argument('--seed', type=int, default=5, help='Global random seed.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Computation device ('cuda' or 'cpu').")

    # Data Generation
    parser.add_argument('--generate_new_data', action='store_true', default=False,
                        help="Flag to generate new dataset.")
    parser.add_argument('--data_dt', type=float, default=0.01, help="Time step for data generation.")
    parser.add_argument('--data_total_steps', type=int, default=400, help="Total steps for generated data.")
    parser.add_argument('--train_ratio', type=float, default=0.75, help="Training data ratio (default 0.7).")

    # Model Hyperparameters
    parser.add_argument('--layers', type=int, default=2, help="Number of layers for NN models.")
    parser.add_argument('--hidden_size', type=int, default=256, help="Hidden layer width.")
    parser.add_argument('--activation', type=str, default='tanh', choices=['relu', 'tanh'],
                        help="Activation function.")
    parser.add_argument('--initializer', type=str, default='xavier', choices=['xavier', 'kaiming'],
                        help="Weight initializer.")

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for MBDNODE training (fixed at 1).")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of parallel workers for data loading (use 0 to avoid CUDA issues).")
    parser.add_argument('--verbose', action='store_true', help="Print detailed training progress.")
    parser.add_argument('--early_stop', type=str, default='False', help="Enable early stopping ('True' or 'False').")
    parser.add_argument('--patience', type=int, default=30, help="Patience for early stopping.")
    parser.add_argument('--step_delay', type=int, default=2, help="Number of steps to delay for MBDNODE training.")
    parser.add_argument('--numerical_methods', type=str, default='rk4', choices=['fe', 'rk4', 'midpoint'],
                        help="Numerical method for integration.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--lr_scheduler', type=str, default='exponential',
                        choices=['exponential', 'step', 'cosine'], help="LR scheduler type.")
    parser.add_argument('--lr_decay_rate', type=float, default=0.98, help="Decay rate for scheduler.")
    parser.add_argument('--lr_decay_steps', type=int, default=100, help="Step size for StepLR or T_max for CosineAnnealingLR.")

    # Testing/Plotting
    parser.add_argument('--skip_train', action='store_true', default=False,
                        help="Skip training, load model and test.")
    parser.add_argument('--model_load_filename', type=str, default="MBDNODE_best",
                        help="MBDNODE state filename to load.")

    args = parser.parse_args()
    args.model_type = 'MBDNODE'
    return args


def main():
    args = parse_arguments()

    # Setup file logging
    log_dir = os.path.join(os.getcwd(), 'log', args.test_case)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'MBDNODE.log')

    # Add file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"=== MBDNODE Training: {args.test_case} ===")
    logger.info(f"Model: {args.layers}-layer network, hidden_size={args.hidden_size}, activation={args.activation}")
    logger.info(f"Training: {args.epochs} epochs, LR={args.lr}, Scheduler={args.lr_scheduler}(gamma={args.lr_decay_rate})")
    logger.info(f"Integration: {args.numerical_methods}, step_delay={args.step_delay}, dt={args.data_dt}")

    device = torch.device(args.device)
    output_paths = get_output_paths(args.test_case, args.model_type)

    os.makedirs(output_paths["model"], exist_ok=True)
    os.makedirs(output_paths["results"], exist_ok=True)
    os.makedirs(output_paths["figures"], exist_ok=True)

    # Data Generation/Loading
    dataset_path = os.path.join('.', 'dataset', args.test_case)
    s_train_file = os.path.join(dataset_path, "s_train.csv")
    t_train_file = os.path.join(dataset_path, "t_train.csv")

    num_steps_train = int(args.data_total_steps * args.train_ratio)

    # Determine system structure
    test_case_config = {
        "Slider_Crank": (1, 2, 2),
        "Single_Mass_Spring_Damper": (1, 2, 2),
        "Double_Pendulum": (2, 2, 4),
        "Triple_Mass_Spring_Damper": (3, 2, 6),
        "Cart_Pole": (2, 2, 4)
    }
    num_bodies, states_per_body, expected_features = test_case_config[args.test_case]

    set_seed(args.seed)
    ground_trajectory = None

    if args.generate_new_data or not os.path.exists(s_train_file):
        logger.info(f"Generating dataset: {args.data_total_steps} total steps, {num_steps_train} train steps")
        if args.test_case == "Slider_Crank":
            generate_slider_crank_dataset(
                total_num_steps=args.data_total_steps*10,
                train_num_steps=num_steps_train*10,
                dt=args.data_dt, seed=args.seed, root_dir='.'
            )
        else:
            generate_dataset(
                test_case=args.test_case, numerical_methods="rk4", dt=args.data_dt,
                num_steps=args.data_total_steps, seed=args.seed,
                gen_train_num_steps=num_steps_train, output_root_dir='.',
                if_noise=False, save_to_file=True
            )

    # Always load from CSV files (matching Justin's approach)
    if True:
        logger.info("Loading existing dataset from CSV files")
        s_train_df = pd.read_csv(s_train_file)
        t_train_df = pd.read_csv(t_train_file)
        s_test_df = pd.read_csv(os.path.join(dataset_path, "s_test.csv"))

        s_train_np = s_train_df.values[:, 1:] if s_train_df.columns[0] in ["idx", "Unnamed: 0"] else s_train_df.values
        s_test_np = s_test_df.values[:, 1:] if s_test_df.columns[0] in ["idx", "Unnamed: 0"] else s_test_df.values

        if s_train_np.shape[1] != expected_features:
            logger.error(f"Data mismatch: expected {expected_features} features, got {s_train_np.shape[1]}")
            return

        s_train = torch.tensor(s_train_np, dtype=torch.float32, device=device).reshape(-1, num_bodies, states_per_body)
        s_test = torch.tensor(s_test_np, dtype=torch.float32, device=device).reshape(-1, num_bodies, states_per_body)
        ground_trajectory = torch.cat((s_train, s_test), dim=0)

    # Split ground_trajectory for training
    s_train = ground_trajectory[:num_steps_train]
    s_test = ground_trajectory[num_steps_train:]

    logger.info(f"Dataset: s_train={s_train.shape}, s_test={s_test.shape}, {num_bodies} bodies, {states_per_body} states/body")

    # Initialize Model
    model = MBDNODE(num_bodys=num_bodies, layers=args.layers, width=args.hidden_size,
                  activation=args.activation, initializer=args.initializer).to(device)
    calculate_model_parameters(model)

    # Training
    if not args.skip_train:
        logger.info("=== Starting Training ===")
        logger.info("=== MBDNODE Training ===")
        training_start_time = time.time()

        trained_model, loss_history = train_trajectory_MBDNODE(
            test_case=args.test_case,
            numerical_methods=args.numerical_methods,
            model=model,
            body_tensor=s_train,
            step_delay=args.step_delay,
            training_size=s_train.shape[0],
            num_epochs=args.epochs,
            dt=args.data_dt,
            device=device,
            verbose=args.verbose,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_paths=output_paths,
            train_ratio=args.train_ratio,
            lr=args.lr,
            lr_scheduler=args.lr_scheduler,
            lr_decay_rate=args.lr_decay_rate,
            lr_decay_steps=args.lr_decay_steps,
            early_stop=(args.early_stop.lower() == 'true'),
            patience=args.patience
        )

        total_training_time = time.time() - training_start_time
        logger.info(f"--- MBDNODE Training Finished: Total Time {total_training_time:.2f}s ---")

        # Save training time to CSV (will be updated with MSE results after testing)
        training_time_data = {
            'model': 'MBDNODE',
            'dataset': args.test_case,
            'training_time': total_training_time,
            'train_size': int(args.data_total_steps * args.train_ratio),
            'test_size': int(args.data_total_steps * (1 - args.train_ratio))
        }

        # Save final model
        final_model_path = os.path.join(output_paths["model"],
                                        f"MBDNODE_final_{args.numerical_methods}_{args.epochs}epochs.pth")
        torch.save(trained_model.state_dict(), final_model_path)
        logger.info(f"Final model saved: {final_model_path}")
        model = trained_model
    else:
        # Load model
        model_path = os.path.join(output_paths["model"], f"{args.model_load_filename}.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model from: {model_path}")

    # Testing
    logger.info("=== Starting Testing ===")
    model.eval()
    initial_state = s_train[0, :]
    num_test_steps = ground_trajectory.shape[0]

    test_trajectory = test_MBDNODE(
        numerical_methods=args.numerical_methods,
        model=model,
        body=initial_state,
        num_steps=num_test_steps,
        dt=args.data_dt,
        device=device
    )

    # Calculate MSE with extrapolation-focused evaluation (Justin's approach)
    mse_loss_fn = nn.MSELoss()

    # Overall MSE across full trajectory
    overall_mse = mse_loss_fn(test_trajectory, ground_trajectory.to(test_trajectory.device))
    logger.info(f"Overall MSE (MBDNODE vs GT, {ground_trajectory.shape[0]} steps): {overall_mse.item():.6e}")

    # Training region MSE (first num_steps_train steps)
    train_region_len = s_train.shape[0]
    if train_region_len > 0 and train_region_len <= ground_trajectory.shape[0] and train_region_len <= test_trajectory.shape[0]:
        train_mse = mse_loss_fn(
            test_trajectory[:train_region_len],
            ground_trajectory[:train_region_len].to(test_trajectory.device)
        )
        logger.info(f"Train Region MSE (first {train_region_len} steps): {train_mse.item():.6e}")
    else:
        train_mse = torch.tensor(float('nan'))
        logger.warning("Could not calculate train region MSE due to length mismatch or zero length.")

    # Extrapolation region MSE (steps beyond training time)
    if ground_trajectory.shape[0] > train_region_len and test_trajectory.shape[0] > train_region_len:
        extrapolation_mse = mse_loss_fn(
            test_trajectory[train_region_len:],
            ground_trajectory[train_region_len:].to(test_trajectory.device)
        )
        logger.info(f"Extrapolation Region MSE (steps {train_region_len} to {ground_trajectory.shape[0]}): {extrapolation_mse.item():.6e}")
        test_mse = extrapolation_mse  # For backward compatibility
    else:
        extrapolation_mse = torch.tensor(float('nan'))
        test_mse = torch.tensor(float('nan'))
        logger.info("No extrapolation region to evaluate or length mismatch.")

    # Save training time and MSE results to CSV
    if 'training_time_data' in locals():
        training_results_csv = os.path.join(output_paths["results"], "training_results.csv")
        import csv
        with open(training_results_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Dataset', 'Training_Time_Seconds', 'Train_Size', 'Test_Size',
                           'Overall_MSE', 'Train_MSE', 'Test_MSE'])
            writer.writerow(['MBDNODE', args.test_case, training_time_data['training_time'],
                           training_time_data['train_size'], training_time_data['test_size'],
                           overall_mse.item(), train_mse.item(), test_mse.item()])

    # Save predictions and metrics
    training_size = int(args.data_total_steps * args.train_ratio)
    test_size = args.data_total_steps - training_size

    save_data(
        data=test_trajectory.reshape(num_test_steps, -1).cpu().detach().numpy(),
        test_case=args.test_case,
        training_size=training_size,
        test_size=test_size,
        model_type=args.model_type,
        dt=args.data_dt
    )
    # Save ground truth for plotting scripts
    # ground_trajectory is already the full [train+test] trajectory, reshape to flat format
    ground_truth_full = ground_trajectory.reshape(-1, num_bodies * 2)
    save_data(
        data=ground_truth_full.cpu().detach().numpy(),
        test_case=args.test_case,
        training_size=training_size,
        test_size=test_size,
        model_type='Ground_truth',
        dt=args.data_dt
    )
    logger.info(f"Saved MBDNODE predictions and ground truth to results/{args.test_case}/")

    # Save metrics
    metrics_df = pd.DataFrame({
        'full_mse': [overall_mse.item()],
        'train_mse': [train_mse.item()],
        'test_mse': [test_mse.item()]
    })
    metrics_path = os.path.join(output_paths["results"], f"MBDNODE_{args.numerical_methods}_test_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Plot comparison
    time_vector = torch.arange(0, ground_trajectory.shape[0]) * args.data_dt
    plot_trajectory_comparison(
        test_case_name=args.test_case,
        model_predictions={f"MBDNODE_{args.numerical_methods}": test_trajectory},
        ground_truth_trajectory=ground_trajectory,
        time_vector=time_vector.cpu().numpy(),
        num_bodies_to_plot=num_bodies,
        num_steps_train=s_train.shape[0],
        output_dir=output_paths["figures"],
        base_filename=f"{args.test_case}_MBDNODE_{args.numerical_methods}_comparison",
        num_epochs=args.epochs
    )
    phase_path = os.path.join(output_paths["figures"], f"{args.test_case}_MBDNODE_{args.numerical_methods}_comparison_phasespace_epochs_{args.epochs}.png")
    time_path = os.path.join(output_paths["figures"], f"{args.test_case}_MBDNODE_{args.numerical_methods}_comparison_timeseries_epochs_{args.epochs}.png")
    logger.info(f"Phase space plot: {phase_path}")
    logger.info(f"Time series plot: {time_path}")

    logger.info("=== MBDNODE Run Complete ===")


if __name__ == '__main__':
    main()