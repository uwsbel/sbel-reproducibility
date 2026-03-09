# FNODE/main_HNN.py
import os
import sys
import argparse
import logging
import torch
import numpy as np
import time

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("HNN_Main")

# --- Ensure Model directory is on path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'Model')
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

# --- Imports ---
from Model.Data_generator import *
from Model.utils import *
from Model.model import *
from Model.force_fun import *


def parse_arguments():
    parser = argparse.ArgumentParser(description="Main Script for Hamiltonian Neural Network (HNN)")
    parser.add_argument('--test_case', type=str, default='Single_Mass_Spring',
                        choices=['Single_Mass_Spring'], help="Dynamical system to simulate.")
    parser.add_argument('--dt', type=float, default=0.01, help="Time step for generated dataset.")
    parser.add_argument('--t_end', type=float, default=30.0, help="End time for dataset generation.")
    parser.add_argument('--samples', type=int, default=1, help="Number of dataset samples to concatenate.")
    parser.add_argument('--test_split', type=float, default=0.1, help="Fraction of data reserved for testing.")
    parser.add_argument('--num_epochs', type=int, default=30000, help="Training iterations.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay for optimizer.")
    parser.add_argument('--num_steps_test', type=int, default=3000, help="Number of rollout steps for testing.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--log_interval', type=int, default=0, help="Logging interval (0 = auto).")
    parser.add_argument('--if_symplectic', action='store_true', default=True,
                        help="Placeholder flag to align with PNODE config (kept for compatibility).")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Computation device.")
    parser.add_argument('--skip_train', action='store_true', default=False,
                        help="Skip training and load existing checkpoint.")
    parser.add_argument('--model_load_filename', type=str, default='HNN_best.pkl',
                        help="Filename of saved HNN checkpoint.")
    args = parser.parse_args()
    args.model_type = 'HNN'
    return args


def main():
    args = parse_arguments()
    start_time = time.time()

    # Setup file logging
    log_dir = os.path.join(os.getcwd(), 'log', args.test_case, 'HNN')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'hnn.log'), mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=== HNN experiment started ===")
    logger.info(f"dt={args.dt}, t_end={args.t_end}, epochs={args.num_epochs}")

    set_seed(args.seed)
    device = torch.device(args.device)
    output_paths = get_output_paths(args.test_case, args.model_type)
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)

    # Dataset
    timescale = int(round(1.0 / args.dt))
    data = generate_hnn_dataset(
        seed=args.seed,
        samples=args.samples,
        test_split=args.test_split,
        t_span=(0.0, args.t_end),
        timescale=timescale
    )

    # Model
    base_model = MLP(input_dim=2, output_dim=1)
    model = HNN(input_dim=2, differentiable_model=base_model).to(device)
    calculate_model_parameters(model)

    # Training
    if not args.skip_train:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
        train_params = {
            'epochs': args.num_epochs,
            'log_interval': args.log_interval if args.log_interval > 0 else max(1, args.num_epochs // 10)
        }
        logger.info(f"Training HNN for {args.num_epochs} steps.")
        train_hnn(model, data, train_params, optimizer, scheduler, output_paths)
        save_model_state(model, output_paths["model"], model_filename="HNN_final.pkl")
    else:
        loaded = load_model_state(model, output_paths["model"], model_filename=args.model_load_filename,
                                  current_device=device)
        if not loaded:
            logger.error("Requested skip_train but no checkpoint was found.")
            return

    # Testing rollout
    t_eval = np.linspace(0.0, args.t_end, args.num_steps_test)
    x0 = data['x'][0]
    pred_traj = hnn_integrate(model, x0, t_eval, step_size=args.dt)

    # Save numerical data
    save_data_np(pred_traj, args.test_case, args.model_type, len(data['x']), args.num_steps_test, args.dt)
    logger.info("Saved rollout trajectory to results.")

    # Convert (q,p) to (x,v) for consistent visualization
    pred_traj_xv = pred_traj.copy()
    pred_traj_xv[:, 1] = pred_traj[:, 1] / 10.0  # v = p/m where m=10.0

    # MSE vs analytical ground truth (Single Mass Spring)
    try:
        m = 10.0
        k = 50.0
        omega = np.sqrt(k / m)
        x_true = np.cos(omega * t_eval)
        v_true = -omega * np.sin(omega * t_eval)
        truth = np.stack([x_true, v_true], axis=1)
        n = min(len(pred_traj_xv), len(truth))
        if n > 0:
            diff = pred_traj_xv[:n, :2] - truth[:n, :2]
            mse_x = float(np.mean(diff[:, 0] ** 2))
            mse_v = float(np.mean(diff[:, 1] ** 2))
            mse_total = float(np.mean(diff ** 2))
            logger.info("MSE (vs analytical GT): mse_total=%.6e, mse_x=%.6e, mse_v=%.6e, N=%d",
                        mse_total, mse_x, mse_v, n)
    except Exception as e:
        logger.warning("Failed to compute MSE: %s", e)

    # Generate simplified plots using unified function
    plot_sms_results(
        pred_traj=pred_traj_xv,
        model_type='HNN',
        training_size=len(data['x']),
        dt=args.dt,
        output_dir=output_paths['figures'],
        num_steps_test=args.num_steps_test,
        start_time=start_time
    )

    logger.info(f"Model saved to: {output_paths['model']}")
    logger.info(f"Results saved to: {output_paths['results']}")
    logger.info(f"Figures saved to: {output_paths['figures']}")


if __name__ == "__main__":
    main()
