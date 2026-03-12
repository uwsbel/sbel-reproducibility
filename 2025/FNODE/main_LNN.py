# FNODE/main_LNN.py
import os
import sys
import argparse
import logging
import torch
import numpy as np

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LNN_Main")

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
import time  # noqa: E402


def parse_arguments():
    parser = argparse.ArgumentParser(description="Main Script for Lagrangian Neural Network (LNN)")
    parser.add_argument('--test_case', type=str, default='Single_Mass_Spring',
                        choices=['Single_Mass_Spring', 'Single_Mass_Spring_Damper', 'Single_Mass_Spring_Symplectic'],
                        help="Dynamical system to simulate.")
    parser.add_argument('--generation_method', type=str, default='analytical',
                        choices=['analytical', 'rk4'],
                        help="Integrator used to generate training data.")
    parser.add_argument('--dt', type=float, default=0.01, help="Time step for training data.")
    parser.add_argument('--num_steps', type=int, default=3000, help="Total steps for generated trajectory.")
    parser.add_argument('--training_size', type=int, default=300, help="Number of steps used for training.")
    parser.add_argument('--num_epochs', type=int, default=400, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--lr_decay', type=float, default=0.98, help="Exponential LR decay.")
    parser.add_argument('--num_steps_test', type=int, default=3000, help="Number of rollout steps for testing.")
    parser.add_argument('--dt_test', type=float, default=0.01, help="Time step used for test rollout.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Computation device.")
    parser.add_argument('--skip_train', action='store_true', default=False,
                        help="Skip training and load existing checkpoint.")
    parser.add_argument('--model_load_filename', type=str, default='LNN_best.pkl',
                        help="Filename of saved LNN checkpoint.")
    args = parser.parse_args()
    args.model_type = 'LNN'
    return args


def main():
    args = parse_arguments()
    start_time = time.time()

    # Setup file logging
    log_dir = os.path.join(os.getcwd(), 'log', args.test_case, 'LNN')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'lnn.log'), mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=== LNN experiment started ===")
    logger.info(f"Test case: {args.test_case}, generation: {args.generation_method}, dt={args.dt}")

    set_seed(args.seed)
    device = torch.device(args.device)
    output_paths = get_output_paths(args.test_case, args.model_type)
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)

    # Data generation
    trajectory = generate_training_data(
        test_case=args.test_case,
        numerical_methods=args.generation_method,
        dt=args.dt,
        num_steps=args.num_steps
    )
    t_full = torch.linspace(0, args.dt * (args.num_steps - 1), args.num_steps)
    derivatives_np = get_xt_anal(trajectory.clone().cpu().numpy(), t_full.cpu().numpy())
    derivatives = torch.tensor(derivatives_np, dtype=torch.float32, device=device)

    # trajectory shape: [num_bodies, num_steps, 2]; train_lnn expects [num_steps, num_bodies, 2]
    # Ensure float32 to match LNN model (avoids "mat1 and mat2 must have the same dtype" error)
    trajectory_permuted = trajectory.permute(1, 0, 2).float()  # -> [num_steps, num_bodies, 2]
    derivatives_permuted = derivatives.permute(1, 0, 2)  # -> [num_steps, num_bodies, 2]
    num_bodys = trajectory.shape[0]  # number of bodies, not steps
    model = LNN(num_bodys=num_bodys).to(device)
    calculate_model_parameters(model)

    # Training
    if not args.skip_train:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
        train_params = {
            'epochs': args.num_epochs,
            'training_size': args.training_size
        }
        logger.info(f"Training LNN for {args.num_epochs} epochs on {args.training_size} steps.")
        train_lnn(model, trajectory_permuted.to(device), derivatives_permuted, train_params, optimizer, scheduler, output_paths)
        save_model_state(model, output_paths["model"], model_filename="LNN_final.pkl")
    else:
        loaded = load_model_state(model, output_paths["model"], model_filename=args.model_load_filename,
                                  current_device=device)
        if not loaded:
            logger.error("Requested skip_train but no checkpoint was found.")
            return

    # Testing rollout: state at t=0 (trajectory shape [num_bodies, num_steps, 2])
    initial_state = trajectory[:, 0, :].to(device)
    t_eval = torch.linspace(0, args.dt_test * (args.num_steps_test - 1), args.num_steps_test)
    pred_traj = test_lnn(model, initial_state, t_eval.cpu().numpy())

    # MSE vs analytical ground truth for Single Mass Spring
    try:
        if args.test_case == 'Single_Mass_Spring':
            t_np = t_eval.detach().cpu().numpy()
            m = 10.0
            k = 50.0
            omega = np.sqrt(k / m)
            x_true = np.cos(omega * t_np)
            v_true = -omega * np.sin(omega * t_np)
            truth = np.stack([x_true, v_true], axis=1)

            pred_np = pred_traj.detach().cpu().numpy() if torch.is_tensor(pred_traj) else np.asarray(pred_traj)
            if pred_np.ndim == 3:
                pred_np = pred_np[:, 0, :]

            n = min(len(pred_np), len(truth))
            if n > 0:
                diff = pred_np[:n, :2] - truth[:n, :2]
                mse_x = float(np.mean(diff[:, 0] ** 2))
                mse_v = float(np.mean(diff[:, 1] ** 2))
                mse_total = float(np.mean(diff ** 2))
                logger.info("MSE (vs analytical GT): mse_total=%.6e, mse_x=%.6e, mse_v=%.6e, N=%d",
                            mse_total, mse_x, mse_v, n)
    except Exception as e:
        logger.warning("Failed to compute MSE: %s", e)

    # Save numerical data
    save_data_np(pred_traj, args.test_case, args.model_type, args.training_size, args.num_steps_test, args.dt_test)
    logger.info("Saved rollout trajectory to results.")

    # Generate simplified plots using unified function
    plot_sms_results(
        pred_traj=pred_traj,
        model_type='LNN',
        training_size=args.training_size,
        dt=args.dt_test,
        output_dir=output_paths['figures'],
        num_steps_test=args.num_steps_test,
        start_time=start_time
    )

    logger.info(f"Model saved to: {output_paths['model']}")
    logger.info(f"Results saved to: {output_paths['results']}")
    logger.info(f"Figures saved to: {output_paths['figures']}")


if __name__ == "__main__":
    main()
