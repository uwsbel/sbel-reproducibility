# FNODE/main_fcnn.py - Simplified with clean logging
import os
import argparse
import sys
import logging
import time
import torch
import pandas as pd

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FCNN_Main")

# --- Ensure Model directory is in Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'Model')
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

# --- Imports ---
from Model.Data_generator import generate_dataset, generate_slider_crank_dataset
from Model.utils import set_seed, get_output_paths, calculate_model_parameters, save_data
from Model.model import FCNN, train_fcnn


def parse_arguments():
    parser = argparse.ArgumentParser(description="Main Script for FCNN Model")

    # Experiment Setup
    parser.add_argument('--test_case', type=str, default='Double_Pendulum',
                        choices=['Single_Mass_Spring_Damper', 'Double_Pendulum',
                                 'Triple_Mass_Spring_Damper', 'Slider_Crank', 'Cart_Pole'],
                        help="Dynamical system to simulate.")
    parser.add_argument('--seed', type=int, default=42, help='Global random seed.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Computation device.")

    # Data Generation
    parser.add_argument('--generate_new_data', action='store_true', default=True,
                        help="Flag to generate new dataset.")
    parser.add_argument('--data_dt', type=float, default=0.01, help="Time step for data generation.")
    parser.add_argument('--data_total_steps', type=int, default=400, help="Total steps for generated data.")
    parser.add_argument('--train_ratio', type=float, default=0.75, help="Training data ratio.")

    # Model Hyperparameters
    parser.add_argument('--layers', type=int, default=2, help="Number of layers for FCNN.")
    parser.add_argument('--hidden_size', type=int, default=256, help="Hidden layer width.")
    parser.add_argument('--activation', type=str, default='tanh', choices=['relu', 'tanh'],
                        help="Activation function.")
    parser.add_argument('--initializer', type=str, default='xavier', choices=['xavier', 'kaiming'],
                        help="Weight initializer.")

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=400, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--lr_scheduler', type=str, default='step',
                        choices=['none', 'exponential', 'step'], help="LR scheduler.")
    parser.add_argument('--lr_decay_rate', type=float, default=0.98, help="Decay rate for scheduler.")

    # Testing/Plotting
    parser.add_argument('--skip_train', action='store_true', default=False,
                        help="Skip training, load model and test.")
    parser.add_argument('--model_load_filename', type=str, default="FCNN_best.pkl",
                        help="FCNN state filename to load.")

    args = parser.parse_args()
    args.model_type = 'FCNN'
    return args


def main():
    args = parse_arguments()

    # Setup file logging
    log_dir = os.path.join(os.getcwd(), 'log', args.test_case)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'fcnn.log')

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"=== FCNN Training: {args.test_case} ===")
    logger.info(f"Model: {args.layers}-layer network, hidden_size={args.hidden_size}, activation={args.activation}")
    logger.info(f"Training: {args.epochs} epochs, LR={args.lr}, Scheduler={args.lr_scheduler}(gamma={args.lr_decay_rate})")

    set_seed(args.seed)
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
                save_to_file=True
            )

    # Load Data
    s_train_df = pd.read_csv(s_train_file)
    t_train_df = pd.read_csv(t_train_file)
    s_test_df = pd.read_csv(os.path.join(dataset_path, "s_test.csv"))
    t_test_df = pd.read_csv(os.path.join(dataset_path, "t_test.csv"))

    s_train_np = s_train_df.values[:, 1:] if s_train_df.columns[0] in ["idx", "Unnamed: 0"] else s_train_df.values
    s_test_np = s_test_df.values[:, 1:] if s_test_df.columns[0] in ["idx", "Unnamed: 0"] else s_test_df.values
    t_train_np = t_train_df['time'].values if 'time' in t_train_df.columns else t_train_df.values.flatten()
    t_test_np = t_test_df['time'].values if 'time' in t_test_df.columns else t_test_df.values.flatten()

    s_train = torch.tensor(s_train_np, dtype=torch.float32)
    t_train = torch.tensor(t_train_np, dtype=torch.float32)
    s_test = torch.tensor(s_test_np, dtype=torch.float32)
    t_test = torch.tensor(t_test_np, dtype=torch.float32)

    num_bodies = s_train.shape[1] // 2 if s_train.dim() == 2 else 1

    logger.info(f"Dataset: s_train={s_train.shape}, s_test={s_test.shape}, {num_bodies} bodies")

    # Initialize Model
    model = FCNN(num_bodys=num_bodies, layers=args.layers, width=args.hidden_size,
                 activation=args.activation, initializer=args.initializer).to(device)
    calculate_model_parameters(model)

    # Training
    if not args.skip_train:
        logger.info("=== Starting Training ===")

        train_params = {
            'epochs': args.epochs,
            'lr': args.lr,
            'lr_scheduler': args.lr_scheduler,
            'lr_decay_rate': args.lr_decay_rate,
            'test_case': args.test_case,
            'outime_log': 1,
            'save_ckpt_freq': 0
        }

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if args.lr_scheduler == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)
        elif args.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=args.lr_decay_rate)
        else:
            scheduler = None

        logger.info("=== FCNN Training ===")
        training_start_time = time.time()

        trained_model, loss_history = train_fcnn(model, s_train, t_train, train_params, optimizer, scheduler, output_paths)

        total_training_time = time.time() - training_start_time
        logger.info(f"--- FCNN Training Finished: Total Time {total_training_time:.2f}s ---")

        # Save training time to CSV (will be updated with MSE results after testing)
        training_time_data = {
            'model': 'FCNN',
            'dataset': args.test_case,
            'training_time': total_training_time,
            'train_size': int(args.data_total_steps * args.train_ratio),
            'test_size': int(args.data_total_steps * (1 - args.train_ratio))
        }

        # Save final model
        final_model_path = os.path.join(output_paths["model"], "FCNN_final.pkl")
        torch.save(trained_model.state_dict(), final_model_path)
        logger.info(f"Final model saved: {final_model_path}")
        model = trained_model
    else:
        # Load model
        model_path = os.path.join(output_paths["model"], args.model_load_filename)
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model from: {model_path}")

    # Testing
    logger.info("=== Starting Testing ===")
    model.eval()

    s_full = torch.cat((s_train, s_test), dim=0)
    t_full = torch.cat((t_train, t_test), dim=0)

    # FCNN uses time as input
    with torch.no_grad():
        predictions = model(t_full.unsqueeze(1).to(device)).cpu()

    # Calculate metrics with extrapolation-focused evaluation (Justin's approach)
    mse_loss = torch.nn.MSELoss()

    # Overall MSE across full trajectory
    overall_mse = mse_loss(predictions, s_full)
    logger.info(f"Overall MSE (FCNN vs GT, {s_full.shape[0]} steps): {overall_mse.item():.6e}")

    # Training region MSE (first len(s_train) steps)
    train_region_len = len(s_train)
    if train_region_len > 0 and train_region_len <= s_full.shape[0] and train_region_len <= predictions.shape[0]:
        train_mse = mse_loss(
            predictions[:train_region_len],
            s_full[:train_region_len]
        )
        logger.info(f"Train Region MSE (first {train_region_len} steps): {train_mse.item():.6e}")
    else:
        train_mse = torch.tensor(float('nan'))
        logger.warning("Could not calculate train region MSE due to length mismatch or zero length.")

    # Extrapolation region MSE (steps beyond training time)
    if s_full.shape[0] > train_region_len and predictions.shape[0] > train_region_len:
        extrapolation_mse = mse_loss(
            predictions[train_region_len:],
            s_full[train_region_len:]
        )
        logger.info(f"Extrapolation Region MSE (steps {train_region_len} to {s_full.shape[0]}): {extrapolation_mse.item():.6e}")
        test_mse = extrapolation_mse  # For backward compatibility
    else:
        extrapolation_mse = torch.tensor(float('nan'))
        test_mse = torch.tensor(float('nan'))
        logger.info("No extrapolation region to evaluate or length mismatch.")

    # Save metrics
    metrics_df = pd.DataFrame({
        'full_mse': [overall_mse.item()],
        'train_mse': [train_mse.item()],
        'test_mse': [test_mse.item()]
    })
    metrics_path = os.path.join(output_paths["results"], "FCNN_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Save training time and MSE results to CSV
    if 'training_time_data' in locals():
        training_results_csv = os.path.join(output_paths["results"], "training_results.csv")
        import csv
        with open(training_results_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Dataset', 'Training_Time_Seconds', 'Train_Size', 'Test_Size',
                           'Overall_MSE', 'Train_MSE', 'Test_MSE'])
            writer.writerow(['FCNN', args.test_case, training_time_data['training_time'],
                           training_time_data['train_size'], training_time_data['test_size'],
                           overall_mse.item(), train_mse.item(), test_mse.item()])

    # Save predictions and ground truth as .npy files for plotting scripts
    training_size = len(s_train)
    test_size = len(s_test)
    save_data(
        data=predictions.cpu().detach().numpy(),
        test_case=args.test_case,
        model_type='FCNN',
        training_size=training_size,
        test_size=test_size,
        dt=args.data_dt
    )
    save_data(
        data=s_full.cpu().detach().numpy(),
        test_case=args.test_case,
        model_type='Ground_truth',
        training_size=training_size,
        test_size=test_size,
        dt=args.data_dt
    )
    logger.info(f"Saved FCNN predictions and ground truth to results/{args.test_case}/")

    # Plot comparison
    from Model.utils import plot_trajectory_comparison
    predictions_3d = predictions.reshape(-1, num_bodies, 2)
    ground_truth_3d = s_full.reshape(-1, num_bodies, 2)
    time_vector = torch.arange(0, ground_truth_3d.shape[0]) * args.data_dt
    plot_trajectory_comparison(
        test_case_name=args.test_case,
        model_predictions={"FCNN": predictions_3d},
        ground_truth_trajectory=ground_truth_3d,
        time_vector=time_vector.cpu().numpy(),
        num_bodies_to_plot=num_bodies,
        num_steps_train=s_train.shape[0],
        output_dir=output_paths["figures"],
        base_filename=f"{args.test_case}_FCNN_comparison",
        num_epochs=args.epochs
    )
    phase_path = os.path.join(output_paths["figures"], f"{args.test_case}_FCNN_comparison_phasespace_epochs_{args.epochs}.png")
    time_path = os.path.join(output_paths["figures"], f"{args.test_case}_FCNN_comparison_timeseries_epochs_{args.epochs}.png")
    logger.info(f"Phase space plot: {phase_path}")
    logger.info(f"Time series plot: {time_path}")

    logger.info("=== FCNN Run Complete ===")


if __name__ == '__main__':
    main()