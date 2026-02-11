# FNODE/main_lstm.py - Simplified with clean logging
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
logger = logging.getLogger("LSTM_Main")

# --- Ensure Model directory is in Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'Model')
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)

# --- Imports ---
from Model.Data_generator import generate_dataset, generate_slider_crank_dataset
from Model.utils import set_seed, get_output_paths, calculate_model_parameters, save_data
from Model.model import LSTMModel, train_lstm


def parse_arguments():
    parser = argparse.ArgumentParser(description="Main Script for LSTM Model")

    # Experiment Setup
    parser.add_argument('--test_case', type=str, default='Single_Mass_Spring_Damper',
                        choices=['Single_Mass_Spring_Damper', 'Slider_Crank',
                                 'Double_Pendulum', 'Triple_Mass_Spring_Damper', 'Cart_Pole'],
                        help="Dynamical system to simulate.")
    parser.add_argument('--seed', type=int, default=42, help='Global random seed.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Computation device.")

    # Data Generation
    parser.add_argument('--generate_new_data', action='store_true', default=False,
                        help="Flag to generate new dataset.")
    parser.add_argument('--data_dt', type=float, default=0.01, help="Time step for data generation.")
    parser.add_argument('--data_total_steps', type=int, default=400, help="Total steps for generated data.")
    parser.add_argument('--train_ratio', type=float, default=0.75, help="Training data ratio.")

    # Model Hyperparameters
    parser.add_argument('--layers', type=int, default=3, help="Number of LSTM layers.")
    parser.add_argument('--hidden_size', type=int, default=256, help="LSTM hidden size.")
    parser.add_argument('--lstm_dropout', type=float, default=0.0, help="Dropout rate for LSTM.")
    parser.add_argument('--lstm_seq_len', type=int, default=16, help="Input sequence length for LSTM.")

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=300, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--lr_scheduler', type=str, default='exponential',
                        choices=['none', 'exponential', 'step'], help="LR scheduler.")
    parser.add_argument('--lr_decay_rate', type=float, default=0.98, help="Decay rate for scheduler.")

    # Testing/Plotting
    parser.add_argument('--skip_train', action='store_true', default=False,
                        help="Skip training, load model and test.")
    parser.add_argument('--model_load_filename', type=str, default="LSTM_best.pkl",
                        help="LSTM state filename to load.")

    args = parser.parse_args()
    args.model_type = 'LSTM'
    return args


def main():
    args = parse_arguments()

    # Setup file logging
    log_dir = os.path.join(os.getcwd(), 'log', args.test_case)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'lstm.log')

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"=== LSTM Training: {args.test_case} ===")
    logger.info(f"Model: {args.layers}-layer LSTM, hidden_size={args.hidden_size}, seq_len={args.lstm_seq_len}")
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
    s_test_df = pd.read_csv(os.path.join(dataset_path, "s_test.csv"))

    s_train_np = s_train_df.values[:, 1:] if s_train_df.columns[0] in ["idx", "Unnamed: 0"] else s_train_df.values
    s_test_np = s_test_df.values[:, 1:] if s_test_df.columns[0] in ["idx", "Unnamed: 0"] else s_test_df.values

    s_train = torch.tensor(s_train_np, dtype=torch.float32)
    s_test = torch.tensor(s_test_np, dtype=torch.float32)

    num_body = s_train.shape[1] // 2 if s_train.dim() == 2 else 1

    logger.info(f"Dataset: s_train={s_train.shape}, s_test={s_test.shape}, {num_body} bodies")

    # Initialize Model
    model = LSTMModel(num_body=num_body, hidden_size=args.hidden_size, num_layers=args.layers,
                      dropout_rate=args.lstm_dropout).to(device)
    calculate_model_parameters(model)

    # Training
    if not args.skip_train:
        logger.info("=== Starting Training ===")

        train_params = {
            'epochs': args.epochs,
            'lr': args.lr,
            'lr_scheduler': args.lr_scheduler,
            'lr_decay_rate': args.lr_decay_rate,
            'lstm_seq_len': args.lstm_seq_len,
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

        logger.info("=== LSTM Training ===")
        training_start_time = time.time()

        trained_model, loss_history = train_lstm(model, s_train, train_params, optimizer, scheduler, output_paths)

        total_training_time = time.time() - training_start_time
        logger.info(f"--- LSTM Training Finished: Total Time {total_training_time:.2f}s ---")

        # Save training time to CSV (will be updated with MSE results after testing)
        training_time_data = {
            'model': 'LSTM',
            'dataset': args.test_case,
            'training_time': total_training_time,
            'train_size': int(args.data_total_steps * args.train_ratio),
            'test_size': int(args.data_total_steps * (1 - args.train_ratio))
        }

        # Save final model
        final_model_path = os.path.join(output_paths["model"], "LSTM_final.pkl")
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

    # Load scaler for denormalization
    import pickle
    scaler_path = os.path.join(output_paths["model"], "lstm_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"Loaded scaler from {scaler_path}")

        # Normalize test data
        s_train_normalized = torch.tensor(scaler.transform(s_train.numpy()), dtype=torch.float32)
        s_test_normalized = torch.tensor(scaler.transform(s_test.numpy()), dtype=torch.float32)
        s_full_normalized = torch.cat((s_train_normalized, s_test_normalized), dim=0)
    else:
        logger.warning(f"Scaler not found at {scaler_path}, using unnormalized data")
        scaler = None
        s_train_normalized = s_train
        s_test_normalized = s_test
        s_full_normalized = torch.cat((s_train, s_test), dim=0)

    # For LSTM, we roll out predictions (on normalized data)
    total_steps = len(s_full_normalized)

    predictions_list = [s_train_normalized[i].unsqueeze(0) for i in range(args.lstm_seq_len)]

    with torch.no_grad():
        for step in range(args.lstm_seq_len, total_steps):
            input_seq = torch.stack(predictions_list[-args.lstm_seq_len:], dim=1).to(device)
            prediction = model(input_seq)  # LSTMModel.forward() returns only the output
            predictions_list.append(prediction.cpu())

    predictions_normalized = torch.cat(predictions_list, dim=0)

    # Denormalize predictions back to original scale
    if scaler is not None:
        predictions = torch.tensor(scaler.inverse_transform(predictions_normalized.numpy()), dtype=torch.float32)
        logger.info("Denormalized predictions back to original scale")
    else:
        predictions = predictions_normalized

    # Use original scale data for comparison
    s_full = torch.cat((s_train, s_test), dim=0)

    # Calculate metrics with extrapolation-focused evaluation (Justin's approach)
    mse_loss = torch.nn.MSELoss()

    # Overall MSE across full trajectory
    overall_mse = mse_loss(predictions, s_full)
    logger.info(f"Overall MSE (LSTM vs GT, {s_full.shape[0]} steps): {overall_mse.item():.6e}")

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
    metrics_path = os.path.join(output_paths["results"], "LSTM_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Save training time and MSE results to CSV
    if 'training_time_data' in locals():
        training_results_csv = os.path.join(output_paths["results"], "training_results.csv")
        import csv
        with open(training_results_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Dataset', 'Training_Time_Seconds', 'Train_Size', 'Test_Size',
                           'Overall_MSE', 'Train_MSE', 'Test_MSE'])
            writer.writerow(['LSTM', args.test_case, training_time_data['training_time'],
                           training_time_data['train_size'], training_time_data['test_size'],
                           overall_mse.item(), train_mse.item(), test_mse.item()])

    # Save predictions and ground truth as .npy files for plotting scripts
    training_size = len(s_train)
    test_size = len(s_test)
    save_data(
        data=predictions.cpu().detach().numpy(),
        test_case=args.test_case,
        model_type='LSTM',
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
    logger.info(f"Saved LSTM predictions and ground truth to results/{args.test_case}/")

    # Plot comparison
    from Model.utils import plot_trajectory_comparison
    predictions_3d = predictions.reshape(-1, num_body, 2)
    ground_truth_3d = s_full.reshape(-1, num_body, 2)
    time_vector = torch.arange(0, ground_truth_3d.shape[0]) * args.data_dt
    plot_trajectory_comparison(
        test_case_name=args.test_case,
        model_predictions={"LSTM": predictions_3d},
        ground_truth_trajectory=ground_truth_3d,
        time_vector=time_vector.cpu().numpy(),
        num_bodies_to_plot=num_body,
        num_steps_train=s_train.shape[0],
        output_dir=output_paths["figures"],
        base_filename=f"{args.test_case}_LSTM_comparison",
        num_epochs=args.epochs
    )
    phase_path = os.path.join(output_paths["figures"], f"{args.test_case}_LSTM_comparison_phasespace_epochs_{args.epochs}.png")
    time_path = os.path.join(output_paths["figures"], f"{args.test_case}_LSTM_comparison_timeseries_epochs_{args.epochs}.png")
    logger.info(f"Phase space plot: {phase_path}")
    logger.info(f"Time series plot: {time_path}")

    logger.info("=== LSTM Run Complete ===")


if __name__ == '__main__':
    main()