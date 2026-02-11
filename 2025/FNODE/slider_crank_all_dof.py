import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
import matplotlib.ticker as ticker

# Optional dataset generator (if available in workspace)
try:
    from Model.Data_generator import generate_slider_crank_dataset
except Exception:
    generate_slider_crank_dataset = None

# -------------------------------------------------------------------
#  Physical parameters
# -------------------------------------------------------------------
L1 = 1.0  # Half-length of crank
L2 = 2.0  # Half-length of rod


# -------------------------------------------------------------------
#  Kinematic Reconstruction Function and Jacobian
# -------------------------------------------------------------------
def constraint_jacobian_slider_crank(q: np.ndarray) -> np.ndarray:
    x1, y1, th1, x2, y2, th2, x3, y3, th3 = q
    J_matrix = np.zeros((8, 9))
    J_matrix[0, 0] = 1.0
    J_matrix[0, 2] = L1 * np.sin(th1)
    J_matrix[1, 1] = 1.0
    J_matrix[1, 2] = -L1 * np.cos(th1)
    J_matrix[2, 0] = 1.0
    J_matrix[2, 2] = -L1 * np.sin(th1)
    J_matrix[2, 3] = -1.0
    J_matrix[2, 5] = -L2 * np.sin(th2)
    J_matrix[3, 1] = 1.0
    J_matrix[3, 2] = L1 * np.cos(th1)
    J_matrix[3, 4] = -1.0
    J_matrix[3, 5] = L2 * np.cos(th2)
    J_matrix[4, 3] = 1.0
    J_matrix[4, 5] = -L2 * np.sin(th2)
    J_matrix[4, 6] = -1.0
    J_matrix[5, 4] = 1.0
    J_matrix[5, 5] = L2 * np.cos(th2)
    J_matrix[5, 7] = -1.0
    J_matrix[6, 7] = 1.0
    J_matrix[7, 8] = 1.0
    return J_matrix


def reconstruct_slider_crank_kinematics(theta1_series: np.ndarray, omega1_series: np.ndarray,
                                        L1_param: float, L2_param: float) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(theta1_series, np.ndarray) or not isinstance(omega1_series, np.ndarray):
        raise TypeError("theta1_series and omega1_series must be NumPy arrays.")
    if theta1_series.ndim != 1 or omega1_series.ndim != 1:
        raise ValueError("theta1_series and omega1_series must be 1D arrays.")
    if len(theta1_series) != len(omega1_series):
        raise ValueError("theta1_series and omega1_series must have the same length.")
    if len(theta1_series) == 0:
        print("Warning: Empty theta1/omega1 series for reconstruction. Returning empty arrays.")
        return np.array([]).reshape(0, 9), np.array([]).reshape(0, 9)

    N_steps = len(theta1_series)
    n_gen_coords = 9
    q_history = np.zeros((N_steps, n_gen_coords))
    v_history = np.zeros((N_steps, n_gen_coords))

    for k in range(N_steps):
        q_k, v_k = np.zeros(n_gen_coords), np.zeros(n_gen_coords)
        th1_k, w1_k = theta1_series[k], omega1_series[k]

        q_k[2] = th1_k
        q_k[0] = L1_param * np.cos(th1_k)
        q_k[1] = L1_param * np.sin(th1_k)
        q_k[7] = 0.0
        q_k[8] = 0.0

        sin_th2_val = -(L1_param / L2_param) * np.sin(th1_k)
        sin_th2 = np.clip(sin_th2_val, -1.0, 1.0)
        if abs(sin_th2_val) > 1.000001:
            pass

        th2_k = np.arcsin(sin_th2)
        q_k[5] = th2_k
        cos_th2_k = np.cos(th2_k)
        q_k[4] = -L2_param * sin_th2
        q_k[3] = 2 * L1_param * np.cos(th1_k) + L2_param * cos_th2_k
        q_k[6] = q_k[3] + L2_param * cos_th2_k
        q_history[k, :] = q_k

        v_k[2] = w1_k
        J_matrix_k = constraint_jacobian_slider_crank(q_k)
        unknown_vel_indices = [0, 1, 3, 4, 5, 6, 7, 8]
        J_mod_k = J_matrix_k[:, unknown_vel_indices]
        rhs_k = -J_matrix_k[:, 2] * w1_k
        try:
            v_unknown_k = np.linalg.solve(J_mod_k, rhs_k)
            for i_store, i_actual in enumerate(unknown_vel_indices):
                v_k[i_actual] = v_unknown_k[i_store]
        except np.linalg.LinAlgError:
            v_k[unknown_vel_indices] = np.nan
        v_history[k, :] = v_k

    return q_history, v_history


def plot_all_models_comparison(time_array: np.ndarray,
                              q_gt: np.ndarray, v_gt: np.ndarray,
                              models_data: dict,
                              output_dir: str,
                              train_size: int = 800,
                              dataset_name: str = "train800_test800"):
    """
    Create 4 subplot figures comparing all models for full DOF system
    Each figure shows one model vs ground truth for all DOFs
    """
    legend_fontsize = 16
    label_fontsize = 16

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Ensure consistent length
    min_len = min(len(time_array), len(q_gt))
    time_plot = time_array[:min_len]

    # Determine split point for ID/OOD
    split_idx = min(train_size, min_len)

    # Create a figure for each model
    for model_idx, (model_name, (q_pred, v_pred)) in enumerate(models_data.items()):
        if q_pred is None or v_pred is None:
            print(f"Skipping {model_name} - no data available")
            continue

        print(f"Creating plot for {model_name}...")

        # Check data consistency
        if len(q_pred) < min_len or len(v_pred) < min_len:
            print(f"  Warning: {model_name} predictions have fewer points than expected.")
            print(f"    Expected: {min_len}, Got q_pred: {len(q_pred)}, v_pred: {len(v_pred)}")

        # Create figure with 6x2 subplots for this model
        fig, axs = plt.subplots(6, 2, figsize=(14, 18), sharex=False)

        # Define what to plot in each subplot
        plot_config = [
            # Left column - positions
            {'ax': axs[0, 0], 'gt': q_gt[:, 0], 'pred': q_pred[:, 0], 'ylabel': r'$x_1$'},
            {'ax': axs[1, 0], 'gt': q_gt[:, 1], 'pred': q_pred[:, 1], 'ylabel': r'$y_1$'},
            {'ax': axs[2, 0], 'gt': q_gt[:, 3], 'pred': q_pred[:, 3], 'ylabel': r'$x_2$'},
            {'ax': axs[3, 0], 'gt': q_gt[:, 4], 'pred': q_pred[:, 4], 'ylabel': r'$y_2$'},
            {'ax': axs[4, 0], 'gt': q_gt[:, 5], 'pred': q_pred[:, 5], 'ylabel': r'$\theta_2$'},
            {'ax': axs[5, 0], 'gt': q_gt[:, 6], 'pred': q_pred[:, 6], 'ylabel': r'$x_3$'},

            # Right column - velocities
            {'ax': axs[0, 1], 'gt': v_gt[:, 0], 'pred': v_pred[:, 0], 'ylabel': r'$\dot{x}_1$'},
            {'ax': axs[1, 1], 'gt': v_gt[:, 1], 'pred': v_pred[:, 1], 'ylabel': r'$\dot{y}_1$'},
            {'ax': axs[2, 1], 'gt': v_gt[:, 3], 'pred': v_pred[:, 3], 'ylabel': r'$\dot{x}_2$'},
            {'ax': axs[3, 1], 'gt': v_gt[:, 4], 'pred': v_pred[:, 4], 'ylabel': r'$\dot{y}_2$'},
            {'ax': axs[4, 1], 'gt': v_gt[:, 5], 'pred': v_pred[:, 5], 'ylabel': r'$\dot{\theta}_2$'},
            {'ax': axs[5, 1], 'gt': v_gt[:, 6], 'pred': v_pred[:, 6], 'ylabel': r'$\dot{x}_3$'},
        ]

        # Plot each subplot
        for idx, config in enumerate(plot_config):
            ax = config['ax']
            gt_data = config['gt'][:min_len]
            pred_data = config['pred']

            # Ensure pred_data and time_plot have same length
            actual_pred_len = min(len(pred_data), min_len)
            pred_data = pred_data[:actual_pred_len]
            time_plot_adjusted = time_plot[:actual_pred_len]
            gt_data_adjusted = gt_data[:actual_pred_len]

            # Plot ground truth
            ax.plot(time_plot_adjusted, gt_data_adjusted, 'b-', linewidth=2, label='Ground truth')

            # Plot ID generalization (training region)
            if split_idx > 0 and split_idx <= actual_pred_len:
                ax.plot(time_plot_adjusted[:split_idx], pred_data[:split_idx], 'r--',
                       linewidth=2.5, label='ID generalization')

            # Plot OOD generalization (test region)
            if split_idx < actual_pred_len:
                ax.plot(time_plot_adjusted[split_idx:], pred_data[split_idx:], 'r:',
                       linewidth=2.5, label='OOD generalization')

            # Set ylabel
            ax.set_ylabel(config['ylabel'], fontsize=label_fontsize)

            # Set x-axis limits and ticks for all subplots
            ax.set_xlim(time_plot_adjusted[0], time_plot_adjusted[-1])
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=12))

            # Add grid
            ax.grid(True, alpha=0.3)

            # Add subplot label (a), (b), etc.
            subplot_label = chr(97 + idx)  # 97 is ASCII for 'a'
            ax.text(0.02, 0.95, f'({subplot_label})', transform=ax.transAxes,
                    fontsize=12, verticalalignment='top')

            # Set xlabel only for bottom row (idx=5 for left column, idx=11 for right column)
            if idx == 5 or idx == 11:
                ax.set_xlabel('t', fontsize=label_fontsize)

        # Add legend above the figure (outside plot area) to avoid overlap with subplots
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels,
                  loc='upper center', ncol=3,
                  bbox_to_anchor=(0.5, 0.98), fontsize=legend_fontsize)

        # Adjust layout and leave extra top margin for legend/title
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])

        # Save figure
        output_path = os.path.join(output_dir, f'{model_name.lower()}_slider_crank_{dataset_name}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {model_name} plot to: {output_path}")


def ensure_ground_truth_and_time(gt_driver_filepath: str,
                                 time_filepath: str,
                                 train_size: int,
                                 test_size: int,
                                 dt: float,
                                 base_results_path: str = 'results/Slider_Crank') -> None:
    """
    Ensure ground-truth npy and time npy exist. If missing, try to build them from
    CSVs in dataset/Slider_Crank or (optionally) call the generator if available.
    This writes files at gt_driver_filepath and time_filepath.
    """
    os.makedirs(base_results_path, exist_ok=True)
    total_steps = train_size + test_size

    # Try to build ground truth npy from dataset CSVs
    if not os.path.exists(gt_driver_filepath):
        dataset_dir = os.path.join('dataset', 'Slider_Crank')
        s_full_csv = os.path.join(dataset_dir, 's_full.csv')
        t_full_csv = os.path.join(dataset_dir, 't_full.csv')

        if os.path.exists(s_full_csv):
            print(f"Building ground-truth from CSV: {s_full_csv}")
            df = pd.read_csv(s_full_csv)
            if 'theta_0_2pi' in df.columns and 'omega' in df.columns:
                arr = df[['theta_0_2pi', 'omega']].values
            else:
                vals = df.values
                start_col = 1 if str(df.columns[0]).startswith('Unnamed') else 0
                if vals.shape[1] < start_col + 2:
                    raise RuntimeError(f"s_full.csv has unexpected shape: {vals.shape}")
                arr = vals[:, start_col:start_col + 2]

            if arr.shape[0] < total_steps:
                raise RuntimeError(f"s_full.csv contains {arr.shape[0]} steps which is less than required {total_steps}")

            arr_trunc = arr[:total_steps, :]
            np.save(gt_driver_filepath, arr_trunc)
            print(f"Saved generated ground-truth to: {gt_driver_filepath}")

        elif generate_slider_crank_dataset is not None:
            print("Ground truth not found; calling generate_slider_crank_dataset to produce dataset CSVs...")
            try:
                generate_slider_crank_dataset(total_num_steps=total_steps,
                                              train_num_steps=train_size,
                                              dt=dt, root_dir='.', seed=42)
            except TypeError:
                # fallback for older signature without seed
                try:
                    generate_slider_crank_dataset(total_num_steps=total_steps,
                                                  train_num_steps=train_size,
                                                  dt=dt, root_dir='.')
                except Exception as e:
                    raise RuntimeError(f"Generator failed: {e}")

            # try to read CSV again
            if os.path.exists(os.path.join('dataset', 'Slider_Crank', 's_full.csv')):
                print("Generator produced CSVs; building ground-truth npy...")
                df = pd.read_csv(os.path.join('dataset', 'Slider_Crank', 's_full.csv'))
                if 'theta_0_2pi' in df.columns and 'omega' in df.columns:
                    arr = df[['theta_0_2pi', 'omega']].values
                else:
                    vals = df.values
                    start_col = 1 if str(df.columns[0]).startswith('Unnamed') else 0
                    arr = vals[:, start_col:start_col + 2]
                arr_trunc = arr[:total_steps, :]
                np.save(gt_driver_filepath, arr_trunc)
                print(f"Saved generated ground-truth to: {gt_driver_filepath}")
            else:
                raise RuntimeError("Generator did not produce expected CSVs (s_full.csv).")
        else:
            raise RuntimeError(f"Ground truth file not found: {gt_driver_filepath}.\n"
                               "Provide dataset CSVs in dataset/Slider_Crank or enable the generator.")

    # Ensure time file exists
    if not os.path.exists(time_filepath):
        dataset_dir = os.path.join('dataset', 'Slider_Crank')
        t_full_csv = os.path.join(dataset_dir, 't_full.csv')
        if os.path.exists(t_full_csv):
            print(f"Building time array from CSV: {t_full_csv}")
            tdf = pd.read_csv(t_full_csv)
            if 'time' in tdf.columns:
                tarr = tdf['time'].values[:total_steps]
            else:
                tarr = tdf.values.flatten()[:total_steps]
        else:
            tarr = np.arange(total_steps) * dt

        np.save(time_filepath, tarr)
        print(f"Saved time array to: {time_filepath}")


# --- Main Script ---
if __name__ == '__main__':
    script_name = "slider_crank_all_dof.py"
    print(f"Running {script_name}: Slider-Crank All Models Full DOF Comparisons")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot Slider-Crank full DOF comparisons')
    parser.add_argument('--train_size', type=int, default=1500,
                       help='Training size for ID/OOD split (default: 1500)')
    parser.add_argument('--test_size', type=int, default=3000,
                       help='Test size (default: 3000)')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Time step (default: 0.01)')
    args = parser.parse_args()

    base_results_path = 'results/Slider_Crank'
    output_base_dir = './figures/Slider_Crank/comparison'

    # Dataset configuration based on arguments
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size
    DT = args.dt
    dataset_suffix = f"train{TRAIN_SIZE}_test{TEST_SIZE}_dt{DT}"

    print(f"Using dataset: {dataset_suffix}")

    # --- File paths ---
    gt_driver_filename = f"Ground_truth_prediction_{dataset_suffix}.npy"
    gt_driver_filepath = os.path.join(base_results_path, gt_driver_filename)

    # Check for time data file or generate it
    time_filename = 'Ground_truth_time_data.npy'
    time_filepath = os.path.join(base_results_path, time_filename)

    # --- Model Prediction Files ---
    # The repository contains several filename conventions; try to locate a matching file
    # for each model by searching results/Slider_Crank (and its subfolders).
    def locate_model_file(model_key: str) -> str | None:
        """Search base_results_path (recursively) for a .npy file matching the model and dataset sizes.

        Returns absolute path if found, otherwise None.
        """
        candidates = []
        search_tokens = [model_key.lower(), str(TRAIN_SIZE), str(TEST_SIZE), str(DT)]
        # Walk the results directory
        for root, dirs, files in os.walk(base_results_path):
            for f in files:
                if not f.lower().endswith('.npy'):
                    continue
                fname = f.lower()
                # skip ground truth/time files
                if 'ground_truth' in fname or 'groundtruth' in fname or 'ground_truth_prediction' in fname:
                    # still allow ground truth elsewhere; models lookup should skip these
                    pass
                # require model name token
                if model_key.lower() not in fname:
                    continue
                # prefer filenames that include train and test numbers
                if str(TRAIN_SIZE) in fname and str(TEST_SIZE) in fname:
                    candidates.append(os.path.join(root, f))
                else:
                    # still accept files that include model name but not sizes
                    candidates.append(os.path.join(root, f))

        # Prefer candidate that contains both train and test numbers and dt
        def score(path: str) -> int:
            s = 0
            low = path.lower()
            if str(TRAIN_SIZE) in low:
                s += 4
            if str(TEST_SIZE) in low:
                s += 4
            if str(DT) in low:
                s += 1
            if model_key.lower() in low:
                s += 8
            return s

        if not candidates:
            return None
        candidates.sort(key=lambda p: score(p), reverse=True)
        return candidates[0]

    model_names = ["FNODE", "LSTM", "MBDNODE", "FCNN"]

    # Locate model prediction files and report
    model_prediction_files: dict[str, str | None] = {}
    print("\nChecking for data files:")
    for model_name in model_names:
        found = locate_model_file(model_name)
        model_prediction_files[model_name] = found
        if found is not None and os.path.exists(found):
            print(f"  ✓ {model_name}: {found}")
        else:
            print(f"  ✗ {model_name}: not found (searched under {base_results_path})")

    # --- Load data ---
    try:
        # Ensure ground truth and time files exist (build from CSVs or generator if necessary)
        ensure_ground_truth_and_time(gt_driver_filepath, time_filepath,
                                     TRAIN_SIZE, TEST_SIZE, DT,
                                     base_results_path)

        print(f"\nLoading time array from: {time_filepath}")
        time_data_loaded = np.load(time_filepath)
        print(f"Time array length: {len(time_data_loaded)}")

        print(f"\nLoading ground truth from: {gt_driver_filepath}")
        gt_driver_data = np.load(gt_driver_filepath)

        if gt_driver_data.ndim != 2 or gt_driver_data.shape[1] != 2:
            raise ValueError(f"Ground Truth file must have 2 columns [theta, omega]. Found shape: {gt_driver_data.shape}")

        print(f"Ground truth data shape: {gt_driver_data.shape}")

        # Determine minimum consistent length
        min_len = min(len(time_data_loaded), gt_driver_data.shape[0])
        time_data = time_data_loaded[:min_len]
        gt_driver_data_truncated = gt_driver_data[:min_len, :]

        theta1_gt_driver = gt_driver_data_truncated[:, 0]
        omega1_gt_driver = gt_driver_data_truncated[:, 1]

        # Reconstruct full 9-DOF Ground Truth
        print("\nReconstructing full ground truth kinematics...")
        q_gt_reconstructed, v_gt_reconstructed = reconstruct_slider_crank_kinematics(
            theta1_gt_driver, omega1_gt_driver, L1, L2
        )

        # Load and process each model's predictions
        models_reconstructed = {}

        for model_name, filepath in model_prediction_files.items():
            if filepath is not None and os.path.exists(filepath):
                print(f"\nProcessing {model_name}...")
                theta_omega_pred = np.load(filepath)

                # Ensure correct shape
                if theta_omega_pred.ndim == 3 and theta_omega_pred.shape[1] == 1:
                    theta_omega_pred = theta_omega_pred.squeeze(1)

                if theta_omega_pred.ndim != 2 or theta_omega_pred.shape[1] != 2:
                    print(f"  Warning: Unexpected shape {theta_omega_pred.shape} for {model_name}. Skipping.")
                    models_reconstructed[model_name] = (None, None)
                    continue

                # Truncate to consistent length
                theta_omega_pred = theta_omega_pred[:min_len, :]

                theta1_pred = theta_omega_pred[:, 0]
                omega1_pred = theta_omega_pred[:, 1]

                # Reconstruct full kinematics
                q_pred_full, v_pred_full = reconstruct_slider_crank_kinematics(
                    theta1_pred, omega1_pred, L1, L2
                )

                models_reconstructed[model_name] = (q_pred_full, v_pred_full)
                print(f"  Reconstructed {model_name} kinematics")
            else:
                print(f"\n{model_name} file not found, skipping...")
                models_reconstructed[model_name] = (None, None)

        # Create comparison plots for all models
        print("\nCreating comparison plots...")
        plot_all_models_comparison(
            time_data,
            q_gt_reconstructed,
            v_gt_reconstructed,
            models_reconstructed,
            output_base_dir,
            train_size=TRAIN_SIZE,
            dataset_name=dataset_suffix
        )

        print(f"\nAll plots saved to: {output_base_dir}")
        print("Done!")

    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        exit(1)