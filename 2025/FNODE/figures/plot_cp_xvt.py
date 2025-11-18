#!/usr/bin/env python3
"""
plot_cp_xvt.py
Creates time series comparison plots for Cart Pole system.
Plots position (x, θ) and velocity (v_x, ω) vs time for different models.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_cart_pole_xvt():
    """
    Loads Cart Pole trajectory data for ground truth and various models,
    then plots position and velocity vs. time in a 4x2 grid.
    Cart Pole has 4 state variables: x (cart position), v_x (cart velocity),
    θ (pole angle), ω (pole angular velocity)
    Row layout: FNODE, MBDNODE, LSTM, FCNN
    Column 1: Positions (x and θ on same plot)
    Column 2: Velocities (v_x and ω on same plot)
    """
    # Get parent directory (FNODE root) since this script is in figures/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    base_results_path = os.path.join(base_dir, 'results', 'Cart_Pole')

    # SPECIFIC CONFIGURATION TO PLOT
    train_steps = 200
    test_steps = 50
    dt = 0.01

    file_suffix = f'_prediction_train{train_steps}_test{test_steps}_dt{dt}.npy'

    model_types_to_plot = ['FNODE', 'MBDNODE', 'LSTM', 'FCNN']
    ground_truth_label = 'Ground_truth'

    total_steps = train_steps + test_steps
    time_vector = np.arange(0, total_steps * dt, dt)

    # Ensure time_vector matches the typical length of loaded data
    if len(time_vector) > total_steps:
        time_vector = time_vector[:total_steps]
    elif len(time_vector) < total_steps:
        logging.warning(
            f"Time vector length ({len(time_vector)}) is less than total_steps ({total_steps}). Data might be truncated in plots.")

    # Load Ground Truth data
    gt_file_path = os.path.join(base_results_path, f"Ground_truth for Cart_Pole with training_size={train_steps}_num_steps_test={test_steps}_dt={dt}.npy")

    ground_truth_data = None
    try:
        if os.path.exists(gt_file_path):
            ground_truth_data = np.load(gt_file_path)
            # Cart Pole data should be (steps, 4): [x, v_x, θ, ω]
            if ground_truth_data.shape[0] != total_steps or ground_truth_data.shape[1] != 4:
                logging.warning(
                    f"Ground truth data shape mismatch: Expected ({total_steps}, 4), Got {ground_truth_data.shape}")
                if ground_truth_data.shape[0] > total_steps:
                    ground_truth_data = ground_truth_data[:total_steps, :]
            logging.info(f"Loaded ground truth data from {gt_file_path}, shape: {ground_truth_data.shape}")
        else:
            logging.error(f"Ground truth file not found at {gt_file_path}")
            return
    except Exception as e:
        logging.error(f"Error loading ground truth data from {gt_file_path}: {e}")
        return

    # Prepare data dictionary for models
    model_data = {}
    for model_type in model_types_to_plot:
        model_file_path = os.path.join(base_results_path, f"{model_type} for Cart_Pole with training_size={train_steps}_num_steps_test={test_steps}_dt={dt}.npy")
        try:
            if os.path.exists(model_file_path):
                data = np.load(model_file_path)
                if data.shape[0] != total_steps or data.shape[1] != 4:
                    logging.warning(
                        f"{model_type} data shape mismatch: Expected ({total_steps}, 4), Got {data.shape}")
                    if data.shape[0] > total_steps:
                        data = data[:total_steps, :]
                model_data[model_type] = data
                logging.info(f"Loaded {model_type} data from {model_file_path}, shape: {data.shape}")
            else:
                logging.warning(f"Data file not found for model: {model_type} at {model_file_path}")
                model_data[model_type] = None
        except Exception as e:
            logging.error(f"Error loading data for model {model_type} from {model_file_path}: {e}")
            model_data[model_type] = None

    # Plotting - 4 models x 2 columns (positions and velocities)
    fig, axs = plt.subplots(len(model_types_to_plot), 2, figsize=(14, 5 * len(model_types_to_plot)),
                             dpi=120, sharex=False, squeeze=False)
    plt.subplots_adjust(hspace=0.5, wspace=0.25)  # Adjusted spacing

    # Add top legend
    overall_legend_lines = [
        plt.Line2D([0], [0], color='blue', linestyle='-', lw=2.5),
        plt.Line2D([0], [0], color='red', linestyle='--', lw=2.5),
        plt.Line2D([0], [0], color='red', linestyle=':', lw=2.5)
    ]
    overall_legend_labels = ['Ground truth', 'ID generalization', 'OOD generalization']
    fig.legend(overall_legend_lines, overall_legend_labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=True, fontsize=20)

    # Define line styles
    gt_style = {'color': 'blue', 'linestyle': '-', 'linewidth': 2.5}
    pred_train_style = {'linestyle': '--', 'linewidth': 2.5}
    pred_test_style = {'linestyle': ':', 'linewidth': 2.5, 'alpha': 0.8}

    model_colors = {
        'FNODE': 'red',
        'MBDNODE': 'red',
        'LSTM': 'red',
        'FCNN': 'red'
    }

    model_display_names = {
        'FNODE': 'FNODE',
        'MBDNODE': 'MBD-NODE',
        'LSTM': 'LSTM',
        'FCNN': 'FCNN'
    }

    # Calculate consistent y-axis limits across all models
    # First, gather all data to find global min/max for each variable
    all_x, all_vx, all_theta, all_omega = [], [], [], []

    # Add ground truth data
    all_x.extend([ground_truth_data[:, 0].min(), ground_truth_data[:, 0].max()])
    all_vx.extend([ground_truth_data[:, 1].min(), ground_truth_data[:, 1].max()])
    all_theta.extend([ground_truth_data[:, 2].min(), ground_truth_data[:, 2].max()])
    all_omega.extend([ground_truth_data[:, 3].min(), ground_truth_data[:, 3].max()])

    # Add all model predictions
    for model_type in model_types_to_plot:
        if model_data.get(model_type) is not None:
            pred_data = model_data[model_type]
            all_x.extend([pred_data[:, 0].min(), pred_data[:, 0].max()])
            all_vx.extend([pred_data[:, 1].min(), pred_data[:, 1].max()])
            all_theta.extend([pred_data[:, 2].min(), pred_data[:, 2].max()])
            all_omega.extend([pred_data[:, 3].min(), pred_data[:, 3].max()])

    # Calculate limits with 10% margin
    x_margin = 0.1 * (max(all_x) - min(all_x))
    vx_margin = 0.1 * (max(all_vx) - min(all_vx))
    theta_margin = 0.1 * (max(all_theta) - min(all_theta))
    omega_margin = 0.1 * (max(all_omega) - min(all_omega))

    x_lim = [min(all_x) - x_margin, max(all_x) + x_margin]
    vx_lim = [min(all_vx) - vx_margin, max(all_vx) + vx_margin]
    theta_lim = [min(all_theta) - theta_margin, max(all_theta) + theta_margin]
    omega_lim = [min(all_omega) - omega_margin, max(all_omega) + omega_margin]

    # Update subplot labels for 4x2 grid
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    for i, model_type in enumerate(model_types_to_plot):
        pred_data = model_data.get(model_type)

        # Current data length for this model
        current_data_len_gt = len(ground_truth_data)
        current_time_vector = time_vector[:current_data_len_gt]

        # Get axes for positions and velocities
        ax_pos = axs[i, 0]  # Positions (x and θ)
        ax_vel = axs[i, 1]  # Velocities (v_x and ω)

        # Add subplot labels - positioned below x-axis to avoid overlap with 't'
        ax_pos.text(0.5, -0.18, subplot_labels[i * 2], transform=ax_pos.transAxes, fontsize=14, ha='center')
        ax_vel.text(0.5, -0.18, subplot_labels[i * 2 + 1], transform=ax_vel.transAxes, fontsize=14, ha='center')

        # Plot ground truth - both x and θ on same plot
        ax_pos.plot(current_time_vector, ground_truth_data[:current_data_len_gt, 0],
                    color='blue', linestyle='-', linewidth=2.5)  # x
        ax_pos.plot(current_time_vector, ground_truth_data[:current_data_len_gt, 2],
                    color='blue', linestyle='-', linewidth=2.5)  # θ

        # Plot ground truth velocities - both v_x and ω on same plot
        ax_vel.plot(current_time_vector, ground_truth_data[:current_data_len_gt, 1],
                    color='blue', linestyle='-', linewidth=2.5)  # v_x
        ax_vel.plot(current_time_vector, ground_truth_data[:current_data_len_gt, 3],
                    color='blue', linestyle='-', linewidth=2.5)  # ω

        if pred_data is not None:
            current_data_len_pred = len(pred_data)
            current_time_vector_pred = time_vector[:current_data_len_pred]

            # Training predictions - all in red with dashed lines
            train_end = min(train_steps, current_data_len_pred)

            # Positions - both x and θ
            ax_pos.plot(current_time_vector_pred[:train_end], pred_data[:train_end, 0],
                        color='red', linestyle='--', linewidth=2.5)  # x
            ax_pos.plot(current_time_vector_pred[:train_end], pred_data[:train_end, 2],
                        color='red', linestyle='--', linewidth=2.5)  # θ

            # Velocities - both v_x and ω
            ax_vel.plot(current_time_vector_pred[:train_end], pred_data[:train_end, 1],
                        color='red', linestyle='--', linewidth=2.5)  # v_x
            ax_vel.plot(current_time_vector_pred[:train_end], pred_data[:train_end, 3],
                        color='red', linestyle='--', linewidth=2.5)  # ω

            # Testing predictions - all in red with dotted lines
            if train_steps < current_data_len_pred:
                # Positions - both x and θ
                ax_pos.plot(current_time_vector_pred[train_steps:], pred_data[train_steps:, 0],
                            color='red', linestyle=':', linewidth=2.5)  # x
                ax_pos.plot(current_time_vector_pred[train_steps:], pred_data[train_steps:, 2],
                            color='red', linestyle=':', linewidth=2.5)  # θ

                # Velocities - both v_x and ω
                ax_vel.plot(current_time_vector_pred[train_steps:], pred_data[train_steps:, 1],
                            color='red', linestyle=':', linewidth=2.5)  # v_x
                ax_vel.plot(current_time_vector_pred[train_steps:], pred_data[train_steps:, 3],
                            color='red', linestyle=':', linewidth=2.5)  # ω

        # Formatting - set y-axis labels
        ax_pos.set_ylabel(r'$x$, $\theta$', fontsize=16)
        ax_vel.set_ylabel(r'$\dot{x}$, $\dot{\theta}$', fontsize=16)

        # Add grids
        ax_pos.grid(True, linestyle=':', alpha=0.3)
        ax_vel.grid(True, linestyle=':', alpha=0.3)

        # Set x-axis labels
        ax_pos.set_xlabel('t', fontsize=14)
        ax_vel.set_xlabel('t', fontsize=14)

        # Always show x-axis tick labels
        ax_pos.tick_params(labelbottom=True)
        ax_vel.tick_params(labelbottom=True)

        # Set consistent y-axis limits for all subplots
        # Use combined limits for positions and velocities
        pos_lim = [min(x_lim[0], theta_lim[0]), max(x_lim[1], theta_lim[1])]
        vel_lim = [min(vx_lim[0], omega_lim[0]), max(vx_lim[1], omega_lim[1])]

        ax_pos.set_ylim(pos_lim)
        ax_vel.set_ylim(vel_lim)

        # Set titles
        ax_pos.set_title(f"{model_display_names.get(model_type, model_type)} - Position", fontsize=16)
        ax_vel.set_title(f"{model_display_names.get(model_type, model_type)} - Velocity", fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0.03, 0.08, 1.0, 0.95])

    # Save the figure
    output_figure_dir = './figures/Cart_Pole/comparison'
    os.makedirs(output_figure_dir, exist_ok=True)
    figure_path = os.path.join(output_figure_dir, 'cp_x_v_time.png')
    try:
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        logging.info(f"Plot saved to {figure_path}")
    except Exception as e:
        logging.error(f"Error saving plot to {figure_path}: {e}")
    plt.close(fig)


if __name__ == '__main__':
    plot_cart_pole_xvt()