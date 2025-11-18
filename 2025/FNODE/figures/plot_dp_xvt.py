import numpy as np
import matplotlib.pyplot as plt
import os
import re


def load_npy_data(file_path):
    """Load numpy data from file and reshape if needed."""
    try:
        data = np.load(file_path, allow_pickle=True)

        # Check if data is 0-dimensional (scalar) or None
        if data.ndim == 0 or data is None:
            print(f"Warning: Data from {file_path} is 0-dimensional or None. Returning empty array.")
            return np.array([])

        # This logic is from the original plot_dp_xvt.py
        if data.ndim == 3 and data.shape[1] == 1 and data.shape[2] == 4:  # Shape like (steps, 1, 4)
            data = data.reshape(data.shape[0], 4)  # Reshape to (steps, 4)
        elif data.ndim == 3:
            # If it's already (steps, 2, 2), it shouldn't be flattened by this.
            # This case in original xvt flattens other 3D arrays.
            if not (data.shape[1] == 2 and data.shape[2] == 2):  # Avoid flattening (steps,2,2)
                data = data.reshape(data.shape[0], -1)  # Flatten multiple bodies if any
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return np.array([])


def extract_file_info(filename):
    """Extract model name, train size, test size, and dt from filename."""
    # New format: MODEL for Test_Case with training_size=XXX_num_steps_test=YYY_dt=Z.ZZ.npy
    pattern_new = r'([A-Za-z_]+) for [A-Za-z_]+ with training_size=(\d+)_num_steps_test=(\d+)_dt=([\d\.]+)\.npy'
    match_new = re.match(pattern_new, filename)
    if match_new:
        model_name = match_new.group(1)
        train_size = int(match_new.group(2))
        test_size = int(match_new.group(3))
        dt = float(match_new.group(4))
        return model_name, train_size, test_size, dt

    # Legacy Pattern: MODEL_prediction_trainXXX_testYYY_dtZ.ZZ.npy
    pattern = r'([A-Za-z_]+)_prediction_train(\d+)_test(\d+)_dt([\d\.]+)\.npy'
    match = re.match(pattern, filename)
    if match:
        model_name = match.group(1)
        train_size = int(match.group(2))
        test_size = int(match.group(3))
        dt = float(match.group(4))
        return model_name, train_size, test_size, dt

    if filename.startswith("Ground_truth"):
        # This case is now handled above by pattern_new, but keep as fallback

        # Legacy formats
        patterns_gt = [
            r'([A-Za-z_]+)_prediction_train(\d+)_test(\d+)_dt([\d\.]+)\.npy',
            r'([A-Za-z_]+)_prediction_train(\d+)_test(\d+)\.npy',
            r'([A-Za-z_]+)_prediction\.npy'
        ]
        for p_idx, pat_gt_str in enumerate(patterns_gt):
            match_gt = re.match(pat_gt_str, filename)
            if match_gt:
                model_name_gt = match_gt.group(1)
                if p_idx == 0:
                    return model_name_gt, int(match_gt.group(2)), int(match_gt.group(3)), float(match_gt.group(4))
                elif p_idx == 1:
                    return model_name_gt, int(match_gt.group(2)), int(match_gt.group(3)), 0.01  # Assume default dt
                elif p_idx == 2:
                    return model_name_gt, 300, 100, 0.01  # Default values if filename is very simple
    return None, None, None, None


def plot_trajectory_comparison(
        test_case_name,
        model_predictions,  # Dict: {'ModelName1': trajectory_data1, 'ModelName2': ...}
        ground_truth_trajectory,  # Numpy array: [num_steps, 2, 2] (pendulums, [angle, velocity])
        time_vector,
        num_steps_train,
        output_dir,
        base_filename="trajectory_comparison",
        plot_phase_space=False,  # Flag to generate phase space plots (set to False)
        plot_time_series=True,  # Flag to generate time series plots
        label_fontsize=16,
        legend_fontsize=20
):
    """
    Generates plots comparing predicted trajectories against ground truth.
    Handles both phase space and time series plots based on flags.
    """
    if len(ground_truth_trajectory) == 0:
        print("Cannot create plots: Ground truth data is empty")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Ensure data is numpy
    gt_np = np.array(ground_truth_trajectory)
    pred_np = {name: np.array(pred) for name, pred in model_predictions.items() if len(pred) > 0}
    time_np = np.array(time_vector)

    standard_models = ['FNODE', 'MBDNODE', 'LSTM', 'FCNN']
    # Ensure models in pred_np are processed, maintaining standard_models order first
    available_models = [m for m in standard_models if m in pred_np] + \
                       [m for m in pred_np if m not in standard_models]

    sorted_models = available_models[:4]  # Max 4 models for plotting
    num_models_to_plot = len(sorted_models)

    # Handling for cases where no models are available for plotting but GT might still be plotted
    if num_models_to_plot == 0:
        if not plot_phase_space and not plot_time_series:
            print("No models to plot and neither phase space nor time series plots are enabled.")
            return
        if plot_phase_space and not plot_time_series:  # Only phase space, but no models
            print("No valid model predictions available for phase space plotting (and time series disabled).")
            return
        # If only time series for GT, or if phase space for GT is desired, proceed with num_models_to_plot = 0 logic

    # --- Legend and Style Definitions ---
    overall_legend_lines = [
        plt.Line2D([0], [0], color='blue', linestyle='-', lw=2.5),
        plt.Line2D([0], [0], color='red', linestyle='--', lw=2.5),
        plt.Line2D([0], [0], color='red', linestyle=':', lw=2.5)
    ]
    overall_legend_labels = ['Ground truth', 'ID generalization', 'OOD generalization']

    line_colors = {
        'gt_p1': 'blue', 'gt_p2': 'blue',  # Colors for Pendulum 1 and 2 Ground Truth
        'pred_p1': 'red', 'pred_p2': 'red'  # Colors for Pendulum 1 and 2 Predictions
    }
    line_styles_main = {'gt': '-', 'id': '--', 'ood': ':'}

    model_display_names = {'FNODE': 'FNODE', 'MBDNODE': 'MBD-NODE', 'LSTM': 'LSTM', 'FCNN': 'FCNN'}
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    # --- Time Series Plotting (if enabled) ---
    if plot_time_series:
        rows_ts = num_models_to_plot if num_models_to_plot > 0 else 1
        fig_ts, axs_ts = plt.subplots(rows_ts, 2, figsize=(14, 5 * rows_ts), dpi=120, sharex=False, squeeze=False)
        plt.subplots_adjust(hspace=0.5, wspace=0.25)  # Increased vertical spacing

        # Position legend closer to the figure
        legend_y_anchor_ts = 0.985 if num_models_to_plot > 0 else 0.95  # Moved legend closer to the plot
        fig_ts.legend(overall_legend_lines, overall_legend_labels, loc='upper center',
                      bbox_to_anchor=(0.5, legend_y_anchor_ts), ncol=3, frameon=True, fontsize=legend_fontsize)

        for i in range(rows_ts):
            model_name_ts = sorted_models[i] if i < num_models_to_plot else "Ground_Truth_Only"

            ax_left_ts = axs_ts[i, 0]
            ax_right_ts = axs_ts[i, 1]

            # Move subplot labels down to avoid overlap with x-axis
            ax_left_ts.text(0.482, -0.23, subplot_labels[i * 2], transform=ax_left_ts.transAxes,
                            fontsize=label_fontsize)
            ax_right_ts.text(0.482, -0.23, subplot_labels[i * 2 + 1], transform=ax_right_ts.transAxes,
                             fontsize=label_fontsize)

            # Plot Ground Truth lines
            ax_left_ts.plot(time_np, gt_np[:, 0, 0], color=line_colors['gt_p1'], linestyle=line_styles_main['gt'],
                            linewidth=2.5, label='GT $\\theta_1$')
            ax_left_ts.plot(time_np, gt_np[:, 1, 0], color=line_colors['gt_p2'], linestyle=line_styles_main['gt'],
                            linewidth=2.5, label='GT $\\theta_2$')

            ax_right_ts.plot(time_np, gt_np[:, 0, 1], color=line_colors['gt_p1'], linestyle=line_styles_main['gt'],
                             linewidth=2.5, label='GT $\\omega_1$')
            ax_right_ts.plot(time_np, gt_np[:, 1, 1], color=line_colors['gt_p2'], linestyle=line_styles_main['gt'],
                             linewidth=2.5, label='GT $\\omega_2$')

            if i < num_models_to_plot and model_name_ts in pred_np:
                model_data_ts = pred_np[model_name_ts]
                # Ensure model_data_ts is (steps, 2, 2) for time series plot
                if not (model_data_ts.ndim == 3 and model_data_ts.shape[1] == 2 and model_data_ts.shape[2] == 2):
                    print(
                        f"Warning: Model {model_name_ts} has unexpected shape {model_data_ts.shape} for time series. Reshaping or Skipping.")
                    if model_data_ts.ndim == 2 and model_data_ts.shape[1] == 4:
                        temp_reshaped_ts = np.zeros((model_data_ts.shape[0], 2, 2))
                        temp_reshaped_ts[:, 0, 0] = model_data_ts[:, 0]
                        temp_reshaped_ts[:, 0, 1] = model_data_ts[:, 2]
                        temp_reshaped_ts[:, 1, 0] = model_data_ts[:, 1]
                        temp_reshaped_ts[:, 1, 1] = model_data_ts[:, 3]
                        model_data_ts = temp_reshaped_ts
                    else:
                        continue  # Skip this model for time series plot

                id_end_ts = min(num_steps_train, len(model_data_ts))

                if id_end_ts > 0:  # ID Predictions
                    ax_left_ts.plot(time_np[:id_end_ts], model_data_ts[:id_end_ts, 0, 0], color=line_colors['pred_p1'],
                                    linestyle=line_styles_main['id'], linewidth=2.5)
                    ax_left_ts.plot(time_np[:id_end_ts], model_data_ts[:id_end_ts, 1, 0], color=line_colors['pred_p2'],
                                    linestyle=line_styles_main['id'], linewidth=2.5)
                    ax_right_ts.plot(time_np[:id_end_ts], model_data_ts[:id_end_ts, 0, 1], color=line_colors['pred_p1'],
                                     linestyle=line_styles_main['id'], linewidth=2.5)
                    ax_right_ts.plot(time_np[:id_end_ts], model_data_ts[:id_end_ts, 1, 1], color=line_colors['pred_p2'],
                                     linestyle=line_styles_main['id'], linewidth=2.5)
                if len(model_data_ts) > num_steps_train:  # OOD Predictions
                    ax_left_ts.plot(time_np[num_steps_train:], model_data_ts[num_steps_train:, 0, 0],
                                    color=line_colors['pred_p1'], linestyle=line_styles_main['ood'], linewidth=2.5)
                    ax_left_ts.plot(time_np[num_steps_train:], model_data_ts[num_steps_train:, 1, 0],
                                    color=line_colors['pred_p2'], linestyle=line_styles_main['ood'], linewidth=2.5)
                    ax_right_ts.plot(time_np[num_steps_train:], model_data_ts[num_steps_train:, 0, 1],
                                     color=line_colors['pred_p1'], linestyle=line_styles_main['ood'], linewidth=2.5)
                    ax_right_ts.plot(time_np[num_steps_train:], model_data_ts[num_steps_train:, 1, 1],
                                     color=line_colors['pred_p2'], linestyle=line_styles_main['ood'], linewidth=2.5)

            # Ensure all axes have proper labels and grid
            ax_left_ts.set_ylabel(r'$\theta$', fontsize=label_fontsize)
            ax_right_ts.set_ylabel(r'$\dot{\theta}$', fontsize=label_fontsize)
            ax_left_ts.grid(True, linestyle=':')
            ax_right_ts.grid(True, linestyle=':')
            ax_left_ts.set_xlabel('t', fontsize=label_fontsize)
            ax_right_ts.set_xlabel('t', fontsize=label_fontsize)

            # Always show x-axis tick labels for all subplots
            ax_left_ts.tick_params(labelbottom=True)
            ax_right_ts.tick_params(labelbottom=True)

            # Set titles for each subplot
            ax_left_ts.set_title(f"{model_display_names.get(model_name_ts, model_name_ts)} - Angular Position", fontsize=16)
            ax_right_ts.set_title(f"{model_display_names.get(model_name_ts, model_name_ts)} - Angular Velocity", fontsize=16)

        # Adjust layout to provide more space at the bottom for labels
        plt.tight_layout(rect=[0.03, 0.08, 1.0, 0.95 if rows_ts > 0 else 0.90])
        save_path_ts = os.path.join(output_dir, f"{base_filename}.png")
        try:
            plt.savefig(save_path_ts, dpi=300, bbox_inches='tight')
            print(f"Time series plot saved to {save_path_ts}")
        except Exception as e:
            print(f"Error saving time series plot: {e}")
        plt.close(fig_ts)


def main():
    test_case = 'Double_Pendulum'

    # SPECIFIC CONFIGURATION TO PLOT
    target_train_size = 300
    target_test_size = 100
    target_dt = 0.01

    # Get parent directory (FNODE root) since this script is in figures/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(base_dir, 'results', test_case)
    # Changed output directory slightly to distinguish from plot_dp_phase.py outputs if they coexist
    output_dir_xvt = os.path.join(base_dir, 'figures', test_case, 'comparison')
    os.makedirs(output_dir_xvt, exist_ok=True)

    print(f"INFO: Base directory: {base_dir}")
    print(f"INFO: Results directory: {results_dir}")
    print(f"INFO: Output directory for corrected xvt plots: {output_dir_xvt}")

    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    print(f"Processing specific configuration: Train: {target_train_size}, Test: {target_test_size}, dt: {target_dt}")

    # Find files for the specific configuration
    file_list = []
    for file_name in os.listdir(results_dir):
        if file_name.endswith('.npy'):
            model_name, train_size, test_size, dt = extract_file_info(file_name)
            if (model_name and
                train_size == target_train_size and
                test_size == target_test_size and
                dt == target_dt):
                file_list.append((model_name, file_name))

    if not file_list:
        print(f"Error: No files found for configuration train={target_train_size}, test={target_test_size}, dt={target_dt}")
        return

    print(f"Found models: {[f[0] for f in file_list]}")

    # Process only the specific configuration
    train_size_conf, test_size_conf, dt_conf = target_train_size, target_test_size, target_dt
    print(
        f"\n--- Processing Group for Corrected xvt: train={train_size_conf}, test={test_size_conf}, dt={dt_conf} ---")

    ground_truth_data_for_plot = None  # Initialize
    model_predictions_for_plot = {}  # Initialize

    # Load and prepare Ground Truth
    gt_file_entry = next((f for f in file_list if f[0] == 'Ground_truth'), None)
    if gt_file_entry:
        gt_file_path = os.path.join(results_dir, gt_file_entry[1])
        raw_gt_data = load_npy_data(gt_file_path)  # Uses the script's load_npy_data
        if raw_gt_data is not None and raw_gt_data.size > 0:
            if raw_gt_data.ndim == 2 and raw_gt_data.shape[1] == 4:
                steps = raw_gt_data.shape[0]
                gt_reshaped = np.zeros((steps, 2, 2))
                gt_reshaped[:, 0, 0] = raw_gt_data[:, 0]  # θ1
                gt_reshaped[:, 0, 1] = raw_gt_data[:, 1]  # ω1
                gt_reshaped[:, 1, 0] = raw_gt_data[:, 2]  # θ2
                gt_reshaped[:, 1, 1] = raw_gt_data[:, 3]  # ω2
                ground_truth_data_for_plot = gt_reshaped
            elif raw_gt_data.ndim == 3 and raw_gt_data.shape[1] == 2 and raw_gt_data.shape[2] == 2:
                ground_truth_data_for_plot = raw_gt_data
            else:
                print(
                    f"Error: Ground Truth data {gt_file_entry[1]} has unexpected shape {raw_gt_data.shape} after loading.")
        else:
            print(f"Error: Ground Truth data {gt_file_entry[1]} is empty.")

    if ground_truth_data_for_plot is None:
        print(f"Critical Error: Ground truth could not be loaded/prepared. Skipping plot.")
        return
    print(f"Ground Truth prepared for plot, shape: {ground_truth_data_for_plot.shape}")

    # Load and prepare Model Predictions
    for model_name_loop, file_name_loop in file_list:
        if model_name_loop == 'Ground_truth':
            continue
        model_file_path = os.path.join(results_dir, file_name_loop)
        raw_model_data = load_npy_data(model_file_path)
        if raw_model_data.any():
            if raw_model_data.ndim == 2 and raw_model_data.shape[1] == 4:
                steps = raw_model_data.shape[0]
                model_reshaped = np.zeros((steps, 2, 2))
                model_reshaped[:, 0, 0] = raw_model_data[:, 0]  # θ1
                model_reshaped[:, 0, 1] = raw_model_data[:, 1]  # ω1
                model_reshaped[:, 1, 0] = raw_model_data[:, 2]  # θ2
                model_reshaped[:, 1, 1] = raw_model_data[:, 3]  # ω2
                model_predictions_for_plot[model_name_loop] = model_reshaped
            elif raw_model_data.ndim == 3 and raw_model_data.shape[1] == 2 and raw_model_data.shape[2] == 2:
                model_predictions_for_plot[model_name_loop] = raw_model_data
            else:
                print(
                    f"Warning: Model {model_name_loop} data {file_name_loop} has unexpected shape {raw_model_data.shape}. Skipping.")
                model_predictions_for_plot[model_name_loop] = np.array(
                    [])  # Store empty to avoid key errors if needed later
        else:
            print(f"Warning: Model {model_name_loop} data {file_name_loop} is empty.")
            model_predictions_for_plot[model_name_loop] = np.array([])

    total_steps = train_size_conf + test_size_conf
    time_steps = np.linspace(0, (total_steps - 1) * dt_conf, total_steps)

    dt_str_suffix = str(dt_conf).replace('.', '_')
    file_suffix = f"_train{train_size_conf}_test{test_size_conf}_dt{dt_str_suffix}"

    # Generate only the time series plot by setting plot_phase_space=False
    plot_trajectory_comparison(
        test_case_name=test_case,
        model_predictions=model_predictions_for_plot,  # Pass the (steps,2,2) data
        ground_truth_trajectory=ground_truth_data_for_plot,  # Pass the (steps,2,2) data
        time_vector=time_steps,
        num_steps_train=train_size_conf,
        output_dir=output_dir_xvt,
        base_filename=f"dp_x_v_time",
        plot_phase_space=False,  # Set to False to only generate time series
        plot_time_series=True
    )

    print(f"\nAll corrected xvt trajectory comparison plots attempted. Check {output_dir_xvt}.")


if __name__ == "__main__":
    main()