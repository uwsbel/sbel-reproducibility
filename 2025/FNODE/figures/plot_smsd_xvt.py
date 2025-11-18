import numpy as np
import matplotlib.pyplot as plt
import os
import re


def load_npy_data(file_path):
    """Load numpy data from file and reshape if needed."""
    data = np.load(file_path, allow_pickle=True)
    if data.ndim == 3 and data.shape[1] == 1:  # Shape like (steps, 1, 2)
        data = data.reshape(data.shape[0], 2)  # Reshape to (steps, 2)
    elif data.ndim == 3:  # Shape like (steps, bodies, 2)
        data = data.reshape(data.shape[0], -1)  # Flatten multiple bodies if any
    return data


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

    # Legacy format: MODEL_prediction_trainXXX_testYYY_dtZ.ZZ.npy
    pattern = r'([A-Za-z_]+)_prediction_train(\d+)_test(\d+)_dt([\d\.]+)\.npy'
    match = re.match(pattern, filename)
    if match:
        model_name = match.group(1)
        train_size = int(match.group(2))
        test_size = int(match.group(3))
        dt = float(match.group(4))
        return model_name, train_size, test_size, dt
    return None, None, None, None


def create_comparison_plots(ground_truth, model_predictions, model_names, time_steps, training_size, output_dir=None,
                            suffix=""):
    """
    Create comparison plots between ground truth and model predictions, distinguishing
    between in-distribution (ID) and out-of-distribution (OOD) regions.
    Improved version with more compact layout and larger fonts.
    """
    # Determine how many models to plot (up to 4)
    plot_models = []
    for name in model_names:
        if name in model_predictions and len(plot_models) < 4:
            plot_models.append(name)

    num_models = len(plot_models)
    if num_models == 0:
        print("No valid models to plot!")
        return

    # Create figure with the correct number of rows
    fig, axs = plt.subplots(num_models, 2, figsize=(12, 4 * num_models), dpi=120)
    if num_models == 1:
        axs = axs.reshape(1, 2)  # Ensure 2D array for single row

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.45, wspace=0.20)

    # Calculate appropriate time range based on data
    max_time = time_steps[-1]
    # Use the full time range for longer sequences
    x_lim = [0, max_time]

    # Define colors and line styles
    colors = {
        'ground_truth': 'blue',
        'id': 'red',
        'ood': 'red'
    }

    line_styles = {
        'ground_truth': '-',
        'id': '--',
        'ood': ':'
    }

    # Create a legend for the entire figure
    custom_lines = [
        plt.Line2D([0], [0], color=colors['ground_truth'], linestyle=line_styles['ground_truth'], lw=2.5),
        plt.Line2D([0], [0], color=colors['id'], linestyle=line_styles['id'], lw=2.5),
        plt.Line2D([0], [0], color=colors['ood'], linestyle=line_styles['ood'], lw=2.5)
    ]

    fig.legend(custom_lines, ['Ground truth', 'ID generalization', 'OOD generalization'],
               loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, fontsize=18)

    # Set subplot labels
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    # Plot each model
    for i, model_name in enumerate(plot_models):
        model_data = model_predictions[model_name]

        # Ensure data lengths match for plotting
        min_len = min(len(ground_truth), len(model_data), len(time_steps))
        if min_len < len(ground_truth) or min_len < len(model_data) or min_len < len(time_steps):
            print(f"Warning: Truncating data to common length {min_len} for {model_name}")
            ground_truth_plot = ground_truth[:min_len]
            model_data_plot = model_data[:min_len]
            time_steps_plot = time_steps[:min_len]
        else:
            ground_truth_plot = ground_truth
            model_data_plot = model_data
            time_steps_plot = time_steps

        # Get position and velocity axes for this model
        ax_pos = axs[i, 0]  # Position (left column)
        ax_vel = axs[i, 1]  # Velocity (right column)

        # Plot ground truth for both position and velocity
        ax_pos.plot(time_steps_plot, ground_truth_plot[:, 0],
                    color=colors['ground_truth'],
                    linestyle=line_styles['ground_truth'],
                    linewidth=2.5)
        ax_vel.plot(time_steps_plot, ground_truth_plot[:, 1],
                    color=colors['ground_truth'],
                    linestyle=line_styles['ground_truth'],
                    linewidth=2.5)

        # Plot ID region (training data)
        id_end = min(training_size, len(model_data_plot))
        if id_end > 0:
            ax_pos.plot(time_steps_plot[:id_end], model_data_plot[:id_end, 0],
                        color=colors['id'],
                        linestyle=line_styles['id'],
                        linewidth=2.5)
            ax_vel.plot(time_steps_plot[:id_end], model_data_plot[:id_end, 1],
                        color=colors['id'],
                        linestyle=line_styles['id'],
                        linewidth=2.5)

        # Plot OOD region (test data)
        if len(model_data_plot) > training_size:
            # Calculate the maximum number of valid points for OOD region
            valid_ood_points = min(len(time_steps_plot) - training_size, len(model_data_plot) - training_size)

            if valid_ood_points > 0:
                ax_pos.plot(time_steps_plot[training_size:training_size + valid_ood_points],
                            model_data_plot[training_size:training_size + valid_ood_points, 0],
                            color=colors['ood'],
                            linestyle=line_styles['ood'],
                            linewidth=2.5)
                ax_vel.plot(time_steps_plot[training_size:training_size + valid_ood_points],
                            model_data_plot[training_size:training_size + valid_ood_points, 1],
                            color=colors['ood'],
                            linestyle=line_styles['ood'],
                            linewidth=2.5)

        # Set titles with display name
        if model_name == 'LSTMModel':
            model_name_display = 'LSTM'
        elif model_name == 'MBDNODE':
            model_name_display = 'MBD-NODE'
        else:
            model_name_display = model_name

        ax_pos.set_title(f"{model_name_display} - Position", fontsize=16)
        ax_vel.set_title(f"{model_name_display} - Velocity", fontsize=16)

        # Set x and y labels
        ax_pos.set_xlabel('$t$', fontsize=16)
        ax_vel.set_xlabel('$t$', fontsize=16)
        ax_pos.set_ylabel('$x$', fontsize=16)
        ax_vel.set_ylabel('$\\dot{x}$', fontsize=16)

        # Increase tick label size
        ax_pos.tick_params(axis='both', labelsize=14)
        ax_vel.tick_params(axis='both', labelsize=14)

        # Set x limits to show full time range
        ax_pos.set_xlim(x_lim)
        ax_vel.set_xlim(x_lim)

        # Set y limits (with some auto-adjustment)
        pos_data = np.concatenate([ground_truth_plot[:, 0], model_data_plot[:, 0]])
        vel_data = np.concatenate([ground_truth_plot[:, 1], model_data_plot[:, 1]])

        pos_margin = 0.1 * (np.max(pos_data) - np.min(pos_data))
        vel_margin = 0.1 * (np.max(vel_data) - np.min(vel_data))

        ax_pos.set_ylim([np.min(pos_data) - pos_margin, np.max(pos_data) + pos_margin])
        ax_vel.set_ylim([np.min(vel_data) - vel_margin, np.max(vel_data) + vel_margin])

        # Add grid
        ax_pos.grid(True, linestyle=':')
        ax_vel.grid(True, linestyle=':')

        # Add subplot labels with larger font, positioned at the bottom center
        ax_pos.text(0.5, -0.3, subplot_labels[2 * i],
                    transform=ax_pos.transAxes,
                    fontsize=16,
                    horizontalalignment='center')
        ax_vel.text(0.5, -0.3, subplot_labels[2 * i + 1],
                    transform=ax_vel.transAxes,
                    fontsize=16,
                    horizontalalignment='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])  # Make room for the legend at the top

    # Save or display
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        figure_name = f'smsd_x_v_time.png'
        plt.savefig(os.path.join(output_dir, figure_name), dpi=300, bbox_inches='tight')
        print(f"Figure saved to {os.path.join(output_dir, figure_name)}")
    else:
        plt.close()

    plt.close()


def main():
    # Set the test case
    test_case = 'Single_Mass_Spring_Damper'

    # SPECIFIC CONFIGURATION TO PLOT
    target_train_size = 300
    target_test_size = 100
    target_dt = 0.01

    # Set base directory for data files with new path structure
    # Get parent directory (FNODE root) since this script is in figures/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(base_dir, 'results', test_case)

    # Set output directory for comparison figures
    output_dir = os.path.join(base_dir, 'figures', test_case, 'comparison')
    os.makedirs(output_dir, exist_ok=True)

    # Check if the results directory exists
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return

    print(f"Processing specific configuration: Train: {target_train_size}, Test: {target_test_size}, dt: {target_dt}")

    # Find files for the specific configuration
    file_list = []
    for file in os.listdir(results_dir):
        if file.endswith('.npy'):
            model_name, train_size, test_size, dt = extract_file_info(file)
            if (model_name is not None and
                train_size == target_train_size and
                test_size == target_test_size and
                dt == target_dt):
                file_list.append((model_name, file))

    if not file_list:
        print(f"Error: No files found for configuration train={target_train_size}, test={target_test_size}, dt={target_dt}")
        return

    print(f"Found models: {[f[0] for f in file_list]}")

    # Find Ground_truth file for this configuration
    ground_truth_file = f'Ground_truth for Single_Mass_Spring_Damper with training_size={target_train_size}_num_steps_test={target_test_size}_dt={target_dt}.npy'
    ground_truth_path = os.path.join(results_dir, ground_truth_file)

    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file not found: {ground_truth_file}")
        return

    # Load ground truth
    try:
        ground_truth_data = load_npy_data(ground_truth_path)
        print(f"Loaded ground truth data with shape {ground_truth_data.shape}")
    except Exception as e:
        print(f"Error loading ground truth data: {e}")
        return

    # Load model data
    model_data = {}
    model_order = ['FNODE', 'MBDNODE', 'LSTMModel', 'FCNN']  # Preferred order

    for model_name, filename in file_list:
        try:
            data = load_npy_data(os.path.join(results_dir, filename))
            model_data[model_name] = data
            print(f"Loaded {model_name} data with shape {data.shape}")
        except Exception as e:
            print(f"Error loading {model_name} data: {e}")

    # Sort models according to model_order
    sorted_models = sorted([m for m in model_data.keys()],
                           key=lambda x: model_order.index(x) if x in model_order else len(model_order))

    # Create time steps array
    total_steps = target_train_size + target_test_size
    time_steps = np.linspace(0, (total_steps - 1) * target_dt, total_steps)

    # Generate plot for this specific configuration
    suffix = f"_train{target_train_size}_test{target_test_size}_dt{target_dt}"
    create_comparison_plots(
        ground_truth=ground_truth_data,
        model_predictions=model_data,
        model_names=sorted_models,
        time_steps=time_steps,
        training_size=target_train_size,
        output_dir=output_dir,
        suffix=suffix
    )

    print(f"Comparison plot created in {output_dir}")


if __name__ == "__main__":
    main()