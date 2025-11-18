import numpy as np
import matplotlib.pyplot as plt
import os
import re


def load_npy_data(file_path):
    """Load numpy data from file and reshape if needed."""
    data = np.load(file_path, allow_pickle=True)
    if data.ndim == 3 and data.shape[1] == 3:  # Shape like (steps, 3, 2) for TMSD
        data = data.reshape(data.shape[0], -1)  # Reshape to (steps, 6)
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
    # Handle both integer and float train/test sizes
    pattern = r'([A-Za-z_]+)_prediction_train([\d\.]+)_test([\d\.]+)_dt([\d\.]+)\.npy'
    match = re.match(pattern, filename)
    if match:
        model_name = match.group(1)
        train_size = int(float(match.group(2)))
        test_size = int(float(match.group(3)))
        dt = float(match.group(4))
        return model_name, train_size, test_size, dt
    return None, None, None, None


def create_comparison_plots(ground_truth, model_predictions, model_names, time_steps, training_size, output_dir=None,
                            suffix=""):
    """
    Create comparison plots between ground truth and model predictions for Triple Mass Spring Damper,
    distinguishing between in-distribution (ID) and out-of-distribution (OOD) regions.
    Shows position and velocity vs time for all three masses in the same plots.
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

    # Create figure with num_models rows and 2 columns (position and velocity)
    fig, axs = plt.subplots(num_models, 2, figsize=(16, 4 * num_models), dpi=120)

    # Handle the case where we have only one model
    if num_models == 1:
        axs = axs.reshape(1, 2)  # Ensure 2D array for single row

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    # Calculate appropriate time range based on data
    max_time = time_steps[-1]
    x_lim = [0, max_time]

    # Define colors for masses and line styles for data types
    mass_colors = ['darkred', 'darkgreen', 'darkblue']  # Different color for each mass

    line_styles = {
        'ground_truth': '-',
        'id': '--',
        'ood': ':'
    }

    # Create a legend for the entire figure - we need to combine styles and masses
    # First create the data type examples (GT, ID, OOD)
    style_lines = [
        plt.Line2D([0], [0], color='blue', linestyle=line_styles['ground_truth'], lw=2.5, label='Ground truth'),
        plt.Line2D([0], [0], color='red', linestyle=line_styles['id'], lw=2.5, label='ID generalization'),
        plt.Line2D([0], [0], color='red', linestyle=line_styles['ood'], lw=2.5, label='OOD generalization')
    ]

    # Combine all legend items
    all_legend_lines = style_lines
    all_legend_labels = [line.get_label() for line in all_legend_lines]

    fig.legend(all_legend_lines, all_legend_labels,
               loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=6, frameon=True, fontsize=18)

    # Set subplot labels
    subplot_labels = [
        '(a)', '(b)',  # First model
        '(c)', '(d)',  # Second model
        '(e)', '(f)',  # Third model
        '(g)', '(h)'  # Fourth model
    ]

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

        # Get axes for this model
        ax_pos = axs[i, 0]  # Position plot (left column)
        ax_vel = axs[i, 1]  # Velocity plot (right column)

        # For each mass, plot position and velocity
        for mass_idx in range(3):
            pos_idx = mass_idx * 2
            vel_idx = pos_idx + 1
            mass_color = mass_colors[mass_idx]

            # Plot ground truth
            ax_pos.plot(time_steps_plot, ground_truth_plot[:, pos_idx],
                        color='blue',
                        linestyle=line_styles['ground_truth'],
                        linewidth=2.5)
            ax_vel.plot(time_steps_plot, ground_truth_plot[:, vel_idx],
                        color='blue',
                        linestyle=line_styles['ground_truth'],
                        linewidth=2.5)

            # Plot ID region (training data)
            id_end = min(training_size, len(model_data_plot))
            if id_end > 0:
                ax_pos.plot(time_steps_plot[:id_end], model_data_plot[:id_end, pos_idx],
                            color='red',
                            linestyle=line_styles['id'],
                            linewidth=2.5)
                ax_vel.plot(time_steps_plot[:id_end], model_data_plot[:id_end, vel_idx],
                            color='red',
                            linestyle=line_styles['id'],
                            linewidth=2.5)

            # Plot OOD region (test data)
            if len(model_data_plot) > training_size:
                # Calculate the maximum number of valid points for OOD region
                valid_ood_points = min(len(time_steps_plot) - training_size, len(model_data_plot) - training_size)

                if valid_ood_points > 0:
                    ax_pos.plot(time_steps_plot[training_size:training_size + valid_ood_points],
                                model_data_plot[training_size:training_size + valid_ood_points, pos_idx],
                                color='red',
                                linestyle=line_styles['ood'],
                                linewidth=2.5)
                    ax_vel.plot(time_steps_plot[training_size:training_size + valid_ood_points],
                                model_data_plot[training_size:training_size + valid_ood_points, vel_idx],
                                color='red',
                                linestyle=line_styles['ood'],
                                linewidth=2.5)

        # Set titles with display name
        if model_name == 'LSTM':
            model_name_display = 'LSTM'
        elif model_name == 'MBDNODE':
            model_name_display = 'MBD-NODE'
        elif model_name == 'FCNN':
            model_name_display = 'FCNN'
        elif model_name == 'FNODE':
            model_name_display = 'FNODE'
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

        # Calculate y limits across all masses
        pos_min, pos_max = float('inf'), float('-inf')
        vel_min, vel_max = float('inf'), float('-inf')

        for mass_idx in range(3):
            pos_idx = mass_idx * 2
            vel_idx = pos_idx + 1

            # Combine ground truth and predictions for each mass
            pos_data = np.concatenate([ground_truth_plot[:, pos_idx], model_data_plot[:, pos_idx]])
            vel_data = np.concatenate([ground_truth_plot[:, vel_idx], model_data_plot[:, vel_idx]])

            # Update min/max values
            pos_min = min(pos_min, np.min(pos_data))
            pos_max = max(pos_max, np.max(pos_data))
            vel_min = min(vel_min, np.min(vel_data))
            vel_max = max(vel_max, np.max(vel_data))

        # Add margins
        pos_margin = 0.1 * (pos_max - pos_min)
        vel_margin = 0.1 * (vel_max - vel_min)

        ax_pos.set_ylim([pos_min - pos_margin, pos_max + pos_margin])
        ax_vel.set_ylim([vel_min - vel_margin, vel_max + vel_margin])

        # Add grid
        ax_pos.grid(True, linestyle=':')
        ax_vel.grid(True, linestyle=':')

        # Add subplot labels
        subplot_idx = i * 2  # Calculate the subplot index
        ax_pos.text(0.03, 0.95, subplot_labels[subplot_idx],
                    transform=ax_pos.transAxes,
                    fontsize=14, fontweight='bold',
                    horizontalalignment='left', verticalalignment='top')
        ax_vel.text(0.03, 0.95, subplot_labels[subplot_idx + 1],
                    transform=ax_vel.transAxes,
                    fontsize=14, fontweight='bold',
                    horizontalalignment='left', verticalalignment='top')

    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Make room for the legend at the top

    # Save or display
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        figure_name = f'tmsd_x_v_t.png'
        plt.savefig(os.path.join(output_dir, figure_name), dpi=300, bbox_inches='tight')
        print(f"Figure saved to {os.path.join(output_dir, figure_name)}")
    else:
        plt.close()

    plt.close()


def main():
    # Set the test case
    test_case = 'Triple_Mass_Spring_Damper'

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

    # Process the specific configuration
    train_size, test_size, dt = target_train_size, target_test_size, target_dt
    print(f"\nProcessing: train_size={train_size}, test_size={test_size}, dt={dt}")

    # Find Ground_truth file
    ground_truth_file = f'Ground_truth for Triple_Mass_Spring_Damper with training_size={train_size}_num_steps_test={test_size}_dt={dt}.npy'
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
    model_order = ['FNODE', 'MBDNODE', 'FCNN', 'LSTM']  # Preferred order - FCNN before LSTM

    for model_name, filename in file_list:
        try:
            data = load_npy_data(os.path.join(results_dir, filename))
            model_data[model_name] = data
            print(f"Loaded {model_name} data with shape {data.shape}")
        except Exception as e:
            print(f"Error loading {model_name} data: {e}")

    # Sort models according to model_order - exclude Ground_truth
    sorted_models = []
    for model in model_order:
        if model in model_data and model != 'Ground_truth':
            sorted_models.append(model)

    # Add any remaining models not in the preferred order (except Ground_truth)
    for model in model_data.keys():
        if model not in sorted_models and model != 'Ground_truth':
            sorted_models.append(model)

    # Create time steps array
    total_steps = train_size + test_size
    time_steps = np.linspace(0, (total_steps - 1) * dt, total_steps)

    # Generate plots for this group
    suffix = f"_train{train_size}_test{test_size}_dt{dt}"
    create_comparison_plots(
        ground_truth=ground_truth_data,
        model_predictions=model_data,
        model_names=sorted_models,
        time_steps=time_steps,
        training_size=train_size,
        output_dir=output_dir,
        suffix=suffix
    )

    print(f"All comparison plots created in {output_dir}")


if __name__ == "__main__":
    main()