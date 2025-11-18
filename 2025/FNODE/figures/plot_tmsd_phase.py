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


def create_phase_space_comparison(ground_truth, model_predictions, model_names, training_size,
                                  output_dir=None, output_filename="phase_space_comparison.png"):
    """
    Create phase space comparison plots between ground truth and model predictions for Triple Mass Spring Damper,
    distinguishing between in-distribution (ID) and out-of-distribution (OOD) regions.

    Args:
        ground_truth: Ground truth data with shape (steps, 6) for 3 masses
        model_predictions: Dictionary of model predictions {model_name: data}
        model_names: List of model names in specific order for plotting
        training_size: Index separating training (ID) and test (OOD) data
        output_dir: Directory to save plots (if None, plots will be displayed)
        output_filename: Name of the output file
    """
    # Create a 4x3 grid (4 models, 3 masses)
    fig, axs = plt.subplots(4, 3, figsize=(15, 20))

    # Colors and line styles exactly as in the example
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
               loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, fontsize=20)

    # Set subplot labels
    subplot_labels = [
        '(a)', '(b)', '(c)',  # First model
        '(d)', '(e)', '(f)',  # Second model
        '(g)', '(h)', '(i)',  # Third model
        '(j)', '(k)', '(l)'  # Fourth model
    ]

    # Plot each model (row) and each mass (column)
    for model_idx, model_name in enumerate(model_names):
        if model_idx >= 4:  # We only have 4 rows
            print(f"Warning: More models than rows. Skipping {model_name}")
            continue

        if model_name is None or model_name not in model_predictions:
            if model_name:
                print(f"Warning: No data found for {model_name}")
            # Still iterate through columns to keep grid consistent
            for mass_idx in range(3):
                ax = axs[model_idx, mass_idx]
                ax.set_visible(False)
            continue

        model_data = model_predictions[model_name]

        # Get the model display name
        if  model_name == 'LSTM':
            model_name_display = 'LSTM'
        elif model_name == 'MBDNODE':
            model_name_display = 'MBD-NODE'
        elif model_name == 'FCNN':
            model_name_display = 'FCNN'
        elif model_name == 'FNODE':
            model_name_display = 'FNODE'
        else:
            model_name_display = model_name

        # Plot each mass in a separate subplot
        for mass_idx in range(3):
            pos_idx = mass_idx * 2
            vel_idx = pos_idx + 1

            ax = axs[model_idx, mass_idx]

            # Plot ground truth phase space
            ax.plot(ground_truth[:, pos_idx], ground_truth[:, vel_idx],
                    color=colors['ground_truth'],
                    linestyle=line_styles['ground_truth'],
                    linewidth=2.5)

            # Plot ID region (training data)
            id_end = min(training_size, len(model_data))
            if id_end > 0:
                ax.plot(model_data[:id_end, pos_idx], model_data[:id_end, vel_idx],
                        color=colors['id'],
                        linestyle=line_styles['id'],
                        linewidth=2.5)

            # Plot OOD region (test data)
            if len(model_data) > training_size:
                ax.plot(model_data[training_size:, pos_idx], model_data[training_size:, vel_idx],
                        color=colors['ood'],
                        linestyle=line_styles['ood'],
                        linewidth=2.5)

            # Use same title format as in the example:
            # {Model} - Mass {number}
            ax.set_title(f"{model_name_display} - Mass {mass_idx + 1}", fontsize=16)

            # Set axes labels
            ax.set_xlabel('$x$', fontsize=16)
            ax.set_ylabel('$\\dot{x}$', fontsize=16)

            # Add grid
            ax.grid(True, linestyle=':')

            # Add subplot label
            ax.text(0.5, -0.2, subplot_labels[model_idx * 3 + mass_idx],
                    transform=ax.transAxes,
                    fontsize=16,
                    horizontalalignment='center',
                    verticalalignment='top')

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Make room for the legend at the top

    # Save or display
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight')
        print(f"Figure saved to {os.path.join(output_dir, output_filename)}")
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

    # Sort models according to model_order
    sorted_models = []
    for model in model_order:
        if model in model_data:
            sorted_models.append(model)

    # Limit to 4 models for the 4x3 grid
    sorted_models = sorted_models[:4]

    # Ensure we have exactly 4 models by adding placeholders if needed
    while len(sorted_models) < 4:
        sorted_models.append(None)

    # Generate phase space plots for this group
    suffix = f"_train{train_size}_test{test_size}_dt{dt}"
    create_phase_space_comparison(
        ground_truth=ground_truth_data,
        model_predictions=model_data,
        model_names=sorted_models,
        training_size=train_size,
        output_dir=output_dir,
        output_filename=f"tmsd_x_v.png"
    )

    print(f"All phase space comparison plots created in {output_dir}")


if __name__ == "__main__":
    main()