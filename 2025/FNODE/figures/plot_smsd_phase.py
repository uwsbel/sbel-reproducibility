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


def create_phase_space_comparison(ground_truth, model_predictions, model_names, training_size,
                                  output_dir=None, output_filename="phase_space_comparison.png"):
    """
    Create phase space comparison plots between ground truth and model predictions, distinguishing
    between in-distribution (ID) and out-of-distribution (OOD) regions.

    Args:
        ground_truth: Ground truth data with shape (steps, features)
        model_predictions: Dictionary of model predictions {model_name: data}
        model_names: List of model names in specific order for plotting
        training_size: Index separating training (ID) and test (OOD) data
        output_dir: Directory to save plots (if None, plots will be displayed)
        output_filename: Name of the output file
    """
    # Create a 2x2 grid for the 4 models
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()  # Flatten to make indexing easier

    # Define labels, line styles, and colors exactly as in the example
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

    # Set up phase space limits for consistency across plots
    x_lim = [-1.1, 1.1]  # Position range
    y_lim = [-2.2, 2.2]  # Velocity range

    # Create a legend for the entire figure
    custom_lines = [
        plt.Line2D([0], [0], color=colors['ground_truth'], linestyle=line_styles['ground_truth'], lw=2.5),
        plt.Line2D([0], [0], color=colors['id'], linestyle=line_styles['id'], lw=2.5),
        plt.Line2D([0], [0], color=colors['ood'], linestyle=line_styles['ood'], lw=2.5)
    ]

    fig.legend(custom_lines, ['Ground truth', 'ID generalization', 'OOD generalization'],
               loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, fontsize=16)

    # Set subplot labels
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    # Plot each model
    for i, model_name in enumerate(model_names):
        if i >= len(axs):  # Make sure we don't try to plot more models than we have axes
            print(f"Warning: More models ({len(model_names)}) than axes ({len(axs)}), skipping {model_name}")
            continue

        if model_name not in model_predictions:
            print(f"Warning: No data found for {model_name}")
            continue

        model_data = model_predictions[model_name]
        ax = axs[i]

        # Plot ground truth phase space (position vs velocity)
        ax.plot(ground_truth[:, 0], ground_truth[:, 1],
                color=colors['ground_truth'],
                linestyle=line_styles['ground_truth'], linewidth=2.5)

        # Plot ID region (training data)
        id_end = min(training_size, len(model_data))
        if id_end > 0:
            ax.plot(model_data[:id_end, 0], model_data[:id_end, 1],
                    color=colors['id'],
                    linestyle=line_styles['id'], linewidth=2.5)

        # Plot OOD region (test data)
        if len(model_data) > training_size:
            ax.plot(model_data[training_size:, 0], model_data[training_size:, 1],
                    color=colors['ood'],
                    linestyle=line_styles['ood'], linewidth=2.5)

        # Set titles
        if model_name == 'LSTMModel':
            model_name_display = 'LSTM'
        elif model_name == 'MBDNODE':
            model_name_display = 'MBD-NODE'
        else:
            model_name_display = model_name

        ax.set_title(f"{model_name_display} - Phase Space", fontsize=14)

        # Set x and y labels
        ax.set_xlabel('$x$', fontsize=16)
        ax.set_ylabel('$\\dot{x}$', fontsize=16)

        # Set limits
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        # Add grid
        ax.grid(True)

        # Add subplot labels in the upper left corner
        ax.text(0.5, -0.25, subplot_labels[i],
                transform=ax.transAxes,
                fontsize=12,
                horizontalalignment='center')

    # If we have fewer than 4 models, hide the unused axes
    for i in range(len(model_names), len(axs)):
        axs[i].set_visible(False)

    plt.tight_layout(rect=[0.0, 0.1, 1.0, 0.92])  # Make room for the legend at the top

    # Save or display
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
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

    # Process the specific configuration
    train_size, test_size, dt = target_train_size, target_test_size, target_dt
    print(f"\nProcessing: train_size={train_size}, test_size={test_size}, dt={dt}")

    # Find Ground_truth file
    ground_truth_file = f'Ground_truth for Single_Mass_Spring_Damper with training_size={train_size}_num_steps_test={test_size}_dt={dt}.npy'
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

    # Generate phase space plots for this group
    suffix = f"_train{train_size}_test{test_size}_dt{dt}"
    create_phase_space_comparison(
        ground_truth=ground_truth_data,
        model_predictions=model_data,
        model_names=sorted_models,
        training_size=train_size,
        output_dir=output_dir,
        output_filename=f"smsd_phase_space.png"
    )

    print(f"All phase space comparison plots created in {output_dir}")


if __name__ == "__main__":
    main()