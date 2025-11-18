import numpy as np
import matplotlib.pyplot as plt
import os
import re
from matplotlib.gridspec import GridSpec
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_npy_data(file_path):
    """Load numpy data from file and reshape if needed."""
    try:
        data = np.load(file_path, allow_pickle=True)
        logger.info(f"Loaded data from {os.path.basename(file_path)} with shape: {data.shape}")

        # Handle different data formats
        if data.ndim == 3:
            if data.shape[1] == 2 and data.shape[2] == 2:
                # Already in correct shape (steps, bodies, 2)
                return data
            else:
                # Try to identify if dimensions need transposing
                logger.info(f"Unusual 3D shape: {data.shape}, trying to infer correct format")
                # If dimensions are swapped, transpose them
                if data.shape[2] == 2:
                    transposed = data
                    logger.info(f"Using data as is: {transposed.shape}")
                    return transposed
        elif data.ndim == 2 and data.shape[1] == 4:
            # For Double Pendulum with format [θ₁, ω₁, θ₂, ω₂]
            steps = data.shape[0]
            reshaped = np.zeros((steps, 2, 2))
            # Important: correct reshaping for double pendulum
            reshaped[:, 0, 0] = data[:, 0]  # θ₁
            reshaped[:, 0, 1] = data[:, 1]  # ω₁
            reshaped[:, 1, 0] = data[:, 2]  # θ₂
            reshaped[:, 1, 1] = data[:, 3]  # ω₂
            logger.info(f"Reshaped data to (steps, bodies, 2): {reshaped.shape}")
            return reshaped

        logger.warning(f"Could not properly reshape data with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return np.array([])  # Return empty array on error


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
                                  output_dir=None, output_filename="dp_phase_space_grid.png"):
    """
    Create phase space plots comparing ground truth and model predictions.

    Args:
        ground_truth: Ground truth data with shape [steps, 2, 2]
        model_predictions: Dictionary of model predictions {model_name: data}
        model_names: List of model names to plot
        training_size: Index separating training (ID) and test (OOD) data
        output_dir: Directory to save plots
        output_filename: Name of the output file
    """
    # Filter out empty model data
    valid_models = [m for m in model_names if m in model_predictions and len(model_predictions[m]) > 0]

    if not valid_models:
        logger.warning("No valid models to plot!")
        return

    # Preferred order for models
    preferred_order = ['FNODE', 'MBDNODE', 'LSTMModel', 'FCNN']
    sorted_models = []

    # First add models in preferred order
    for model in preferred_order:
        if model in valid_models:
            sorted_models.append(model)

    # Then add any remaining models
    for model in valid_models:
        if model not in sorted_models:
            sorted_models.append(model)

    num_models = min(len(sorted_models), 4)  # Limit to 4 models
    logger.info(f"Creating phase space plots for {num_models} models: {sorted_models[:num_models]}")

    # Set up figure with up to 4 rows and 2 columns (one per pendulum)
    fig = plt.figure(figsize=(14, 5 * num_models))

    # Set up GridSpec for more control over subplot positioning
    gs = GridSpec(num_models, 2, figure=fig, hspace=0.4, wspace=0.25)



    # Create a single legend for the entire figure
    legend_handles = [
        plt.Line2D([0], [0], color='blue', linestyle='-', lw=2.5, label='Ground truth'),
        plt.Line2D([0], [0], color='red', linestyle='--', lw=2.5, label='ID generalization'),
        plt.Line2D([0], [0], color='red', linestyle=':', lw=2.5, label='OOD generalization')
    ]

    fig.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, 0.94), ncol=3, fontsize=20)

    # Model display names
    model_display_names = {
        'FNODE': 'FNODE',
        'MBDNODE': 'MBD-NODE',
        'LSTMModel': 'LSTM',
        'LSTM': 'LSTM',  # Add both variants
        'FCNN': 'FCNN'
    }

    # Subplot labels
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    # Auto-calculate axis limits from all data (ground truth + models)
    if len(ground_truth) > 0:
        # Start with ground truth limits
        x1_min, x1_max = ground_truth[:, 0, 0].min(), ground_truth[:, 0, 0].max()
        v1_min, v1_max = ground_truth[:, 0, 1].min(), ground_truth[:, 0, 1].max()
        x2_min, x2_max = ground_truth[:, 1, 0].min(), ground_truth[:, 1, 0].max()
        v2_min, v2_max = ground_truth[:, 1, 1].min(), ground_truth[:, 1, 1].max()

        # Extend limits based on all model predictions
        for model_data in model_predictions.values():
            if len(model_data) > 0:
                x1_min = min(x1_min, model_data[:, 0, 0].min())
                x1_max = max(x1_max, model_data[:, 0, 0].max())
                v1_min = min(v1_min, model_data[:, 0, 1].min())
                v1_max = max(v1_max, model_data[:, 0, 1].max())
                x2_min = min(x2_min, model_data[:, 1, 0].min())
                x2_max = max(x2_max, model_data[:, 1, 0].max())
                v2_min = min(v2_min, model_data[:, 1, 1].min())
                v2_max = max(v2_max, model_data[:, 1, 1].max())

        # Add 10% padding
        x1_pad = 0.1 * (x1_max - x1_min)
        v1_pad = 0.1 * (v1_max - v1_min)
        x2_pad = 0.1 * (x2_max - x2_min)
        v2_pad = 0.1 * (v2_max - v2_min)

        x1_lim = [x1_min - x1_pad, x1_max + x1_pad]
        v1_lim = [v1_min - v1_pad, v1_max + v1_pad]
        x2_lim = [x2_min - x2_pad, x2_max + x2_pad]
        v2_lim = [v2_min - v2_pad, v2_max + v2_pad]
    else:
        # Default limits if ground truth is empty
        x1_lim = [-1.5, 1.5]
        v1_lim = [-5, 5]
        x2_lim = [-3, 3]
        v2_lim = [-8, 8]

    # Plot each model
    for i, model_name in enumerate(sorted_models[:num_models]):

        # Create axes for pendulum 1 and 2
        ax1 = fig.add_subplot(gs[i, 0])  # Pendulum 1
        ax2 = fig.add_subplot(gs[i, 1])  # Pendulum 2

        # Plot ground truth for both pendulums
        ax1.plot(ground_truth[:, 0, 0], ground_truth[:, 0, 1],
                 color='blue', linewidth=2.5, label='Ground Truth')
        ax2.plot(ground_truth[:, 1, 0], ground_truth[:, 1, 1],
                 color='blue', linewidth=2.5)

        # Plot model predictions
        model_data = model_predictions[model_name]

        # Training data (ID)
        id_end = min(training_size, len(model_data))
        if model_name=='MBDNODE':
            if id_end > 0:
                ax1.plot(model_data[:id_end, 0, 0], model_data[:id_end, 0, 1],
                         color='red', linestyle='--', linewidth=2.5)
                ax2.plot(model_data[:id_end, 1, 0], model_data[:id_end, 1, 1],
                         color='red', linestyle='--', linewidth=2.5)

            # Testing data (OOD)
            if len(model_data) > training_size:
                ax1.plot(model_data[training_size:, 0, 0], model_data[training_size:, 0, 1],
                         color='red', linestyle=':', linewidth=2.5)
                ax2.plot(model_data[training_size:, 1, 0], model_data[training_size:, 1, 1],
                         color='red', linestyle=':', linewidth=2.5)
            ax1.set_title(f"{model_display_names[model_name]} - Pendulum 1", fontsize=16)
            ax2.set_title(f"{model_display_names[model_name]} - Pendulum 2", fontsize=16)
        else:
            if id_end > 0:
                ax1.plot(model_data[:id_end, 0, 0], model_data[:id_end, 0, 1],
                         color='red', linestyle='--', linewidth=2.5)
                ax2.plot(model_data[:id_end, 1, 0], model_data[:id_end, 1, 1],
                         color='red', linestyle='--', linewidth=2.5)

            # Testing data (OOD)
            if len(model_data) > training_size:
                ax1.plot(model_data[training_size:, 0, 0], model_data[training_size:, 0, 1],
                         color='red', linestyle=':', linewidth=2.5)
                ax2.plot(model_data[training_size:, 1, 0], model_data[training_size:, 1, 1],
                         color='red', linestyle=':', linewidth=2.5)
            ax1.set_title(f"{model_display_names[model_name]} - Pendulum 1", fontsize=16)
            ax2.set_title(f"{model_display_names[model_name]} - Pendulum 2", fontsize=16)

        # Set axis labels
        ax1.set_xlabel(r'$\theta_1$ ', fontsize=16)
        ax1.set_ylabel(r'$\dot{\theta}_1$ ', fontsize=16)
        ax2.set_xlabel(r'$\theta_2$ ', fontsize=16)
        ax2.set_ylabel(r'$\dot{\theta}_2$ ', fontsize=16)

        # Set consistent axis limits
        ax1.set_xlim(x1_lim)
        ax1.set_ylim(v1_lim)
        ax2.set_xlim(x2_lim)
        ax2.set_ylim(v2_lim)

        # Set aspect ratio to 'equal' for better phase space visualization
        ax1.set_aspect('auto')
        ax2.set_aspect('auto')

        # Add grid
        ax1.grid(True, linestyle=':', alpha=0.7)
        ax2.grid(True, linestyle=':', alpha=0.7)

        # Add subplot labels - positioned below x-axis to avoid overlap
        ax1.text(0.5, -0.25, subplot_labels[i * 2], transform=ax1.transAxes,
                 fontsize=14, ha='center')
        ax2.text(0.5, -0.25, subplot_labels[i * 2 + 1], transform=ax2.transAxes,
                 fontsize=14, ha='center')

    # Adjust layout
    plt.tight_layout(rect=[0.03, 0.08, 1.0, 0.95])

    # Save figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Figure saved to {output_path}")
    else:
        plt.close(fig)

    plt.close(fig)


def main():
    """Main function to create phase space plots for Double Pendulum models."""
    # Set the test case
    test_case = 'Double_Pendulum'

    # SPECIFIC CONFIGURATION TO PLOT
    target_train_size = 300
    target_test_size = 100
    target_dt = 0.01

    # Set base directory for data files
    # Get parent directory (FNODE root) since this script is in figures/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(base_dir, 'results', test_case)

    # Set output directory for comparison figures
    output_dir = os.path.join(base_dir, 'figures', test_case, 'comparison')
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Looking for results in: {results_dir}")
    if not os.path.exists(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        return

    logger.info(f"Processing specific configuration: Train: {target_train_size}, Test: {target_test_size}, dt: {target_dt}")

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
        logger.error(f"No files found for configuration train={target_train_size}, test={target_test_size}, dt={target_dt}")
        return

    logger.info(f"Found models: {[f[0] for f in file_list]}")

    # Process the specific configuration
    train_size, test_size, dt = target_train_size, target_test_size, target_dt
    logger.info(f"Processing: train_size={train_size}, test_size={test_size}, dt={dt}")

    # Find Ground truth file
    ground_truth_file = f'Ground_truth for Double_Pendulum with training_size={train_size}_num_steps_test={test_size}_dt={dt}.npy'
    ground_truth_path = os.path.join(results_dir, ground_truth_file)

    if not os.path.exists(ground_truth_path):
        logger.error(f"Ground truth file not found: {ground_truth_file}")
        return

    # Load ground truth data
    ground_truth_data = load_npy_data(ground_truth_path)

    if len(ground_truth_data) == 0:
        logger.error("Ground truth data is empty or failed to load")
        return

    # Load model predictions
    model_data = {}
    for model_name, filename in file_list:
        if model_name != 'Ground_truth':  # Skip ground truth file in this loop
            model_path = os.path.join(results_dir, filename)
            data = load_npy_data(model_path)
            if len(data) > 0:
                model_data[model_name] = data
                logger.info(f"Loaded {model_name} data with shape {data.shape}")

    # Create file suffix for this configuration
    suffix = f"_train{train_size}_test{test_size}_dt{dt}"

    # Create the grid comparison plot
    if model_data:
        try:
            create_phase_space_comparison(
                ground_truth=ground_truth_data,
                model_predictions=model_data,
                model_names=list(model_data.keys()),
                training_size=train_size,
                output_dir=output_dir,
                output_filename=f"dp_x_v.png"
            )
            logger.info(f"Created comparison plot for train={train_size}, test={test_size}, dt={dt}")
        except Exception as e:
            logger.error(f"Error creating comparison plot: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.warning("No valid model data found for plotting")

    logger.info(f"All phase space plots created in {output_dir}")


if __name__ == "__main__":
    main()