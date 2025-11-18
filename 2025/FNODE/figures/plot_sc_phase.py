import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_slider_crank_phase():
    """
    Loads Slider Crank trajectory data for ground truth and various models,
    then plots theta vs omega phase space in a 2x2 grid layout:
    FNODE    MBD-NODE
    LSTM     FCNN
    """
    # Get parent directory (FNODE root) since this script is in figures/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    base_results_path = os.path.join(base_dir, 'results', 'Slider_Crank')

    # SPECIFIC CONFIGURATION TO PLOT
    train_steps = 1500
    test_steps = 3000
    dt = 0.01

    file_suffix = f'_prediction_train{train_steps}_test{test_steps}_dt{dt}.npy'

    model_types_to_plot = ['FNODE', 'MBDNODE', 'LSTM', 'FCNN']
    ground_truth_label = 'Ground_truth'

    total_steps = train_steps + test_steps

    # Load Ground Truth data
    gt_file_path = os.path.join(base_results_path, ground_truth_label, f"{ground_truth_label}{file_suffix}")
    gt_file_path_alt = os.path.join(base_results_path, f"Ground_truth for Slider_Crank with training_size={train_steps}_num_steps_test={test_steps}_dt={dt}.npy")

    ground_truth_data = None
    try:
        if os.path.exists(gt_file_path):
            ground_truth_data = np.load(gt_file_path)
        elif os.path.exists(gt_file_path_alt):
            ground_truth_data = np.load(gt_file_path_alt)
            gt_file_path = gt_file_path_alt
        else:
            logging.error(f"Ground truth file not found at {gt_file_path} or {gt_file_path_alt}")
            return

        # Reshape if it's (steps, 1, 2) -> (steps, 2) for single body
        if ground_truth_data.ndim == 3 and ground_truth_data.shape[1] == 1:
            ground_truth_data = ground_truth_data.squeeze(axis=1)
        if ground_truth_data.shape[0] != total_steps or ground_truth_data.shape[1] != 2:
            logging.warning(
                f"Ground truth data shape mismatch: Expected ({total_steps}, 2), Got {ground_truth_data.shape}. Truncating/Padding may occur.")
            if ground_truth_data.shape[0] > total_steps:
                ground_truth_data = ground_truth_data[:total_steps, :]
        logging.info(f"Loaded ground truth data from {gt_file_path}, shape: {ground_truth_data.shape}")
    except Exception as e:
        logging.error(f"Error loading ground truth data from {gt_file_path}: {e}")
        return

    # Prepare data dictionary for models
    model_data = {}
    for model_type in model_types_to_plot:
        model_file_path = os.path.join(base_results_path, f"{model_type} for Slider_Crank with training_size={train_steps}_num_steps_test={test_steps}_dt={dt}.npy")
        try:
            if os.path.exists(model_file_path):
                data = np.load(model_file_path)
                # Reshape if it's (steps, 1, 2) -> (steps, 2)
                if data.ndim == 3 and data.shape[1] == 1:
                    data = data.squeeze(axis=1)
                if data.shape[0] != total_steps or data.shape[1] != 2:
                    logging.warning(
                        f"{model_type} data shape mismatch: Expected ({total_steps}, 2), Got {data.shape}. Truncating/Padding may occur.")
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

    # Create 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=120)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Add top legend
    overall_legend_lines = [
        plt.Line2D([0], [0], color='blue', linestyle='-', lw=2.5),
        plt.Line2D([0], [0], color='red', linestyle='--', lw=2.5),
        plt.Line2D([0], [0], color='red', linestyle=':', lw=2.5)
    ]
    overall_legend_labels = ['Ground truth', 'ID generalization', 'OOD generalization']
    fig.legend(overall_legend_lines, overall_legend_labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, fontsize=20)

    # Define line styles
    gt_style = {'color': 'blue', 'linestyle': '-', 'linewidth': 2.5}
    pred_train_style = {'color': 'red', 'linestyle': '--', 'linewidth': 2.5}
    pred_test_style = {'color': 'red', 'linestyle': ':', 'linewidth': 2.5, 'alpha': 0.8}

    # Model positions and display names
    model_positions = {
        'FNODE': (0, 0),  # Top-left
        'MBDNODE': (0, 1),  # Top-right
        'LSTM': (1, 0),  # Bottom-left
        'FCNN': (1, 1)  # Bottom-right
    }

    model_display_names = {
        'FNODE': 'FNODE',
        'MBDNODE': 'MBD-NODE',
        'LSTM': 'LSTM',
        'FCNN': 'FCNN'
    }

    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    # Plot each model in its designated position
    for idx, model_type in enumerate(['FNODE', 'MBDNODE', 'LSTM', 'FCNN']):
        row, col = model_positions[model_type]
        ax = axs[row, col]

        pred_data = model_data.get(model_type)

        # Get current data length
        current_data_len_gt = len(ground_truth_data)

        # Add subplot label
        ax.text(0.5, -0.25, subplot_labels[idx], transform=ax.transAxes,
                fontsize=14, horizontalalignment='center')

        # Plot ground truth (theta vs omega phase space)
        ax.plot(ground_truth_data[:current_data_len_gt, 0], ground_truth_data[:current_data_len_gt, 1], **gt_style)

        if pred_data is not None:
            current_data_len_pred = len(pred_data)

            # Training predictions (ID region)
            train_end = min(train_steps, current_data_len_pred)
            ax.plot(pred_data[:train_end, 0], pred_data[:train_end, 1], **pred_train_style)

            # Testing predictions (OOD region)
            if train_steps < current_data_len_pred:
                ax.plot(pred_data[train_steps:current_data_len_pred, 0],
                        pred_data[train_steps:current_data_len_pred, 1], **pred_test_style)

        # Formatting
        ax.set_xlabel(r'$\theta_1$', fontsize=16)
        ax.set_ylabel(r'$\dot{\theta}_1$', fontsize=16)
        ax.grid(True, linestyle=':')
        ax.tick_params(labelbottom=True)

        # Set title
        display_name = model_display_names.get(model_type, model_type)
        ax.set_title(f"{display_name}", fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0.03, 0.08, 1.0, 0.92])

    # Save the figure
    output_figure_dir = './figures/Slider_Crank/comparison'
    os.makedirs(output_figure_dir, exist_ok=True)
    figure_path = os.path.join(output_figure_dir, 'sc_theta_omega.png')
    try:
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        logging.info(f"Plot saved to {figure_path}")
    except Exception as e:
        logging.error(f"Error saving plot to {figure_path}: {e}")
    plt.close(fig)


if __name__ == '__main__':
    plot_slider_crank_phase()