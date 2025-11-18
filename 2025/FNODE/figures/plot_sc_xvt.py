# plot_sc_xvt.py
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_slider_crank_xvt():
    """
    Loads Slider Crank trajectory data for ground truth and various models,
    then plots theta_1 (angle) vs. time and omega_1 (angular velocity) vs. time
    in a 4x2 grid with subplot labels and top legend.
    """
    # Get parent directory (FNODE root) since this script is in figures/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    base_results_path = os.path.join(base_dir, 'results', 'Slider_Crank')
    file_suffix = '_prediction_train1500_test3000_dt0.01.npy'

    model_types_to_plot = ['FNODE', 'MBDNODE', 'LSTM', 'FCNN']
    ground_truth_label = 'Ground_truth'

    train_steps = 1500
    test_steps = 3000
    total_steps = train_steps + test_steps
    dt = 0.01
    time_vector = np.arange(0, total_steps * dt, dt)

    # Ensure time_vector matches the typical length of loaded data
    if len(time_vector) > total_steps:
        time_vector = time_vector[:total_steps]
    elif len(time_vector) < total_steps:
        logging.warning(
            f"Time vector length ({len(time_vector)}) is less than total_steps ({total_steps}). Data might be truncated in plots.")

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

    # Plotting
    fig, axs = plt.subplots(len(model_types_to_plot), 2, figsize=(14, 5 * len(model_types_to_plot)), dpi=120, sharex=False, squeeze=False)
    plt.subplots_adjust(hspace=0.5, wspace=0.25)  # Increased vertical spacing

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

    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    for i, model_type in enumerate(model_types_to_plot):
        pred_data = model_data.get(model_type)

        # Current data length for this model, could be less than total_steps if truncated
        current_data_len_gt = len(ground_truth_data)
        current_time_vector = time_vector[:current_data_len_gt]

        # Theta plot
        ax_theta = axs[i, 0]
        ax_omega = axs[i, 1]

        # Add subplot labels
        ax_theta.text(0.5, -0.18, subplot_labels[i * 2], transform=ax_theta.transAxes, fontsize=14, ha='center')
        ax_omega.text(0.5, -0.18, subplot_labels[i * 2 + 1], transform=ax_omega.transAxes, fontsize=14, ha='center')

        # Plot ground truth
        ax_theta.plot(current_time_vector, ground_truth_data[:current_data_len_gt, 0], **gt_style)
        ax_omega.plot(current_time_vector, ground_truth_data[:current_data_len_gt, 1], **gt_style)

        if pred_data is not None:
            current_data_len_pred = len(pred_data)
            current_time_vector_pred = time_vector[:current_data_len_pred]

            # Training predictions
            train_end = min(train_steps, current_data_len_pred)
            ax_theta.plot(current_time_vector_pred[:train_end],
                          pred_data[:train_end, 0],
                          color=model_colors.get(model_type, 'gray'), **pred_train_style)
            ax_omega.plot(current_time_vector_pred[:train_end],
                          pred_data[:train_end, 1],
                          color=model_colors.get(model_type, 'gray'), **pred_train_style)

            # Testing predictions
            if train_steps < current_data_len_pred:
                ax_theta.plot(current_time_vector_pred[train_steps:current_data_len_pred],
                              pred_data[train_steps:current_data_len_pred, 0],
                              color=model_colors.get(model_type, 'gray'), **pred_test_style)
                ax_omega.plot(current_time_vector_pred[train_steps:current_data_len_pred],
                              pred_data[train_steps:current_data_len_pred, 1],
                              color=model_colors.get(model_type, 'gray'), **pred_test_style)

        # Formatting
        ax_theta.set_ylabel(r'$\theta$', fontsize=16)
        ax_omega.set_ylabel(r'$\dot{\theta}$', fontsize=16)
        ax_theta.grid(True, linestyle=':')
        ax_omega.grid(True, linestyle=':')
        ax_theta.set_xlabel('t', fontsize=16)
        ax_omega.set_xlabel('t', fontsize=16)

        # Always show x-axis tick labels
        ax_theta.tick_params(labelbottom=True)
        ax_omega.tick_params(labelbottom=True)

        # Set titles
        ax_theta.set_title(f"{model_display_names.get(model_type, model_type)} - Angle", fontsize=16)
        ax_omega.set_title(f"{model_display_names.get(model_type, model_type)} - Angular Velocity", fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0.03, 0.08, 1.0, 0.95])

    # Save the figure
    output_figure_dir = './figures/Slider_Crank/comparison'
    os.makedirs(output_figure_dir, exist_ok=True)
    figure_path = os.path.join(output_figure_dir, 'sc_x_v_time.png')
    try:
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        logging.info(f"Plot saved to {figure_path}")
    except Exception as e:
        logging.error(f"Error saving plot to {figure_path}: {e}")
    plt.close(fig)


if __name__ == '__main__':
    plot_slider_crank_xvt()