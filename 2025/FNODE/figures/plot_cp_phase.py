#!/usr/bin/env python3
"""
plot_cp_phase.py
Creates phase space comparison plots for Cart Pole system.
Plots x vs v_x (cart phase space) and θ vs ω (pole phase space) for different models.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def plot_cart_pole_phase():
    """
    Loads Cart Pole trajectory data for ground truth and various models,
    then plots phase space (x vs v_x and θ vs ω) in a 4x2 grid layout.
    Left column: Cart phase space (x vs v_x)
    Right column: Pole phase space (θ vs ω)
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

    # Plotting - 4 models x 2 columns (cart and pole phase spaces)
    fig, axs = plt.subplots(len(model_types_to_plot), 2, figsize=(14, 5 * len(model_types_to_plot)),
                             dpi=120, squeeze=False)
    plt.subplots_adjust(hspace=0.6, wspace=0.25)  # Adjusted spacing

    # Add top legend - consistent with other plot scripts
    overall_legend_lines = [
        plt.Line2D([0], [0], color='blue', linestyle='-', lw=2.5),
        plt.Line2D([0], [0], color='red', linestyle='--', lw=2.5),
        plt.Line2D([0], [0], color='red', linestyle=':', lw=2.5)
    ]
    overall_legend_labels = ['Ground truth', 'ID generalization', 'OOD generalization']
    fig.legend(overall_legend_lines, overall_legend_labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.985), ncol=3, frameon=True, fontsize=20)

    # Define line styles - consistent with other scripts
    gt_style = {'color': 'blue', 'linestyle': '-', 'linewidth': 2.5}
    pred_train_style = {'color': 'red', 'linestyle': '--', 'linewidth': 2.5}
    pred_test_style = {'color': 'red', 'linestyle': ':', 'linewidth': 2.5, 'alpha': 0.8}

    model_display_names = {
        'FNODE': 'FNODE',
        'MBDNODE': 'MBD-NODE',
        'LSTM': 'LSTM',
        'FCNN': 'FCNN'
    }

    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    for i, model_type in enumerate(model_types_to_plot):
        pred_data = model_data.get(model_type)

        # Get axes for cart and pole phase spaces
        ax_cart = axs[i, 0]  # Cart phase space (x vs v_x)
        ax_pole = axs[i, 1]  # Pole phase space (θ vs ω)

        # Add subplot labels - positioned below x-axis to avoid overlap
        ax_cart.text(0.5, -0.22, subplot_labels[i * 2], transform=ax_cart.transAxes, fontsize=14, ha='center')
        ax_pole.text(0.5, -0.22, subplot_labels[i * 2 + 1], transform=ax_pole.transAxes, fontsize=14, ha='center')

        # Plot ground truth phase space with solid blue line
        ax_cart.plot(ground_truth_data[:, 0], ground_truth_data[:, 1], **gt_style)
        ax_pole.plot(ground_truth_data[:, 2], ground_truth_data[:, 3], **gt_style)

        if pred_data is not None:
            # Plot training region with dashed red line
            ax_cart.plot(pred_data[:train_steps, 0], pred_data[:train_steps, 1], **pred_train_style)
            ax_pole.plot(pred_data[:train_steps, 2], pred_data[:train_steps, 3], **pred_train_style)

            # Plot testing region with dotted red line
            if len(pred_data) > train_steps:
                ax_cart.plot(pred_data[train_steps:, 0], pred_data[train_steps:, 1], **pred_test_style)
                ax_pole.plot(pred_data[train_steps:, 2], pred_data[train_steps:, 3], **pred_test_style)

        # Formatting
        ax_cart.set_xlabel(r'$x$', fontsize=16)
        ax_cart.set_ylabel(r'$\dot{x}$', fontsize=16)
        ax_pole.set_xlabel(r'$\theta$', fontsize=16)
        ax_pole.set_ylabel(r'$\dot{\theta}$', fontsize=16)

        ax_cart.grid(True, linestyle=':', alpha=0.3)
        ax_pole.grid(True, linestyle=':', alpha=0.3)

        # Set titles
        ax_cart.set_title(f"{model_display_names.get(model_type, model_type)} - Cart Phase Space", fontsize=16)
        ax_pole.set_title(f"{model_display_names.get(model_type, model_type)} - Pole Phase Space", fontsize=16)

        # Increase tick label size
        ax_cart.tick_params(axis='both', labelsize=12)
        ax_pole.tick_params(axis='both', labelsize=12)

    # Adjust layout
    plt.tight_layout(rect=[0.03, 0.08, 1.0, 0.95])

    # Save the figure
    output_figure_dir = './figures/Cart_Pole/comparison'
    os.makedirs(output_figure_dir, exist_ok=True)
    figure_path = os.path.join(output_figure_dir, 'cp_phase_space.png')
    try:
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        logging.info(f"Plot saved to {figure_path}")
    except Exception as e:
        logging.error(f"Error saving plot to {figure_path}: {e}")
    plt.close(fig)


if __name__ == '__main__':
    plot_cart_pole_phase()