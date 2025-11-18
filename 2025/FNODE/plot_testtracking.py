#!/usr/bin/env python3
"""
plot_testtracking.py
Plots test tracking comparison from existing CSV files.
This script ONLY generates plots - it does NOT train models or save data.
"""

import os
import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Plot test tracking comparison from CSV files')
    parser.add_argument('--test_case', type=str, default='Triple_Mass_Spring_Damper',
                        help='Test case name (default: Triple_Mass_Spring_Damper)')
    parser.add_argument('--skip_mbdnode', action='store_true',
                        help='Skip MBDNODE in plots')
    parser.add_argument('--skip_fnode', action='store_true',
                        help='Skip FNODE in plots')
    parser.add_argument('--skip_lstm', action='store_true',
                        help='Skip LSTM in plots')
    parser.add_argument('--skip_fcnn', action='store_true',
                        help='Skip FCNN in plots')
    return parser.parse_args()


def plot_comparison(mbdnode_df, fnode_df, lstm_df, fcnn_df, test_case):
    """Generate Test MSE vs Training Time comparison plots (both minutes and seconds)"""
    logger.info("="*80)
    logger.info("GENERATING COMPARISON PLOTS")
    logger.info("="*80)

    # Create output directory
    comparison_dir = os.path.join(os.getcwd(), 'figures', test_case, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)

    # Plot each model
    models_data = [
        ('MBD-NODE', mbdnode_df, 'blue', 'o'),
        ('FNODE', fnode_df, 'green', 's'),
        ('LSTM', lstm_df, 'red', '^'),
        ('FCNN', fcnn_df, 'orange', 'D')
    ]

    # Generate both plots: minutes and seconds
    for time_unit in ['minutes', 'seconds']:
        plt.figure(figsize=(12, 8))

        for model_name, df, color, marker in models_data:
            # Skip if DataFrame is None (model was skipped)
            if df is None:
                continue
            # Filter out None values
            valid_data = df[df['test_mse'].notna()]

            # Convert time based on unit
            if time_unit == 'minutes':
                time_data = valid_data['cumulative_time'] / 60
                xlabel = 'Cumulative Training Time (minutes)'
                filename = 'test_mse_vs_time_all_models.png'
            else:  # seconds
                time_data = valid_data['cumulative_time']
                xlabel = 'Cumulative Training Time (seconds)'
                filename = 'test_mse_vs_time_all_models_seconds.png'

            test_mse = valid_data['test_mse']

            # Get final test MSE
            final_mse = test_mse.iloc[-1] if len(test_mse) > 0 else float('nan')

            plt.plot(time_data, test_mse,
                    label=f'{model_name} (final: {final_mse:.4f})',
                    color=color, marker=marker, markevery=max(1, len(time_data)//20),
                    linewidth=2, markersize=6, alpha=0.8)

        plt.xlabel(xlabel, fontsize=14, fontweight='bold')
        plt.ylabel('Test MSE', fontsize=14, fontweight='bold')
        plt.title(f'Test MSE vs Training Time - All Models ({test_case})',
                  fontsize=16, fontweight='bold', pad=20)
        plt.yscale('log')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=12, loc='best', framealpha=0.9)
        plt.tight_layout()

        # Save
        output_path = os.path.join(comparison_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot ({time_unit}) saved to: {output_path}")
        plt.close()


def load_tracking_data(test_case, model_name):
    """Load existing test tracking CSV data for a model"""
    csv_path = os.path.join(os.getcwd(), 'results', test_case, model_name, f'{model_name}_test_tracking.csv')
    if os.path.exists(csv_path):
        logger.info(f"Loading {model_name} data from: {csv_path}")
        return pd.read_csv(csv_path)
    else:
        logger.warning(f"CSV not found for {model_name}: {csv_path}")
        return None


def main():
    args = parse_arguments()

    logger.info("="*80)
    logger.info("PLOTTING TEST TRACKING COMPARISON (FROM EXISTING CSV FILES)")
    logger.info("="*80)

    # Determine which models to plot (all enabled by default unless skipped)
    plot_mbdnode = not args.skip_mbdnode
    plot_fnode = not args.skip_fnode
    plot_lstm = not args.skip_lstm
    plot_fcnn = not args.skip_fcnn

    logger.info(f"Models to plot: MBDNODE={plot_mbdnode}, FNODE={plot_fnode}, "
               f"LSTM={plot_lstm}, FCNN={plot_fcnn}")

    # Load existing CSV data instead of training
    mbdnode_df = load_tracking_data(args.test_case, 'MBDNODE') if plot_mbdnode else None
    fnode_df = load_tracking_data(args.test_case, 'FNODE') if plot_fnode else None
    lstm_df = load_tracking_data(args.test_case, 'LSTM') if plot_lstm else None
    fcnn_df = load_tracking_data(args.test_case, 'FCNN') if plot_fcnn else None

    # Generate comparison plot only if at least 2 models were loaded
    run_models = [df for df in [mbdnode_df, fnode_df, lstm_df, fcnn_df] if df is not None]
    if len(run_models) >= 2:
        plot_comparison(mbdnode_df, fnode_df, lstm_df, fcnn_df, args.test_case)
    else:
        logger.info("Skipping comparison plot (need at least 2 models with data)")

    logger.info("="*80)
    logger.info("PLOTTING COMPLETE!")
    logger.info("="*80)
    logger.info("\nPlots saved to:")
    logger.info(f"  - figures/{args.test_case}/comparison/test_mse_vs_time_all_models.png (minutes)")
    logger.info(f"  - figures/{args.test_case}/comparison/test_mse_vs_time_all_models_seconds.png (seconds)")
    logger.info("="*80)

if __name__ == "__main__":
    main()