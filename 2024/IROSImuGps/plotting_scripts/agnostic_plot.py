import os
import sys
from typing import Union, List
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import wasserstein_distance


plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 14,  # Changed from 18
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'font.family': 'sans-serif',
})


base_dir = "../data2/misc2/"


def extract_data(file_path: str):
    """
    Extract data from the file and return lists of all the columns
    """
    df = pd.read_csv(file_path)
    se_time = df["se_time"].to_list()
    gt_time = df["gt_time"].to_list()
    se_vel = df["se_vel"].to_list()
    gt_vel = df["gt_vel"].to_list()

    # Drop the zeroes from gt_time and gt_vel
    gt_time = [x for x in gt_time if x != 0.0]
    gt_vel = [x for x in gt_vel if x != 0.0]

    return se_time, gt_time, se_vel, gt_vel


def moving_average(data: Union[List[float], np.ndarray], window_size: int) -> np.ndarray:
    """
    Computes the moving average of the given list using a window of size n.
    """
    # Create a window (kernel) for convolution that represents the uniform weights of the moving average
    window = np.ones(int(window_size)) / float(window_size)

    # Use np.convolve to compute the moving average, 'valid' mode returns output only for complete windows
    return np.convolve(data, window, 'valid')


def plot(ax, time_se, se_vel, color=sns.color_palette("Reds", 8)[-1]):
    """
    Plot the given data
    """
    ax.plot(time_se, se_vel,
            linewidth=2, color=color)


if __name__ == "__main__":
    # provide path to the csv file that needs to be quickly plotted
    # pathSim = sys.argv[1]
    # pathGrass = sys.argv[2]
    # pathConc = sys.argv[3]

    sim_base_path = "../data2/misc2/flat/sim/sine_"
    grass_base_path = "../data2/misc2/flat/grass/sine_"
    conc_base_path = "../data2/misc2/flat/conc/flat_"

    rmse_sim_ = []
    rmse_grass_ = []
    rmse_conc_ = []

    sf_diff_sim_ = []
    sf_diff_grass_ = []
    sf_diff_conc_ = []
    for i in range(1, 10):
        pathSim = sim_base_path + str(i) + ".csv"
        pathGrass = grass_base_path + str(i) + ".csv"
        pathConc = conc_base_path + str(i) + ".csv"
        se_time, gt_time, se_vel, gt_vel = extract_data(pathSim)
        se_time_grass, gt_time_grass, se_vel_grass, gt_vel_grass = extract_data(
            pathGrass)
        se_time_conc, gt_time_conc, se_vel_conc, gt_vel_conc = extract_data(
            pathConc)

        se_time = [x - se_time[0] for x in se_time]
        gt_time = [x - gt_time[0] for x in gt_time]

        se_time_grass = [x - se_time_grass[0] for x in se_time_grass]
        gt_time_grass = [x - gt_time_grass[0] for x in gt_time_grass]

        se_time_conc = [x - se_time_conc[0] for x in se_time_conc]
        gt_time_conc = [x - gt_time_conc[0] for x in gt_time_conc]
        # Cut conc data at 25 seconds
        se_time_conc = [x for x in se_time_conc if x <= 25]
        gt_time_conc = [x for x in gt_time_conc if x <= 25]
        se_vel_conc = [se_vel_conc[i]
                       for i, x in enumerate(se_time_conc) if x <= 25]
        gt_vel_conc = [gt_vel_conc[i]
                       for i, x in enumerate(gt_time_conc) if x <= 25]

        figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))

        ax1.plot(se_time, se_vel,
                 linewidth=2, color=sns.color_palette("Oranges", 8)[-3], label="Sim.")
        ax1.plot(gt_time, gt_vel, linewidth=2,
                 color="darkgray", label="Ground Truth")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Velocity (m/s)")
        ax1.set_yticks(np.arange(0, 2.6, 0.5))
        ax1.set_xticks(np.arange(0, 31, 5))
        ax1.legend(loc='lower right', fontsize=10)

        ax2.plot(se_time_grass, se_vel_grass, linewidth=2,
                 color=sns.color_palette("Oranges", 8)[-3], label="Grass")
        ax2.plot(gt_time_grass, gt_vel_grass, linewidth=2,
                 color="darkgray", label="Ground Truth")
        ax2.set_xlabel("Time (s)")
        ax2.set_yticks([])
        ax2.set_xticks(np.arange(0, 31, 5))
        ax2.legend(loc='lower right', fontsize=10)

        ax3.plot(se_time_conc, se_vel_conc, linewidth=2,
                 color=sns.color_palette("Oranges", 8)[-3], label="Concrete")
        ax3.plot(gt_time_conc, gt_vel_conc, linewidth=2,
                 color="darkgray", label="Ground Truth")
        ax3.set_xlabel("Time (s)")
        ax3.set_yticks([])
        ax3.set_xticks(np.arange(0, 31, 5))
        ax3.legend(loc='lower right', fontsize=10)

        figure.tight_layout(pad=1.0)
        plt.savefig(f"../plots_3/agnostic_{i}.png", dpi=300)
        plt.show()
