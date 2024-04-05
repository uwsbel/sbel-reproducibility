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

        # figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

        # ax1.plot(se_time, se_vel,
        #          linewidth=2, color=sns.color_palette("Oranges", 8)[-3], label="Sim. IMU + grass GPS")
        # ax1.plot(se_time, se_vel, linewidth=2,
        #          color=sns.color_palette("Oranges", 8)[-3])
        # ax1.plot(gt_time, gt_vel, linewidth=2,
        #          color="darkgray", label="Ground Truth")
        # ax2.plot(se_time_grass, se_vel_grass, linewidth=2,
        #          color=sns.color_palette("Oranges", 8)[-3], label="grass IMU + grass GPS")
        # ax2.plot(gt_time_grass, gt_vel_grass, linewidth=2,
        #          color="darkgray", label="Ground Truth")
        # ax1.set_xlabel("Time (s)")
        # ax1.set_ylabel("Velocity (m/s)")
        # ax2.set_xlabel("Time (s)")
        # ax2.set_yticks(np.arange(0, 2.6, 0.5))
        # ax1.set_yticks(np.arange(0, 2.6, 0.5))
        # ax1.set_xticks(np.arange(0, 21, 5))
        # ax2.set_xticks(np.arange(0, 21, 5))
        # ax2.set_yticks([])
        # ax1.legend(loc='upper left', fontsize=10)
        # ax2.legend(loc='upper left', fontsize=10)
        # figure.tight_layout(pad=1.0)
        # plt.savefig("../plots_3/grassGPS_simIMU.png", dpi=300)
        # plt.show()

        # ===================================
        # Compute RMSE
        # ==================================
        # Sim
        hz = 15
        min_time = min(gt_time[-1], se_time[-1])
        time = np.arange(0, min_time, 1/hz)
        gt_vel = np.interp(time, gt_time, gt_vel)
        window_size = 2
        se_vel = moving_average(se_vel, window_size)
        # Do same with se_time
        se_time = moving_average(se_time, window_size)
        # Interpolate all the data to the new time array
        se_vel = np.interp(time, se_time, se_vel)

        # Find RMSE between the two velocities
        rmse_sim = np.sqrt(np.square(np.subtract(gt_vel, se_vel)).mean())
        rmse_sim_.append(rmse_sim)
        # Normalize by the larger mean
        # rmse_sim = rmse_sim / max(np.max(gt_vel), np.max(se_vel))
        # print(f"Max GT vel: {np.max(gt_vel)}")

        print(f"Normalized RMSE in sim is: {rmse_sim}")

        # grass
        hz = 15
        min_time = min(gt_time_grass[-1], se_time_grass[-1])
        time = np.arange(0, min_time, 1/hz)
        gt_vel_grass = np.interp(time, gt_time_grass, gt_vel_grass)
        window_size = 2
        se_vel_grass = moving_average(se_vel_grass, window_size)
        # Do same with se_time
        se_time_grass = moving_average(se_time_grass, window_size)
        # Interpolate all the data to the new time array
        se_vel_grass = np.interp(time, se_time_grass, se_vel_grass)
        # Find RMSE between the two velocities
        rmse_grass = np.sqrt(
            np.square(np.subtract(gt_vel_grass, se_vel_grass)).mean())
        rmse_grass_.append(rmse_grass)

        # Normalize by the larger mean
        # rmse_grass = rmse_grass / max(np.max(gt_vel_grass), np.max(se_vel_grass))
        # print(f"Max GT vel: {np.max(gt_vel_grass)}")
        print(f"Normalized RMSE in grass is: {rmse_grass}")

        # conc
        hz = 15
        min_time = min(gt_time_conc[-1], se_time_conc[-1])
        time = np.arange(0, min_time, 1/hz)
        gt_vel_conc = np.interp(time, gt_time_conc, gt_vel_conc)
        window_size = 2
        se_vel_conc = moving_average(se_vel_conc, window_size)
        # Do same with se_time
        se_time_conc = moving_average(se_time_conc, window_size)
        # Interpolate all the data to the new time array
        se_vel_conc = np.interp(time, se_time_conc, se_vel_conc)
        # Find RMSE between the two velocities
        rmse_conc = np.sqrt(
            np.square(np.subtract(gt_vel_conc, se_vel_conc)).mean())
        rmse_conc_.append(rmse_conc)
        # Normalize by the larger mean
        # rmse_grass = rmse_grass / max(np.max(gt_vel_grass), np.max(se_vel_grass))
        # print(f"Max GT vel: {np.max(gt_vel_grass)}")
        print(f"Normalized RMSE in conc is: {rmse_conc}")

        # =====================================
        # Compute Spectral Flatness
        # =====================================
        # Sim
        gt_fft = np.fft.fft(gt_vel)
        se_fft = np.fft.fft(se_vel)

        # Compute Spectral flatness
        gt_sf = np.exp(np.mean(np.log(np.abs(gt_fft) + 1e-10))) / \
            (np.mean(np.abs(gt_fft)))
        se_sf = np.exp(np.mean(np.log(np.abs(se_fft) + 1e-10))) / \
            (np.mean(np.abs(se_fft)))

        # Compute absoulte difference in the specral flatness
        sf_diff = np.abs(gt_sf - se_sf)
        sf_diff_sim_.append(sf_diff)
        print(f"Spectral flatness difference in sim is: {sf_diff}")

        # grass
        gt_fft_grass = np.fft.fft(gt_vel_grass)
        se_fft_grass = np.fft.fft(se_vel_grass)

        # Compute Spectral flatness
        gt_sf_grass = np.exp(np.mean(np.log(np.abs(gt_fft_grass) + 1e-10))) / \
            (np.mean(np.abs(gt_fft_grass)))
        se_sf_grass = np.exp(np.mean(np.log(np.abs(se_fft_grass) + 1e-10))) / \
            (np.mean(np.abs(se_fft_grass)))

        # Compute absoulte difference in the specral flatness
        sf_diff_grass = np.abs(gt_sf_grass - se_sf_grass)
        sf_diff_grass_.append(sf_diff_grass)
        print(f"Spectral flatness difference in grass is: {sf_diff_grass}")

        # conc
        gt_fft_conc = np.fft.fft(gt_vel_conc)
        se_fft_conc = np.fft.fft(se_vel_conc)
        # Compute Spectral flatness
        gt_sf_conc = np.exp(np.mean(np.log(np.abs(gt_fft_conc) + 1e-10))) / \
            (np.mean(np.abs(gt_fft_conc)))

        se_sf_conc = np.exp(np.mean(np.log(np.abs(se_fft_conc) + 1e-10))) / \
            (np.mean(np.abs(se_fft_conc)))
        # Compute absoulte difference in the specral flatness
        sf_diff_conc = np.abs(gt_sf_conc - se_sf_conc)
        sf_diff_conc_.append(sf_diff_conc)
        print(f"Spectral flatness difference in conc is: {sf_diff_conc}")

    # W_1
    epd_sim_grass = wasserstein_distance(rmse_sim_, rmse_grass_)
    print(f"Wasserstein distance between sim and grass is: {epd_sim_grass}")
    epd_sim_conc = wasserstein_distance(rmse_sim_, rmse_conc_)
    print(f"Wasserstein distance between sim and conc is: {epd_sim_conc}")
    # W_2
    epd_sf_sim_grass = wasserstein_distance(sf_diff_sim_, sf_diff_grass_)
    print(
        f"Wasserstein distance between spectral flatness of sim and grass is: {epd_sf_sim_grass}")
    epd_sf_sim_conc = wasserstein_distance(sf_diff_sim_, sf_diff_conc_)
    print(
        f"Wasserstein distance between spectral flatness of sim and conc is: {epd_sf_sim_conc}")

    # VEPD
    vepd_sim_grass = (epd_sim_grass + epd_sf_sim_grass) / 2
    print(f"VEPD between sim and grass is: {vepd_sim_grass}")

    vepd_sim_conc = (epd_sim_conc + epd_sf_sim_conc) / 2
    print(f"VEPD between sim and conc is: {vepd_sim_conc}")
