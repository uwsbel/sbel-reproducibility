import os
from typing import Union, List
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt


"""
FOR COMMAND LINE ARGUMENTS ACCEPTED PLEASE USE python3 computeEPD.py -h
"""

plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 12,  # Changed from 18
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'font.family': 'sans-serif',
})


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

    print(f"Last GT time recorded: {gt_time[-1]}")
    print(f"Last SE time recorded: {se_time[-1]}")

    return se_time, gt_time, se_vel, gt_vel


def moving_average(data: Union[List[float], np.ndarray], window_size: int) -> np.ndarray:
    """
    Computes the moving average of the given list using a window of size n.
    """
    # Create a window (kernel) for convolution that represents the uniform weights of the moving average
    window = np.ones(int(window_size)) / float(window_size)

    # Use np.convolve to compute the moving average, 'valid' mode returns output only for complete windows
    return np.convolve(data, window, 'valid')

# We have 4 scenarios in both sim and real.
# In sim, each scenario has 6 models associated with it
# In real, only 1 model is associated with each scenario
# In sim and real, within each scenarion and model, we have 10 test runs


color_pallette = {
    # "baseline_results": sns.color_palette("Reds", 8)[-1],
    "chNormal_results": sns.color_palette("Greens", 8)[-2],
    "chWalk_results": sns.color_palette("Oranges", 8)[-3],
    "AirSim_results": sns.color_palette("Purples", 8)[-4],
    "chNormal_AirSim_results": sns.color_palette("Blues", 8)[-5],
    "chWalk_AirSim_results": (0, 0, 0),
    # "gazebo_results": sns.color_palette("YlOrBr", 8)[-1]
}

scenario = {
    0: "rest",
    1: "path",
    2: "circle",
    3: "sine"
}
scenario_list = list(scenario.values())

model_folders = [
    # "baseline_results",
    "chNormal_results",
    "chWalk_results",
    "AirSim_results",
    "chNormal_AirSim_results",
    "chWalk_AirSim_results",
    # "gazebo_results"
]


# Map folders to plot model names
model_names = {
    # "baseline_results": "Baseline",
    "chNormal_results": "Ch:Gauss",
    "chWalk_results": "Ch:RW",
    "AirSim_results": "AirSim",
    "chNormal_AirSim_results": "Ch:Gauss+AirSim",
    "chWalk_AirSim_results": "Ch:RW+AirSim",
    # "gazebo_results": "Gazebo"
}

models_list = list(model_names.values())

sim_baseline_paths = {0: [], 1: [], 2: [], 3: []}
sim_baseline_data = {0: [], 1: [], 2: [], 3: []}
sim_baseline_data_fft = {0: [], 1: [], 2: [], 3: []}

sim_chNormal_paths = {0: [], 1: [], 2: [], 3: []}
sim_chNormal_data = {0: [], 1: [], 2: [], 3: []}
sim_chNormal_data_fft = {0: [], 1: [], 2: [], 3: []}

sim_chWalk_paths = {0: [], 1: [], 2: [], 3: []}
sim_chWalk_data = {0: [], 1: [], 2: [], 3: []}
sim_chWalk_data_fft = {0: [], 1: [], 2: [], 3: []}

sim_AirSim_paths = {0: [], 1: [], 2: [], 3: []}
sim_AirSim_data = {0: [], 1: [], 2: [], 3: []}
sim_AirSim_data_fft = {0: [], 1: [], 2: [], 3: []}

sim_chNormal_AirSim_paths = {0: [], 1: [], 2: [], 3: []}
sim_chNormal_AirSim_data = {0: [], 1: [], 2: [], 3: []}
sim_chNormal_AirSim_data_fft = {0: [], 1: [], 2: [], 3: []}

sim_chWalk_AirSim_paths = {0: [], 1: [], 2: [], 3: []}
sim_chWalk_AirSim_data = {0: [], 1: [], 2: [], 3: []}
sim_chWalk_AirSim_data_fft = {0: [], 1: [], 2: [], 3: []}

sim_gazebo_paths = {0: [], 1: [], 2: [], 3: []}
sim_gazebo_data = {0: [], 1: [], 2: [], 3: []}
sim_gazebo_data_fft = {0: [], 1: [], 2: [], 3: []}


sim = {
    # "baseline_results": sim_baseline_paths,
    "chNormal_results": sim_chNormal_paths,
    "chWalk_results": sim_chWalk_paths,
    "AirSim_results": sim_AirSim_paths,
    "chNormal_AirSim_results": sim_chNormal_AirSim_paths,
    "chWalk_AirSim_results": sim_chWalk_AirSim_paths,
    # "gazebo_results": sim_gazebo_paths
}

sim_data = {
    # "baseline_results": sim_baseline_data,
    "chNormal_results": sim_chNormal_data,
    "chWalk_results": sim_chWalk_data,
    "AirSim_results": sim_AirSim_data,
    "chNormal_AirSim_results": sim_chNormal_AirSim_data,
    "chWalk_AirSim_results": sim_chWalk_AirSim_data,
    # "gazebo_results": sim_gazebo_data
}

sim_data_fft = {
    # "baseline_results": sim_baseline_data_fft,
    "chNormal_results": sim_chNormal_data_fft,
    "chWalk_results": sim_chWalk_data_fft,
    "AirSim_results": sim_AirSim_data_fft,
    "chNormal_AirSim_results": sim_chNormal_AirSim_data_fft,
    "chWalk_AirSim_results": sim_chWalk_AirSim_data_fft,
    # "gazebo_results": sim_gazebo_data_fft
}

real = {
    0: [],
    1: [],
    2: [],
    3: []
}

real_data = {
    0: [],
    1: [],
    2: [],
    3: []
}

real_data_fft = {
    0: [],
    1: [],
    2: [],
    3: []
}

# First we need to generate sim and real data paths to get the data
base_dir = "../data2"
sim_path = os.path.join(base_dir, "Sim")
real_path = os.path.join(base_dir, "Real")


# Add sim paths based on sensor models
for folder in model_folders:
    model_path = os.path.join(sim_path, folder)
    # Add all the paths to the dictionary
    for i in range(0, 4):
        # generate a path that is model_path + scenario[i] + test_run (from 1 to 10)
        for j in range(1, 11):
            sim[folder][i].append(os.path.join(
                model_path, f"{scenario[i]}_{j}.csv"))

    print(f"Added {folder} paths to {sim[folder]}")

# Add all the paths to the dictionary
for i in range(0, 4):
    # generate a path that is model_path + scenario[i] + test_run (from 1 to 10)
    for j in range(1, 11):
        real[i].append(os.path.join(
            real_path, f"{scenario[i]}_{j}.csv"))

print(f"Added real paths to {real}")


# Go through sim and real and load data frames into sim_data and real_data
for folder in model_folders:
    for i in range(1, 4):
        for path in sim[folder][i]:
            # Process the data to make it on the same time scale
            # by linear interpolation
            se_time, gt_time, se_vel, gt_vel = extract_data(path)
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
            # Normalize by the larger mean
            # rmse_sim = rmse_sim / max(np.max(gt_vel), np.max(se_vel))
            # print(f"Max GT vel: {np.max(gt_vel)}")

            print(f"Normalized RMSE for {path}: {rmse_sim}")

            # Make new numpy array with time, gt_vel and se_vel
            sim_data[folder][i].append(rmse_sim)

            # ===================================
            # FFT
            # ===================================

            # Compute FFT of gt_vel and se_vel
            gt_fft = np.fft.fft(gt_vel)
            se_fft = np.fft.fft(se_vel)

            # Compute Spectral flatness
            gt_sf = np.exp(np.mean(np.log(np.abs(gt_fft) + 1e-10))) / \
                (np.mean(np.abs(gt_fft)))
            se_sf = np.exp(np.mean(np.log(np.abs(se_fft) + 1e-10))) / \
                (np.mean(np.abs(se_fft)))

            # Compute absoulte difference in the specral flatness
            sf_diff = np.abs(gt_sf - se_sf)

            sim_data_fft[folder][i].append(sf_diff)

            print(f"Spectral flatness difference for {path}: {sf_diff}")


for i in range(1, 4):
    for path in real[i]:
        # Process the data to make it on the same time scale
        # by linear interpolation
        se_time, gt_time, se_vel, gt_vel = extract_data(path)
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
        rmse_real = np.sqrt(np.square(np.subtract(gt_vel, se_vel)).mean())
        # Normalize by the larger mean
        # rmse_real = rmse_real / max(np.max(gt_vel), np.max(se_vel))

        # print(f"Max GT vel: {np.max(gt_vel)}")
        print(
            f"Normalized RMSE for {path}: {rmse_real}")

        real_data[i].append(rmse_real)

        # ===================================
        # FFT
        # ===================================
        # Compute FFT of gt_vel and se_vel
        gt_fft = np.fft.fft(gt_vel)
        se_fft = np.fft.fft(se_vel)
        # Compute Spectral flatness
        gt_sf = np.exp(np.mean(np.log(np.abs(gt_fft) + 1e-10))) / \
            (np.mean(np.abs(gt_fft)))
        se_sf = np.exp(np.mean(np.log(np.abs(se_fft) + 1e-10))) / \
            (np.mean(np.abs(se_fft)))
        # Compute absoulte difference in the specral flatness
        sf_diff = np.abs(gt_sf - se_sf)
        real_data_fft[i].append(sf_diff)
        print(f"Spectral flatness difference for {path}: {sf_diff}")

# Now calculate EPD for each model within each scenario
# with each scenario in real
# zs = np.zeros((len(models_list), len(scenario_list)))

df = pd.DataFrame(np.zeros((len(models_list), len(scenario_list))),
                  index=models_list, columns=scenario_list)
df_fft = pd.DataFrame(np.zeros((len(models_list), len(
    scenario_list))), index=models_list, columns=scenario_list)
# zs = np.zeros((len(models_list), 1))
df_combined = pd.DataFrame(
    np.zeros((len(models_list), 1)), index=models_list, columns=["EPD"])
df_combined_fft = pd.DataFrame(
    np.zeros((len(models_list), 1)), index=models_list, columns=["EPD"])
# Combined all scenarios for each model
sim_combined = {
    "baseline_results": [],
    "chNormal_results": [],
    "chWalk_results": [],
    "AirSim_results": [],
    "chNormal_AirSim_results": [],
    "chWalk_AirSim_results": []
}

sim_combined_fft = {
    "baseline_results": [],
    "chNormal_results": [],
    "chWalk_results": [],
    "AirSim_results": [],
    "chNormal_AirSim_results": [],
    "chWalk_AirSim_results": []
}
real_combined = []
real_combined_fft = []
for ct, model in enumerate(model_folders):
    for scenario_ in range(1, 4):
        # Combine all scenarios for each model
        sim_combined[model].append(sim_data[model][scenario_])
        sim_combined_fft[model].append(sim_data_fft[model][scenario_])
        if ct == 0:
            real_combined.append(real_data[scenario_])
            real_combined_fft.append(real_data_fft[scenario_])

        epd = wasserstein_distance(
            sim_data[model][scenario_], real_data[scenario_])

        epd_fft = wasserstein_distance(
            sim_data_fft[model][scenario_], real_data_fft[scenario_])

        df.at[model_names[model],
              scenario[scenario_]] = epd
        df_fft.at[model_names[model],
                  scenario[scenario_]] = epd_fft
        print(f"EPD for {model_names[model]} in {scenario[scenario_]}: {epd}")
        print(
            f"EFD Spectral Flatness for {model_names[model]} in {scenario[scenario_]}: {epd_fft}")

    # Combine
    sim_combined[model] = np.concatenate(sim_combined[model])
    sim_combined_fft[model] = np.concatenate(sim_combined_fft[model])
    print(f"Combined {model} data: {sim_combined[model]}")
    print(f"Combined real data: {np.concatenate(real_combined)}")
    epd = wasserstein_distance(
        sim_combined[model], np.concatenate(real_combined))

    epd_fft = wasserstein_distance(
        sim_combined_fft[model], np.concatenate(real_combined_fft))

    df_combined.at[model_names[model], "EPD"] = epd
    df_combined_fft.at[model_names[model], "EPD"] = epd_fft
    print(f"EPD for {model_names[model]}: {epd}")
    print(f"EFD Spectral Flatness for {model_names[model]}: {epd_fft}")

print(df)
print(df_fft)
print(df_combined)
print(df_combined_fft)

df_combined_final = pd.DataFrame()
df_combined_final["Model"] = list(model_names.values())
df_combined_final["W_1"] = df_combined["EPD"].values
df_combined_final["W_2"] = df_combined_fft["EPD"].values
df_combined_final["Mean"] = df_combined_final[["W_1", "W_2"]].mean(axis=1)
df_combined_final = df_combined_final.round(4)
df_combined_final.to_csv('output_combined_final.csv', index=True)

# Create a new data frame that takes the average of df and df_fft
df_average = pd.DataFrame(np.zeros((len(models_list), len(scenario_list))),
                          index=models_list, columns=scenario_list)
df_average = (df + df_fft) / 2.


# Arrange df_average based on column and print the rank from lowest to highest with values
df_average_rank = df_average.drop('rest', axis=1).rank(axis=0, ascending=True)
df_average_rank = df_average_rank.round(4)
# print(df_average_rank)
df_average_rank.to_csv('output_average_rank.csv', index=True)


df_average = df_average.round(4)
df_average.to_csv('output_average.csv', index=True)
df = df.round(4)
df.to_csv('output.csv', index=True)

df_fft = df_fft.round(4)
df_fft.to_csv('output_fft.csv', index=True)

df_combined = df_combined.round(4)
df_combined.to_csv('output_combined.csv', index=True)

df_combined_fft = df_combined_fft.round(4)
df_combined_fft.to_csv('output_combined_fft.csv', index=True)


plot_dir = "../plots_3"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


fig, ax = plt.subplots()
sns.kdeplot(np.concatenate(real_combined),
            label="Real", ax=ax, color="darkgray",  bw_adjust=1.)
for model in model_folders:
    # Plot sim_combined[model] and np.concatenate(real_combined) as distributions with kde
    sns.kdeplot(sim_combined[model], label=f"{model_names[model]}",
                ax=ax, color=color_pallette[model],  bw_adjust=1.)

    ax.set_xlabel("RMSE")
    ax.set_ylabel("Density")
    ax.set_xlim(-max(np.abs(ax.get_xlim())), max(np.abs(ax.get_xlim())))
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "combined_kde.png"), dpi=300)
plt.show()

# Plot real
legend_properties = {'size': 8}  # Font size of 12
fig, axs = plt.subplots(1, len(model_folders) + 1,
                        figsize=(10, 1.5), constrained_layout=True)
bin_width = 0.025
bins = np.arange(0, 0.8 + bin_width, bin_width)
axs[0].hist(np.concatenate(real_combined),
            bins=bins, color="darkgray", alpha=0.7)
# axs[0].set_xlabel("RMSE")
axs[0].set_title("Real", fontsize=10)
axs[0].set_ylabel("Count", fontsize=10)
axs[0].set_xlim(0, 0.8)
axs[0].set_ylim(0, 15)
# axs[0].legend(prop=legend_properties)
axs[0].set_xticks(np.arange(0, 0.8 + 0.2, 0.2))
axs[0].set_yticks(np.arange(0, 15 + 5, 5))
axs[0].tick_params(axis='both', which='both', labelsize=9, rotation=45)

for i, model in enumerate(model_folders):
    axs[i+1].hist(sim_combined[model], bins=bins, color=color_pallette[model],
                  alpha=0.7)
    axs[i+1].set_xlim(0, 0.8)
    axs[i+1].set_ylim(0, 15)
    axs[i+1].set_title(model_names[model], fontsize=10)
    axs[i+1].set_xticks(np.arange(0, 0.8 + 0.2, 0.2))
    axs[i+1].tick_params(axis='both', which='both', labelsize=9)
    axs[i+1].tick_params(axis='x', rotation=45)
    axs[i+1].tick_params(axis='x')
    axs[i+1].set_yticks([])
plt.tight_layout(pad=0.01)
plt.subplots_adjust(wspace=0.2)
plt.savefig(os.path.join(plot_dir, "hist_rmse.png"), dpi=300)
plt.show()

# FFT
legend_properties = {'size': 8}
bin_width = 0.00675
bins = np.arange(0, 0.2 + bin_width, bin_width)

fig, axs = plt.subplots(1, len(model_folders) + 1,
                        figsize=(10, 1.5), constrained_layout=True)

# Real data
axs[0].hist(np.concatenate(real_combined_fft), bins=bins,
            color="darkgray", alpha=0.7, label="Real")
axs[0].set_ylabel("Count", fontsize=10)
axs[0].set_title("Real", fontsize=10)
axs[0].set_yticks(np.arange(0, 25 + 5, 5))

# Model data (combined in a loop)
for i, model in enumerate(model_folders):
    axs[i + 1].hist(sim_combined_fft[model], bins=bins,
                    color=color_pallette[model], alpha=0.7)
    axs[i + 1].set_title(model_names[model], fontsize=10)
    axs[i+1].set_yticks([])

# Common formatting for all subplots
for ax in axs:
    ax.set_xlim(0, 0.2)
    ax.set_ylim(0, 25)
    ax.set_xticks(np.arange(0, 0.2 + 0.05, 0.05))
    ax.tick_params(axis='both', which='both', labelsize=9, rotation=45)

# Place legend outside
# plt.legend(prop=legend_properties, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout(pad=0.01)
plt.subplots_adjust(wspace=0.2)
plt.savefig(os.path.join(plot_dir, "hist_fft.png"), dpi=300)
plt.show()
