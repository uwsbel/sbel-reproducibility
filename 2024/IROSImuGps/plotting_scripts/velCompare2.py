import os
from typing import Union, List
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


"""
FOR COMMAND LINE ARGUMENTS ACCEPTED PLEASE USE python3 velCompare.py -h
"""

plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'font.family': 'sans-serif',
})

# ------------------------------------
# Useful mappings for args
# ------------------------------------
# Mode mapping to folder names
mode_folder = {0: "Sim", 1: "Real"}

# Model mapping to folder names for simulation mode
model_folders = {
    0: "baseline_results",
    1: "chNormal_results",
    2: "chWalk_results",
    3: "AirSim_results",
    4: "chNormal_AirSim_results",
    5: "chWalk_AirSim_results",
    6: "gazebo_results"
}


# Map the scenario type to file name
scenario = {
    0: "rest",
    1: "path",
    2: "circle",
    3: "sine"
}

# scenario = {
#     0: "standstill",
#     1: "line",
#     2: "circle",
#     3: "sine"
# }

# Map folders to plot model names
model_names = {
    "baseline_results": "Baseline",
    "chNormal_results": "Ch:N",
    "chWalk_results": "Ch:W",
    "AirSim_results": "AirSim",
    "chNormal_AirSim_results": "Ch:N+AirSim",
    "chWalk_AirSim_results": "Ch:W+AirSim",
    "gazebo_results": "Gazebo"
}

# Generate colors for each of the models
color_pallette = {
    "baseline_results": sns.color_palette("Reds", 8)[-1],
    "chNormal_results": sns.color_palette("Greens", 8)[-2],
    "chWalk_results": sns.color_palette("Oranges", 8)[-3],
    "AirSim_results": sns.color_palette("Purples", 8)[-4],
    "chNormal_AirSim_results": sns.color_palette("Blues", 8)[-5],
    "chWalk_AirSim_results": (0, 0, 0),
    "gazebo_results": sns.color_palette("YlOrBr", 8)[-1]
}


# Axis Limits based on scenario type
# Only Y
axis_limits = {
    "rest": [0, 0.8],
    "path": [0, 2.3],
    "circle": [0, 2.3],
    "sine": [0, 2.3]
}

# axis_limits = {
#     "standstill": [0, 0.8],
#     "line": [0, 2.3],
#     "circle": [0, 2.3],
#     "sine": [0, 2.3]
# }


# Base directory
base_dir = "../data2"

# ------------------------------------
# Data extraction utils
# ------------------------------------


def construct_path(args, mode_folder, model_folders, scenario):
    """
    Construct the path to the data file based on the command line arguments
    """

    # Start with the mode-specific folder
    path = os.path.join(base_dir, mode_folder[args.mode])

    # If simulation mode, add model folders to the path
    if args.mode == 0:
        # If specific models are chosen, iterate through each
        if args.models != [0, 1, 2, 3, 4, 5]:
            paths = [os.path.join(path, model_folders[model])
                     for model in args.models]
        else:
            # If no specific model is chosen (default case), use all models
            paths = [os.path.join(path, model_folder)
                     for model_folder in model_folders.values()]
    else:
        # For real mode, there are no model-specific folders
        paths = [path]

    # Append scenario and test number to each path for Sim mode, for Real mode it's not applicable
    if args.mode == 0:
        final_paths = [os.path.join(
            path, f"{scenario[args.scenario_type]}_{args.test_number}.csv") for path in paths],
    else:
        final_paths = [os.path.join(
            path, f"{scenario[args.scenario_type]}_{args.test_number}.csv") for path in paths]
    return final_paths


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

# ------------------------------------
# Data processing utils
# ------------------------------------


def plot(ax, time_se, se_vel, path, color=sns.color_palette("Reds", 8)[-1]):
    """
    Plot the given data
    """

    # Get the plot model name
    try:  # This will only work in sim where we have a model
        # From path extract the model
        model = path.split("/")[3]
        model_name = model_names[model]
        ax.plot(time_se, se_vel, label=model_name,
                linewidth=2, color=color)
    except:
        ax.plot(time_se, se_vel, label="State Estimation",
                linewidth=2, color=color)


def moving_average(data: Union[List[float], np.ndarray], window_size: int) -> np.ndarray:
    """
    Computes the moving average of the given list using a window of size n.
    """
    # Create a window (kernel) for convolution that represents the uniform weights of the moving average
    window = np.ones(int(window_size)) / float(window_size)

    # Use np.convolve to compute the moving average, 'valid' mode returns output only for complete windows
    return np.convolve(data, window, 'valid')


# ------------------------------------
# Data plotting utils
# ------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Command line argument parser for simulation or real scenario")

    # Sim or real option
    parser.add_argument("mode", type=int, choices=[
                        0, 1], help="Simulation mode (0) or Real mode (1)")

    # Scenario type
    parser.add_argument("scenario_type", type=int, choices=range(
        4), help="Scenario type: 0=rest, 1=straight line, 2=circle, 3=sine")

    # Test number
    parser.add_argument("test_number", type=int, help="Test number")

    # Optional model numbers for simulation mode
    parser.add_argument("--models", nargs="*", type=int, choices=range(6), default=[0, 1, 2, 3, 4, 5],
                        help="Model numbers to plot (default: all models)")

    parser.add_argument("--show", action="store_true",
                        help="Show the plot")

    args = parser.parse_args()

    # Check if sim mode is selected and specific models are not provided
    if args.mode == 0 and args.models == [0, 1, 2, 3, 4, 5]:
        print("Simulation mode selected. Plotting all models by default.")
    elif args.mode == 0:
        print(f"Simulation mode selected")
        print("Models selected are:")
        for model in args.models:
            print(f"{model_folders[model]}")
    else:
        print("Real mode selected. Model selection is ignored.")

    print(
        f"Scenario type: {scenario[args.scenario_type]}, Test number: {args.test_number}")

    # Construct paths of all files we need
    paths = construct_path(args, mode_folder, model_folders, scenario)

    # Make paths tuple of lists into a single list
    if args.mode == 0:
        paths = paths[0]

        # ------------------------------------
        # First go through all the files in paths and get the min gt, se time
        min_gt_time = 100000
        min_se_time = 100000

        for path in paths:
            se_time, gt_time, se_vel, gt_vel = extract_data(path)
            # Check if lists are empty
            if not se_time:
                print(f"Empty list for {path}")
                continue
            se_time = [x - se_time[0] for x in se_time]
            gt_time = [x - gt_time[0] for x in gt_time]

            if (gt_time[-1] < min_gt_time):
                min_gt_time = gt_time[-1]

            if (se_time[-1] < min_se_time):
                min_se_time = se_time[-1]

        min_time = min(min_gt_time, min_se_time)
        # ------------------------------------

    # ------------------------------------
    # Plot set up
    # ------------------------------------
    # get the figure and axis and assign a figure size
    figure, ax = plt.subplots(figsize=(10, 6))
    # Set the axis limit
    ax.set_ylim(axis_limits[scenario[args.scenario_type]])

    # Set yaxis ticks to 0.25 increments
    ax.set_yticks(np.arange(0, 2.3, 0.25))

    # ------------------------------------
    # Just take the first path and get the gt vel (supposed to be the same)
    # ------------------------------------
    if args.mode == 0:
        se_time, gt_time, se_vel, gt_vel = extract_data(paths[0])
        # Zero se_time and gt_time
        se_time = [x - se_time[0] for x in se_time]
        gt_time = [x - gt_time[0] for x in gt_time]

        # print(se_time[0], se_time[-1])
        # print(gt_time[0], gt_time[-1])
        # hz = 15
        # time = np.arange(0, min(gt_time[-1], se_time[-1]), 1/hz)
        # time = np.arange(0, min_time, 1/hz)
        # gt_vel = np.interp(time, gt_time, gt_vel)
        ax.plot(gt_time, gt_vel, label="Ground Truth",
                color="darkgray", linewidth=2, linestyle='--')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")

        # hz = 15
        # time = np.arange(0, min_time, 1/hz)
        # for path in paths:
        #     se_time, gt_time, se_vel, gt_vel = extract_data(path)
        #     se_time = [x - se_time[0] for x in se_time]
        #     gt_time = [x - gt_time[0] for x in gt_time]

        #     gt_vel = np.interp(time, gt_time, gt_vel)
        #     if path == paths[0]:
        #         mean_gt_vel = gt_vel
        #     else:
        #         mean_gt_vel += gt_vel
        # mean_gt_vel = mean_gt_vel / len(paths)
        # ax.plot(time, mean_gt_vel, label="Ground Truth",
        #         color="darkgray", linewidth=2, linestyle='--')

        # ax.set_xlabel("Time (s)")
        # ax.set_ylabel("Velocity (m/s)")
    else:
        se_time, gt_time, se_vel, gt_vel = extract_data(paths[0])
        # Zero se_time and gt_time
        se_time = [x - se_time[0] for x in se_time]
        gt_time = [x - gt_time[0] for x in gt_time]
        # hz = 15
        # min_time = min(gt_time[-1], se_time[-1])
        # time = np.arange(0, min_time, 1/hz)

        # gt_vel = np.interp(time, gt_time, gt_vel)
        ax.plot(gt_time, gt_vel, label="Ground Truth",
                color="darkgray", linewidth=2, linestyle='--')

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")

    for path in paths:
        print("File Paths are")
        print(path)
        se_time, gt_time, se_vel, gt_vel = extract_data(path)
        # Zero se_time and gt_time
        se_time = [x - se_time[0] for x in se_time]
        gt_time = [x - gt_time[0] for x in gt_time]

        print(f"Last GT time recorded: {gt_time[-1]}")
        print(f"Last SE time recorded: {se_time[-1]}")

        # Compute rough frequency of se_time and gt_time
        se_freq = 1 / np.mean(np.diff(se_time))
        gt_freq = 1 / np.mean(np.diff(gt_time))
        print(f"Rough SE Frequency: {se_freq} Hz\n"
              f"Rough GT Frequency: {gt_freq} Hz")

        # Do moving average of the se to smooth noise
        # window_size = 7
        # se_vel = moving_average(se_vel, window_size)
        # # Do same with se_time
        # se_time = moving_average(se_time, window_size)

        # Create a time array with frequncy 15 Hz uptil the least of gt_time and se_time
        # time = np.arange(0, min(gt_time[-1], se_time[-1]), 1/hz)
        # time = np.arange(0, min_time, 1/hz)

        # Interpolate all the data to the new time array
        # se_vel = np.interp(time, se_time, se_vel)

        # Limit the time arrays and the corresponding velocity arrays to min_time

        if (args.mode == 0):
            # based on the model get the color
            color = color_pallette[path.split("/")[3]]

            plot(ax, se_time, se_vel, path, color=color)
        else:
            plot(ax, se_time, se_vel, path)

    # From path extract Sim or Real
    mode = paths[0].split("/")[2]
    # From path extract scenario type
    if args.mode == 0:
        scenario_ = paths[0].split("/")[4].split("_")[0]
        test_number = paths[0].split("/")[4].split("_")[1].split(".")[0]
    else:
        scenario_ = paths[0].split("/")[3].split("_")[0]
        test_number = paths[0].split("/")[3].split("_")[1].split(".")[0]
    # From path extract test number

    # save the plot
    # create plots folder if it doesn't exist
    if not os.path.exists("../plots_3"):
        os.makedirs("../plots_3")
        # create mode folder if it doesn't exist
    if not os.path.exists(f"../plots_3/{mode}"):
        os.makedirs(f"../plots_3/{mode}")
    plt.legend()
    plt.tight_layout()
    if (len(args.models) == 1):
        plt.savefig(
            f"../plots_3/{mode}/{scenario_}_{test_number}_{args.models[0]}.png", dpi=300)
    else:
        plt.savefig(
            f"../plots_3/{mode}/{scenario_}_{test_number}.png", dpi=300
        )
    # Optional argument for show

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
