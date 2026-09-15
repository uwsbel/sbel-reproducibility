import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
from matplotlib.ticker import ScalarFormatter, FuncFormatter

slope_dict = {
    "1": r"$0^\circ$",
    "2": r"$2.5^\circ$",
    "3": r"$5^\circ$",
    "4": r"$10^\circ$",
    "5": r"$15^\circ$",
    "6": r"$20^\circ$",
    "7": r"$25^\circ$",
}

# Set the common rcParams for all plots
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['font.weight'] = 'bold'


def read_position_data(file_path):
    print(f"Reading file: {file_path}")
    # Read data with header row
    data = pd.read_csv(file_path, sep='\t')

    # Drop the initial row (row index 0)
    data = data.iloc[1:].reset_index(drop=True)

    # Convert columns to numeric (they may be read as strings due to scientific notation)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Rename columns to match our expected format
    column_mapping = {
        'time': 'time',
        'x': 'w_pos_x',
        'y': 'w_pos_y',
        'z': 'w_pos_z',
        'vx': 'w_vel_x',
        'vy': 'w_vel_y',
        'vz': 'w_vel_z',
        'ax': 'angvel_x',
        'ay': 'angvel_y',
        'az': 'angvel_z',
        'fx': 'force_x',
        'fy': 'force_y',
        'fz': 'force_z',
        'tx': 'torque_x',
        'ty': 'torque_y',
        'tz': 'torque_z'
    }

    data = data.rename(columns=column_mapping)
    return data


def scientific_notation_formatter(x, pos):
    return f'{x:.0e}'


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Plot Viper wheel data comparing MCC and MU_OF_I rheology models')
    parser.add_argument('--vary_ps', action='store_true',
                        help='Plot multiple lines for different frequency values')
    args = parser.parse_args()

    # Set the style and context for the plot
    sns.set(style="ticks", context="talk")

    slopes = ["1", "2", "3", "4", "5", "6", "7"]
    spacing = "0.010"
    d0 = 1.3
    av = "0.10"
    var_time_step = True
    rad_plus_grouser = 0.225 + 0.03
    calc_start_time = 0.0
    
    # Define ps values to use
    default_ps = 1
    ps_values = [1, 2, 5, 10] if args.vary_ps else [default_ps]

    # MCC parameters (matching the directory structure)
    pre_pressure_scale = 2.0
    kappa = 0.2
    lambda_val = 0.8

    # Create a figure with proper dimensions
    plt.figure(figsize=(6, 6))

    # Define colors for rheology models
    mu_of_i_color = (0.35, 0.6, 0.6)  # Matte teal for MU_OF_I
    mcc_color = (0.8, 0.2, 0.4)  # Crimson/magenta for MCC
    
    # Define color for experimental data
    exp_color = (0.5, 0.5, 0.5)  # Gray for experimental data

    # Create a nested dictionary to store data for each rheology model, ps and slope
    data_dict = {"MU_OF_I": {ps: {} for ps in ps_values}, 
                 "MCC": {ps: {} for ps in ps_values}}
    slip_dict = {"MU_OF_I": {ps: {} for ps in ps_values}, 
                 "MCC": {ps: {} for ps in ps_values}}

    # Base directory
    base_dir = "./DEMO_OUTPUT/FSI_SlopedSingleWheelTest"

    # Load data for each rheology model, ps and slope
    for rheology in ["MU_OF_I", "MCC"]:
        for ps in ps_values:
            for slope in slopes:
                # Construct directory path based on rheology model
                if rheology == "MU_OF_I":
                    # MU_OF_I: no MCC parameters in directory name
                    dir_path = f'{base_dir}/ps_{ps}_s_{spacing}_d0_{d0}_av_{av}/{slope}'
                else:
                    # MCC: includes MCC parameters in directory name
                    dir_path = f'{base_dir}/ps_{ps}_s_{spacing}_d0_{d0}_av_{av}_pre_pressure_scale_{pre_pressure_scale}_kappa_{kappa:.2f}_lambda_{lambda_val:.2f}/{slope}'
                
                if var_time_step:
                    file_path = f'{dir_path}/results_variable_time_step.txt'
                else:
                    file_path = f'{dir_path}/results_fixed_time_step.txt'
                try:
                    data = read_position_data(file_path)
                    data_dict[rheology][ps][slope] = data
                except FileNotFoundError:
                    print(f"Warning: File not found for {rheology}, ps={ps}, slope={slope}. Skipping.")
                    continue

    # Calculate slip for each rheology model, ps and slope
    for rheology in ["MU_OF_I", "MCC"]:
        for ps in ps_values:
            for slope in slopes:
                if slope not in data_dict[rheology][ps]:
                    continue

                data = data_dict[rheology][ps][slope]

                # Find the closest time value to calc_start_time
                closest_start_idx = (data['time'] - calc_start_time).abs().idxmin()
                pos_at_start = data.loc[closest_start_idx, 'w_pos_x']

                # Now get the pos at the last time step
                pos_at_last_step = data.iloc[-1]['w_pos_x']
                # Then the averaged velocity is
                avg_vel = (pos_at_last_step - pos_at_start) / \
                    (data.iloc[-1]['time'] - data.loc[closest_start_idx, 'time'])

                # For this duration, compute the average angular velocity in z of the wheel
                # Compute average angular velocity from start time to last time step
                avg_ang_vel = data['angvel_z'][closest_start_idx:].mean()
                # ang_vel_z = data['angvel_z']
                # avg_ang_vel = ang_vel_z.mean()
                print(f"{rheology}, PS={ps}, Slope={slope}: Total time: {data.iloc[-1]['time'] - data.loc[closest_start_idx, 'time']}")
                print(f"{rheology}, PS={ps}, Slope={slope}: Average Velocity = {avg_vel}")

                print(f"{rheology}, PS={ps}, Slope={slope}: Average Angular Velocity = {avg_ang_vel}")

                # Compute the slip
                slip = 1 - avg_vel / (avg_ang_vel * rad_plus_grouser)
                print(f"{rheology}, PS={ps}, Slope={slope}: Slip = {slip}")
                slip_dict[rheology][ps][slope] = slip

    # Plot slip vs slope for each rheology model
    for rheology in ["MU_OF_I", "MCC"]:
        for ps in ps_values:
            if not slip_dict[rheology][ps]:  # Skip if no data for this ps
                continue

            # Get slopes and slips for this rheology and ps
            valid_slopes = [s for s in slopes if s in slip_dict[rheology][ps]]
            slopes_numeric = [float(slope_dict[slope].strip('$^\\circ$'))
                              for slope in valid_slopes]
            slips = [slip_dict[rheology][ps][slope] for slope in valid_slopes]

            # Choose color based on rheology model
            color = mu_of_i_color if rheology == "MU_OF_I" else mcc_color
            
            # Create label
            # Display name for legend
            display_name = r'$\mu(I)$' if rheology == "MU_OF_I" else rheology
            if args.vary_ps:
                label = f'{display_name} (PS={ps})'
            else:
                label = display_name

            # Plot with different marker and line styles for each rheology model
            linestyle = '-' if rheology == "MU_OF_I" else '--'
            marker = 'o' if rheology == "MU_OF_I" else 's'
            plt.plot(slips, slopes_numeric, linestyle=linestyle, marker=marker, 
                    color=color, markersize=8, label=label, linewidth=2)

    # Use the original experimental data
    experimental2 = np.array([
        [3.1504614500604013, 0.27322413552669644],
        [5.154396334609679, 5.237445584511807],
        [13.465437029246175, 10.026680600615933],
        [43.41343674026234, 14.961499624489019],
        [75.59749928174078, 20.23066622535026],
        [88.45956494491666, 25.09928778925685]
    ])

    dem_data = np.array([
        [0.0, 0.04],
        [2.5, 0.06],
        [5.0, 0.105],
        [10.0, 0.21],
        [15.0, 0.415],
        [20.0, 0.62],
        [25.0, 0.72],
    ])

    # Convert slip percentages to decimal (0-1 range)
    experimental2[:, 0] /= 100

    # Plot the experimental data
    plt.plot(experimental2[:, 0], experimental2[:, 1], linestyle='--',
             marker='^', label='Experimental', color=exp_color, linewidth=2, markersize=10)
             
    # Plot the DEM data
    plt.plot(dem_data[:, 1], dem_data[:, 0], linestyle='-.', 
             marker='s', label='DEM', color='#4A90D9', linewidth=2, markersize=8)

    # Set labels and title with consistent font styling
    plt.xlabel('Slip Ratio', fontsize=14, fontweight='bold')
    plt.ylabel('Slope (deg)', fontsize=14, fontweight='bold')
    
    # Set y-ticks using slope values
    plt.yticks(ticks=[0, 2.5, 5, 10, 15, 20, 25],
               labels=[r'$0^\circ$', r'$2.5^\circ$', r'$5^\circ$', r'$10^\circ$',
                       r'$15^\circ$', r'$20^\circ$', r'$25^\circ$'])

    # Remove grid to match cone_penetration_plotting style
    plt.grid(False)
    plt.xlim(0, 1.0)
    plt.ylim(0, 30)

    # Increase tick label size
    plt.tick_params(axis='both', which='major', labelsize=12)

    # Add legend with better positioning and formatting
    plt.legend(fontsize=7, loc='lower right',
              frameon=True, framealpha=0.9)

    # Apply tight layout for better formatting
    plt.tight_layout()

    # Save the plot with a descriptive filename
    os.makedirs("./paper_plots", exist_ok=True)
    
    if args.vary_ps:
        output_filename = f"./paper_plots/Viper_slip_vs_slope_rheology_comparison_ps_vary.png"
    else:
        output_filename = f"./paper_plots/Viper_slip_vs_slope_rheology_comparison_ps{default_ps}_s{spacing}_d0{d0}_av{av}.png"
    
    plt.savefig(output_filename, dpi=600)
    print(f"Plot saved as: {output_filename}")

    # Show the plot
    plt.show()