import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

slope_dict = {
    "1": 0.0,
    "2": 2.5,
    "3": 5.0,
    "4": 10.0,
    "5": 15.0,
    "6": 20.0,
    "7": 25.0,
}

# Edit this list to include only the depths you want to plot.
# PLOT_DEPTHS = [0.1, 0.2, 0.4, 0.6, 0.8]
PLOT_DEPTHS = [0.1, 0.5, 1.0]

EXPERIMENTAL_DATA = np.array(
    [
        [3.1504614500604013, 0.27322413552669644],
        [5.154396334609679, 5.237445584511807],
        [13.465437029246175, 10.026680600615933],
        [43.41343674026234, 14.961499624489019],
        [75.59749928174078, 20.23066622535026],
        [88.45956494491666, 25.09928778925685],
    ]
)

DEM_DATA = np.array(
    [
        [0.0, 0.04],
        [2.5, 0.06],
        [5.0, 0.105],
        [10.0, 0.21],
        [15.0, 0.415],
        [20.0, 0.62],
        [25.0, 0.72],
    ]
)

RHEOLOGY_GROUPS = {
    "both": ["MU_OF_I", "MCC"],
    "mu_i": ["MU_OF_I"],
    "mcc": ["MCC"],
}

plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.weight"] = "bold"


def depth_tag(depth: float) -> str:
    return f"{depth:.3f}"


def read_position_data(file_path: str) -> pd.DataFrame:
    print(f"Reading file: {file_path}")
    data = pd.read_csv(file_path, sep="\t")

    # Keep parity with existing scripts.
    data = data.iloc[1:].reset_index(drop=True)

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    column_mapping = {
        "time": "time",
        "x": "w_pos_x",
        "y": "w_pos_y",
        "z": "w_pos_z",
        "vx": "w_vel_x",
        "vy": "w_vel_y",
        "vz": "w_vel_z",
        "ax": "angvel_x",
        "ay": "angvel_y",
        "az": "angvel_z",
        "fx": "force_x",
        "fy": "force_y",
        "fz": "force_z",
        "tx": "torque_x",
        "ty": "torque_y",
        "tz": "torque_z",
    }

    return data.rename(columns=column_mapping)


def compute_slip(data: pd.DataFrame, wheel_radius_eff: float, calc_start_time: float) -> float:
    if data.empty:
        return np.nan

    start_idx = (data["time"] - calc_start_time).abs().idxmin()
    t_start = data.loc[start_idx, "time"]
    t_end = data.iloc[-1]["time"]
    dt = t_end - t_start
    if dt <= 0.0:
        return np.nan

    x_start = data.loc[start_idx, "w_pos_x"]
    x_end = data.iloc[-1]["w_pos_x"]
    avg_vel = (x_end - x_start) / dt

    avg_ang_vel = data["angvel_z"][start_idx:].mean()
    if abs(avg_ang_vel) < 1e-10:
        return np.nan

    return 1.0 - avg_vel / (avg_ang_vel * wheel_radius_eff)


def format_curve_label(rheology: str, depth: float, vary_ps: bool, ps: int, include_model_name: bool) -> str:
    depth_label = f"d={depth:.1f} m"
    if include_model_name:
        model_label = r"$\mu(I)$" if rheology == "MU_OF_I" else "MCC"
        depth_label = f"{model_label}, {depth_label}"
    if vary_ps:
        depth_label = f"{depth_label}, PS={ps}"
    return depth_label


def add_plot_styling(ax: plt.Axes) -> None:
    ax.set_xlabel("Slip Ratio", fontsize=18, fontweight="bold")
    ax.set_ylabel("Slope (deg)", fontsize=18, fontweight="bold")
    ax.set_yticks(
        [0, 2.5, 5, 10, 15, 20, 25],
        [r"$0^\circ$", r"$2.5^\circ$", r"$5^\circ$", r"$10^\circ$", r"$15^\circ$", r"$20^\circ$", r"$25^\circ$"],
    )
    ax.grid(False)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 30)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.legend(fontsize=12, loc="lower right", frameon=True, framealpha=0.9, ncol=2)


def save_plot(
    rheologies: list[str],
    slip_dict: dict,
    ps_values: list[int],
    depth_values: list[float],
    slopes: list[str],
    depth_colors: np.ndarray,
    rheology_styles: dict,
    rheology_markers: dict,
    experimental_data: np.ndarray,
    vary_ps: bool,
    output_filename: str,
    include_model_name: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for rheology in rheologies:
        for ps in ps_values:
            for depth_idx, depth in enumerate(depth_values):
                valid_slopes = [s for s in slopes if s in slip_dict[rheology][ps][depth]]
                if not valid_slopes:
                    continue

                slips = [slip_dict[rheology][ps][depth][s] for s in valid_slopes]
                slopes_numeric = [slope_dict[s] for s in valid_slopes]
                label = format_curve_label(rheology, depth, vary_ps, ps, include_model_name)

                ax.plot(
                    slips,
                    slopes_numeric,
                    linestyle=rheology_styles[rheology],
                    marker=rheology_markers[rheology],
                    color=depth_colors[depth_idx],
                    markersize=6,
                    linewidth=2.1,
                    label=label,
                )

    ax.plot(
        experimental_data[:, 0],
        experimental_data[:, 1],
        linestyle="--",
        marker="^",
        label="Experimental",
        color=(0.5, 0.5, 0.5),
        linewidth=2.1,
        markersize=7,
    )

    ax.plot(
        DEM_DATA[:, 1],
        DEM_DATA[:, 0],
        linestyle=(0, (1, 1)),
        marker="s",
        label="DEM",
        color="#1F5A91",
        linewidth=2.3,
        markersize=7,
    )

    add_plot_styling(ax)
    fig.tight_layout()
    fig.savefig(output_filename, dpi=600)
    print(f"Plot saved as: {output_filename}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot slip-slope curves for depth-variation sloped-wheel runs"
    )
    parser.add_argument(
        "--vary_ps",
        action="store_true",
        help="Plot multiple lines for different proximity-search frequencies",
    )
    parser.add_argument(
        "--rheology",
        type=str.lower,
        choices=list(RHEOLOGY_GROUPS.keys()),
        default="both",
        help="Select which rheology curves to plot: mu_i, mcc, or both (default: both)",
    )
    args = parser.parse_args()

    sns.set(style="ticks", context="talk")

    slopes = list(slope_dict.keys())
    depth_values = PLOT_DEPTHS
    if not depth_values:
        raise ValueError("PLOT_DEPTHS is empty. Add at least one depth to plot.")
    rheologies_to_plot = RHEOLOGY_GROUPS[args.rheology]
    spacing = "0.010"
    d0 = 1.3
    av = "0.10"
    var_time_step = True
    rad_plus_grouser = 0.225 + 0.03
    calc_start_time = 0.0

    default_ps = 1
    ps_values = [1, 2, 5, 10] if args.vary_ps else [default_ps]

    pre_pressure_scale = 1.5
    kappa = 0.2
    lambda_val = 0.8

    depth_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(depth_values)))
    rheology_styles = {
        "MU_OF_I": "-",
        "MCC": "--",
    }
    rheology_markers = {
        "MU_OF_I": "o",
        "MCC": "s",
    }

    slip_dict = {
        rheology: {ps: {d: {} for d in depth_values} for ps in ps_values}
        for rheology in rheologies_to_plot
    }
    data_dict = {
        rheology: {ps: {d: {} for d in depth_values} for ps in ps_values}
        for rheology in rheologies_to_plot
    }
    base_dir = "./DEMO_OUTPUT/FSI_SlopedSingleWheelTest"

    for rheology in rheologies_to_plot:
        for ps in ps_values:
            for depth in depth_values:
                dtag = depth_tag(depth)
                for slope in slopes:
                    if rheology == "MU_OF_I":
                        dir_path = (
                            f"{base_dir}/ps_{ps}_s_{spacing}_d0_{d0}_av_{av}_depth_{dtag}/{slope}"
                        )
                    else:
                        dir_path = (
                            f"{base_dir}/ps_{ps}_s_{spacing}_d0_{d0}_av_{av}"
                            f"_pre_pressure_scale_{pre_pressure_scale:.1f}"
                            f"_kappa_{kappa:.2f}_lambda_{lambda_val:.2f}"
                            f"_depth_{dtag}/{slope}"
                        )

                    file_name = "results_variable_time_step.txt" if var_time_step else "results_fixed_time_step.txt"
                    file_path = f"{dir_path}/{file_name}"

                    try:
                        data = read_position_data(file_path)
                        data_dict[rheology][ps][depth][slope] = data
                    except FileNotFoundError:
                        print(
                            f"Warning: missing file for {rheology}, ps={ps}, depth={depth}, slope={slope}. Skipping."
                        )

    for rheology in rheologies_to_plot:
        for ps in ps_values:
            for depth in depth_values:
                for slope in slopes:
                    if slope not in data_dict[rheology][ps][depth]:
                        continue

                    data = data_dict[rheology][ps][depth][slope]
                    slip = compute_slip(data, rad_plus_grouser, calc_start_time)
                    print(f"{rheology}, PS={ps}, depth={depth}, slope={slope_dict[slope]} deg: slip={slip}")

                    if not np.isnan(slip):
                        slip_dict[rheology][ps][depth][slope] = slip

    experimental_data = EXPERIMENTAL_DATA.copy()
    experimental_data[:, 0] /= 100.0

    os.makedirs("./paper_plots", exist_ok=True)
    if args.vary_ps:
        output_suffix = "_ps_vary"
    else:
        output_suffix = ""

    output_specs = []
    if args.rheology in {"both", "mcc"}:
        output_specs.append(
            {
                "rheologies": ["MCC"],
                "output_filename": (
                    f"./paper_plots/Viper_slip_vs_slope_depth_comparison_mcc_only{output_suffix}.png"
                ),
                "include_model_name": False,
            }
        )
    if args.rheology in {"both", "mu_i"}:
        output_specs.append(
            {
                "rheologies": ["MU_OF_I"],
                "output_filename": (
                    f"./paper_plots/Viper_slip_vs_slope_depth_comparison_mu_i_only{output_suffix}.png"
                ),
                "include_model_name": False,
            }
        )
    if args.rheology == "both":
        output_specs.append(
            {
                "rheologies": ["MU_OF_I", "MCC"],
                "output_filename": (
                    f"./paper_plots/Viper_slip_vs_slope_depth_comparison{output_suffix}.png"
                ),
                "include_model_name": True,
            }
        )

    for output_spec in output_specs:
        save_plot(
            rheologies=output_spec["rheologies"],
            slip_dict=slip_dict,
            ps_values=ps_values,
            depth_values=depth_values,
            slopes=slopes,
            depth_colors=depth_colors,
            rheology_styles=rheology_styles,
            rheology_markers=rheology_markers,
            experimental_data=experimental_data,
            vary_ps=args.vary_ps,
            output_filename=output_spec["output_filename"],
            include_model_name=output_spec["include_model_name"],
        )