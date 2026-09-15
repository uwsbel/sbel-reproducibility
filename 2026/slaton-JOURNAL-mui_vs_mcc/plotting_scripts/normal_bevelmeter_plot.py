#!/usr/bin/env python3
"""
Plot plate penetration vs time by loading specific DEMO_OUTPUT runs
with fully-qualified parameter folders (similar to the penetrometer
comparison helper).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Plot display options
SHOW_LEGEND = False
SHOW_GRID = False

# Set IEEE journal-style font sizes
IEEE_FONT_SIZES = {
    'label': 16,
    'title': 18,
    'tick': 14,
    'legend': 12,
    'linewidth': 2.5
}


RUN_CONFIGS = [
    {
        "label": r"$\mu(I)$",
        "path_components": [
            "FSI_NormalBevameter_GRC1_24.0cm_MU_OF_I_58.4cm",
            "maxPressure_30.00",
            # "plateDiameter_10.20",
            "plateDiameter_19.00",
            "youngsModulus_1e+06",
            "density_1670",
            "mu_s_0.6593",
            "mu_2_0.6593",
            "cohesion_0",
            "boundaryType_adami",
            "viscosityType_artificial_bilateral",
            "kernelType_wendland",
            "ps_1_s_0.002_d0_1.3_t_2e-05_av_0.2",
            "force_vs_time.txt",
        ],
    },
    {
        "label": fr"Pc = 20*P, M = 1.3, $\kappa$ = 0.00625, $\lambda_s$ = 0.025",
        "path_components": [
            "FSI_NormalBevameter_GRC1_24.0cm_MCC_58.4cm_pre_pressure_scale_20",
            "maxPressure_30.00",
            # "plateDiameter_10.20",
            "plateDiameter_19.00",
            "youngsModulus_1e+06",
            "density_1670",
            "mu_s_0.6593",
            "mu_2_0.6593",
            "cohesion_0",
            "boundaryType_adami",
            "viscosityType_artificial_bilateral",
            "kernelType_wendland",
            "ps_1_s_0.002_d0_1.3_t_2e-05_av_0.2",
            "force_vs_time.txt",
        ],
    },
    
]


def resolve_force_file(base_dir, path_components):
    """
    Join the configured path components relative to DEMO_OUTPUT and ensure the
    expected force_vs_time.txt exists.

    If the exact path does not exist, emit a warning and then walk the
    top-level folder as a fallback so minor parameter differences can still be
    picked up.
    """
    candidate = os.path.join(base_dir, *path_components)
    if os.path.isfile(candidate):
        return candidate

    # Warn when the exact requested path is missing and we are about to fall back.
    print(
        f"[WARN] Requested path not found, falling back to search:\n"
        f"       {candidate}"
    )

    top_level = path_components[0]
    terminal = path_components[-1]
    search_root = os.path.join(base_dir, top_level)
    if not os.path.isdir(search_root):
        raise FileNotFoundError(f"Missing folder: {search_root}")

    for dirpath, _, filenames in os.walk(search_root):
        if terminal in filenames:
            return os.path.join(dirpath, terminal)

    raise FileNotFoundError(
        f"Unable to locate {terminal} for config rooted at {top_level}"
    )


def load_penetration_data(file_path):
    """Load the penetration data CSV and validate the required columns."""
    df = pd.read_csv(file_path)
    required = {"Time", "penetration-depth"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{file_path} missing columns: {', '.join(sorted(missing))}")
    return df


def plot_penetration_vs_time():
    # Apply seaborn style and context
    sns.set_style("white")
    sns.set_context("talk")
    plt.rcParams.update({
        'font.size': IEEE_FONT_SIZES['tick'],
        'axes.labelsize': IEEE_FONT_SIZES['label'],
        'axes.titlesize': IEEE_FONT_SIZES['title'],
        'xtick.labelsize': IEEE_FONT_SIZES['tick'],
        'ytick.labelsize': IEEE_FONT_SIZES['tick'],
        'legend.fontsize': IEEE_FONT_SIZES['legend'],
        'figure.titlesize': IEEE_FONT_SIZES['title']
    })

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "DEMO_OUTPUT")

    series = []
    for config in RUN_CONFIGS:
        label = config["label"]
        try:
            force_file = resolve_force_file(base_dir, config["path_components"])
            df = load_penetration_data(force_file)
            series.append({"label": label, "data": df, "path": force_file})
            print(f"Loaded {label} from {force_file}")
        except Exception as exc:
            print(f"Skipping {label}: {exc}")

    if not series:
        print("No plate penetration data could be loaded. Nothing to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("tab10", n_colors=max(3, len(series)))

    # Pressure ramps from 0 to 30 kPa over 3 seconds (10 kPa/s)
    for idx, item in enumerate(series):
        data = item["data"]
        ax.plot(
            data["Time"] * 10,  # Convert time (s) to pressure (kPa): 10 kPa/s
            -data["penetration-depth"] * 100,  # Convert m to cm, offset by 0.1 cm
            label=item["label"],
            linewidth=IEEE_FONT_SIZES['linewidth'],
            color=palette[idx % len(palette)],
        )

    ax.set_xlabel(r"Pressure, kN/m$^2$", fontsize=IEEE_FONT_SIZES['label'])
    ax.set_ylabel("Penetration, cm", fontsize=IEEE_FONT_SIZES['label'])
    if SHOW_GRID:
        ax.grid(True, linestyle='--', alpha=0.7, linewidth=1.2)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 0.7)
    ax.invert_yaxis()  # 0 at top, 0.7 at bottom
    if SHOW_LEGEND:
        ax.legend(loc="lower left", fontsize=IEEE_FONT_SIZES['legend'], 
                  frameon=True, fancybox=True, shadow=True)
    ax.tick_params(axis='both', which='major', labelsize=IEEE_FONT_SIZES['tick'], 
                   width=1.5, length=5)
    plt.tight_layout()

    out_dir = os.path.join(script_dir, "plate_penetration_plots")
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, "penetration_vs_time.png")
    plt.savefig(output_file, dpi=600, bbox_inches="tight")
    print(f"Plot saved as {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_penetration_vs_time()