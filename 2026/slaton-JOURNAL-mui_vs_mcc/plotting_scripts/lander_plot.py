#!/usr/bin/env python3
"""Analyze the lunar lander CRM parameter sweep and generate paper-ready plots.

This script parses the MCC sweep stored under DEMO_OUTPUT_NREL_LANDER, computes
run-level sinkage metrics from the recorded lander body trajectory, and writes a
compact set of figures geared toward a rheology-comparison paper:

1. Time histories grouped by the three swept MCC controls:
   - pre_pressure_scale (OCR proxy)
   - kappa
   - lambda / kappa
2. Peak and final penetration heat maps relative to the MU_OF_I baseline.
3. Main-effect summaries for the three MCC controls.

Important caveat for this existing sweep:
The current demo regenerates a random height map and rock field for every run,
but the output folder does not record the terrain seed or the local surface
height under each footpad. Absolute penetration is therefore reported relative to
the nominal flat reference surface z = TERRAIN_HEIGHT_MAX. The exact body
settlement is also stored in the summary table and should be treated as the
cleanest cross-run metric.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MCC_PATTERN = re.compile(
    r"pre_pressure_scale_(?P<pre_pressure_scale>[0-9.]+)_kappa_(?P<kappa>[0-9.]+)_lambda_(?P<lambda_value>[0-9.]+)"
)


@dataclass(frozen=True)
class GeometryConfig:
    cylinder_length: float = 4.0
    leg_length: float = 1.5
    leg_angle_deg: float = 30.0
    footpad_height: float = 0.1
    footpad_offset: float = 0.02
    ground_clearance: float = 0.05
    reference_surface_z: float = 0.3

    @property
    def body_to_lowest_offset(self) -> float:
        leg_vertical = self.leg_length * math.cos(math.radians(self.leg_angle_deg))
        footpad_extension = self.footpad_height / 2.0 + self.footpad_offset
        return self.cylinder_length / 2.0 + leg_vertical + footpad_extension


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("build_new/bin/DEMO_OUTPUT_NREL_LANDER"),
        help="Directory containing ROBOT_Lander_CRM_* output folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for figures and summary tables. Defaults to <input-dir>/analysis.",
    )
    parser.add_argument(
        "--time-round-decimals",
        type=int,
        default=6,
        help="Decimal places used when grouping trajectory samples by time.",
    )
    parser.add_argument(
        "--metric",
        choices=("nominal_penetration_m", "settlement_m"),
        default="nominal_penetration_m",
        help="Primary metric used in the generated plots.",
    )
    parser.add_argument(
        "--max-peak-penetration",
        type=float,
        default=None,
        help="Exclude runs whose peak nominal penetration exceeds this threshold [m].",
    )
    return parser


def load_single_run(csv_path: Path, geometry: GeometryConfig) -> tuple[pd.DataFrame, dict]:
    run_name = csv_path.parent.name
    is_mui = "MU_OF_I" in run_name

    if is_mui:
        params = {
            "rheology_model": "MU_OF_I",
            "pre_pressure_scale": np.nan,
            "kappa": np.nan,
            "lambda_value": np.nan,
            "lambda_over_kappa": np.nan,
        }
    else:
        match = MCC_PATTERN.search(run_name)
        if not match:
            raise ValueError(f"Could not parse MCC parameters from '{run_name}'")
        pre_pressure_scale = float(match.group("pre_pressure_scale"))
        kappa = float(match.group("kappa"))
        lambda_value = float(match.group("lambda_value"))
        lambda_over_kappa = round(lambda_value / kappa, 6)
        params = {
            "rheology_model": "MCC",
            "pre_pressure_scale": pre_pressure_scale,
            "kappa": kappa,
            "lambda_value": lambda_value,
            "lambda_over_kappa": lambda_over_kappa,
        }

    df = pd.read_csv(csv_path).copy()
    if df.empty:
        raise ValueError(f"Empty trajectory file: {csv_path}")

    initial_pz = float(df["pz"].iloc[0])
    df["run_name"] = run_name
    df["csv_path"] = str(csv_path)
    df["time_s"] = df["time"].round(6)
    df["lowest_point_z_m"] = df["pz"] - geometry.body_to_lowest_offset
    df["settlement_m"] = initial_pz - df["pz"]
    df["nominal_penetration_m"] = geometry.reference_surface_z - df["lowest_point_z_m"]
    df["reference_contact_m"] = df["nominal_penetration_m"].clip(lower=0.0)
    for key, value in params.items():
        df[key] = value

    max_idx = int(df["settlement_m"].idxmax())
    max_row = df.loc[max_idx]
    contact_mask = df["nominal_penetration_m"] >= 0.0
    first_contact_time = float(df.loc[contact_mask, "time_s"].iloc[0]) if contact_mask.any() else np.nan

    summary = {
        "run_name": run_name,
        "csv_path": str(csv_path),
        **params,
        "num_samples": int(len(df)),
        "t_end_s": float(df["time_s"].iloc[-1]),
        "initial_pz_m": initial_pz,
        "initial_lowest_point_z_m": float(df["lowest_point_z_m"].iloc[0]),
        "peak_settlement_m": float(df["settlement_m"].max()),
        "final_settlement_m": float(df["settlement_m"].iloc[-1]),
        "peak_nominal_penetration_m": float(df["nominal_penetration_m"].max()),
        "final_nominal_penetration_m": float(df["nominal_penetration_m"].iloc[-1]),
        "rebound_after_peak_m": float(df["settlement_m"].max() - df["settlement_m"].iloc[-1]),
        "time_to_peak_s": float(max_row["time_s"]),
        "time_to_reference_contact_s": first_contact_time,
        "peak_vz_m_per_s": float(df["vz"].abs().max()),
    }

    return df, summary


def load_dataset(input_dir: Path, geometry: GeometryConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_paths = sorted(input_dir.glob("ROBOT_Lander_CRM*/lander_body_data.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No lander_body_data.csv files found under {input_dir}")

    trajectories: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for csv_path in csv_paths:
        trajectory, summary = load_single_run(csv_path, geometry)
        trajectories.append(trajectory)
        summaries.append(summary)

    trajectories_df = pd.concat(trajectories, ignore_index=True)
    summaries_df = pd.DataFrame(summaries).sort_values(
        ["rheology_model", "pre_pressure_scale", "kappa", "lambda_value"], na_position="last"
    )
    return trajectories_df, summaries_df


def filter_dataset(
    trajectories_df: pd.DataFrame, summaries_df: pd.DataFrame, max_peak_penetration: float | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_peak_penetration is None:
        return trajectories_df, summaries_df

    keep_runs = summaries_df.loc[
        summaries_df["peak_nominal_penetration_m"] <= max_peak_penetration, "run_name"
    ].tolist()
    filtered_summaries = summaries_df[summaries_df["run_name"].isin(keep_runs)].copy()
    filtered_trajectories = trajectories_df[trajectories_df["run_name"].isin(keep_runs)].copy()

    return filtered_trajectories, filtered_summaries


def format_group_label(group_column: str, value: float) -> str:
    if group_column == "pre_pressure_scale":
        return rf"$\mathrm{{OCR}} = {value:g}$"
    if group_column == "kappa":
        return rf"$\kappa = {value:g}$"
    if group_column == "lambda_over_kappa":
        if abs(value - round(value)) < 1e-9:
            return rf"$\lambda/\kappa = {int(round(value))}$"
        return rf"$\lambda/\kappa = {value:g}$"
    return f"{group_column} = {value:g}"


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 16,
            "axes.titlesize": 15,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )


def compute_group_profiles(
    trajectories_df: pd.DataFrame, group_column: str, metric: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mcc = trajectories_df[trajectories_df["rheology_model"] == "MCC"].copy()
    grouped = (
        mcc.groupby([group_column, "time_s"])[metric]
        .agg(mean="mean", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        .reset_index()
    )
    overall = (
        mcc.groupby("time_s")[metric]
        .agg(overall_min="min", overall_max="max")
        .reset_index()
        .sort_values("time_s")
    )
    return grouped, overall


def plot_profile_panel(
    ax: plt.Axes,
    trajectories_df: pd.DataFrame,
    mui_df: pd.DataFrame,
    group_column: str,
    metric: str,
    palette_name: str,
) -> None:
    grouped, overall = compute_group_profiles(trajectories_df, group_column, metric)
    values = sorted(grouped[group_column].dropna().unique())
    colors = plt.get_cmap(palette_name)(np.linspace(0.15, 0.9, len(values)))

    ax.fill_between(
        overall["time_s"],
        overall["overall_min"],
        overall["overall_max"],
        color="0.88",
        linewidth=0.0,
        label="MCC sweep span",
    )

    for color, value in zip(colors, values):
        subset = grouped[grouped[group_column] == value].sort_values("time_s")
        ax.plot(subset["time_s"], subset["mean"], color=color, linewidth=2.0, label=format_group_label(group_column, value))
        ax.fill_between(subset["time_s"], subset["q25"], subset["q75"], color=color, alpha=0.15, linewidth=0.0)

    ax.plot(
        mui_df["time_s"],
        mui_df[metric],
        color="black",
        linestyle="--",
        linewidth=2.3,
        label=r"$\mu(I)$ baseline",
    )
    ax.set_xlabel("Time [s]")


def plot_grouped_profiles(
    trajectories_df: pd.DataFrame,
    summaries_df: pd.DataFrame,
    output_path: Path,
    metric: str,
) -> None:
    mui_df = trajectories_df[trajectories_df["rheology_model"] == "MU_OF_I"].copy().sort_values("time_s")
    fig, axes = plt.subplots(3, 1, figsize=(9, 14), sharex=True, sharey=True, constrained_layout=True)

    plot_profile_panel(
        axes[0],
        trajectories_df,
        mui_df,
        "pre_pressure_scale",
        metric,
        "viridis",
    )
    plot_profile_panel(
        axes[1],
        trajectories_df,
        mui_df,
        "kappa",
        metric,
        "plasma",
    )
    plot_profile_panel(
        axes[2],
        trajectories_df,
        mui_df,
        "lambda_over_kappa",
        metric,
        "cividis",
    )

    ylabel = "Nominal penetration depth [m]" if metric == "nominal_penetration_m" else "Body settlement [m]"
    for ax in axes:
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0.0)
        ax.legend(loc="lower right", frameon=True)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_heatmap_matrix(
    summaries_df: pd.DataFrame,
    metric_column: str,
    ratio_value: float,
    pps_values: Iterable[float],
    kappa_values: Iterable[float],
    mui_metric_value: float,
) -> np.ndarray:
    subset = summaries_df[
        (summaries_df["rheology_model"] == "MCC")
        & np.isclose(summaries_df["lambda_over_kappa"], ratio_value, atol=1e-6)
    ]
    pivot = subset.pivot(index="pre_pressure_scale", columns="kappa", values=metric_column)
    pivot = pivot.reindex(index=list(pps_values), columns=list(kappa_values))
    return pivot.to_numpy() - mui_metric_value


def plot_delta_heatmaps(summaries_df: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("peak_nominal_penetration_m", r"Peak nominal penetration minus $\mu(I)$ [m]"),
        ("final_nominal_penetration_m", r"Final nominal penetration minus $\mu(I)$ [m]"),
    ]
    ratios = sorted(
        summaries_df.loc[summaries_df["rheology_model"] == "MCC", "lambda_over_kappa"].dropna().unique()
    )
    pps_values = sorted(summaries_df.loc[summaries_df["rheology_model"] == "MCC", "pre_pressure_scale"].dropna().unique())
    kappa_values = sorted(summaries_df.loc[summaries_df["rheology_model"] == "MCC", "kappa"].dropna().unique())
    mui_summary = summaries_df[summaries_df["rheology_model"] == "MU_OF_I"].iloc[0]

    matrices: list[np.ndarray] = []
    for metric_column, _ in metrics:
        mui_value = float(mui_summary[metric_column])
        for ratio in ratios:
            matrices.append(build_heatmap_matrix(summaries_df, metric_column, ratio, pps_values, kappa_values, mui_value))

    max_abs = max(float(np.nanmax(np.abs(matrix))) for matrix in matrices)
    fig, axes = plt.subplots(2, len(ratios), figsize=(15.5, 7.0), constrained_layout=True)

    for row_idx, (metric_column, row_title) in enumerate(metrics):
        mui_value = float(mui_summary[metric_column])
        for col_idx, ratio in enumerate(ratios):
            ax = axes[row_idx, col_idx]
            matrix = build_heatmap_matrix(summaries_df, metric_column, ratio, pps_values, kappa_values, mui_value)
            image = ax.imshow(matrix[::-1], cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto")
            ax.set_xticks(range(len(kappa_values)))
            ax.set_xticklabels([f"{value:g}" for value in kappa_values], rotation=45, ha="right")
            ax.set_yticks(range(len(pps_values)))
            ax.set_yticklabels([f"{value:g}" for value in reversed(pps_values)])
            ax.text(
                0.03,
                0.97,
                format_group_label("lambda_over_kappa", ratio),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.7},
            )
            if col_idx == 0:
                ax.set_ylabel(row_title + "\n" + r"$\mathrm{OCR}$")
            else:
                ax.set_ylabel(r"$\mathrm{OCR}$")
            ax.set_xlabel(r"$\kappa$")

    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.95, pad=0.015)
    cbar.set_label(r"MCC minus $\mu(I)$ [m]", fontsize=13)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_main_effects(summaries_df: pd.DataFrame, output_path: Path) -> None:
    mcc = summaries_df[summaries_df["rheology_model"] == "MCC"].copy()
    mui_summary = summaries_df[summaries_df["rheology_model"] == "MU_OF_I"].iloc[0]
    effect_specs = [
        ("pre_pressure_scale", r"$\mathrm{OCR}$"),
        ("kappa", r"$\kappa$"),
        ("lambda_over_kappa", r"$\lambda/\kappa$"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 13.5), constrained_layout=True)
    for ax, (column, xlabel) in zip(axes, effect_specs):
        grouped = (
            mcc.groupby(column)[["final_nominal_penetration_m", "peak_nominal_penetration_m"]]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        x = grouped[column].to_numpy(dtype=float)
        final_mean = grouped[("final_nominal_penetration_m", "mean")].to_numpy(dtype=float)
        final_min = grouped[("final_nominal_penetration_m", "min")].to_numpy(dtype=float)
        final_max = grouped[("final_nominal_penetration_m", "max")].to_numpy(dtype=float)
        peak_mean = grouped[("peak_nominal_penetration_m", "mean")].to_numpy(dtype=float)

        ax.fill_between(x, final_min, final_max, color="#cfe0f2", linewidth=0.0, alpha=0.9, label="Final MCC range")
        ax.plot(x, final_mean, color="#1f77b4", linewidth=2.5, marker="o", label="Final MCC mean")
        ax.plot(x, peak_mean, color="#d62728", linewidth=2.0, marker="s", label="Peak MCC mean")
        ax.axhline(
            float(mui_summary["final_nominal_penetration_m"]),
            color="black",
            linestyle="--",
            linewidth=1.8,
            label=r"$\mu(I)$ final",
        )
        ax.axhline(
            float(mui_summary["peak_nominal_penetration_m"]),
            color="0.35",
            linestyle=":",
            linewidth=1.8,
            label=r"$\mu(I)$ peak",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Nominal penetration depth [m]")
        if column == "kappa":
            ax.set_xscale("log")
            ax.set_xticks(x)
            ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:g}"))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.02))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_caveats(
    output_path: Path, summaries_df: pd.DataFrame, geometry: GeometryConfig, max_peak_penetration: float | None
) -> None:
    mcc = summaries_df[summaries_df["rheology_model"] == "MCC"]
    mui = summaries_df[summaries_df["rheology_model"] == "MU_OF_I"].iloc[0]

    lines = [
        "# Lander CRM sweep analysis notes",
        "",
        f"- MCC runs found: {len(mcc)}",
        "- MU_OF_I baselines found: 1",
        f"- Body-to-lowest-point offset used: {geometry.body_to_lowest_offset:.6f} m",
        f"- Ground clearance used: {geometry.ground_clearance:.3f} m",
        f"- Reference surface used for nominal penetration: z = {geometry.reference_surface_z:.3f} m",
        f"- MU_OF_I final settlement: {mui['final_settlement_m']:.6f} m",
        f"- MU_OF_I final nominal penetration: {mui['final_nominal_penetration_m']:.6f} m",
    ]
    if max_peak_penetration is not None:
        lines.extend(
            [
                f"- Applied run filter: peak nominal penetration <= {max_peak_penetration:.3f} m",
                f"- MCC runs remaining after filter: {len(mcc)}",
            ]
        )
    lines.extend(
        [
            "",
            "## Important caveat",
            "",
            "- Each run appears to regenerate a random terrain height map and rock layout.",
            "- The current output does not record the terrain seed or the local initial surface height under the footpads.",
            "- Absolute penetration is therefore measured relative to the nominal top surface, not the exact local terrain.",
            "- Cross-run parameter trends are still visible, but any publication claim about absolute sinkage should preferably be confirmed with a fixed-seed rerun.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    args = build_argument_parser().parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or (input_dir / "analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = GeometryConfig()
    set_plot_style()

    trajectories_df, summaries_df = load_dataset(input_dir, geometry)
    trajectories_df, summaries_df = filter_dataset(trajectories_df, summaries_df, args.max_peak_penetration)
    trajectories_df["time_s"] = trajectories_df["time_s"].round(args.time_round_decimals)

    if summaries_df.empty:
        raise ValueError("No runs remain after applying the requested filter")
    if (summaries_df["rheology_model"] == "MU_OF_I").sum() == 0:
        raise ValueError("The requested filter removed the MU_OF_I baseline; cannot generate comparison plots")

    summaries_path = output_dir / "lander_sweep_summary.csv"
    trajectories_path = output_dir / "lander_sweep_trajectories.csv"
    grouped_profiles_path = output_dir / "lander_grouped_profiles.png"
    heatmaps_path = output_dir / "lander_penetration_delta_heatmaps.png"
    main_effects_path = output_dir / "lander_main_effects.png"
    notes_path = output_dir / "analysis_notes.md"

    summaries_df.to_csv(summaries_path, index=False)
    trajectories_df.to_csv(trajectories_path, index=False)

    plot_grouped_profiles(trajectories_df, summaries_df, grouped_profiles_path, args.metric)
    plot_delta_heatmaps(summaries_df, heatmaps_path)
    plot_main_effects(summaries_df, main_effects_path)
    write_caveats(notes_path, summaries_df, geometry, args.max_peak_penetration)

    mcc = summaries_df[summaries_df["rheology_model"] == "MCC"].copy()
    mui = summaries_df[summaries_df["rheology_model"] == "MU_OF_I"].iloc[0]

    print(f"Loaded {len(mcc)} MCC runs and 1 MU_OF_I baseline from {input_dir}")
    print(f"Summary table: {summaries_path}")
    print(f"Trajectory table: {trajectories_path}")
    print(f"Grouped profiles: {grouped_profiles_path}")
    print(f"Delta heat maps: {heatmaps_path}")
    print(f"Main effects: {main_effects_path}")
    print(f"Notes: {notes_path}")
    if args.max_peak_penetration is not None:
        print(f"Applied peak nominal penetration filter <= {args.max_peak_penetration:.5f} m")
    print("")
    print("MU_OF_I reference:")
    print(f"  final settlement          = {mui['final_settlement_m']:.5f} m")
    print(f"  final nominal penetration = {mui['final_nominal_penetration_m']:.5f} m")
    print("")
    print("MCC sweep extremes:")
    print(f"  final nominal penetration min = {mcc['final_nominal_penetration_m'].min():.5f} m")
    print(f"  final nominal penetration max = {mcc['final_nominal_penetration_m'].max():.5f} m")
    print(f"  peak nominal penetration  min = {mcc['peak_nominal_penetration_m'].min():.5f} m")
    print(f"  peak nominal penetration  max = {mcc['peak_nominal_penetration_m'].max():.5f} m")


if __name__ == "__main__":
    main()