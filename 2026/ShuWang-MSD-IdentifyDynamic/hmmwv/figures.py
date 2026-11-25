"""Redraw the two HMMWV result figures from the files of this folder.

make_figures(case_dir, out_dir) writes HMMWV_test_predictions.pdf and HMMWV_parameter_recovery.pdf.
prediction_arrays and recovery_arrays return the plotted arrays.
"""
import json
from pathlib import Path

import numpy as np

PARAMETERS = ("mu", "M", "COMx", "COMz", "Ixx", "Iyy", "Izz")
SETTINGS = ("different_geometry", "same_geometry")
OUTPUT_INTERVAL_S = 0.002
SKIPPED_SAMPLES = 100
FAMILIES = [("lane_change_brake", "Lane change, brake"), ("longitudinal_chirp", "Longitudinal chirp"),
            ("multisine_slalom", "Multisine slalom"), ("sine_dwell_coast", "Sine dwell, coast"),
            ("offset_brake_slalom", "Offset-brake slalom"), ("late_sweep_release", "Late sweep, release")]
CHANNELS = [("position_y", 1.0, "Position $y$ (m)"),
            ("front_left_wheel_speed", 1.0, "Front-left wheel\nspeed (rad/s)"),
            ("front_left_upright_y", 1000.0, "Front-left upright\n$y$ rel. chassis (mm)"),
            ("yaw_rate", 1.0, "Yaw rate (rad/s)")]
TRUE = dict(color="#c62828", lw=1.2, ls="-", label=r"Noise-free teacher output ($\theta^\star$, $g^\star$)")
PRED = dict(color="#1669a7", lw=1.2, ls="--", label=r"Prediction, selected endpoint ($\hat\theta$, $\hat g$)")
TITLE = {"different_geometry": "(a) Different geometry", "same_geometry": "(b) Same geometry"}
SYMBOL = {"mu": r"$\mu$", "M": r"$M$", "COMx": r"$\mathrm{COMx}$", "COMz": r"$\mathrm{COMz}$",
          "Ixx": r"$I_{xx}$", "Iyy": r"$I_{yy}$", "Izz": r"$I_{zz}$"}
SEL = dict(s=34, marker="o", facecolors="#1669a7", edgecolors="k", linewidths=0.6, zorder=4)
OTHER = dict(s=9, marker="o", color="0.40", linewidths=0, zorder=3)
RANGE = "#bcd4ea"


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prediction_arrays(case_dir):
    """Curves of the prediction figure: one entry per family (column) and channel (row)."""
    out = []
    for family, _ in FAMILIES:
        with np.load(Path(case_dir) / "signals" / "figure_curves" / f"test_{family}_1.npz", allow_pickle=False) as f:
            for name, scale, _ in CHANNELS:
                true = np.array(f["noise_free_" + name]) * scale
                pred = np.array(f["prediction_" + name]) * scale
                out.append({"family": family, "channel": name, "time_s": np.arange(len(true)) * OUTPUT_INTERVAL_S,
                            "noise_free": true, "prediction": pred})
    return out


def recovery_arrays(case_dir):
    """Signed errors (estimate - teacher) / half-range of the four endpoints of each geometry setting."""
    case_dir = Path(case_dir)
    bounds = _json(case_dir / "settings" / "bounds.json")["parameters"]
    half = np.array([b["half_range"] for b in bounds])
    teacher = np.array([p["value"] for p in _json(case_dir / "teacher" / "teacher0.json")["parameters"]])
    out = {}
    for setting in SETTINGS:
        end = _json(case_dir / "results" / setting / "endpoints.json")
        errors = np.array([(np.array(r["estimate"]) - teacher) / half for r in end["runs"]])
        k = next(i for i, r in enumerate(end["runs"]) if r["seed"] == end["selected_seed"])
        out[setting] = {"errors": errors, "selected_index": k, "low": errors.min(axis=0), "high": errors.max(axis=0)}
    return out


def _predictions(case_dir, out_dir, plt):
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    curves = prediction_arrays(case_dir)
    fig, axes = plt.subplots(len(CHANNELS), len(FAMILIES), figsize=(7.4, 6.2), sharex=True)
    for n, c in enumerate(curves):
        j, i = divmod(n, len(CHANNELS))
        ax = axes[i, j]
        ax.plot(c["time_s"], c["noise_free"], **TRUE)
        ax.plot(c["time_s"], c["prediction"], **PRED)
        ax.axvline(SKIPPED_SAMPLES * OUTPUT_INTERVAL_S, color="0.3", ls=":", lw=0.7)
        ax.tick_params(length=2, pad=1.5)
        if j == 0:
            ax.set_ylabel(CHANNELS[i][2])
        if i == 0:
            ax.set_title(FAMILIES[j][1], fontsize=7)
        if i == len(CHANNELS) - 1:
            ax.set_xlabel("Time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=0.4, h_pad=0.6)
    fig.savefig(Path(out_dir) / "HMMWV_test_predictions.pdf", bbox_inches="tight", dpi=170)
    plt.close(fig)


def _recovery(case_dir, out_dir, plt, patches):
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    data = recovery_arrays(case_dir)
    x = np.arange(len(PARAMETERS))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, setting in zip(axes, SETTINGS):
        d = data[setting]
        errors, k = d["errors"], d["selected_index"]
        others = np.delete(errors, k, axis=0)
        ax.axhline(0, color="0.35", lw=0.8, zorder=1)
        ax.bar(x, d["high"] - d["low"], bottom=d["low"], width=0.34, color=RANGE, lw=0, zorder=2)
        ax.scatter(np.repeat(x[None], len(others), axis=0).ravel(), others.ravel(), **OTHER)
        ax.scatter(x, errors[k], **SEL)
        ax.use_sticky_edges = False
        ax.margins(y=0.06)
        ax.set_xticks(x)
        ax.set_xticklabels([SYMBOL[n] for n in PARAMETERS])
        ax.set_title(f"{TITLE[setting]}; selected: seed {k + 1}", fontsize=8)
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel(r"Error $\Delta z_j$ (half-ranges)")
    handles = [plt.Line2D([], [], color="0.35", lw=0.8),
               plt.Line2D([], [], ls="", marker="o", mfc=SEL["facecolors"], mec="k", mew=0.6, ms=5.5),
               patches.Patch(color=RANGE),
               plt.Line2D([], [], ls="", marker="o", mfc=OTHER["color"], mec="none", ms=3)]
    fig.legend(handles, ["Teacher value", "Selected endpoint", "Range of the four endpoints", "Other endpoints"],
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06), fontsize=7)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(Path(out_dir) / "HMMWV_parameter_recovery.pdf", bbox_inches="tight", dpi=170)
    plt.close(fig)


def make_figures(case_dir, out_dir):
    """Draw the two figures into out_dir and return the paths of the files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _predictions(case_dir, out_dir, plt)
    _recovery(case_dir, out_dir, plt, patches)
    return [out_dir / "HMMWV_test_predictions.pdf", out_dir / "HMMWV_parameter_recovery.pdf"]


if __name__ == "__main__":
    for path in make_figures(Path(__file__).resolve().parent, Path.cwd() / "figures_out"):
        print(path)
