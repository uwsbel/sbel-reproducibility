"""Draw the two Polaris result figures of the paper from the files of this folder.

make_figures writes Polaris_test_predictions.pdf and Polaris_parameter_recovery.pdf.
numpy and matplotlib only. No simulation.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

FAMILIES = ["Lane change, brake", "Longitudinal chirp", "Multisine slalom", "Sine dwell, coast",
            "Offset-brake slalom", "Late sweep, release"]
# plotted channels: array name in signals/figure_curves, factor, axis label
CHANNELS = [("position_y_m", 1.0, "Position $y$ (m)"),
            ("wheel_speed_front_left_rad_s", 1.0, "Front-left wheel\nspeed (rad/s)"),
            ("upright_y_front_left_m", 1000.0, "Front-left upright\n$y$ rel. chassis (mm)"),
            ("yaw_rate_rad_s", 1.0, "Yaw rate (rad/s)")]
DT, SKIP = 0.002, 100
NAMES = ["mu", "M", "COMx", "COMz", "Ixx", "Iyy", "Izz"]
SYMBOL = {"mu": r"$\mu$", "M": r"$M$", "COMx": r"$\mathrm{COMx}$", "COMz": r"$\mathrm{COMz}$",
          "Ixx": r"$I_{xx}$", "Iyy": r"$I_{yy}$", "Izz": r"$I_{zz}$"}
TITLE = {"different_geometry": "(a) Different geometry", "same_geometry": "(b) Same geometry"}
TRUE = dict(color="#c62828", lw=1.2, ls="-", label=r"Noise-free teacher output ($\theta^\star$, $g^\star$)")
PRED = dict(color="#1669a7", lw=1.2, ls="--", label=r"Prediction, selected endpoint ($\hat\theta$, $\hat g$)")
SEL = dict(s=34, marker="o", facecolors="#1669a7", edgecolors="k", linewidths=0.6, zorder=4)
OTHER = dict(s=9, marker="o", color="0.40", linewidths=0, zorder=3)
RANGE = "#bcd4ea"


def load_module(name):
    """Load a module of this folder by its path (the three cases have modules with the same names)."""
    path = Path(__file__).resolve().parent / (name + ".py")
    spec = importlib.util.spec_from_file_location("polaris_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prediction_curves(case_dir):
    """Plotted curves of the prediction figure: {(row, column): [(time, teacher output), (time, prediction)]}."""
    case_dir = Path(case_dir)
    test = json.loads((case_dir / "settings" / "motions.json").read_text(encoding="utf-8"))["test"]
    first = {m["family_index"]: m["name"] for m in test if m["variant"] == 1}
    assert sorted(first) == list(range(1, 7))
    curves = {}
    for j in range(6):
        with np.load(case_dir / "signals" / "figure_curves" / (first[j + 1] + ".npz"), allow_pickle=False) as f:
            for i, (name, factor, _) in enumerate(CHANNELS):
                curves[(i, j)] = []
                for who in ("teacher", "model"):
                    y = np.array(f[name + "_" + who]) * factor
                    curves[(i, j)].append((np.arange(len(y)) * DT, y))
    return curves


def recovery_errors(case_dir):
    """Signed errors (estimate - teacher) / half-range: {setting: (4 x 7 array, index of the selected run)}."""
    tables = load_module("tables")
    data = tables.load_results(case_dir)["teacher0"]
    out = {}
    for s in ("different_geometry", "same_geometry"):
        runs = sorted(data[s]["runs"], key=lambda x: x["seed"])
        errors = np.array([tables.errors(data[s], run)[1] for run in runs])
        out[s] = (errors, runs.index(tables.selected_run(runs)))
    return out


def predictions(case_dir, out_dir, plt):
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    curves = prediction_curves(case_dir)
    fig, axes = plt.subplots(len(CHANNELS), 6, figsize=(7.4, 6.2), sharex=True)
    for j in range(6):
        for i, (_, _, ylabel) in enumerate(CHANNELS):
            ax = axes[i, j]
            for (t, y), style in zip(curves[(i, j)], (TRUE, PRED)):
                ax.plot(t, y, **style)
            ax.axvline(SKIP * DT, color="0.3", ls=":", lw=0.7)
            ax.tick_params(length=2, pad=1.5)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(FAMILIES[j], fontsize=7)
            if i == len(CHANNELS) - 1:
                ax.set_xlabel("Time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=0.4, h_pad=0.6)
    fig.savefig(Path(out_dir) / "Polaris_test_predictions.pdf", bbox_inches="tight", dpi=170)
    plt.close(fig)


def recovery(case_dir, out_dir, plt, matplotlib):
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    errors = recovery_errors(case_dir)
    x = np.arange(len(NAMES))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, setting in zip(axes, ("different_geometry", "same_geometry")):
        D, k = errors[setting]
        lo, hi = D.min(axis=0), D.max(axis=0)
        others = np.delete(D, k, axis=0)
        ax.axhline(0, color="0.35", lw=0.8, zorder=1)
        ax.bar(x, hi - lo, bottom=lo, width=0.34, color=RANGE, lw=0, zorder=2)
        ax.scatter(np.repeat(x[None], len(others), axis=0).ravel(), others.ravel(), **OTHER)
        ax.scatter(x, D[k], **SEL)
        ax.use_sticky_edges = False
        ax.margins(y=0.06)
        ax.set_xticks(x)
        ax.set_xticklabels([SYMBOL[n] for n in NAMES])
        ax.set_title(f"{TITLE[setting]}; selected: seed {k + 1}", fontsize=8)
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel(r"Error $\Delta z_j$ (half-ranges)")
    handles = [plt.Line2D([], [], color="0.35", lw=0.8),
               plt.Line2D([], [], ls="", marker="o", mfc=SEL["facecolors"], mec="k", mew=0.6, ms=5.5),
               matplotlib.patches.Patch(color=RANGE),
               plt.Line2D([], [], ls="", marker="o", mfc=OTHER["color"], mec="none", ms=3)]
    fig.legend(handles, ["Teacher value", "Selected endpoint", "Range of the four endpoints", "Other endpoints"],
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.06), fontsize=7)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(Path(out_dir) / "Polaris_parameter_recovery.pdf", bbox_inches="tight", dpi=170)
    plt.close(fig)


def make_figures(case_dir, out_dir):
    """Write the two figures as PDF files into out_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches
    import matplotlib.pyplot as plt
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    predictions(case_dir, out_dir, plt)
    recovery(case_dir, out_dir, plt, matplotlib)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    make_figures(here, here / "figures_out")
