"""Result figures of the PUMA case, drawn from the files of this folder.

make_figures writes PUMA_test_predictions.pdf and PUMA_parameter_recovery.pdf.
recovery_data and prediction_data return the plotted arrays.
The module needs numpy and matplotlib. It runs no simulation.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SETTINGS = ("different_geometry", "same_geometry")
TITLE = {"different_geometry": "(a) Different geometry", "same_geometry": "(b) Same geometry"}
SELECTED_COLOR = "#1669a7"
SYMBOL = {"m2": "m_2", "m3": "m_3", "m2c2x": "m_2c_{2x}", "m2c2z": "m_2c_{2z}", "m3c3x": "m_3c_{3x}",
          "m3c3z": "m_3c_{3z}", "I1z": "I_{1z}", "I2y": "I_{2y}", "I2z": "I_{2z}", "I3y": "I_{3y}", "I3z": "I_{3z}",
          "b1": "b_1", "b2": "b_2", "b3": "b_3", "fc1": "f_{c1}", "fc2": "f_{c2}", "fc3": "f_{c3}"}
FAMILIES = [("HG", "Gravity-dominated"), ("HF", "Friction-dominated"), ("HI", "Inertia-rich"), ("HC", "Coupling"),
            ("CS", "Spline path"), ("CH", "Chirp")]
SHOWN_VARIANT = 4          # first test variant of each family
JOINT_ANGLE_NOISE_URAD = 100.0
MICRO = chr(181)           # micro sign of the axis labels
RECOVERY_YLIM = (-0.05, 0.10)   # common axis of both panels; only the selected endpoints are drawn


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cost_module(case_dir):
    spec = importlib.util.spec_from_file_location("puma_cost", Path(case_dir) / "cost.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def recovery_data(case_dir):
    """Signed errors of the four endpoints in half-ranges, per setting: array seeds x parameters."""
    case_dir = Path(case_dir)
    par = _json(case_dir / "settings" / "bounds.json")["parameters"]
    names = [p["name"] for p in par]
    centre = np.array([0.5 * (p["lower"] + p["upper"]) for p in par])
    half = np.array([0.5 * (p["upper"] - p["lower"]) for p in par])
    teacher = {p["name"]: p["value"] for p in _json(case_dir / "teacher" / "teacher0.json")["parameters"]}
    z_teacher = (np.array([teacher[n] for n in names]) - centre) / half
    out = {"names": names}
    for s in SETTINGS:
        end = _json(case_dir / "results" / s / "endpoints.json")
        runs = sorted(end["runs"], key=lambda r: r["seed"])
        out[s] = {"seeds": [r["seed"] for r in runs], "selected_seed": end["selected_seed"],
                  "errors": np.array([np.array(r["z"]) - z_teacher for r in runs])}
    return out


def prediction_data(case_dir):
    """Joint angles of the noise-free teacher output and of the selected endpoint, and their
    difference in micro-rad, for one test variant of each family, different-geometry setting."""
    case_dir = Path(case_dir)
    cost = _cost_module(case_dir)
    out = []
    for prefix, family in FAMILIES:
        motion = f"{prefix}{SHOWN_VARIANT}"
        teacher = cost.load(case_dir / "signals" / "test_noise_free" / "different_geometry" / f"{motion}.npz")
        pred = cost.load(case_dir / "signals" / "predictions_selected" / "different_geometry" / f"{motion}.npz")
        e = pred["q"] - teacher["q"]
        e = np.arctan2(np.sin(e), np.cos(e))
        out.append({"motion": motion, "family": family, "teacher_time": teacher["time"], "teacher_q": teacher["q"],
                    "prediction_time": pred["time"], "prediction_q": pred["q"], "error_urad": 1e6 * e})
    return out, cost.SCORING_START_S


def recovery_figure(case_dir, out_dir):
    data = recovery_data(case_dir)
    names = data["names"]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.3), sharex=True, sharey=True)
    x = np.arange(len(names))
    for ax, s in zip(axes, SETTINGS):
        ax.axhline(0, color="0.35", lw=0.8, zorder=1)
        errors = data[s]["errors"]
        k = data[s]["seeds"].index(data[s]["selected_seed"])
        ax.scatter(x, errors[k], s=34, marker="o", facecolors=SELECTED_COLOR, edgecolors="k", linewidths=0.6, zorder=4)
        ax.set_ylim(*RECOVERY_YLIM)
        ax.set_yticks([-0.05, 0.0, 0.05, 0.10])
        ax.set_title(f"{TITLE[s]}; selected: seed {data[s]['selected_seed']}", fontsize=8)
        ax.set_ylabel(r"Error $\Delta z_j$ (half-ranges)")
        ax.grid(axis="y", alpha=.18)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"${SYMBOL[n]}$" for n in names])
    handles = [plt.Line2D([], [], color="0.35", lw=0.8),
               plt.Line2D([], [], ls="", marker="o", mfc=SELECTED_COLOR, mec="k", mew=0.6, ms=5.5)]
    fig.legend(handles, ["Teacher value", "Selected endpoint"],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=7)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(Path(out_dir) / "PUMA_parameter_recovery.pdf", bbox_inches="tight")
    plt.close(fig)


def prediction_figure(case_dir, out_dir):
    data, scoring_start = prediction_data(case_dir)
    teacher_style = dict(color="#c62828", lw=1.0, ls="-", label=r"Noise-free teacher output ($\theta^\star$, $g^\star$)")
    pred_style = dict(color="#1669a7", lw=1.0, ls="--", label=r"Prediction, selected endpoint ($\hat\theta$, $\hat g$)")
    fig = plt.figure(figsize=(7.4, 7.6))
    gs = fig.add_gridspec(7, 6, height_ratios=[1, 1, 1, 0.16, 1, 1, 1], hspace=0.32, wspace=0.42)
    grid_rows = [0, 1, 2, 4, 5, 6]
    axes = np.empty((6, 6), dtype=object)
    for j in range(6):
        for i, g in enumerate(grid_rows):
            axes[i, j] = fig.add_subplot(gs[g, j], sharex=axes[0, j] if i else None)
    for j, d in enumerate(data):
        for joint in range(3):
            ax = axes[joint, j]
            ax.plot(d["teacher_time"], d["teacher_q"][:, joint], **teacher_style)
            ax.plot(d["prediction_time"], d["prediction_q"][:, joint], **pred_style)
            ax = axes[joint + 3, j]
            ax.axhspan(-JOINT_ANGLE_NOISE_URAD, JOINT_ANGLE_NOISE_URAD, color="0.88", lw=0)
            ax.axhline(0, **teacher_style)
            ax.plot(d["teacher_time"], d["error_urad"][:, joint], **pred_style)
        axes[0, j].set_title(d["family"], fontsize=7)
        for i in range(6):
            axes[i, j].axvline(scoring_start, color="0.3", ls=":", lw=0.7)
            axes[i, j].tick_params(length=2, pad=1.5, labelbottom=i in (2, 5))
        axes[5, j].set_xlabel("Time (s)")
    for joint in range(3):
        axes[joint, 0].set_ylabel(f"Joint {joint + 1} angle\n(rad)")
        axes[joint + 3, 0].set_ylabel(f"Joint {joint + 1} error\n({MICRO}rad)")
    for i, lab in ((0, "(a) Joint angles"), (3, "(b) Joint-angle errors")):
        box = axes[i, 0].get_position()
        fig.text(0.02, box.y1 + (0.035 if i == 0 else 0.012), lab, fontsize=8, ha="left", va="bottom")
    handles, labels = axes[3, 0].get_legend_handles_labels()
    handles.append(matplotlib.patches.Patch(color="0.88"))
    labels.append(r"$\pm$ one noise standard deviation")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.965), fontsize=7)
    fig.savefig(Path(out_dir) / "PUMA_test_predictions.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figures(case_dir=HERE, out_dir="figures_out"):
    """Draw the two result figures of the PUMA case into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = {"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(base):
        recovery_figure(case_dir, out_dir)
    with plt.rc_context({**base, "font.size": 7}):
        prediction_figure(case_dir, out_dir)
