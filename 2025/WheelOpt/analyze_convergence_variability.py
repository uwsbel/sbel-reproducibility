#!/usr/bin/env python
"""Convergence and run-to-run variability analysis of the BO campaigns.

Added during the journal revision to address a reviewer request for best-so-far
convergence curves and uncertainty estimates. It reproduces:
  - the manuscript's 4-panel best-so-far convergence figure (bo_convergence.png), and
  - the variability numbers quoted in the revision (results.json, notes.md):
    bootstrap re-initialization spread of each campaign's Sobol stage, and the
    scatter of the metric components among near-optimal designs.

Inputs: the archived per-trial campaign records from the BayesianOptimizationData
folder in the paper's Box data share (see README.md). Point --data-root at your
local download of that folder:

    python analyze_convergence_variability.py --data-root /path/to/BayesianOptimizationData

Dependencies: numpy, pandas, matplotlib (all in requirements.txt).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# n_sobol per campaign as reported in the paper: pull 1200, joint 1800,
# looped Stage 1 1200, looped Stage 2 600.
CAMPAIGNS = {
    "pull": dict(csv="Pull/trials.csv", n_sobol=1200,
                 label="(a) Wheel-only, pull test", ylab="cost (time-only)"),
    "joint": dict(csv="SineSideSlip_wControl_Dec22/trials_wControl.csv", n_sobol=1800,
                  label="(b) Joint, sine test", ylab="cost (composite)"),
    "stage1": dict(csv="SineSideSlip_global_1700_0_0.6_25_3_5_24Dec/trials.csv", n_sobol=1200,
                   label="(c) Looped Stage 1 (wheel), sine test", ylab="cost (composite)"),
    "stage2": dict(csv="SineSideSlip_global_1700_0_0.6_25_3_5_24Dec/"
                       "trials_stPIDControllerOnly_0.0_0.0_0.0_1.0.csv", n_sobol=600,
                   label="(d) Looped Stage 2 (steering), sine test", ylab="cost (tracking-only)"),
}

B = 10_000
SEED = 20260708  # fixed so the reported numbers regenerate exactly


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=Path("."),
                    help="Path to the downloaded BayesianOptimizationData folder")
    ap.add_argument("--out", type=Path, default=Path("analysis_out"),
                    help="Output directory (default: ./analysis_out)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    results = {}
    frames = {}

    for key, c in CAMPAIGNS.items():
        df = pd.read_csv(args.data_root / c["csv"]).sort_values("trial_index").reset_index(drop=True)
        frames[key] = df
        metric = df["metric"].to_numpy()

        inc_i = int(np.argmin(metric))
        inc = df.iloc[inc_i]

        # Bootstrap over the Sobol stage: what would a re-initialized quasi-random
        # stage of the same size have found? (initialization variability only)
        sobol = df[df["trial_index"] < c["n_sobol"]]
        pool = sobol["metric"].to_numpy()
        n = len(pool)
        idx = rng.integers(0, n, size=(B, n))
        boot_min_pos = np.argmin(pool[idx], axis=1)
        boot_best = pool[idx[np.arange(B), boot_min_pos]]

        boot = {
            "n_sobol_completed": n,
            "best_mean": float(boot_best.mean()),
            "best_sd": float(boot_best.std(ddof=1)),
            "best_p2_5": float(np.percentile(boot_best, 2.5)),
            "best_p97_5": float(np.percentile(boot_best, 97.5)),
        }
        if "rms_error" in df.columns:
            rms_pool = sobol["rms_error"].to_numpy()
            boot_rms = rms_pool[idx[np.arange(B), boot_min_pos]]
            boot["selected_rms_mean"] = float(boot_rms.mean())
            boot["selected_rms_sd"] = float(boot_rms.std(ddof=1))

        # Near-optimal plateau: trials within x% of the best cost, and the spread of
        # the metric components among the top 1% of trials.
        best = float(metric.min())
        plateau = {f"within_{p}pct": int((metric <= best * (1 + p / 100)).sum()) for p in (1, 2, 5)}
        top1 = df[df["metric"] <= np.percentile(metric, 1)]
        comp = {}
        for col in ("total_time_to_reach", "rms_error", "average_power"):
            if col in df.columns:
                comp[col] = dict(mean=float(top1[col].mean()), sd=float(top1[col].std(ddof=1)),
                                 min=float(top1[col].min()), max=float(top1[col].max()))

        results[key] = {
            "rows_completed": int(len(df)),
            "max_trial_index": int(df["trial_index"].max()),
            "incumbent": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                          for k, v in inc.items() if k != "timestamp"},
            "incumbent_found_at_trial": int(inc["trial_index"]),
            "best_metric": best,
            "sobol_stage_best": float(pool.min()),
            "bo_stage_gain_pct": float((pool.min() - best) / pool.min() * 100),
            "bootstrap_sobol": boot,
            "plateau": plateau,
            "top1pct_components": comp,
            "top1pct_n": int(len(top1)),
        }

    # ---- figure -------------------------------------------------------------
    ACCENT = "#0072B2"  # one hue for the one entity (best-so-far); gray for raw trials
    RAW = "#B0B0B0"
    INK = "#1A1A1A"

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    for ax, (key, c) in zip(axes.flat, CAMPAIGNS.items()):
        df = frames[key]
        metric = df["metric"].to_numpy()
        ti = df["trial_index"].to_numpy()
        bsf = np.minimum.accumulate(metric)

        ylo = bsf.min() - 0.06 * (np.percentile(metric, 90) - bsf.min())
        yhi = np.percentile(metric, 90)
        ax.scatter(ti, metric, s=3, color=RAW, alpha=0.35,
                   linewidths=0, rasterized=True, label="evaluations")
        ax.plot(ti, bsf, color=ACCENT, lw=1.8, label="best so far")
        ax.axvline(c["n_sobol"], color="#555555", lw=1.0, ls=(0, (4, 3)))
        ax.text(c["n_sobol"], 0.985, "  Sobol $\\rightarrow$ BO",
                transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=7.5, color="#555555")
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(0, int(df["trial_index"].max()) + 1)
        ax.set_title(c["label"], loc="left")
        ax.set_xlabel("evaluation index")
        ax.set_ylabel(c["ylab"])
        ax.grid(True, color="#DDDDDD", lw=0.5, alpha=0.8)
        ax.set_axisbelow(True)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out / "bo_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- reports ------------------------------------------------------------
    with open(args.out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    lines = ["# Convergence and variability summary\n"]
    for key, r in results.items():
        b = r["bootstrap_sobol"]
        lines.append(f"## {CAMPAIGNS[key]['label']}")
        lines.append(f"- completed {r['rows_completed']} of {r['max_trial_index'] + 1} trials; "
                     f"incumbent found at trial {r['incumbent_found_at_trial']}")
        lines.append(f"- best cost {r['best_metric']:.4f}; Sobol-stage best {r['sobol_stage_best']:.4f}; "
                     f"BO stage improved the Sobol best by {r['bo_stage_gain_pct']:.1f}%")
        lines.append(f"- bootstrap ({B} replicates) over the {b['n_sobol_completed']}-point Sobol stage: "
                     f"best cost {b['best_mean']:.3f} +/- {b['best_sd']:.3f} "
                     f"(95% band [{b['best_p2_5']:.3f}, {b['best_p97_5']:.3f}])")
        if "selected_rms_mean" in b:
            lines.append(f"- tracking error of the bootstrap-selected design: "
                         f"{b['selected_rms_mean']:.4f} +/- {b['selected_rms_sd']:.4f} m")
        lines.append(f"- plateau: {r['plateau']['within_1pct']} / {r['plateau']['within_2pct']} / "
                     f"{r['plateau']['within_5pct']} trials within 1/2/5% of the best cost")
        for col, s in r["top1pct_components"].items():
            lines.append(f"- top-1% ({r['top1pct_n']} trials) {col}: mean {s['mean']:.4f}, sd {s['sd']:.4f}, "
                         f"range [{s['min']:.4f}, {s['max']:.4f}]")
        lines.append("")

    (args.out / "notes.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
