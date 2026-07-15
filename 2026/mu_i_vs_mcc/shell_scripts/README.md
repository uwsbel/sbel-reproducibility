# Experiment Shell Scripts

This folder is now restricted to the shell scripts that map to the experiment families shown in the paper results.

These are copied from `build_new/bin`, so they still assume a build-tree working directory and relative executable paths like `./demo_FSI_*` and `./demo_ROBOT_*`.

| Script | Original source | Paper section / figure family | What it runs |
| --- | --- | --- | --- |
| `cone_nrel.sh` | `build_new/bin/cone_nrel.sh` | Cone Penetrometer Test | Runs `demo_FSI_ConePenetrometer` for the GRC-1 cone-penetration comparison with MCC (`OCR=10`) and a `MU_OF_I` cohesion sweep at `container_height=0.24 m`. |
| `bev_nrel.sh` | `build_new/bin/bev_nrel.sh` | Normal Bevameter Test | Runs `demo_FSI_NormalBevameter` with `OCR=20`, `kappa=0.00625`, `lambda=0.025`, cohesion sweep `{0,100,1000,5000}`, and bin heights `0.024, 0.12, 0.24 m`. The paper figure itself uses the `0.24 m` cases. |
| `viper_wheel_paperRuns.sh` | `build_new/bin/viper_wheel_paperRuns.sh` | MGRU3 single-wheel constant-depth results | Runs `demo_FSI_SlopedSingleWheelTest` over slope angle for the constant-depth wheel study. |
| `viper_wheel_paperRuns_DepthVar.sh` | `build_new/bin/viper_wheel_paperRuns_DepthVar.sh` | MGRU3 depth-sensitivity results | Runs `demo_FSI_SlopedSingleWheelTest` with depth variation for the variable-depth wheel figure family. |
| `lander_runs_sbatch.sh` | `build_new/bin/lander_runs_sbatch.sh` | Lunar lander touchdown sensitivity study | Runs the 168-case MCC SLURM array sweep plus the single `MU_OF_I` baseline for the lander paper figures. |

## Notes

- I removed the cone-drop scripts, actual-slope wheel scripts, lander utility scripts, and render/video helpers because they do not map to the figures in `results.tex`.
- The lander sweep script still calls `demo_ROBOT_Lander_CRM`. If you want this curated copy to rerun the restored flat-bed executable, change that command to `demo_ROBOT_Lander_CRM_Flat`.
- The two least-certain mappings are `cone_nrel.sh` and `bev_nrel.sh`. They are the closest matches in `build_new/bin` to the paper’s CPT and bevameter results. If you remember a different runner for either one, I can swap it in.