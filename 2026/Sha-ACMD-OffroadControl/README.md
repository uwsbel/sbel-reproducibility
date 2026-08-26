# Terrain- and Latency-Aware Shared and Autonomous Control on Deformable Terrain

Kyle Sha, Zhenhao Zhou, Radu Serban, Dan Negrut.
The 12th Asian Conference on Multibody Dynamics (ACMD 2026),
October 25-29, 2026, Okinawa, Japan.

## Where everything lives

- **Implementation, benchmarks, and evidence**:
  https://github.com/ksha23/acmd-offroad-control (public upon
  publication). The repository is the paper-value closure: every tracked
  file participates in producing a figure, table, or number in the
  manuscript, `docs/PAPER_TABLE_VALUE_PROVENANCE.md` maps each published
  value to the generation and command that regenerate it, and the
  publish boundary (`benchmarking/publish_paper_figures.py`) refuses any
  result that cannot verify its recorded source digests against the
  commit it names. `benchmarking/verify_provenance_chain.py` enforces
  the data-provenance contract: every training corpus is DERIVED (with a
  manifest naming its collector commit and binary hash) or PRIMARY (with
  a stated reason).
- **Chrono**: the vendored `third_party/chrono` submodule pins commit
  `81d8f249170f6179021e54ce1bf32437ccf84845` (upstream Project Chrono,
  10.0.0-dev; build flags in the repository's `SETUP.md`). acados is
  pinned at `8d6cd69ff133b16ad8dc903afaa4835d587544ab`.
- **Training corpora**: tracked in-repo. The deployed checkpoints
  (`nn_models/tire_force_static`, `nn_models/tire_force_rate`) train on
  `data/tire_rig_commanded` (DERIVED, collected end to end by the
  tracked rig binaries whose hashes the corpus MANIFEST records); the
  scalar-parent comparison checkpoint trains on `data/tire_rig_static`
  (declared PRIMARY, recovered with an exact hash match). Every
  checkpoint's recorded `training_csv_sha256` verifies against the
  tracked corpus.
- **Raw result generations** (multi-GB, back the published CSVs): stored
  as release assets on a companion data repository; restore with
  `data_sync/data_sync.sh pull` per the repository's `DATA.md`.

## Reproducing the paper

```bash
source /opt/ros/jazzy/setup.bash
export PYTHONPATH=$CHRONO_BUILD/bin:$PYTHONPATH
export ACADOS_SOURCE_DIR=<acados checkout>
conda run -n scm-terrain python benchmarking/run.py --tier smoke   # wiring check
conda run -n scm-terrain python benchmarking/run.py --tier paper   # every manuscript number
conda run -n scm-terrain python benchmarking/verify_provenance_chain.py
```

The paper tier collects every study serially or with per-study
parallelism as each study's contract requires, publishes the canonical
CSVs and figures through the fail-closed boundary, and each collected
study records the commit it ran from, whether the worktree was clean,
and the digest of every contract source file.

## Contact

kasha2@wisc.edu
