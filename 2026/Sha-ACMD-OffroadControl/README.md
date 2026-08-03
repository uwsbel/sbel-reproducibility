# Terrain- and Latency-Aware Shared and Autonomous Control on Deformable Terrain

Kyle Sha, Zhenhao Zhou, Radu Serban, Dan Negrut.
The 12th Asian Conference on Multibody Dynamics (ACMD 2026),
October 25-29, 2026, Okinawa, Japan.

## Where everything lives

- **Implementation, benchmarks, and evidence**:
  https://github.com/ksha23/offroad-control (branch `acmd-rig`; public
  upon publication). The repository is the strict paper-value closure:
  every tracked file participates in producing a figure, table, or number
  in the manuscript, and two verifiers (`benchmarking/verify_rig_only.py`,
  `benchmarking/verify_provenance_chain.py`) enforce the paper's
  rig-only supervision and data-provenance contracts.
- **Chrono**: the vendored `third_party/chrono` submodule pins commit
  `81d8f249170f6179021e54ce1bf32437ccf84845` (upstream Project Chrono,
  10.0.0-dev; build flags in the repository's `SETUP.md`). acados is
  pinned at `8d6cd69ff133b16ad8dc903afaa4835d587544ab`.
- **Training corpora**: tracked in-repo — `data/tire_rig_v5` (DERIVED,
  collected end to end by the tracked `build_rig_cmd` binaries whose
  hashes the corpus MANIFEST records) and
  `data/tire_rig/scm_static_100k_v4.csv` (declared PRIMARY).
- **Raw result generations** (multi-GB, back the published CSVs): stored
  as release assets on a companion data repository; restore with
  `data_sync/data_sync.sh pull` per the repository's `DATA.md`.

## Reproducing the paper

```bash
conda env create -f environment.yml   # or see SETUP.md
conda run -n scm-terrain python benchmarking/run.py --tier paper
conda run -n scm-terrain python benchmarking/make_paper_figures.py
conda run -n scm-terrain python benchmarking/verify_rig_only.py
conda run -n scm-terrain python benchmarking/verify_provenance_chain.py
```

The paper tier is 8 commands, one per results section; the dry-run
manifest (`--tier paper --dry-run`) is the authority for exact command
lines. A bare clone (without the raw-data restore) can still rebuild the
three manuscript figures from the committed evidence:
`python benchmarking/make_paper_figures.py --from-published`.

Questions: open an issue on the implementation repository or contact
kasha2@wisc.edu.
