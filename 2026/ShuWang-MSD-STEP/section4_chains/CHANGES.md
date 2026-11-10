# Changes made for the release (section4_chains)

The computations behind the paper were not changed. Apart from the modified copies below (including those listed under
"Configurations not used in the paper") and the text edits described under "Comments and descriptive text", every
copied file is byte-identical to its source (see SOURCES.json, which lists the source path and SHA-256 of every
file as copied, and for each modified copy also the original file and its SHA-256; `<workspace>` stands for the root
of the authors' workspace).

Two kinds of change were made, each marked in the file with a `# RELEASE:` comment:

- **path changed**: an import path or data path pointed to a folder of the authors' workspace; it now points to the
  folder of this package that holds the same files.
- **guard removed**: the script refused to run unless a file from a run that is not part of this repository existed (a
  reproduction check against saved results of that run). Only the guard was removed; where the guard step also created
  the output folder, a `mkdir` was added.

Folder names in this package and the workspace folders they come from:

| Package folder | Workspace folder |
|---|---|
| `step/` | `stage1_final_261015/code_s4f/` (STEP code, JT core `jt_checked.py`, configs, three-link data) |
| `review/` | `review_261020/` (JT, JT variants, JT+bias, the 10 s Stage I runs) |
| `nbody/` | `stage1_final_261015/nbody_compare/` (N-body chains, JT-BS, JT-BHS, JT-B) |
| `parity/` | `stage1_final_261015/step_parity_261103/` (1 deg runs, second motion, timing) |

## Modified copies

| File | Change | Reason |
|---|---|---|
| `review/d_common.py` | path changed | `CODE` pointed to `../stage1_final_261015/code_s4f`; now `../step`. |
| `review/jt_t1_diag.py` | path changed | `ROOT` pointed to `../three_link_jtoc_261013/code_oc`; now `../step` (its `jt_checked.py`, `comparison_noise_bias.json` and `step.json` were byte-identical to the copies in `step/` before the text edits below). |
| `review/jt_bias.py` | path changed | `sys.path` entry `../three_link_jtoc_261013/code_oc/src` is now `../step/src`. |
| `review/c3f_mc_stepR.py` | guard removed | Required `out_f/REPRO_CHECK.json` (made from about 1 GB of saved JT state histories that are not part of this repository). `mkdir` of `out_f/mc` uses `parents=True`, because the guard step created `out_f`. |
| `nbody/nbody_common.py` | path changed | `CODE` and `REVIEW` pointed to `../code_s4f` and `../../review_261020`; now `../step` and `../review`. |
| `parity/parity_common.py` | path changed | `NB` pointed to `../nbody_compare`; now `../nbody`. |

## Configurations not used in the paper

After the changes above, code paths, runs and saved outputs for configurations that are not used in the paper were
removed. The jobs that remain compute the paper results with the same operations in the same order, and every value
that remains in `expected/` is unchanged (entries were only deleted). These edits are not marked with `# RELEASE:` and
are not shown in the diffs below.

| File | Change |
|---|---|
| `review/c3_mc.py` | Code paths and saved outputs for configurations that are not used in the paper were removed. |
| `review/d_common.py` | Code paths for configurations that are not used in the paper were removed. |
| `review/jt_t1_diag.py` | Code paths for configurations that are not used in the paper were removed; the module now only provides `QK`, `cfg` and the JT core module. |
| `review/jt_stepR.py` | Code paths for configurations that are not used in the paper were removed. |
| `nbody/nbody_jobs.py` | Code paths and saved outputs for configurations that are not used in the paper were removed. |
| `parity/calibration_fix.py`, `parity/parity_fix.py` | Code paths for configurations that are not used in the paper were removed. |
| `parity/run_mc_b.py` | Code paths for configurations that are not used in the paper were removed; the module now only provides `job()`, used by `run_mc_c_base.py` and `run_mc_d.py`. |
| `parity/run_mc_c_base.py` | Code paths and saved outputs for configurations that are not used in the paper were removed; it now contains the `work()` function of the workspace script `run_mc_c.py`, for the paper jobs. |
| `parity/run_mc_d.py` | Code paths and saved outputs for configurations that are not used in the paper were removed. |
| `parity/gen_data_c.py` | Runs for configurations that are not used in the paper were removed. |
| `parity/parity_c.py` | Descriptive text only (docstring). |
| `parity/run_mc_c.py`, `parity/compiled_step_fix2.py` | Removed; they are not needed for the paper results (their entries in `SOURCES.json` were removed). |
| `step/configs/comparison_noise_bias.json` | The key `author_confirmation`, which no code reads, was removed. |
| `expected/review/out_d/mc/`, `expected/nbody/out/mc/`, `expected/parity/out/fixc/mc/t2/base/`, `expected/parity/out/paper/mc/t2_p5/` | Saved outputs of configurations that are not used in the paper, and the keys `est`, `fix2` and `config`, which the code no longer writes, were removed. |

## Scripts that read outputs of other producers

- `review/c3c_mc_aprime_oc.py` checks its Stage I levers against `out_d/mc/` from `c3_mc.py`; run `c3_mc.py` first (the
  STEP rows of Table 8 come from it anyway).
- `review/c3_mc.py` checks that the paper seeds regenerate the shipped `step/data/noise_only` and `noise_bias` files.
- `parity/run_mc_d.py gate` compares with `nbody/out/mc/N03_r01_noise_bias.json` from `nbody/run_mc.py`; the gate is optional.

## New files (not modified copies)

- `run_section4_chains.py`: lists and runs the producers of Table 8, Table 11 and the Section 4.2.4 sentence.
- `pathcheck.py`: stops a run early when the folder path is too long for the Numba cache under Windows without long-path support.
- `parity/run_mc_c_base.py`: runs the second-motion jobs of Table 11 at 1 deg (N = 3, 5, 10).
- `tables_common.py`, `make_table8.py`, `make_table11.py`, `make_s424.py`: table scripts. `make_s424.py` computes the
  Section 4.2.4 bootstrap intervals with the algorithm used for the paper; the script that computed them for the paper
  is not part of this repository.
- `.gitignore`.

## Comments and descriptive text

After copying, comments, docstrings and descriptive strings were edited in many files, including descriptive values
that scripts write to output JSON (and the same values in `expected/` and `step/configs/`): references to internal
documents, run codes and dates were removed or replaced by a description of what the code does, Chinese comments were
translated, and the local path prefix was replaced by `<workspace>`. No statement, identifier, number, key, label or path
used by the code was changed; this was checked by comparing the syntax trees of every `.py` file with docstrings and
string constants masked, and the non-string values of every JSON file. The SHA-256 values in `SOURCES.json` are those of
the files as copied, before these edits. In the diffs below, internal document references in comments were removed in
the same way.

## Diffs

The diffs show the path and guard changes; the removals listed under "Configurations not used in the paper" are not
shown.

```diff
### review/d_common.py (path; source <workspace>\Diagnostics\review_261020\d_common.py)
--- <workspace>\Diagnostics\review_261020\d_common.py
+++ review/d_common.py
@@ -8,3 +8,3 @@
 HERE = Path(__file__).parent
-CODE = HERE.parent / 'stage1_final_261015' / 'code_s4f'
+CODE = HERE.parent / 'step'  # RELEASE: path changed for this repository
 sys.path.insert(0, str(HERE)); sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / 'src'))

### review/jt_t1_diag.py (path; source <workspace>\Diagnostics\review_261020\jt_t1_diag.py)
--- <workspace>\Diagnostics\review_261020\jt_t1_diag.py
+++ review/jt_t1_diag.py
@@ -9,3 +9,3 @@
 HERE = Path(__file__).parent; DG = HERE.parent
-J = DG / 'three_link_jtoc_261013'; ROOT = J / 'code_oc'
+ROOT = DG / 'step'  # RELEASE: path changed for this repository
 sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))

### review/jt_bias.py (path; source <workspace>\Diagnostics\review_261020\jt_bias.py)
--- <workspace>\Diagnostics\review_261020\jt_bias.py
+++ review/jt_bias.py
@@ -7,3 +7,3 @@
 from numba import njit
-sys.path.insert(0, str(Path(__file__).parent.parent / 'three_link_jtoc_261013' / 'code_oc' / 'src'))
+sys.path.insert(0, str(Path(__file__).parent.parent / 'step' / 'src'))  # RELEASE: path changed for this repository
 import jt_checked as jtc

### review/c3f_mc_stepR.py (guard; source <workspace>\Diagnostics\review_261020\c3f_mc_stepR.py)
--- <workspace>\Diagnostics\review_261020\c3f_mc_stepR.py
+++ review/c3f_mc_stepR.py
@@ -36,5 +36,4 @@
 if __name__ == '__main__':
-    chk = json.loads((d.HERE / 'out_f' / 'REPRO_CHECK.json').read_text())
-    if not chk.get('all_ok'): raise SystemExit('reproduction check not passed')
-    MCF.mkdir(exist_ok=True)
+    # RELEASE: guard removed (out_f/REPRO_CHECK.json, a check against saved state histories that are not part of this repository)
+    MCF.mkdir(parents=True, exist_ok=True)  # RELEASE: parents=True also creates out_f
     jobs = [(r, s) for r in range(1, 21) for s in ('noise_only', 'noise_bias')]

### nbody/nbody_common.py (path; source <workspace>\Diagnostics\stage1_final_261015\nbody_compare\nbody_common.py)
--- <workspace>\Diagnostics\stage1_final_261015\nbody_compare\nbody_common.py
+++ nbody/nbody_common.py
@@ -10,3 +10,3 @@
 from scipy.spatial.transform import Rotation as R
-CODE = HERE.parent / 'code_s4f'; REVIEW = HERE.parent.parent / 'review_261020'
+CODE = HERE.parent / 'step'; REVIEW = HERE.parent / 'review'  # RELEASE: paths changed for this repository
 sys.path.insert(0, str(HERE)); sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / 'src')); sys.path.append(str(REVIEW))

### parity/parity_common.py (path; source <workspace>\Diagnostics\stage1_final_261015\step_parity_261103\parity_common.py)
--- <workspace>\Diagnostics\stage1_final_261015\step_parity_261103\parity_common.py
+++ parity/parity_common.py
@@ -5,3 +5,3 @@
 HERE = Path(__file__).resolve().parent
-NB = HERE.parent / 'nbody_compare'
+NB = HERE.parent / 'nbody'  # RELEASE: path changed for this repository
 os.environ['NUMBA_CACHE_DIR'] = str(HERE / '.numba_cache')
```

Data: the records that ../prepare_data.py generates are not stored in this repository (see section4_README.md, "Data"); the SOLVER.json files of the paper records are in expected/.
