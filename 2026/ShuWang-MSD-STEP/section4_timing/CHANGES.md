# Changes made for the release (section4_timing)

No numerical code was changed. Apart from the modified copies below and the text edits described under "Comments and
descriptive text", every copied file is byte-identical to its source (see SOURCES.json, which lists the source path and
SHA-256 of every file as copied, and for each modified copy also the original file and its SHA-256; `<workspace>` stands
for the root of the authors' workspace).

One kind of change was made, marked in the file with a `# RELEASE:` comment:

- **path changed**: an import path or data path pointed to a folder of the authors' workspace; it now points to the
  folder of this package that holds the same files.

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
| `review/jt_bias.py` | path changed | `sys.path` entry `../three_link_jtoc_261013/code_oc/src` is now `../step/src` (same `jt_checked.py`). Only `prop_bias` is used here. |
| `nbody/nbody_common.py` | path changed | `CODE` and `REVIEW` pointed to `../code_s4f` and `../../review_261020`; now `../step` and `../review`. |
| `parity/parity_common.py` | path changed | `NB` pointed to `../nbody_compare`; now `../nbody`. |

`parity/timing_paper.py` is unchanged. It counts running Python processes with the Windows command `tasklist`, so as
shipped it runs on Windows only (see section4_README.md).

## New files (not modified copies)

- `tables_common.py`, `make_table12.py`: table script for Table 12 and the Stage I times of Section 4.5.
- `run_timing.py`: runs `parity/timing_paper.py` after `pathcheck.py` (Windows path-length check for the Numba cache).
- `.gitignore`.

## Comments and descriptive text

After copying, comments, docstrings and descriptive strings were edited in many files, including descriptive values
that scripts write to output JSON: references to internal documents, run codes and dates were removed or replaced by a
description of what the code does, Chinese comments were translated, and the local path prefix was replaced by
`<workspace>`. No statement, identifier, number, key, label or path used by the code was changed; this was checked by
comparing the syntax trees of every `.py` file with docstrings and string constants masked, and the non-string values of
every JSON file. The SHA-256 values in `SOURCES.json` are those of the files as copied, before these edits.

## Diffs

```diff
### review/jt_bias.py (path; source <workspace>\Diagnostics\review_261020\jt_bias.py)
--- <workspace>\Diagnostics\review_261020\jt_bias.py
+++ review/jt_bias.py
@@ -7,3 +7,3 @@
 from numba import njit
-sys.path.insert(0, str(Path(__file__).parent.parent / 'three_link_jtoc_261013' / 'code_oc' / 'src'))
+sys.path.insert(0, str(Path(__file__).parent.parent / 'step' / 'src'))  # RELEASE: path changed for this repository
 import jt_checked as jtc

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
