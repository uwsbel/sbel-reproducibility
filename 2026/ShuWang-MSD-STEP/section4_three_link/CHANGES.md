# Changes made for the release (section4_three_link)

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
| `review/jt_t1_diag.py` | path changed | `ROOT` pointed to `../three_link_jtoc_261013/code_oc`; now `../step`. Only `src/jt_checked.py`, `configs/comparison_noise_bias.json` and `configs/step.json` were used from there, and they were byte-identical to the copies in `step/` before the text edits below. |
| `review/jt_bias.py` | path changed | `sys.path` entry `../three_link_jtoc_261013/code_oc/src` is now `../step/src` (same `jt_checked.py`). |
| `review/c2_step10.py` | guard removed | A Stage I reproduction check on [0, 30] s against `code_s4f/runs/eiv_noise_bias/body*_stageI.json` (a run that is not part of this repository). It ran before, and independently of, the 10 s runs; all random generators in Stage I are seeded per call, and `calibration.STAGE1_KEEP` is reset after every run. The output `C2_STEP10.json` therefore lacks the keys `check_30s_*` (also removed from `expected/review/out_d/C2_STEP10.json`). |
| `review/c1f_jt_variants_stepR.py` | guard removed | Required `out_f/REPRO_CHECK.json` (a check against about 1 GB of saved JT state histories; neither is part of this repository). Adds `out_f.mkdir`, which the guard step did. |
| `review/run_Aprime_oc_stepR.py` | guard removed | Required `out_e/APRIME_OC.json` (a run with residual SD 0.005/0.005 that is not part of this repository) and reran that setting only to check its reproduction. The check run and its report line are removed, and the output no longer has the key `check_reproduces_previous` (also removed from `expected/review/out_e/APRIME_OC_STEPR.json`). Adds `out_e.mkdir`. |
| `nbody/nbody_common.py` | path changed | `CODE` and `REVIEW` pointed to `../code_s4f` and `../../review_261020`; now `../step` and `../review`. |

## Configurations not used in the paper

After the changes above, code paths, runs and saved outputs for configurations that are not used in the paper were
removed. The jobs that remain compute the paper results with the same operations in the same order, and every value
that remains in `expected/` is unchanged (entries were only deleted). These edits are not marked with `# RELEASE:` and
are not shown in the diffs below.

| File | Change |
|---|---|
| `review/d_common.py` | Code paths for configurations that are not used in the paper were removed. |
| `review/jt_t1_diag.py` | Code paths for configurations that are not used in the paper were removed; the module now only provides `QK`, `cfg` and the JT core module. |
| `review/jt_stepR.py` | Code paths and saved outputs for configurations that are not used in the paper were removed (`c1f_jt_variants_stepR.py` runs its `variants()`). |
| `review/c1g_jt_prior_mid.py` | A check against an output of a configuration that is not used in the paper was removed, with its output key `repro_max_abs_diff`. |
| `review/run_Aprime_oc_stepR.py` | The key `check_reproduces_previous` is no longer written (see above). |
| `nbody/nbody_jobs.py` | Code paths and saved outputs for configurations that are not used in the paper were removed. |
| `step/configs/comparison_noise_bias.json` | The key `author_confirmation`, which no code reads, was removed. |
| `expected/review/out_f/C1F_JT_VARIANTS_STEPR.json`, `expected/review/out_g/C1G_JT_PRIOR_MID.json`, `expected/nbody/out/single/` | Saved outputs of configurations that are not used in the paper, and the key `repro_max_abs_diff`, were removed. |
| `expected/review/out_d/C2_STEP10.json`, `expected/review/out_e/APRIME_OC_STEPR.json` | The keys `check_30s_*` and `check_reproduces_previous`, which the code no longer writes, were removed. |

## Scripts that read outputs of other producers

- `review/run_Aprime_oc_stepR.py` reads `out_d/C2_STEP10.json` (step 1 of the same replay) to count where STEP is smaller.

## New files (not modified copies)

- `run_section4_three_link.py`: runs the producers in order.
- `pathcheck.py`: stops a run early when the folder path is too long for the Numba cache under Windows without long-path support.
- `nbody/run_jtbs_single.py`: calls `run_single.work` unchanged for N = 3 only (the N = 5 and 10 records are not in this section).
- `run_table9_spatial.py`: reruns the spatial double-pendulum rows of Table 9 with `../section3_spatial/run_case.py`.
- `tables_common.py`, `make_table5.py`, `make_table6.py`, `make_table7.py`, `make_table9.py`, `make_table10.py`: table scripts.
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
shown (for example, the line `ok = None` of `review/run_Aprime_oc_stepR.py` below was removed later).

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

### review/c2_step10.py (guard; source <workspace>\Diagnostics\review_261020\c2_step10.py)
--- <workspace>\Diagnostics\review_261020\c2_step10.py
+++ review/c2_step10.py
@@ -8,17 +8,4 @@
 out = {}
-# check: calibration_s = 30 reproduces the saved Stage I levers of runs/eiv_noise_bias
-m, ref = d.data('noise_bias')
-r30 = d.run_step(m, d.scfg('step_eiv.json', 30.0), stage2=False)
-run = d.CODE / 'runs' / 'eiv_noise_bias'
-saved = []
-for b in (1, 2, 3):
-    s = json.loads((run / f'body{b}_stageI.json').read_text())
-    saved += [s['lever']] if b == 1 else [s['parent_lever'], s['lever']]
-dev = max(float(np.max(np.abs(np.array(a) - np.array(b)))) for a, b in zip(r30['levers'], saved))
-out['check_30s_max_abs_lever_diff_m'] = dev
-out['check_30s_ok_predeclared_1e-9'] = bool(dev <= 1e-9)
-out['check_30s_ok'] = bool(dev <= 1e-6)  # BFGS stops at precision loss; tolerance 1e-6 m
-print('30 s reproduction: max |dl| =', dev, flush=True)
-if not out['check_30s_ok']:
-    (d.OUT / 'C2_STEP10.json').write_text(json.dumps(out, indent=1)); raise SystemExit('reproduction check failed')
+# RELEASE: guard removed (a Stage I check on [0, 30] s against saved levers that are not part of this repository;
+# it ran before, and independently of, the 10 s runs below, so its removal does not change their results)
 

### review/c1f_jt_variants_stepR.py (guard; source <workspace>\Diagnostics\review_261020\c1f_jt_variants_stepR.py)
--- <workspace>\Diagnostics\review_261020\c1f_jt_variants_stepR.py
+++ review/c1f_jt_variants_stepR.py
@@ -39,4 +39,4 @@
 if __name__ == '__main__':
-    chk = json.loads((F / 'REPRO_CHECK.json').read_text())
-    if not chk.get('all_ok'): raise SystemExit('reproduction check not passed')
+    # RELEASE: guard removed (out_f/REPRO_CHECK.json, a check against saved state histories that are not part of this repository)
+    F.mkdir(exist_ok=True)  # RELEASE: creates out_f
     target = F / 'C1F_JT_VARIANTS_STEPR.json'

### review/run_Aprime_oc_stepR.py (guard; source <workspace>\Diagnostics\review_261020\run_Aprime_oc_stepR.py)
--- <workspace>\Diagnostics\review_261020\run_Aprime_oc_stepR.py
+++ review/run_Aprime_oc_stepR.py
@@ -13,3 +13,4 @@
 c2 = json.loads((d.OUT / 'C2_STEP10.json').read_text())
-prev = json.loads((OUTE / 'APRIME_OC.json').read_text())
+# RELEASE: guard removed (out_e/APRIME_OC.json, a run with residual SD 0.005/0.005 that is not part of this repository)
+OUTE.mkdir(exist_ok=True)  # RELEASE: creates out_e
 res, L = {'STEP_residual_sd': STEP_R}, []
@@ -34,8 +35,6 @@
     lev = dict(zip(d.LABELS, [np.array(x) for x in r['levers']]))
-    hc = jo.run(m, st, lev, 0.0, d.jtd.QK, (0.005, 0.005), d.G_VEC, d.PIVOT, oc=True, motion_sd=STEP_P0)
-    ok = bool(np.array_equal(np.array(d.state_rows(hc, t, m['time'], jt=True)['t>=10']), np.array(prev[s]["A' (JT+bias+OC)"]['t10'])))
-    if not ok: raise SystemExit(f'(0.005, 0.005) does not reproduce out_e/APRIME_OC.json ({s})')
+    ok = None  # RELEASE: guard removed (the 0.005/0.005 run only checked the reproduction of out_e/APRIME_OC.json)
     h = jo.run(m, st, lev, 0.0, d.jtd.QK, STEP_R, d.G_VEC, d.PIVOT, oc=True, motion_sd=STEP_P0)
     sr = d.state_rows(h, t, m['time'], jt=True); hs, ts = heading(h, t, w)
-    a = np.array(sr['t>=10']); stp = np.array(c2[s]['state']['t>=10']); old = np.array(prev[s]["A' (JT+bias+OC)"]['t10'])
+    a = np.array(sr['t>=10']); stp = np.array(c2[s]['state']['t>=10'])  # RELEASE: guard removed ('old' came from out_e/APRIME_OC.json)
     res[s] = dict(check_reproduces_previous=ok, finite=bool(all(np.isfinite(h[k]).all() for k in h)),
@@ -44,3 +43,3 @@
     L.append(f'== {s} (check: residual SD 0.005/0.005 reproduces previous A\'+OC: {ok}; finite {res[s]["finite"]})')
-    for name, x in (('STEP', stp), ("A'+OC, JT residual SD 0.005/0.005", old), (f"A'+OC, STEP residual SD {STEP_R[0]}/{STEP_R[1]}", a)):
+    for name, x in (('STEP', stp), (f"A'+OC, STEP residual SD {STEP_R[0]}/{STEP_R[1]}", a)):
         L.append(f'  {name:40s} e_p ' + '/'.join(f'{v:.4f}' for v in x[:, 0]) + '  e_v ' + '/'.join(f'{v:.4f}' for v in x[:, 1])

### nbody/nbody_common.py (path; source <workspace>\Diagnostics\stage1_final_261015\nbody_compare\nbody_common.py)
--- <workspace>\Diagnostics\stage1_final_261015\nbody_compare\nbody_common.py
+++ nbody/nbody_common.py
@@ -10,3 +10,3 @@
 from scipy.spatial.transform import Rotation as R
-CODE = HERE.parent / 'code_s4f'; REVIEW = HERE.parent.parent / 'review_261020'
+CODE = HERE.parent / 'step'; REVIEW = HERE.parent / 'review'  # RELEASE: paths changed for this repository
 sys.path.insert(0, str(HERE)); sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / 'src')); sys.path.append(str(REVIEW))
```

Data: the records that ../prepare_data.py generates are not stored in this repository (see section4_README.md, "Data"); the SOLVER.json files of the paper records are in expected/.
