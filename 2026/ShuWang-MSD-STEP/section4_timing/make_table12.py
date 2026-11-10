"""Table 12 (tab:efficiency) and the Stage I times of Section 4.5, from parity/out/paper/timing/TIMING_P.json
(parity/timing_paper.py: paper draw, Motion 1, biased measurements, initial attitude SD 1 deg, one process, one BLAS
thread, ten alternated full-record runs per filter after warm-up; Stage I three runs).

Timing depends on the machine. Absolute times (ms per sample, Stage I seconds) are reported but not judged. The ratios to
STEP and the total-time ratios match if they are within 20 % of the paper value, the exponents if within 0.15 (the paper
notes that ratios differed by up to 13 % between repeated runs on the same machine). Internal names: STEP-S = STEP
Stage II (compiled), JT-BHS = JT-BHSF.

Usage: python make_table12.py [--source replay|expected] [--paper ../paper_values_ch4.json]
"""
import numpy as np
import tables_common as tc

Ns = (3, 5, 10)
NAMES = [('JT-B', 'JT-B'), ('JT-BS', 'JT-BS'), ('JT-BHSF', 'JT-BHS'), ('STEP', 'STEP-S')]


def rules(key):
    if key.endswith('ms per sample') or key.endswith('|s'):
        return ('info',)
    if key.endswith('exponent'):
        return ('abs', 0.15)
    return ('rel', 0.20)


def cells(b):
    T = tc.jload(b / 'parity' / 'out' / 'paper' / 'timing' / 'TIMING_P.json')
    R = T['by_N']
    v, w, info = {}, {}, {}
    for lab, k in NAMES:
        for N in Ns:
            r = R[str(N)]
            v[f'{lab}|N={N}|ms per sample'] = r[k]['median_ms']
            v[f'{lab}|N={N}|ratio to STEP'] = r[k]['median_ms'] / r['STEP-S']['median_ms']
            info[f'{lab}|N={N}|IQR / median'] = (r[k]['q3_ms'] - r[k]['q1_ms']) / r[k]['median_ms']
        v[f'{lab}|exponent'] = float(np.polyfit(np.log(Ns), np.log([R[str(N)][k]['median_ms'] for N in Ns]), 1)[0])
    par = []
    for N in Ns:
        r = R[str(N)]
        n = r['samples']
        s1 = r['stage1_s']['median']
        pp = r['parents_stage2_10s_s']['median']
        par.append(pp)
        v[f'Stage I|N={N}|s'] = s1
        t = {k: r[k]['median_ms'] * n / 1e3 for _, k in NAMES}
        total_step = s1 + t['STEP-S']
        if N in (3, 10):
            w[f'total time ratio|JT-BHSF|N={N}'] = (s1 + pp + t['JT-BHS']) / total_step
            w[f'total time ratio|JT-BS|N={N}'] = (s1 + pp + t['JT-BS']) / total_step
        info[f'N={N}|STEP-S vs reference implementation, largest state difference'] = r['STEP_S_vs_reference_max']
        info[f'N={N}|share of the 1 ms sample interval used: STEP, JT-BS'] = (r['STEP-S']['median_ms'], r['JT-BS']['median_ms'])
    w["parents' Stage II on [0, 10] s|min|s"] = min(par)
    w["parents' Stage II on [0, 10] s|max|s"] = max(par)
    w.update({k: x for k, x in v.items() if k.startswith('Stage I')})
    return {k: x for k, x in v.items() if not k.startswith('Stage I')}, w, info, T


def show(v, w, info, T):
    print(f"Timing run {T.get('start')} to {T.get('end')}; software {T.get('software')}; p0_deg {T.get('p0_deg')}")
    print('Table 12: median ms per sample (ratio to STEP Stage II), exponent')
    for lab, _ in NAMES:
        print(f'  {lab:8s} ' + '  '.join(f"{v[f'{lab}|N={N}|ms per sample']:.3f} ({v[f'{lab}|N={N}|ratio to STEP']:.1f})" for N in Ns)
              + f"   {v[f'{lab}|exponent']:.2f}")
    print('Section 4.5 text:')
    for k, x in w.items():
        print(f'  {k}: {x:.2f}')
    for k, x in info.items():
        print(f'  {k}: {x}')


if __name__ == '__main__':
    a = tc.parse_args(__doc__)
    v, w, info, T = cells(tc.base(a.source))
    show(v, w, info, T)
    ok1 = tc.compare('tab:efficiency', v, a, rules)
    ok2 = tc.compare('sec:computational_cost text', w, a, rules)
    raise SystemExit(0 if ok1 is not False and ok2 is not False else 1)
