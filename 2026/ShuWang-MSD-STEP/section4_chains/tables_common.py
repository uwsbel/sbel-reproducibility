"""Shared helpers of the Section 4 table scripts: where the outputs are, comparison with the paper, printing.

Every make_table*.py script accepts
  --source replay     outputs of the replay in this folder (default)
  --source expected   saved outputs of the run behind the paper (expected/)
  --paper PATH        paper values (default: ../paper_values_ch4.json)
A paper cell matches when the recomputed value, rounded to the printed number of decimals, equals the printed value
(the rule of check_replay.py). Cells marked 'true'/'false' are logical statements of the text.
"""
import json
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent


def parse_args(description, extra=None):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--source', choices=['replay', 'expected'], default='replay')
    p.add_argument('--paper', default=str(PKG / 'paper_values_ch4.json'))
    if extra:
        extra(p)
    return p.parse_args()


def base(source):
    """Folder under which review/, nbody/ and parity/ outputs are found."""
    return HERE if source == 'replay' else HERE / 'expected'


def jload(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def decimals(printed):
    return len(printed.split('.')[1]) if '.' in printed else 0


def judge(printed, value, rule=None):
    """(match or None, shown). rule None: rounding to the printed decimals; ('rel', tol) or ('abs', tol): tolerance;
    ('info',): reported only (machine-dependent)."""
    if printed in ('true', 'false'):
        return bool(value) == (printed == 'true'), str(bool(value)).lower()
    d = decimals(printed)
    shown = f'{float(value):.{d + 2}f}'
    if rule is None:
        return round(float(value), d) == float(printed), shown
    if rule[0] == 'info':
        return None, shown
    dev = abs(float(value) - float(printed))
    return (dev <= rule[1] * abs(float(printed)) if rule[0] == 'rel' else dev <= rule[1]), shown


def compare(label, cells, args, rules=None):
    """Compare the recomputed cells with every paper cell of one table. Writes tables/<label>_<source>.json.
    Returns True or False, or None if the paper file is missing."""
    pf = Path(args.paper)
    out_dir = HERE / 'tables'
    out_dir.mkdir(exist_ok=True)
    safe = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in label)
    if not pf.exists():
        print(f'{pf} not found: values printed without comparison')
        (out_dir / f'{safe}_{args.source}.json').write_text(json.dumps(dict(label=label, source=args.source, cells=cells), indent=1))
        return None
    paper = jload(pf)[label]
    rows, ok, n_info = [], True, 0
    for key, printed in paper['values'].items():
        if key not in cells:
            rows.append(dict(cell=key, paper=printed, replay=None, match=False))
            ok = False
            continue
        rule = rules(key) if rules else None
        match, shown = judge(printed, cells[key], rule)
        if match is None:
            n_info += 1
        else:
            ok &= match
        rows.append(dict(cell=key, paper=printed, replay=shown, match=match, rule=None if rule is None else list(rule)))
    bad = [r for r in rows if r['match'] is False]
    print(f"\n{paper['table']} ({label}): {len(rows)} paper cells, {len(bad)} mismatched, {n_info} reported only")
    for r in bad:
        print(f"  MISMATCH  {r['cell']}: paper {r['paper']}, {args.source} {r['replay']}")
    print(f"{paper['table']}: {'PASS' if ok else 'FAIL'} (source: {args.source})")
    (out_dir / f'{safe}_{args.source}.json').write_text(json.dumps(dict(label=label, source=args.source, all_match=ok, cells=rows), indent=1))
    return ok


def fmt(x, d):
    return '---' if x is None else f'{x:.{d}f}'


def g3(x):
    """Number format of Tables 7 and 8: 4 decimals below 0.1, else 3."""
    return f'{x:.4f}' if x < 0.1 else f'{x:.3f}'
