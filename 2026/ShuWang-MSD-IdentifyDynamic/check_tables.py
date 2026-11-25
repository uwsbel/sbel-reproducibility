"""Compare the recomputed table values with the printed values of the paper.

Usage:
    python check_tables.py                      all cases, printed values from paper_values.json
    python check_tables.py --case puma          one case
    python check_tables.py --signals            also recompute the costs from the stored signals

Each case folder has a module tables.py with
    build_tables(case_dir)      -> {table label: {row index: [numbers]}}
    build_conclusions(case_dir) -> {key: value}
The row index counts the data rows of the printed table from 0. The numbers
are not rounded. A recomputed number x matches a printed number p with d
decimals if abs(x - p) <= 0.5 * 10**(-d). Integers must be equal.
The two tables of the observability map print their values to three
significant digits (for example 1450 for 1453.98). There the tolerance is half
a unit of the third significant digit, or of the last printed decimal if that
is coarser.
The script needs numpy only. It runs no simulation.
"""
import math
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
CASES = ("puma", "hmmwv", "polaris")
THREE_DIGIT_TABLES = {"tab:puma_map_snr", "tab:vehicle_map_snr"}   # printed to three significant digits


def load_module(path, name):
    """Load a module of one case folder. The folder stays on the import path
    until forget_case() is called, so the module can import its neighbours."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def forget_case(case_dir):
    """Drop the modules of one case, so that the next case imports its own."""
    case_dir = str(Path(case_dir).resolve())
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and str(Path(f).resolve()).startswith(case_dir):
            del sys.modules[name]
    while case_dir in sys.path:
        sys.path.remove(case_dir)


def tolerance(tok, label):
    tol = tok["tol"]
    if label in THREE_DIGIT_TABLES and tok["value"] != 0:
        tol = max(tol, 0.5 * 10.0 ** (math.floor(math.log10(abs(tok["value"]))) - 2))
    return tol


def matches(x, tok, label=None):
    slack = 1e-9 * max(1.0, abs(tok["value"]))
    return abs(float(x) - tok["value"]) <= tolerance(tok, label) + slack


def check_tables(paper, cases, with_signals=False):
    covered = {}
    differ = []
    n_ok = 0
    concl = {}
    sig = []
    for case in cases:
        mod = load_module(HERE / case / "tables.py", case + "_tables")
        tabs = mod.build_tables(HERE / case)
        concl[case] = mod.build_conclusions(HERE / case)
        if with_signals:
            smod = load_module(HERE / case / "signals.py", case + "_signals")
            sig.extend(dict(case=case, **r) for r in smod.check_signals(HERE / case))
        forget_case(HERE / case)
        for label, rows in tabs.items():
            if label not in paper["tables"]:
                differ.append({"case": case, "table": label, "problem": "label not in the paper"})
                continue
            prow = paper["tables"][label]["rows"]
            for idx, values in rows.items():
                idx = int(idx)
                key = (label, idx)
                if key in covered:
                    differ.append({"case": case, "table": label, "row": idx,
                                  "problem": "row given by two cases"})
                    continue
                covered[key] = case
                if idx >= len(prow):
                    differ.append({"case": case, "table": label, "row": idx, "problem": "no such row"})
                    continue
                toks = prow[idx]["tokens"]
                if len(values) != len(toks):
                    differ.append({"case": case, "table": label, "row": idx,
                                  "first_cell": prow[idx]["first_cell"],
                                  "problem": f"{len(values)} values for {len(toks)} printed numbers"})
                    continue
                for col, (x, tok) in enumerate(zip(values, toks)):
                    if matches(x, tok, label):
                        n_ok += 1
                    else:
                        differ.append({"case": case, "table": label, "row": idx, "column": col,
                                      "first_cell": prow[idx]["first_cell"],
                                      "printed": tok["printed"], "recomputed": float(x)})
    missing = []
    for label, tab in paper["tables"].items():
        for idx, row in enumerate(tab["rows"]):
            if (label, idx) not in covered:
                missing.append({"table": label, "row": idx, "first_cell": row["first_cell"],
                                "numbers": len(row["tokens"])})
    return n_ok, differ, missing, concl, sig


def check_conclusions(paper, c):
    """Numbers of the conclusions. Needs all three cases."""
    out = []
    if set(c) != set(CASES):
        return out
    p = paper["conclusions"]

    def rounds_to(x, printed):
        dec = len(printed.split(".")[1]) if "." in printed else 0
        return abs(float(x) - float(printed)) <= 0.5 * 10.0 ** (-dec) + 1e-9

    out.append(("puma share removed (%)", p["puma_share_removed_pct"],
                [c["puma"]["share_removed_pct"]],
                rounds_to(c["puma"]["share_removed_pct"], p["puma_share_removed_pct"][0])))
    for case, key in (("polaris", "polaris_share_removed_pct_range"),
                      ("hmmwv", "hmmwv_share_removed_pct_range")):
        lo, hi = c[case]["share_removed_pct_range"]
        out.append((f"{case} share removed, range over the teachers (%)", p[key], [lo, hi],
                    rounds_to(lo, p[key][0]) and rounds_to(hi, p[key][1])))
    worst = max(c["hmmwv"]["mass_error_pct_max"], c["polaris"]["mass_error_pct_max"])
    out.append(("vehicle mass error, largest over all teachers and runs (%)",
                p["vehicle_mass_error_pct_bound"], [worst],
                rounds_to(worst, p["vehicle_mass_error_pct_bound"][0])))
    # test cost of the run with the lowest validation cost / lowest test cost of its four runs
    ratios = list(c["hmmwv"]["selected_over_lowest_test_cost"]) + \
        list(c["polaris"]["selected_over_lowest_test_cost"])
    lowest = [r <= 1.0 + 1e-12 for r in ratios]
    out.append(("vehicle comparisons: the selected run has the lowest test cost",
                p["vehicle_comparisons"], [sum(lowest), len(ratios)],
                [str(sum(lowest)), str(len(ratios))] == list(p["vehicle_comparisons"])))
    others = [100.0 * (r - 1.0) for r, low in zip(ratios, lowest) if not low]
    bound = float(p["vehicle_comparisons_other_within_pct"][0])
    out.append(("vehicle comparisons: excess of the other selected runs over the lowest test cost (%)",
                p["vehicle_comparisons_other_within_pct"], others, all(x <= bound for x in others)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=CASES)
    ap.add_argument("--paper-values", default=str(HERE / "paper_values.json"))
    ap.add_argument("--signals", action="store_true")
    ap.add_argument("--report", default=str(HERE / "check_report.json"))
    a = ap.parse_args()
    cases = (a.case,) if a.case else CASES
    paper = json.loads(Path(a.paper_values).read_text(encoding="utf-8"))
    n_ok, differ, missing, concl_values, sig = check_tables(paper, cases, a.signals)
    if a.case:
        missing = []      # other cases own the remaining rows
    concl = check_conclusions(paper, concl_values)
    sig_diff = [r for r in sig if not r["ok"]]
    concl_diff = [r for r in concl if not r[3]]
    print(f"table numbers that match: {n_ok}")
    print(f"table numbers that do not match: {len(differ)}")
    for f in differ:
        print("  ", f)
    print(f"printed rows without a recomputed row: {len(missing)}")
    for m in missing:
        print("  ", m)
    for name, printed, got, ok in concl:
        print(f"conclusions: {name}: printed {printed}, recomputed {got}: {'ok' if ok else 'DIFFERENT'}")
    if a.signals:
        print(f"values recomputed from the signals: {len(sig)}, different: {len(sig_diff)}")
        for r in sig_diff:
            print("  ", r)
    passed = not differ and not missing and not concl_diff and not sig_diff
    Path(a.report).write_text(json.dumps({
        "cases": list(cases), "matching_numbers": n_ok, "different": differ,
        "rows_not_covered": missing,
        "conclusions": [{"name": n, "printed": p, "recomputed": g, "ok": ok} for n, p, g, ok in concl],
        "signals": {"values": len(sig), "different": sig_diff},
        "result": "PASS" if passed else "FAIL"}, indent=1), encoding="utf-8")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
