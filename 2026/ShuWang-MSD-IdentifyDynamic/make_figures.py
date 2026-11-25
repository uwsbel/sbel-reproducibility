"""Redraw the six result figures of the paper from the files of this repository.

Usage:
    python make_figures.py              all cases, output in figures_out/
    python make_figures.py --case puma  one case
Each case folder has a module figures.py with make_figures(case_dir, out_dir).
"""
import argparse
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
CASES = ("puma", "hmmwv", "polaris")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=CASES)
    ap.add_argument("--out", default=str(HERE / "figures_out"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for case in ((a.case,) if a.case else CASES):
        path = HERE / case / "figures.py"
        spec = importlib.util.spec_from_file_location(case + "_figures", path)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(path.parent))
        try:
            spec.loader.exec_module(mod)
            mod.make_figures(HERE / case, out)
        finally:
            sys.path.remove(str(path.parent))
            # drop the modules of this case, so that the next case imports its own
            for name, m in list(sys.modules.items()):
                f = getattr(m, "__file__", None)
                if f and str(Path(f).resolve()).startswith(str(path.parent.resolve())):
                    del sys.modules[name]
        print("figures of", case, "written to", out)


if __name__ == "__main__":
    main()
