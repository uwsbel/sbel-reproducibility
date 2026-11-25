"""Read the printed numbers of the manuscript tables from the LaTeX sources.

Each table is found by its label. Each data row is a line of the tabular
body that ends with a double backslash. A row gives a list of numeric tokens,
read from left to right:

  1.234        one number with three decimals
  4.14--4.35   two numbers (a range)
  119/144      two integers
  $-0.132$     one number
  $2.2\\times10^{-2}$   one number, 0.022, with one mantissa decimal
  48 (ii)      the integer 48 (the event label is not a number)
  $0.003$ m    the number 0.003 (the unit is not a number)

A cell that starts with a letter or a symbol name gives no token, and so
does a cell with several numbers mixed with text. The functions use the standard library only.
"""
import json
import re
from pathlib import Path

NUM = r"[-+]?\d+(?:\.\d+)?"
SCI = re.compile(r"(" + NUM + r")\s*\\times\s*10\^\{?\s*(-?\d+)\s*\}?")
PLAIN = re.compile(NUM)

SECTION_FILES = [
    "examples.tex", "discussion.tex", "appendix_puma.tex",
    "appendix_vehicle_settings.tex", "appendix_vehicle_estimates.tex",
]


def _strip_markup(cell):
    """Remove LaTeX markup that carries no table value."""
    c = cell
    c = re.sub(r"\\(?:label|ref|eqref|cite)\{[^}]*\}", " ", c)
    c = re.sub(r"\$\^\{?\\(?:ast|dagger)\}?\$", " ", c)       # selection marks
    c = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", " ", c)       # keep the content
    c = re.sub(r"\\(?:cmidrule|cline)(?:\([^)]*\))?\{[^}]*\}", " ", c)
    c = re.sub(r"\\(?:midrule|toprule|bottomrule|hline)", " ", c)
    c = c.replace("\\,", "").replace("~", " ")
    c = re.sub(r"\\(?:mathrm|text|emph|textit|textbf|hat|mathit)\b", " ", c)
    c = c.replace("\\%", " ")
    return c


def _text_only(cell):
    """True for a cell that names something (a symbol, a motion, a label)."""
    c = _strip_markup(cell)
    c = SCI.sub(" ", c)
    c = PLAIN.sub(" ", c)
    c = re.sub(r"[\s$(){}\[\]\\/,.;:&^+-]|--", "", c)
    return len(c) > 0


def tokens_of_cell(cell):
    """Numeric tokens of one cell. A cell with letters gives no token,
    except the patterns 'number (label)' and 'number unit' handled below."""
    c = _strip_markup(cell)
    out = []
    # scientific notation first
    pos = 0
    pieces = []
    for m in SCI.finditer(c):
        pieces.append((c[pos:m.start()], None))
        pieces.append((None, m))
        pos = m.end()
    pieces.append((c[pos:], None))
    for text, sci in pieces:
        if sci is not None:
            mant, expo = sci.group(1), int(sci.group(2))
            dec = len(mant.split(".")[1]) if "." in mant else 0
            out.append({"printed": sci.group(0), "value": float(mant) * 10.0 ** expo,
                        "tol": 0.5 * 10.0 ** (expo - dec)})
            continue
        t = text.replace("--", " ").replace("/", " ")
        t = re.sub(r"(?<=\d)-(?=\d)", " ", t)
        for m in PLAIN.finditer(t):
            s = m.group(0)
            dec = len(s.split(".")[1]) if "." in s else 0
            out.append({"printed": s, "value": float(s), "tol": 0.5 * 10.0 ** (-dec)})
    return out


def _is_name_cell(cell):
    """Cells such as '$I_{2y}$', 'H-D04, P-D01', 'HF4', '$m_3c_{3x}$' name a row.
    Their digits are not table values."""
    c = _strip_markup(cell)
    letters = re.sub(r"\\times|\\ast", "", c)
    letters = SCI.sub(" ", letters)
    return re.search(r"[A-Za-z]", letters) is not None


def rows_of_table(tex, label):
    """Data rows of the tabular that belongs to the given label."""
    i = tex.find("\\label{" + label + "}")
    if i < 0:
        return None
    b = tex.find("\\begin{tabular}", i)
    e = tex.find("\\end{tabular}", b)
    body = tex[b:e]
    body = body[body.find("\n") + 1:]
    rows = []
    for raw in re.split(r"\\\\", body):
        line = " ".join(x for x in raw.split("\n") if not x.strip().startswith("%"))
        if "&" not in line:
            continue
        cells = line.split("&")
        toks = []
        for k, cell in enumerate(cells):
            if _is_name_cell(cell):
                # A leading number followed by text without digits, such as
                # '48 (ii)', '20 (val.)', '$0.003$ m', '$0.5^\circ$': keep the number.
                plain = _strip_markup(cell).replace("$", "")
                m = re.match(r"\s*(" + NUM + r")([^\d]*)$", plain)
                if m:
                    toks.extend(tokens_of_cell(m.group(1)))
                continue
            toks.extend(tokens_of_cell(cell))
        first = re.sub(r"\s+", " ", _strip_markup(cells[0])).strip()
        if toks:
            rows.append({"first_cell": first, "tokens": toks})
    return rows


def read_tables(manuscript_dir):
    """All labelled tables of the manuscript: label -> list of rows."""
    sec = Path(manuscript_dir) / "sections"
    out = {}
    for name in SECTION_FILES:
        tex = (sec / name).read_text(encoding="utf-8")
        for label in re.findall(r"\\label\{(tab:[^}]+)\}", tex):
            rows = rows_of_table(tex, label)
            if rows is not None:
                out[label] = {"file": name, "rows": rows}
    return out


def read_conclusions(manuscript_dir):
    """The numbers of the conclusions, by pattern."""
    tex = (Path(manuscript_dir) / "sections" / "conclusions.tex").read_text(encoding="utf-8")
    pats = {
        "puma_share_removed_pct": r"removes (\d+)\\% of the test cost",
        "polaris_share_removed_pct_range": r"(\d+)--(\d+)\\% for Polaris",
        "hmmwv_share_removed_pct_range": r"(\d+)--(\d+)\\% for the HMMWV",
        "vehicle_mass_error_pct_bound": r"within (\d+(?:\.\d+)?)\\% for every teacher",
        "vehicle_comparisons": r"in (\d+) of the (\d+) vehicle comparisons",
        "vehicle_comparisons_other_within_pct": r"vehicle comparisons and is within (\d+)\\% of it in the other",
    }
    out = {}
    for key, pat in pats.items():
        m = re.search(pat, tex)
        out[key] = None if m is None else [x for x in m.groups()]
    return out


def main():
    import sys
    man = sys.argv[1]
    dest = Path(sys.argv[2])
    data = {"tables": read_tables(man), "conclusions": read_conclusions(man)}
    dest.write_text(json.dumps(data, indent=1), encoding="utf-8")
    n = sum(len(r["tokens"]) for t in data["tables"].values() for r in t["rows"])
    print("tables", len(data["tables"]), "numeric tokens", n)
    for lab, t in data["tables"].items():
        print(f"{lab:36s} rows {len(t['rows']):3d} tokens {sum(len(r['tokens']) for r in t['rows']):4d}")
    print(data["conclusions"])


if __name__ == "__main__":
    main()
