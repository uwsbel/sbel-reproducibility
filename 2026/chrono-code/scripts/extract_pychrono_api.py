"""Extract PyChrono API via Python introspection and write pychrono_api.json.

Mirrors the format of the existing pychrono_api.json so that
scripts/build_api_index.py can consume it unchanged.

Usage:
    python scripts/extract_pychrono_api.py
    python scripts/extract_pychrono_api.py --out data/pychrono_docs/pychrono_api.json
    python scripts/extract_pychrono_api.py --dry-run   # print stats only

The script auto-detects the PyChrono share/ path from the conda environment
and adds it to sys.path if needed.
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import pydoc
import re
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------------------
# PyChrono module list (each maps to a .py + _<name>.so in share/chrono/python)
# ---------------------------------------------------------------------------
CHRONO_MODULES = [
    "core",
    "vehicle",
    "fea",
    "sensor",
    "robot",
    "fsi",
    "irrlicht",
    "postprocess",
    "parsers",
    "pardisomkl",
    "cascade",
]

# These names appear in every SWIG module but are boilerplate, not real API.
_SWIG_BOILERPLATE = frozenset({
    "thisown",
    "this",
    "acquire",
    "disown",
    "next",
    "previous",
})

# Regex to pull the first line of a SWIG docstring (the Python-facing signature).
_FIRST_LINE_RE = re.compile(r"^([^\n]+)")


def _find_pychrono_path() -> Path | None:
    """Look for share/chrono/python in the active conda prefix."""
    import os

    candidates: list[Path] = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "share" / "chrono" / "python")

    # Fallback: walk up from current interpreter
    py_exec = Path(sys.executable).resolve()
    for parent in py_exec.parents:
        candidate = parent / "share" / "chrono" / "python"
        if candidate.exists():
            candidates.append(candidate)

    for c in candidates:
        if (c / "pychrono" / "_core.so").exists() or (c / "pychrono" / "core.py").exists():
            return c

    return None


def _ensure_pychrono_on_path() -> None:
    chrono_path = _find_pychrono_path()
    if chrono_path is None:
        print(
            "WARNING: could not auto-detect PyChrono share path. "
            "Add it to PYTHONPATH manually.",
            file=sys.stderr,
        )
        return
    path_str = str(chrono_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        print(f"Added to sys.path: {path_str}")


def _capture_help(obj: object) -> str:
    """Return pydoc help text for obj as a string."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buf
        pydoc.help(obj)
    except Exception:
        pass
    finally:
        sys.stdout = old_stdout
    return buf.getvalue().strip()


def _get_signature(obj: object, name: str) -> str:
    """Best-effort signature string from inspect or docstring."""
    try:
        sig = inspect.signature(obj)  # type: ignore[arg-type]
        return f"{name}{sig}"
    except (TypeError, ValueError):
        pass
    doc = getattr(obj, "__doc__", "") or ""
    m = _FIRST_LINE_RE.match(doc.strip())
    if m:
        first = m.group(1).strip()
        if "(" in first:
            return first
    return f"{name}(*args)"


def _get_constructor_overloads(cls: type) -> list[str]:
    """Extract constructor overload signatures from __init__.__doc__."""
    init = getattr(cls, "__init__", None)
    if init is None:
        return []
    doc = getattr(init, "__doc__", "") or ""
    # SWIG __init__ docs look like:
    #   __init__(self) -> ChBody
    #   __init__(self, other: ChBody const &) -> ChBody
    overloads = []
    for line in doc.splitlines():
        line = line.strip()
        if line.startswith("__init__(") or line.startswith(f"{cls.__name__}("):
            # Normalise __init__ → ClassName
            line = re.sub(r"^__init__", cls.__name__, line)
            # Strip return type annotation " -> Type"
            line = re.sub(r"\s*->\s*\S+\s*$", "", line).strip()
            if line not in overloads:
                overloads.append(line)
    return overloads


def _is_public_method(name: str, obj: object) -> bool:
    if name.startswith("_"):
        return False
    if name in _SWIG_BOILERPLATE:
        return False
    if isinstance(obj, (int, float, str, bytes, bool)):
        return False  # class-level constants — handled separately
    return callable(obj) or isinstance(obj, property)


def _extract_method(cls: type, name: str, obj: object) -> dict:
    doc = ""
    if isinstance(obj, property):
        raw_doc = getattr(obj.fget, "__doc__", "") or ""
    else:
        raw_doc = getattr(obj, "__doc__", "") or ""

    # First line = Python-facing signature (SWIG puts it there)
    lines = raw_doc.strip().splitlines()
    if lines:
        first = lines[0].strip()
        sig = first if "(" in first else f"{name}(self)"
        # Description = rest after the overload block
        desc_lines = []
        past_sigs = False
        for l in lines[1:]:
            stripped = l.strip()
            if not past_sigs and (not stripped or "(" in stripped[:40]):
                continue
            past_sigs = True
            desc_lines.append(stripped)
        description = " ".join(desc_lines).strip()[:200]
    else:
        sig = f"{name}(self)"
        description = ""

    # wrapper = full help text (includes C++ type hints for SWIG methods)
    wrapper = _capture_help(obj) if not isinstance(obj, property) else ""

    return {
        "signature": sig,
        "description": description,
        "wrapper": wrapper,
    }


def _extract_class(module_name: str, name: str, cls: type) -> dict:
    """Build a class entry matching the existing JSON schema."""
    # Class-level description from __doc__
    raw_doc = getattr(cls, "__doc__", "") or ""
    lines = raw_doc.strip().splitlines()
    description = lines[0].strip() if lines else ""

    # Constructor signature
    sig = _get_signature(cls, name)
    overloads = _get_constructor_overloads(cls)

    # Methods
    methods: dict[str, dict] = {}
    for attr_name in dir(cls):
        try:
            attr = getattr(cls, attr_name)
        except Exception:
            continue
        if not _is_public_method(attr_name, attr):
            continue
        methods[attr_name] = _extract_method(cls, attr_name, attr)

    entry: dict = {
        "type": "class",
        "signature": sig,
        "description": description,
    }
    if overloads:
        entry["constructor_overloads"] = overloads
    if methods:
        entry["methods"] = methods
    return entry


def _extract_module_function(module_name: str, name: str, fn: object) -> dict:
    """Build a module-level function entry."""
    sig = _get_signature(fn, name)
    raw_doc = getattr(fn, "__doc__", "") or ""
    lines = raw_doc.strip().splitlines()
    desc_lines = [l.strip() for l in lines if l.strip() and "(" not in l[:40]]
    description = " ".join(desc_lines[:3]).strip()[:200]
    wrapper = _capture_help(fn)
    return {
        "type": "module_function",
        "signature": sig,
        "description": description,
        "methods": {
            name: {
                "signature": sig,
                "description": description,
                "wrapper": wrapper,
            }
        },
    }


def extract_module(module_name: str) -> dict[str, dict]:
    """Import pychrono.<module_name> and extract all classes and functions."""
    full_name = f"pychrono.{module_name}"
    try:
        mod = __import__(full_name, fromlist=[module_name])
    except ImportError as exc:
        print(f"  SKIP {full_name}: {exc}", file=sys.stderr)
        return {}

    entries: dict[str, dict] = {}
    for attr_name in dir(mod):
        if attr_name.startswith("_"):
            continue
        try:
            obj = getattr(mod, attr_name)
        except Exception:
            continue

        if inspect.isclass(obj):
            entries[attr_name] = _extract_class(module_name, attr_name, obj)
        elif callable(obj) and not isinstance(obj, types.ModuleType):
            entries[attr_name] = _extract_module_function(module_name, attr_name, obj)
        # Skip plain module-level constants (ints, strings) — they add noise
        # but little value for the RAG use case.

    return entries


def build_api(modules: list[str]) -> dict:
    data: dict = {}
    total_classes = 0
    total_methods = 0
    total_funcs = 0

    for mod_name in modules:
        print(f"Extracting pychrono.{mod_name}...")
        entries = extract_module(mod_name)
        data[mod_name] = entries

        n_cls = sum(1 for v in entries.values() if v.get("type") == "class")
        n_fn = sum(1 for v in entries.values() if v.get("type") == "module_function")
        n_meth = sum(len(v.get("methods", {})) for v in entries.values() if v.get("type") == "class")
        print(f"  {n_cls} classes, {n_fn} functions, {n_meth} methods")
        total_classes += n_cls
        total_funcs += n_fn
        total_methods += n_meth

    data["_metadata"] = {
        "format": "module_class_method_hierarchy_v1",
        "modules": modules,
        "module_count": len(modules),
        "total_classes": total_classes,
        "total_methods": total_methods,
        "total_module_functions": total_funcs,
        "source": "python_introspection",
        "chrono_share_path": str(_find_pychrono_path() or ""),
    }
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PyChrono API to JSON.")
    parser.add_argument(
        "--out",
        default="data/pychrono_docs/pychrono_api.json",
        help="Output path (default: data/pychrono_docs/pychrono_api.json)",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=CHRONO_MODULES,
        metavar="MODULE",
        help=f"Modules to extract (default: {' '.join(CHRONO_MODULES)})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing output",
    )
    args = parser.parse_args()

    _ensure_pychrono_on_path()

    data = build_api(args.modules)
    meta = data["_metadata"]
    print(
        f"\nTotal: {meta['total_classes']} classes, "
        f"{meta['total_module_functions']} functions, "
        f"{meta['total_methods']} methods"
    )

    if args.dry_run:
        print("(dry-run: not writing output)")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Written to {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
