"""Runtime signature digest for chrono_code.utils public API.

Scans the symbols in `chrono_code.utils.__all__` once per process and exposes:

- `build_reference_block()` — a markdown block injected into the codegen system
  prompt, so the code-generation agent sees the signatures of every public util
  without needing a round-trip tool call.
- `get_signatures()` — a `{fqname: inspect.Signature}` mapping consumed by the
  AST call validator to hard-fail on arg/kwarg mismatches.

The digest is derived from the module at import time (no hand-maintained .md),
so it never drifts from the real code.
"""

from __future__ import annotations

import inspect
import logging
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


_UTILS_PACKAGE = "chrono_code.utils"


class _UtilsIndex:
    """Cached lookup of public utils symbols."""

    _built: bool = False
    # Canonical name -> (qualified_import_path, Signature or None, doc_first_line, kind)
    _entries: Dict[str, Tuple[str, Optional[inspect.Signature], str, str]] = {}

    @classmethod
    def build(cls, force: bool = False) -> None:
        if cls._built and not force:
            return
        cls._entries.clear()

        try:
            utils_mod = import_module(_UTILS_PACKAGE)
        except Exception as exc:
            logger.warning("Failed to import %s for signature digest: %s", _UTILS_PACKAGE, exc)
            cls._built = True
            return

        public_names = list(getattr(utils_mod, "__all__", []) or [])
        lazy_map: Dict[str, Tuple[str, str]] = getattr(utils_mod, "_LAZY_IMPORTS", {}) or {}

        for name in public_names:
            try:
                obj = getattr(utils_mod, name)
            except Exception as exc:
                logger.debug("Could not resolve %s.%s: %s", _UTILS_PACKAGE, name, exc)
                continue

            module_path, attr_name = lazy_map.get(
                name, (getattr(obj, "__module__", _UTILS_PACKAGE), name)
            )
            sig = _safe_signature(obj)
            doc = _first_doc_line(obj)
            kind = _classify(obj)
            cls._entries[name] = (f"{module_path}.{attr_name}", sig, doc, kind)

        cls._built = True

    @classmethod
    def entries(
        cls,
    ) -> Dict[str, Tuple[str, Optional[inspect.Signature], str, str]]:
        cls.build()
        return dict(cls._entries)


def _safe_signature(obj: Any) -> Optional[inspect.Signature]:
    """Return inspect.Signature if obj is callable with introspectable sig."""
    if not callable(obj):
        return None
    target = obj
    if inspect.isclass(obj):
        init = getattr(obj, "__init__", None)
        if init is None:
            return None
        target = init
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return None
    # For __init__ signatures, drop the implicit `self` parameter so validator
    # binds user-supplied args directly against ClassName(...) call sites.
    if inspect.isclass(obj):
        params = [p for n, p in sig.parameters.items() if n != "self"]
        sig = sig.replace(parameters=params)
    return sig


def _first_doc_line(obj: Any) -> str:
    doc = inspect.getdoc(obj) or ""
    first = doc.strip().split("\n", 1)[0].strip()
    if len(first) > 160:
        first = first[:157] + "..."
    return first


def _classify(obj: Any) -> str:
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "func"
    return "const"


def get_signatures() -> Dict[str, inspect.Signature]:
    """Return {canonical_name: Signature} for callables only.

    Consumed by the AST call validator: lookup is by the **local name** the
    agent imported (e.g. `setup_preview_camera`), because imports are resolved
    to canonical names when parsing.
    """
    out: Dict[str, inspect.Signature] = {}
    for name, (_fqname, sig, _doc, _kind) in _UtilsIndex.entries().items():
        if sig is not None:
            out[name] = sig
    return out


def get_fq_signatures() -> Dict[str, inspect.Signature]:
    """Return {fully_qualified_name: Signature} for callables.

    Used when the agent imports via `import chrono_code.utils.scene_assets`
    and then calls `chrono_code.utils.scene_assets.add_visual_assets(...)`.
    """
    out: Dict[str, inspect.Signature] = {}
    for _name, (fqname, sig, _doc, _kind) in _UtilsIndex.entries().items():
        if sig is not None:
            out[fqname] = sig
    return out


def build_reference_block() -> str:
    """Render the utils signature digest as a markdown block for system prompt.

    Format is compact (one line per symbol) so the whole block stays well
    under ~2k tokens for the current ~23-symbol public surface.
    """
    entries = _UtilsIndex.entries()
    if not entries:
        return ""

    # Group by submodule (everything after `chrono_code.utils.`) so related
    # symbols cluster together.
    by_submodule: Dict[str, List[Tuple[str, str, Optional[inspect.Signature], str, str]]] = {}
    for name, (fqname, sig, doc, kind) in entries.items():
        submodule = fqname.rsplit(".", 1)[0]
        short = submodule.replace(f"{_UTILS_PACKAGE}.", "")
        by_submodule.setdefault(short or "<root>", []).append((name, fqname, sig, doc, kind))

    lines: List[str] = []
    lines.append("UTILS API REFERENCE (chrono_code.utils — always prefer these over reinventing):")
    lines.append(
        "Import: `from chrono_code.utils import <name>` or "
        "`from chrono_code.utils.<submodule> import <name>`."
    )
    lines.append("")

    for submodule in sorted(by_submodule):
        lines.append(f"### {_UTILS_PACKAGE}.{submodule}")
        for name, fqname, sig, doc, kind in sorted(by_submodule[submodule]):
            if kind == "class":
                rendered = f"class {name}{sig}" if sig is not None else f"class {name}"
            elif kind == "func":
                rendered = f"{name}{sig}" if sig is not None else f"{name}(...)"
            else:
                rendered = f"{name} = <const>"
            line = f"- `{rendered}`"
            if doc:
                line += f" — {doc}"
            lines.append(line)
        lines.append("")

    lines.append(
        "Arg/kwarg names are validated after generation (AST check); mismatches "
        "hard-fail and are fed back for correction."
    )
    return "\n".join(lines).rstrip() + "\n"
