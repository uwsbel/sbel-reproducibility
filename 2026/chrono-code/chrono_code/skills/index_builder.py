"""Build a lightweight API reference index from SKILL.md files + utils
signatures, for retrieval-augmented injection at codegen time (Tier 3b).

Design (kept deliberately simple — no embeddings / vector DB):

* Skills: regex-scan every ``SKILL.md`` for mentions of PyChrono API
  symbols (``chrono.ChXxx``, ``chronovsg.Yyy``, ``sens.Zzz``,
  ``veh.Ww``) and project utils (``setup_preview_camera``, etc.).
  For each symbol mention, capture the containing paragraph as a
  snippet (clipped to ~300 chars).

* Utils: AST-scan ``chrono_code/utils/*.py`` for public ``def``s.
  Extract ``name``, call signature, and the first line of docstring.

The produced index is a plain ``dict[str, list[IndexEntry]]`` that we
serialize to ``chrono_code/skills/.skill_index.json`` so rebuilds are
cheap (~tens of ms per run; rebuilt on every process start for
simplicity — this is not a hot path).

Used by :mod:`chrono_code.tools.signature_injector` (Step 8) at
``write_file`` / ``edit_file`` time to append relevant snippets to the
tool return so the LLM sees authoritative signatures on the next turn.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# --- Configuration ---------------------------------------------------------

SKILLS_ROOT = Path(__file__).resolve().parent
UTILS_ROOT = Path(__file__).resolve().parents[1] / "utils"
INDEX_CACHE_PATH = SKILLS_ROOT / ".skill_index.json"

# PyChrono modules the codegen emits against. Order matters for regex
# grouping — longer prefixes go first.
CHRONO_MODULE_PREFIXES = ("chronovsg", "chrono_sens", "chrono", "veh", "sens")

# API symbol regex: captures e.g. `chrono.ChBody`, `chronovsg.ChVisualSystemVSG`,
# `sens.ChCameraSensor`, `veh.HMMWV_Full`. Allows trailing attr access.
_API_SYMBOL_RE = re.compile(
    r"\b(?:" + "|".join(CHRONO_MODULE_PREFIXES) + r")\.[A-Z][A-Za-z0-9_]+"
)

# Utility function names we want to detect in user code (set later from
# the utils scan; matched as standalone identifiers).
_UTILS_NAMES_RE_CACHE: Optional[re.Pattern] = None

SNIPPET_MAX_CHARS = 300
MAX_SNIPPETS_PER_SYMBOL = 3


@dataclass
class IndexEntry:
    """One retrieved piece of reference context for an API symbol."""
    source: str            # e.g. "skill:sens/camera" or "util:scene_assets"
    snippet: str           # up to SNIPPET_MAX_CHARS
    kind: str = "skill"    # "skill" | "util"


@dataclass
class UtilEntry:
    """A project utility function's signature + first doc line."""
    module: str     # e.g. "chrono_code.utils.scene_assets"
    name: str       # e.g. "make_contact_material"
    signature: str  # "(friction: float, restitution: float, method: str) -> ChContactMaterial"
    docline: str    # First non-empty docstring line
    source: str = field(init=False)

    def __post_init__(self) -> None:
        self.source = f"util:{self.module.rsplit('.', 1)[-1]}"


# --- Skill scanning --------------------------------------------------------


def _extract_snippet_for_symbol(text: str, symbol: str) -> Optional[str]:
    """Return the paragraph (first ~300 chars) containing ``symbol``.

    Splits the text on blank lines to find the paragraph block; if the
    paragraph is too long, truncates. This is a coarse but robust
    heuristic — SKILL.md files are structured with clear paragraph
    breaks around API documentation.
    """
    idx = text.find(symbol)
    if idx < 0:
        return None
    # Walk backwards to start of paragraph (blank line or doc start)
    para_start = text.rfind("\n\n", 0, idx)
    para_start = para_start + 2 if para_start >= 0 else 0
    # Walk forwards to end of paragraph
    para_end = text.find("\n\n", idx)
    if para_end < 0:
        para_end = len(text)
    snippet = text[para_start:para_end].strip()
    if len(snippet) > SNIPPET_MAX_CHARS:
        snippet = snippet[:SNIPPET_MAX_CHARS - 3] + "..."
    return snippet


def build_skill_index(skills_root: Optional[Path] = None) -> Dict[str, List[IndexEntry]]:
    """Scan every SKILL.md in ``skills_root`` (defaults to the project
    skills directory) and return ``api_symbol → [IndexEntry]``."""
    root = skills_root or SKILLS_ROOT
    index: Dict[str, List[IndexEntry]] = {}

    for skill_md in root.rglob("SKILL.md"):
        try:
            text = skill_md.read_text(encoding="utf-8")
        except OSError:
            continue
        skill_name = str(skill_md.relative_to(root).parent).replace("\\", "/")
        seen_symbols: set = set()
        for match in _API_SYMBOL_RE.finditer(text):
            symbol = match.group(0)
            if symbol in seen_symbols:
                continue  # dedupe within a single skill
            seen_symbols.add(symbol)
            snippet = _extract_snippet_for_symbol(text, symbol)
            if not snippet:
                continue
            entries = index.setdefault(symbol, [])
            if len(entries) >= MAX_SNIPPETS_PER_SYMBOL:
                continue
            entries.append(IndexEntry(
                source=f"skill:{skill_name}",
                snippet=snippet,
                kind="skill",
            ))
    return index


# --- Utils signature scanning ---------------------------------------------


def _format_signature(func: ast.FunctionDef) -> str:
    """Produce a compact single-line signature string."""
    args = []
    positional_kw_only = func.args.posonlyargs + func.args.args + func.args.kwonlyargs
    num_defaults = len(func.args.defaults)
    defaults_for_args = [None] * (len(func.args.args) - num_defaults) + list(func.args.defaults)
    for i, a in enumerate(func.args.posonlyargs + func.args.args):
        piece = a.arg
        if a.annotation is not None:
            piece += f": {ast.unparse(a.annotation)}"
        # Matching default if any
        defaults_index = i - len(func.args.posonlyargs)
        if 0 <= defaults_index < len(defaults_for_args) and defaults_for_args[defaults_index] is not None:
            piece += f" = {ast.unparse(defaults_for_args[defaults_index])}"
        args.append(piece)
    if func.args.vararg:
        args.append("*" + func.args.vararg.arg)
    for i, a in enumerate(func.args.kwonlyargs):
        piece = a.arg
        if a.annotation is not None:
            piece += f": {ast.unparse(a.annotation)}"
        kd = func.args.kw_defaults[i]
        if kd is not None:
            piece += f" = {ast.unparse(kd)}"
        args.append(piece)
    if func.args.kwarg:
        args.append("**" + func.args.kwarg.arg)
    ret = f" -> {ast.unparse(func.returns)}" if func.returns is not None else ""
    return f"({', '.join(args)}){ret}"


def _first_docline(func: ast.FunctionDef) -> str:
    ds = ast.get_docstring(func) or ""
    for line in ds.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:SNIPPET_MAX_CHARS]
    return ""


def build_utils_signature_index(utils_root: Optional[Path] = None) -> Dict[str, UtilEntry]:
    """Scan ``chrono_code/utils/*.py`` for public ``def``s and return
    a ``function_name → UtilEntry`` map.

    Skips functions starting with ``_`` (treated as private).
    """
    root = utils_root or UTILS_ROOT
    index: Dict[str, UtilEntry] = {}

    for py_file in sorted(root.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = f"chrono_code.utils.{py_file.stem}"
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                index.setdefault(node.name, UtilEntry(
                    module=module_name,
                    name=node.name,
                    signature=_format_signature(node),
                    docline=_first_docline(node),
                ))
    return index


# --- Aggregated cache ------------------------------------------------------


def build_full_index() -> Dict[str, any]:
    """Build both skill and utils indexes and return a single dict."""
    return {
        "skills": {
            symbol: [asdict(e) for e in entries]
            for symbol, entries in build_skill_index().items()
        },
        "utils": {
            name: asdict(entry)
            for name, entry in build_utils_signature_index().items()
        },
    }


def write_index_cache(path: Optional[Path] = None) -> Path:
    """Build + write the full index to ``.skill_index.json`` for cheap
    downstream reuse. Returns the path it wrote to."""
    target = path or INDEX_CACHE_PATH
    data = build_full_index()
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return target


def load_index_cache(path: Optional[Path] = None) -> Optional[Dict[str, any]]:
    """Load the cached index from ``.skill_index.json``. Returns None if
    the cache doesn't exist or is malformed."""
    target = path or INDEX_CACHE_PATH
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def get_or_build_index() -> Dict[str, any]:
    """Return a usable index, building + caching it if needed."""
    cached = load_index_cache()
    if cached is not None:
        return cached
    write_index_cache()
    result = load_index_cache()
    return result or {"skills": {}, "utils": {}}
